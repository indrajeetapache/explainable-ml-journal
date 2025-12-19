import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import sys
import os
import argparse

# Add pipeline folder to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

try:
    from custom_transformers import CategoricalEncoder, ModalityFeatureSelector
except ImportError:
    from pipeline.custom_transformers import CategoricalEncoder, ModalityFeatureSelector

warnings.filterwarnings('ignore')

class TCGASurvivalPredictor:
    def __init__(self, processed_data_path: str, target_column='vital_status', random_state=42):
        self.data_path = processed_data_path
        self.target_column = target_column
        self.random_state = random_state
        self.pipeline = None
        
    def load_and_prep(self):
        print(f"📂  Loading data from: {self.data_path}")
        df = pd.read_parquet(self.data_path)
        
        # Ensure target exists
        initial_len = len(df)
        df = df[df[self.target_column].notna()].copy()
        print(f"    Dropped {initial_len - len(df)} rows with missing target.")
        
        # Encode Target
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(df[self.target_column])
        X = df.drop(columns=[self.target_column, 'patient_id'])
        
        # Print mapping
        mapping = dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))
        print(f"    Target Mapping: {mapping}")
        
        # Calculate Class Weights
        n_neg, n_pos = (y == 0).sum(), (y == 1).sum()
        self.scale_pos_weight = n_neg / n_pos
        print(f"    Class Imbalance: {n_pos} Positive / {n_neg} Negative (Scale Weight: {self.scale_pos_weight:.2f})")
        
        return X, pd.Series(y, index=df.index)

    def run(self, output_dir: str):
        print("\n" + "="*60)
        print(f"🚀  STARTING STEP 2: MODEL TRAINING")
        print("="*60)
        
        X, y = self.load_and_prep()
        
        # 1. Define Pipeline
        print(f"\n⚙️   Building XGBoost Pipeline...")
        xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'scale_pos_weight': self.scale_pos_weight,
            'random_state': self.random_state,
            'n_jobs': -1
        }
        
        self.pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('encoder', CategoricalEncoder()),
            ('selector', ModalityFeatureSelector(n_rna_genes=150)),
            ('clf', XGBClassifier(**xgb_params))
        ])

        # 2. Cross Validation
        print(f"🔄  Running 5-Fold Cross-Validation (Stratified)...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        scores = cross_validate(self.pipeline, X, y, cv=cv, scoring=['roc_auc', 'f1', 'accuracy'], return_train_score=True)
        
        print("-" * 40)
        print(f"    Avg Test AUC:      {scores['test_roc_auc'].mean():.4f}")
        print(f"    Avg Test F1 Score: {scores['test_f1'].mean():.4f}")
        print(f"    Avg Test Accuracy: {scores['test_accuracy'].mean():.4f}")
        print("-" * 40)

        # 3. Final Fit & SHAP
        print(f"\n🧠  Retraining on full dataset & computing SHAP explanations...")
        self.pipeline.fit(X, y)
        model = self.pipeline.named_steps['clf']
        
        # Transform data manually for SHAP
        X_trans = X.copy()
        for name, step in self.pipeline.steps[:-1]:
            X_trans = step.transform(X_trans)
            
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_trans)

        # 4. Save & Visualize
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾  Saving results to: {out_path}")
        
        # SHAP Summary Plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_trans, show=False)
        plt.savefig(out_path / 'shap_summary.png', bbox_inches='tight')
        print(f"    Saved: shap_summary.png")
        
        # Save Features List
        selector = self.pipeline.named_steps['selector']
        with open(out_path / 'selected_features.txt', 'w') as f:
            f.write('\n'.join(selector.selected_features_))
        print(f"    Saved: selected_features.txt")
            
        print("="*60)
        print(f"✅  STEP 2 COMPLETE. Check the 'results' folder.")
        print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to processed parquet file')
    parser.add_argument('--output', type=str, required=True, help='Directory to save results')
    args = parser.parse_args()
    
    predictor = TCGASurvivalPredictor(args.input)
    predictor.run(args.output)
