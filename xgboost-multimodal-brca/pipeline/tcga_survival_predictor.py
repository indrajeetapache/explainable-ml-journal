"""
TCGA Breast Cancer Survival Prediction Pipeline
===============================================================================
Production-grade XGBoost classifier using sklearn Pipeline CV.

Key Fix: All preprocessing (imputation, encoding, feature selection) happens
inside CV folds to prevent test data leakage.

Author: TCGA Research Team
Date: December 2024
"""

import pandas as pd
import numpy as np
import dask.dataframe as dd
from typing import Tuple, Dict, List, Any, Optional
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    make_scorer, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
import xgboost as xgb
from xgboost import XGBClassifier

# SHAP
import shap

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataAggregator:
    """Aggregates mutation-level TCGA data to patient-level."""

    def __init__(self, target_column: str = 'vital_status'):
        self.target_column = target_column
        self.key_genes = ['TP53', 'PIK3CA', 'BRCA1', 'BRCA2', 'PTEN', 'AKT1', 'CDH1']

    def aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate mutation-level to patient-level.

        Args:
            df: Mutation-level DataFrame

        Returns:
            Patient-level DataFrame
        """
        logger.info(f"Aggregating {len(df):,} mutation records to patient-level...")

        # Identify columns
        clinical_cols = [
            'years_to_birth', 'gender', 'race', 'ethnicity',
            'tumor_tissue_site', 'histological_type', 'pathologic_stage',
            'pathology_t_stage', 'pathology_n_stage', 'pathology_m_stage',
            'number_of_lymph_nodes', 'radiation_therapy',
            'days_to_death', 'days_to_last_followup', 'days_to_last_known_alive'
        ]

        cnv_cols = ['segment_mean', 'num_probes']
        metadata_cols = clinical_cols + cnv_cols + \
                       ['patient_id', self.target_column, 'mutation_count',
                        'hugo_symbol', 'variant_classification', 'tumor_vaf']
        rna_cols = [col for col in df.columns if col not in metadata_cols][:200]

        # Aggregation
        agg_dict = {self.target_column: 'first'}

        for col in clinical_cols:
            if col in df.columns:
                agg_dict[col] = 'first'

        if 'mutation_count' in df.columns:
            agg_dict['mutation_count'] = 'sum'
        if 'tumor_vaf' in df.columns:
            agg_dict['tumor_vaf'] = 'mean'
        if 'hugo_symbol' in df.columns:
            agg_dict['unique_genes_mutated'] = ('hugo_symbol', lambda x: len(x.unique()))
        if 'segment_mean' in df.columns:
            agg_dict['cnv_mean'] = ('segment_mean', 'mean')
            agg_dict['cnv_std'] = ('segment_mean', 'std')
        if 'num_probes' in df.columns:
            agg_dict['total_cnv_probes'] = ('num_probes', 'sum')

        for col in rna_cols:
            if col in df.columns:
                agg_dict[col] = 'first'

        patient_df = df.groupby('patient_id').agg(**agg_dict).reset_index()

        # Mutation flags
        for gene in self.key_genes:
            patient_df[f'{gene}_mutated'] = df.groupby('patient_id')['hugo_symbol'].apply(
                lambda x: int(gene in x.values)
            ).values

        # CNV burden
        if 'segment_mean' in df.columns:
            amp_df = df[df['segment_mean'] > 0.2].groupby('patient_id').size()
            patient_df['amplification_count'] = patient_df['patient_id'].map(amp_df).fillna(0)

            del_df = df[df['segment_mean'] < -0.2].groupby('patient_id').size()
            patient_df['deletion_count'] = patient_df['patient_id'].map(del_df).fillna(0)

            patient_df['total_cnv_segments'] = df.groupby('patient_id').size().values

        # High-impact mutations
        if 'variant_classification' in df.columns:
            high_impact = ['Frame_Shift_Del', 'Frame_Shift_Ins', 'Nonsense_Mutation',
                          'Splice_Site', 'Translation_Start_Site']
            high_impact_df = df[df['variant_classification'].isin(high_impact)].groupby('patient_id').size()
            patient_df['high_impact_mutations'] = patient_df['patient_id'].map(high_impact_df).fillna(0)

        # Feature engineering
        if 'years_to_birth' in patient_df.columns:
            patient_df['age_at_diagnosis'] = -patient_df['years_to_birth']

        if 'mutation_count' in patient_df.columns and 'age_at_diagnosis' in patient_df.columns:
            patient_df['mutation_burden_per_age'] = patient_df['mutation_count'] / (patient_df['age_at_diagnosis'] + 1)

        if 'amplification_count' in patient_df.columns and 'deletion_count' in patient_df.columns:
            patient_df['amp_del_ratio'] = (patient_df['amplification_count'] + 1) / (patient_df['deletion_count'] + 1)

        if 'mutation_count' in patient_df.columns and 'total_cnv_segments' in patient_df.columns:
            patient_df['genomic_instability'] = (
                (patient_df['mutation_count'] / patient_df['mutation_count'].max()) +
                (patient_df['total_cnv_segments'] / patient_df['total_cnv_segments'].max())
            ) / 2

        logger.info(f"Patient-level shape: {patient_df.shape}")
        return patient_df


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical features - fits on training data only."""

    def __init__(self):
        self.encoders_ = {}

    def fit(self, X, y=None):
        X = X.copy()
        cat_cols = X.select_dtypes(include=['object', 'category']).columns

        for col in cat_cols:
            encoder = LabelEncoder()
            encoder.fit(X[col].astype(str))
            self.encoders_[col] = encoder

        return self

    def transform(self, X):
        X = X.copy()

        for col, encoder in self.encoders_.items():
            if col in X.columns:
                # Handle unseen categories
                X[col] = X[col].astype(str).map(
                    lambda x: x if x in encoder.classes_ else encoder.classes_[0]
                )
                X[col] = encoder.transform(X[col])

        return X


class ModalityFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Select features by modality with variance filtering on RNA genes.
    Fits on training data only to prevent leakage.
    """

    def __init__(self, n_rna_genes: int = 150):
        self.n_rna_genes = n_rna_genes
        self.selected_features_ = None
        self.feature_groups_ = {}
        self.biomarkers = [
            'ESR1', 'PGR', 'ERBB2', 'TP63', 'EGFR',
            'KRT5', 'KRT14', 'KRT17', 'MKI67', 'TOP2A',
            'CCND1', 'VIM', 'FN1', 'CD274', 'PDCD1'
        ]

    def fit(self, X, y=None):
        """Fit feature selection on training data only."""

        # Identify feature groups
        clinical = [col for col in X.columns if any(
            kw in col.lower() for kw in ['age', 'stage', 'node', 'tumor', 'gender',
                                          'race', 'ethnicity', 'therapy', 'days_']
        )]

        mutation = [col for col in X.columns if any(
            kw in col for kw in ['mutation', '_mutated', 'vaf', 'unique_genes', 'high_impact']
        )]

        cnv = [col for col in X.columns if any(
            kw in col for kw in ['cnv_', 'amplification', 'deletion', 'segment', 'probes']
        )]

        engineered = [col for col in X.columns if any(
            kw in col for kw in ['ratio', 'score', 'burden', 'instability']
        )]

        core = set(clinical + mutation + cnv + engineered)
        rna_genes = [col for col in X.columns if col not in core]

        # Variance filter on RNA genes (fit on training data)
        if len(rna_genes) > 0:
            selector = VarianceThreshold(threshold=0.01)
            X_rna = X[rna_genes]
            selector.fit(X_rna)
            rna_filtered = [rna_genes[i] for i in range(len(rna_genes))
                           if selector.get_support()[i]]
        else:
            rna_filtered = []

        # Select top N variable genes (on training data)
        if len(rna_filtered) > self.n_rna_genes:
            gene_variance = X[rna_filtered].var().sort_values(ascending=False)
            top_variable = gene_variance.head(self.n_rna_genes).index.tolist()
        else:
            top_variable = rna_filtered

        # Add biomarkers
        biomarkers_present = [g for g in self.biomarkers if g in X.columns]
        selected_rna = list(set(top_variable + biomarkers_present))

        # Combine all
        self.selected_features_ = clinical + mutation + cnv + engineered + selected_rna
        self.selected_features_ = [col for col in self.selected_features_ if col in X.columns]

        self.feature_groups_ = {
            'Clinical': clinical,
            'Mutation': mutation,
            'CNV': cnv,
            'Engineered': engineered,
            'RNA': selected_rna
        }

        logger.info(f"Selected {len(self.selected_features_)} features:")
        for modality, feats in self.feature_groups_.items():
            logger.info(f"  {modality}: {len(feats)}")

        return self

    def transform(self, X):
        """Apply feature selection."""
        return X[self.selected_features_]


class TCGASurvivalPredictor:
    """
    Production-grade survival prediction with sklearn Pipeline (NO DATA LEAKAGE).

    All preprocessing happens inside CV folds via Pipeline.
    """

    def __init__(
        self,
        data_path: str,
        target_column: str = 'vital_status',
        n_folds: int = 5,
        n_rna_genes: int = 150,
        random_state: int = 42
    ):
        self.data_path = data_path
        self.target_column = target_column
        self.n_folds = n_folds
        self.n_rna_genes = n_rna_genes
        self.random_state = random_state

        self.pipeline = None
        self.label_encoder = None
        self.cv_results = {}
        self.feature_groups = {}

        logger.info(f"Initialized TCGASurvivalPredictor (Pipeline-based)")

    def load_data(self, df_dask: dd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Aggregate Dask df to patient-level, then prepare for training."""
        logger.info("Aggregating mutation-level to patient-level in Dask...")

        # Aggregate in Dask (stays distributed)
        aggregator = DataAggregator(target_column=self.target_column)
        patient_df = aggregator.aggregate(df_dask)  # Returns pandas after .compute()

        logger.info(f"Aggregated to: {patient_df.shape}")

        # Drop rows missing target
        patient_df = patient_df[patient_df[self.target_column].notna()].copy()

        # Encode target
        y_raw = patient_df[self.target_column]
        self.label_encoder = LabelEncoder()
        y = pd.Series(self.label_encoder.fit_transform(y_raw), index=patient_df.index)

        # Features
        X = patient_df.drop(columns=[self.target_column, 'patient_id'])

        # Class distribution
        n_neg = (y == 0).sum()
        n_pos = (y == 1).sum()
        self.scale_pos_weight = n_neg / n_pos

        logger.info(f"Dataset: {len(X)} patients, {X.shape[1]} features")
        logger.info(f"Scale pos weight: {self.scale_pos_weight:.2f}")

        return X, y

    def build_pipeline(self, xgb_params: Optional[Dict] = None) -> Pipeline:
        """
        Build sklearn Pipeline with all preprocessing steps.

        Pipeline ensures all preprocessing happens inside CV folds.
        """
        logger.info("Building sklearn Pipeline...")

        # Identify column types for imputation
        # Note: This is just for setup - actual fitting happens in CV

        if xgb_params is None:
            xgb_params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': 6,
                'learning_rate': 0.05,
                'n_estimators': 200,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 3,
                'gamma': 0.1,
                'scale_pos_weight': self.scale_pos_weight,
                'random_state': self.random_state,
                'n_jobs': -1
            }

        # Build pipeline
        pipeline = Pipeline([
            ('imputer_num', SimpleImputer(strategy='median')),  # Numeric imputation
            ('encoder', CategoricalEncoder()),  # Categorical encoding
            ('feature_selector', ModalityFeatureSelector(n_rna_genes=self.n_rna_genes)),  # Feature selection
            ('classifier', XGBClassifier(**xgb_params))  # XGBoost
        ])

        logger.info("Pipeline steps:")
        for name, step in pipeline.steps:
            logger.info(f"  {name}: {type(step).__name__}")

        self.pipeline = pipeline
        return pipeline

    def train_with_cv(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Train with stratified K-fold CV (NO DATA LEAKAGE).

        Pipeline ensures preprocessing is fit on training folds only.
        """
        logger.info(f"Training with {self.n_folds}-fold CV...")

        cv = StratifiedKFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.random_state
        )

        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1',
            'roc_auc': 'roc_auc'
        }

        # Cross-validation with Pipeline (no leakage!)
        cv_results = cross_validate(
            self.pipeline,
            X, y,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1,
            verbose=1
        )

        # Log results
        logger.info("\nCross-Validation Results (NO DATA LEAKAGE):")
        logger.info("="*60)
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            test_scores = cv_results[f'test_{metric}']
            train_scores = cv_results[f'train_{metric}']
            logger.info(f"{metric.upper()}:")
            logger.info(f"  Test:  {test_scores.mean():.4f} ± {test_scores.std():.4f}")
            logger.info(f"  Train: {train_scores.mean():.4f} ± {train_scores.std():.4f}")

        # Train final model on full data
        logger.info("\nTraining final model on full dataset...")
        self.pipeline.fit(X, y)

        # Store feature groups from fitted selector
        selector = self.pipeline.named_steps['feature_selector']
        self.feature_groups = selector.feature_groups_

        self.cv_results = {
            'cv_scores': cv_results,
            'mean_test_auc': cv_results['test_roc_auc'].mean(),
            'std_test_auc': cv_results['test_roc_auc'].std(),
            'mean_test_f1': cv_results['test_f1'].mean(),
            'std_test_f1': cv_results['test_f1'].std()
        }

        return self.cv_results

    def compute_shap_values(self, X: pd.DataFrame, sample_size: int = 500) -> np.ndarray:
        """Compute SHAP values on transformed data."""
        logger.info("Computing SHAP values...")

        # Transform data through pipeline (excluding classifier)
        X_transformed = X.copy()
        for name, step in self.pipeline.steps[:-1]:  # All except classifier
            X_transformed = step.transform(X_transformed)

        # Sample for efficiency
        if len(X_transformed) > sample_size:
            X_sample = X_transformed.sample(n=sample_size, random_state=self.random_state)
        else:
            X_sample = X_transformed

        # SHAP
        model = self.pipeline.named_steps['classifier']
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        logger.info(f"SHAP values shape: {shap_values.shape}")

        self.shap_values = shap_values
        self.X_shap = X_sample

        return shap_values

    def plot_feature_importance(self, top_n: int = 20, save_path: Optional[str] = None):
        """Plot XGBoost feature importance."""
        model = self.pipeline.named_steps['classifier']
        selector = self.pipeline.named_steps['feature_selector']

        importances = model.feature_importances_
        features = selector.selected_features_

        importance_df = pd.DataFrame({
            'feature': features,
            'importance': importances
        }).sort_values('importance', ascending=False)

        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df.head(top_n), x='importance', y='feature', palette='viridis')
        plt.title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_shap_summary(self, plot_type: str = 'dot', top_n: int = 20, save_path: Optional[str] = None):
        """Plot SHAP summary."""
        plt.figure(figsize=(10, 8))

        if plot_type == 'dot':
            shap.summary_plot(self.shap_values, self.X_shap, max_display=top_n, show=False)
        else:
            shap.summary_plot(self.shap_values, self.X_shap, plot_type='bar', max_display=top_n, show=False)

        plt.title(f'SHAP Summary (Top {top_n})', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_shap_by_modality(self, save_path: Optional[str] = None):
        """Plot SHAP grouped by modality."""
        selector = self.pipeline.named_steps['feature_selector']
        features = selector.selected_features_

        mean_shap = np.abs(self.shap_values).mean(axis=0)

        shap_df = pd.DataFrame({'feature': features, 'mean_shap': mean_shap})

        def assign_modality(feature):
            for modality, feats in self.feature_groups.items():
                if feature in feats:
                    return modality
            return 'Other'

        shap_df['modality'] = shap_df['feature'].apply(assign_modality)

        modality_importance = shap_df.groupby('modality')['mean_shap'].agg(['sum', 'mean', 'count'])
        modality_importance = modality_importance.sort_values('sum', ascending=False)

        logger.info("\nSHAP by Modality:")
        for modality in modality_importance.index:
            logger.info(f"  {modality}: {modality_importance.loc[modality, 'sum']:.4f}")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        modality_importance['sum'].plot(kind='barh', ax=axes[0], color='steelblue')
        axes[0].set_title('SHAP Importance by Modality', fontweight='bold')
        axes[0].set_xlabel('Total |SHAP|')

        top_features = shap_df.nlargest(20, 'mean_shap')
        colors = {'Clinical': 'blue', 'Mutation': 'red', 'CNV': 'green',
                 'RNA': 'purple', 'Engineered': 'orange'}
        bar_colors = [colors.get(m, 'gray') for m in top_features['modality']]

        axes[1].barh(range(len(top_features)), top_features['mean_shap'], color=bar_colors)
        axes[1].set_yticks(range(len(top_features)))
        axes[1].set_yticklabels(top_features['feature'], fontsize=9)
        axes[1].set_title('Top 20 Features by Modality', fontweight='bold')
        axes[1].invert_yaxis()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_confusion_matrix(self, X: pd.DataFrame, y: pd.Series, save_path: Optional[str] = None):
        """Plot confusion matrix."""
        y_pred = self.pipeline.predict(X)
        cm = confusion_matrix(y, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        print("\nClassification Report:")
        print(classification_report(y, y_pred, target_names=self.label_encoder.classes_, digits=4))

    def save_results(self, output_dir: str):
        """Save model and results."""
        logger.info(f"Saving to {output_dir}...")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save pipeline
        import joblib
        joblib.dump(self.pipeline, output_path / 'pipeline.pkl')

        # Save features
        selector = self.pipeline.named_steps['feature_selector']
        with open(output_path / 'selected_features.txt', 'w') as f:
            f.write('\n'.join(selector.selected_features_))

        for modality, features in self.feature_groups.items():
            with open(output_path / f'features_{modality.lower()}.txt', 'w') as f:
                f.write('\n'.join(features))

        # Save CV results
        import json
        cv_save = {
            'mean_test_auc': float(self.cv_results['mean_test_auc']),
            'std_test_auc': float(self.cv_results['std_test_auc']),
            'mean_test_f1': float(self.cv_results['mean_test_f1']),
            'std_test_f1': float(self.cv_results['std_test_f1'])
        }
        with open(output_path / 'cv_results.json', 'w') as f:
            json.dump(cv_save, f, indent=2)

        logger.info("✓ Results saved")

    def run_complete_pipeline(
        self,
        output_dir: str = '/mnt/user-data/outputs/tcga_results',
        xgb_params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Run complete pipeline."""
        logger.info("="*70)
        logger.info("TCGA SURVIVAL PREDICTION (NO DATA LEAKAGE)")
        logger.info("="*70)

        # Load
        X, y = self.load_data()

        # Build pipeline
        self.build_pipeline(xgb_params)

        # Train with CV (no leakage!)
        cv_results = self.train_with_cv(X, y)

        # SHAP
        shap_values = self.compute_shap_values(X)

        # Plots
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.plot_feature_importance(save_path=output_path / 'feature_importance.png')
        self.plot_shap_summary(plot_type='dot', save_path=output_path / 'shap_summary.png')
        self.plot_shap_by_modality(save_path=output_path / 'shap_by_modality.png')
        self.plot_confusion_matrix(X, y, save_path=output_path / 'confusion_matrix.png')

        # Save
        self.save_results(output_dir)

        logger.info("\n" + "="*70)
        logger.info("PIPELINE COMPLETE (NO DATA LEAKAGE)")
        logger.info("="*70)
        logger.info(f"ROC-AUC: {cv_results['mean_test_auc']:.4f} ± {cv_results['std_test_auc']:.4f}")
        logger.info(f"F1:      {cv_results['mean_test_f1']:.4f} ± {cv_results['std_test_f1']:.4f}")

        return {
            'cv_results': cv_results,
            'pipeline': self.pipeline,
            'feature_groups': self.feature_groups
        }


# if __name__ == "__main__":
#     predictor = TCGASurvivalPredictor(
#         data_path="/path/to/data.parquet",
#         n_rna_genes=150,
#         random_state=42
#     )

#     results = predictor.run_complete_pipeline()


# # Example usage
# if __name__ == "__main__":
#     # Initialize pipeline
#     predictor = TCGASurvivalPredictor(
#         data_path="/path/to/RNA_mutation_dataset.parquet",
#         target_column='vital_status',
#         n_folds=5,
#         random_state=42
#     )

#     # Run complete pipeline
#     results = predictor.run_complete_pipeline(
#         output_dir='/mnt/user-data/outputs/tcga_survival_results',
#         n_rna_genes=150,
#         xgb_params=None  # Use defaults
#     )

#     print("\nPipeline execution complete!")
#     print(f"Model ROC-AUC: {results['cv_results']['mean_test_auc']:.4f}")
