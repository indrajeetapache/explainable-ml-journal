"""
TCGA Breast Cancer Survival Prediction Pipeline
================================================
Production-grade XGBoost classifier for predicting patient vital status (LIVING/DECEASED)
using multimodal genomic data: Clinical, Mutation, CNV, and RNA-seq features.

Author: TCGA Research Team
Date: December 2024
Purpose: Domain-aware SHAP analysis for cross-modality feature importance testing
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
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, RFECV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import xgboost as xgb
from xgboost import XGBClassifier

# SHAP for interpretability
import shap

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TCGASurvivalPredictor:
    """
    Production-grade XGBoost classifier for TCGA breast cancer survival prediction.
    
    Pipeline Components:
    1. Data aggregation from mutation-level to patient-level
    2. Feature engineering across 4 modalities (Clinical, Mutation, CNV, RNA-seq)
    3. Missing data imputation with domain-appropriate strategies
    4. Multi-stage feature selection (variance + RFE + domain knowledge)
    5. Stratified K-fold cross-validation with class imbalance handling
    6. SHAP-based interpretability analysis
    7. Comprehensive performance evaluation and visualization
    """
    
    def __init__(
        self,
        data_path: str,
        target_column: str = 'vital_status',
        n_folds: int = 5,
        random_state: int = 42
    ):
        """
        Initialize the survival prediction pipeline.
        
        Args:
            data_path: Path to parquet file with mutation-level TCGA data
            target_column: Name of target variable (default: 'vital_status')
            n_folds: Number of CV folds (default: 5)
            random_state: Random seed for reproducibility (default: 42)
        """
        self.data_path = data_path
        self.target_column = target_column
        self.n_folds = n_folds
        self.random_state = random_state
        
        # Pipeline components (initialized during fit)
        self.model = None
        self.label_encoder = None
        self.imputers = {}
        self.selected_features = None
        self.feature_groups = {}  # Track features by modality
        self.cv_results = {}
        self.shap_values = None
        
        # Cancer biomarkers for domain-guided feature selection
        self.cancer_biomarkers = [
            'ESR1', 'PGR', 'ERBB2',  # Hormone receptors
            'TP63', 'EGFR',  # Basal markers
            'KRT5', 'KRT14', 'KRT17',  # Basal cytokeratins
            'MKI67', 'TOP2A', 'CCND1',  # Proliferation
            'VIM', 'FN1',  # Mesenchymal
            'CD274', 'PDCD1'  # Immune checkpoint
        ]
        
        logger.info(f"Initialized TCGASurvivalPredictor")
        logger.info(f"Data: {data_path}")
        logger.info(f"Target: {target_column}, CV folds: {n_folds}")
    
    
    def load_and_aggregate_data(self) -> pd.DataFrame:
        """
        Load mutation-level data and aggregate to patient-level.
        
        Strategy:
        - Clinical features: Take first value (identical across mutation rows)
        - Mutation features: Sum counts, average VAF, create gene-specific flags
        - CNV features: Aggregate segments (mean, std, burden counts)
        - RNA-seq features: Take first value (identical per patient sample)
        
        Returns:
            Patient-level DataFrame (977 patients × ~180+ features)
        
        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If required columns missing
        """
        logger.info("Loading mutation-level data from parquet...")
        
        try:
            # Load with Dask for memory efficiency
            df_dask = dd.read_parquet(self.data_path)
            
            # Convert to pandas for aggregation (manageable size after groupby)
            df = df_dask.compute()
            logger.info(f"Loaded {len(df):,} mutation records")
            
        except FileNotFoundError:
            logger.error(f"Data file not found: {self.data_path}")
            raise
        
        # Validate required columns exist
        required_cols = ['patient_id', self.target_column, 'hugo_symbol']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        logger.info("Aggregating to patient-level...")
        n_patients_before = df['patient_id'].nunique()
        
        # Identify column types for proper aggregation
        clinical_cols = [
            'years_to_birth', 'gender', 'race', 'ethnicity',
            'tumor_tissue_site', 'histological_type', 'pathologic_stage',
            'pathology_t_stage', 'pathology_n_stage', 'pathology_m_stage',
            'number_of_lymph_nodes', 'radiation_therapy',
            'days_to_death', 'days_to_last_followup', 'days_to_last_known_alive'
        ]
        
        mutation_cols = ['hugo_symbol', 'variant_classification', 'variant_type', 'tumor_vaf']
        cnv_cols = ['segment_mean', 'num_probes', 'start', 'end']
        
        # Get RNA gene columns (all columns not in other categories)
        metadata_cols = clinical_cols + mutation_cols + cnv_cols + \
                       ['patient_id', self.target_column, 'mutation_count',
                        'patient_id_chromosome', 'chromosome_x', 'chromosome_y']
        rna_cols = [col for col in df.columns if col not in metadata_cols]
        
        logger.info(f"Identified {len(rna_cols)} RNA expression genes")
        
        # Build aggregation dictionary
        agg_dict = {self.target_column: 'first'}
        
        # Clinical: take first (should be identical per patient)
        for col in clinical_cols:
            if col in df.columns:
                agg_dict[col] = 'first'
        
        # Mutation features
        if 'mutation_count' in df.columns:
            agg_dict['mutation_count'] = 'sum'
        if 'tumor_vaf' in df.columns:
            agg_dict['tumor_vaf'] = 'mean'
        if 'hugo_symbol' in df.columns:
            agg_dict['unique_genes_mutated'] = ('hugo_symbol', lambda x: len(x.unique()))
        
        # CNV features
        if 'segment_mean' in df.columns:
            agg_dict['cnv_mean'] = ('segment_mean', 'mean')
            agg_dict['cnv_std'] = ('segment_mean', 'std')
        if 'num_probes' in df.columns:
            agg_dict['total_cnv_probes'] = ('num_probes', 'sum')
        
        # RNA-seq: take first (identical per patient)
        for col in rna_cols[:200]:  # Limit to top 200 to avoid excessive features initially
            if col in df.columns:
                agg_dict[col] = 'first'
        
        # Perform aggregation
        patient_df = df.groupby('patient_id').agg(**agg_dict).reset_index()
        
        logger.info(f"Aggregated {n_patients_before} patients")
        logger.info(f"Patient-level data shape: {patient_df.shape}")
        
        # Create mutation flags for key cancer genes
        logger.info("Creating gene-specific mutation flags...")
        key_genes = ['TP53', 'PIK3CA', 'BRCA1', 'BRCA2', 'PTEN', 'AKT1', 'CDH1']
        
        for gene in key_genes:
            flag_col = f'{gene}_mutated'
            patient_df[flag_col] = df.groupby('patient_id')['hugo_symbol'].apply(
                lambda x: int(gene in x.values)
            ).values
            logger.info(f"  {gene}: {patient_df[flag_col].sum()} patients mutated")
        
        # Create CNV burden features
        logger.info("Computing CNV burden metrics...")
        if 'segment_mean' in df.columns:
            # Amplifications (segment_mean > 0.2)
            amp_df = df[df['segment_mean'] > 0.2].groupby('patient_id').size()
            patient_df['amplification_count'] = patient_df['patient_id'].map(amp_df).fillna(0)
            
            # Deletions (segment_mean < -0.2)
            del_df = df[df['segment_mean'] < -0.2].groupby('patient_id').size()
            patient_df['deletion_count'] = patient_df['patient_id'].map(del_df).fillna(0)
            
            # Total CNV segments
            seg_df = df.groupby('patient_id').size()
            patient_df['total_cnv_segments'] = patient_df['patient_id'].map(seg_df).fillna(0)
            
            logger.info(f"  Mean amplifications: {patient_df['amplification_count'].mean():.1f}")
            logger.info(f"  Mean deletions: {patient_df['deletion_count'].mean():.1f}")
        
        # Create high-impact mutation count
        if 'variant_classification' in df.columns:
            high_impact = ['Frame_Shift_Del', 'Frame_Shift_Ins', 'Nonsense_Mutation',
                          'Splice_Site', 'Translation_Start_Site']
            high_impact_df = df[df['variant_classification'].isin(high_impact)].groupby('patient_id').size()
            patient_df['high_impact_mutations'] = patient_df['patient_id'].map(high_impact_df).fillna(0)
            
            logger.info(f"  Mean high-impact mutations: {patient_df['high_impact_mutations'].mean():.1f}")
        
        logger.info(f"Final patient-level features: {patient_df.shape[1]}")
        
        return patient_df
    
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer additional features from aggregated patient data.
        
        Creates:
        - Age at diagnosis (from years_to_birth)
        - TNM stage combinations
        - Mutation/CNV burden ratios
        - RNA expression ratios for biomarker panels
        
        Args:
            df: Patient-level DataFrame
        
        Returns:
            DataFrame with engineered features added
        """
        logger.info("Engineering additional features...")
        df = df.copy()
        
        # Age at diagnosis (TCGA uses negative years_to_birth)
        if 'years_to_birth' in df.columns:
            df['age_at_diagnosis'] = -df['years_to_birth']
            logger.info(f"  Age range: {df['age_at_diagnosis'].min():.0f}-{df['age_at_diagnosis'].max():.0f}")
        
        # Mutation burden per age (normalized)
        if 'mutation_count' in df.columns and 'age_at_diagnosis' in df.columns:
            df['mutation_burden_per_age'] = df['mutation_count'] / (df['age_at_diagnosis'] + 1)
        
        # CNV burden ratio (amplifications vs deletions)
        if 'amplification_count' in df.columns and 'deletion_count' in df.columns:
            df['amp_del_ratio'] = (df['amplification_count'] + 1) / (df['deletion_count'] + 1)
        
        # Genomic instability score (combined mutation + CNV)
        if 'mutation_count' in df.columns and 'total_cnv_segments' in df.columns:
            df['genomic_instability'] = (
                (df['mutation_count'] / df['mutation_count'].max()) +
                (df['total_cnv_segments'] / df['total_cnv_segments'].max())
            ) / 2
        
        # ER/HER2 expression ratio (luminal vs HER2-enriched)
        if 'ESR1' in df.columns and 'ERBB2' in df.columns:
            df['ESR1_ERBB2_ratio'] = (df['ESR1'] + 0.1) / (df['ERBB2'] + 0.1)
        
        # Proliferation score (MKI67 + TOP2A + CCND1)
        proliferation_genes = ['MKI67', 'TOP2A', 'CCND1']
        if all(gene in df.columns for gene in proliferation_genes):
            df['proliferation_score'] = df[proliferation_genes].mean(axis=1)
        
        logger.info(f"Added engineered features, new shape: {df.shape}")
        
        return df
    
    
    def handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values with domain-appropriate imputation strategies.
        
        Strategy:
        - Clinical categorical: Mode imputation
        - Clinical numerical: Median imputation
        - Mutation features: Fill 0 (no mutation detected)
        - CNV features: Fill 0 (no CNV detected)
        - RNA expression: Median imputation (biological variation)
        - Drop patients missing target variable
        
        Args:
            df: Patient-level DataFrame
        
        Returns:
            DataFrame with missing values imputed
        """
        logger.info("Handling missing data...")
        df = df.copy()
        
        # Check initial missingness
        missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
        high_missing = missing_pct[missing_pct > 30]
        if len(high_missing) > 0:
            logger.warning(f"  {len(high_missing)} features with >30% missing:")
            for col, pct in high_missing.head().items():
                logger.warning(f"    {col}: {pct:.1f}%")
        
        # Drop rows missing target variable
        if df[self.target_column].isnull().any():
            n_missing = df[self.target_column].isnull().sum()
            logger.warning(f"  Dropping {n_missing} patients with missing target")
            df = df[df[self.target_column].notna()]
        
        # Identify column types
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target and ID from numerical
        numerical_cols = [col for col in numerical_cols if col not in [self.target_column, 'patient_id']]
        
        # Mutation features: Fill 0
        mutation_features = [col for col in df.columns if 
                            '_mutated' in col or 'mutation_' in col or 
                            'high_impact' in col]
        for col in mutation_features:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # CNV features: Fill 0
        cnv_features = [col for col in df.columns if 
                       'cnv_' in col or 'amplification' in col or 
                       'deletion' in col or 'segment' in col]
        for col in cnv_features:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # Categorical imputation (mode)
        cat_to_impute = [col for col in categorical_cols if 
                        col not in [self.target_column, 'patient_id'] and 
                        df[col].isnull().any()]
        
        if cat_to_impute:
            logger.info(f"  Imputing {len(cat_to_impute)} categorical features (mode)")
            self.imputers['categorical'] = SimpleImputer(strategy='most_frequent')
            df[cat_to_impute] = self.imputers['categorical'].fit_transform(df[cat_to_impute])
        
        # Numerical imputation (median)
        num_to_impute = [col for col in numerical_cols if 
                        col not in mutation_features + cnv_features and 
                        df[col].isnull().any()]
        
        if num_to_impute:
            logger.info(f"  Imputing {len(num_to_impute)} numerical features (median)")
            self.imputers['numerical'] = SimpleImputer(strategy='median')
            df[num_to_impute] = self.imputers['numerical'].fit_transform(df[num_to_impute])
        
        # Final check
        remaining_missing = df.isnull().sum().sum()
        logger.info(f"  Remaining missing values: {remaining_missing}")
        
        return df
    
    
    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_rna_genes: int = 150
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Multi-stage feature selection strategy.
        
        Stage 1: Variance threshold (remove near-zero variance)
        Stage 2: Select top N most variable RNA genes
        Stage 3: Add cancer biomarkers (domain knowledge)
        Stage 4: Keep all clinical, mutation, CNV features
        Stage 5: Recursive Feature Elimination (optional refinement)
        
        Args:
            X: Feature matrix
            y: Target variable
            n_rna_genes: Number of RNA genes to keep (default: 150)
        
        Returns:
            Tuple of (selected features DataFrame, feature names list)
        """
        logger.info("Performing feature selection...")
        
        # Identify feature groups
        clinical_features = [col for col in X.columns if any(
            kw in col.lower() for kw in ['age', 'stage', 'node', 'tumor', 'gender', 
                                          'race', 'ethnicity', 'therapy', 'days_']
        )]
        
        mutation_features = [col for col in X.columns if any(
            kw in col for kw in ['mutation', '_mutated', 'vaf', 'unique_genes', 'high_impact']
        )]
        
        cnv_features = [col for col in X.columns if any(
            kw in col for kw in ['cnv_', 'amplification', 'deletion', 'segment', 'probes']
        )]
        
        engineered_features = [col for col in X.columns if any(
            kw in col for kw in ['ratio', 'score', 'burden', 'instability']
        )]
        
        # Remaining columns are RNA genes
        core_features = set(clinical_features + mutation_features + cnv_features + engineered_features)
        rna_genes = [col for col in X.columns if col not in core_features]
        
        logger.info(f"Feature groups identified:")
        logger.info(f"  Clinical: {len(clinical_features)}")
        logger.info(f"  Mutation: {len(mutation_features)}")
        logger.info(f"  CNV: {len(cnv_features)}")
        logger.info(f"  Engineered: {len(engineered_features)}")
        logger.info(f"  RNA genes: {len(rna_genes)}")
        
        # Store feature groups
        self.feature_groups = {
            'Clinical': clinical_features,
            'Mutation': mutation_features,
            'CNV': cnv_features,
            'Engineered': engineered_features,
            'RNA': []  # Will be populated after selection
        }
        
        # Stage 1: Variance threshold on RNA genes only
        if len(rna_genes) > 0:
            logger.info(f"Stage 1: Variance filtering on {len(rna_genes)} RNA genes...")
            selector = VarianceThreshold(threshold=0.01)
            X_rna = X[rna_genes]
            selector.fit(X_rna)
            rna_genes_filtered = [rna_genes[i] for i in range(len(rna_genes)) 
                                 if selector.get_support()[i]]
            logger.info(f"  Retained {len(rna_genes_filtered)} genes after variance filter")
        else:
            rna_genes_filtered = []
        
        # Stage 2: Select top N most variable RNA genes
        if len(rna_genes_filtered) > n_rna_genes:
            logger.info(f"Stage 2: Selecting top {n_rna_genes} most variable genes...")
            gene_variance = X[rna_genes_filtered].var().sort_values(ascending=False)
            top_variable_genes = gene_variance.head(n_rna_genes).index.tolist()
            logger.info(f"  Selected {len(top_variable_genes)} variable genes")
        else:
            top_variable_genes = rna_genes_filtered
        
        # Stage 3: Add cancer biomarkers (force include if present)
        logger.info("Stage 3: Adding cancer biomarkers...")
        biomarkers_present = [gene for gene in self.cancer_biomarkers if gene in X.columns]
        logger.info(f"  Found {len(biomarkers_present)} biomarkers: {biomarkers_present}")
        
        # Combine RNA features (top variable + biomarkers)
        selected_rna = list(set(top_variable_genes + biomarkers_present))
        logger.info(f"  Total RNA genes selected: {len(selected_rna)}")
        
        # Update feature groups
        self.feature_groups['RNA'] = selected_rna
        
        # Stage 4: Combine all selected features
        selected_features = (clinical_features + mutation_features + 
                           cnv_features + engineered_features + selected_rna)
        
        # Remove duplicates while preserving order
        selected_features = list(dict.fromkeys(selected_features))
        
        # Filter to columns that actually exist in X
        selected_features = [col for col in selected_features if col in X.columns]
        
        logger.info(f"Total features selected: {len(selected_features)}")
        logger.info(f"  Clinical: {len(clinical_features)}")
        logger.info(f"  Mutation: {len(mutation_features)}")
        logger.info(f"  CNV: {len(cnv_features)}")
        logger.info(f"  Engineered: {len(engineered_features)}")
        logger.info(f"  RNA: {len(selected_rna)}")
        
        self.selected_features = selected_features
        
        return X[selected_features], selected_features
    
    
    def prepare_target(self, df: pd.DataFrame) -> Tuple[pd.Series, Dict[str, int]]:
        """
        Encode target variable to binary format.
        
        Args:
            df: DataFrame containing target column
        
        Returns:
            Tuple of (encoded target series, label mapping dict)
        """
        logger.info(f"Encoding target variable: {self.target_column}")
        
        y = df[self.target_column].copy()
        
        # Check unique values
        unique_vals = y.unique()
        logger.info(f"  Unique values: {unique_vals}")
        
        # Encode to binary (0=LIVING, 1=DECEASED)
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Create mapping
        label_mapping = dict(zip(
            self.label_encoder.classes_,
            self.label_encoder.transform(self.label_encoder.classes_)
        ))
        
        logger.info(f"  Label mapping: {label_mapping}")
        
        # Check class distribution
        class_counts = pd.Series(y_encoded).value_counts()
        logger.info(f"  Class distribution:")
        for label, encoded in label_mapping.items():
            count = class_counts.get(encoded, 0)
            pct = count / len(y_encoded) * 100
            logger.info(f"    {label} ({encoded}): {count} ({pct:.1f}%)")
        
        # Calculate scale_pos_weight for XGBoost
        n_neg = class_counts.get(0, 1)
        n_pos = class_counts.get(1, 1)
        scale_pos_weight = n_neg / n_pos
        logger.info(f"  XGBoost scale_pos_weight: {scale_pos_weight:.2f}")
        
        return pd.Series(y_encoded, index=df.index), label_mapping
    
    
    def encode_categorical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features using Label Encoding.
        
        Args:
            X: Feature matrix
        
        Returns:
            DataFrame with encoded categorical features
        """
        logger.info("Encoding categorical features...")
        X = X.copy()
        
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if len(categorical_cols) > 0:
            logger.info(f"  Found {len(categorical_cols)} categorical features")
            
            for col in categorical_cols:
                encoder = LabelEncoder()
                X[col] = encoder.fit_transform(X[col].astype(str))
                logger.info(f"    {col}: {len(encoder.classes_)} classes")
        
        return X
    
    
    def train_with_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        xgb_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Train XGBoost model with stratified K-fold cross-validation.
        
        Args:
            X: Feature matrix
            y: Target variable (encoded)
            xgb_params: XGBoost hyperparameters (optional)
        
        Returns:
            Dictionary containing CV results and trained model
        """
        logger.info(f"Training XGBoost with {self.n_folds}-fold CV...")
        
        # Calculate scale_pos_weight for class imbalance
        n_neg = (y == 0).sum()
        n_pos = (y == 1).sum()
        scale_pos_weight = n_neg / n_pos
        
        # Default XGBoost parameters
        if xgb_params is None:
            xgb_params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'scale_pos_weight': scale_pos_weight,
                'random_state': self.random_state,
                'n_jobs': -1
            }
        
        logger.info(f"XGBoost parameters:")
        for param, value in xgb_params.items():
            logger.info(f"  {param}: {value}")
        
        # Initialize model
        self.model = XGBClassifier(**xgb_params)
        
        # Stratified K-fold
        cv = StratifiedKFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.random_state
        )
        
        # Scoring metrics
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1',
            'roc_auc': 'roc_auc'
        }
        
        # Perform cross-validation
        logger.info("Running cross-validation...")
        cv_results = cross_validate(
            self.model,
            X, y,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1,
            verbose=0
        )
        
        # Log results
        logger.info("\nCross-Validation Results:")
        logger.info("=" * 60)
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            test_scores = cv_results[f'test_{metric}']
            train_scores = cv_results[f'train_{metric}']
            logger.info(f"{metric.upper()}:")
            logger.info(f"  Test:  {test_scores.mean():.4f} ± {test_scores.std():.4f}")
            logger.info(f"  Train: {train_scores.mean():.4f} ± {train_scores.std():.4f}")
        
        # Train final model on full dataset
        logger.info("\nTraining final model on full dataset...")
        self.model.fit(X, y)
        
        # Store results
        self.cv_results = {
            'cv_scores': cv_results,
            'mean_test_auc': cv_results['test_roc_auc'].mean(),
            'std_test_auc': cv_results['test_roc_auc'].std(),
            'mean_test_f1': cv_results['test_f1'].mean(),
            'std_test_f1': cv_results['test_f1'].std()
        }
        
        return self.cv_results
    
    
    def compute_shap_values(
        self,
        X: pd.DataFrame,
        sample_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute SHAP values for model interpretability.
        
        Uses TreeExplainer for efficient SHAP computation on XGBoost models.
        
        Args:
            X: Feature matrix
            sample_size: Number of samples for SHAP (default: all, or 500 if >500)
        
        Returns:
            SHAP values array (n_samples × n_features)
        """
        logger.info("Computing SHAP values...")
        
        if self.model is None:
            raise ValueError("Model not trained. Call train_with_cv() first.")
        
        # Sample data if too large (SHAP can be slow)
        if sample_size is None:
            sample_size = min(500, len(X))
        
        if len(X) > sample_size:
            logger.info(f"  Sampling {sample_size} instances for SHAP computation")
            X_sample = X.sample(n=sample_size, random_state=self.random_state)
        else:
            X_sample = X
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(self.model)
        
        # Compute SHAP values
        logger.info("  Computing SHAP values (this may take a few minutes)...")
        shap_values = explainer.shap_values(X_sample)
        
        self.shap_values = shap_values
        
        logger.info(f"  SHAP values shape: {shap_values.shape}")
        
        return shap_values
    
    
    def plot_feature_importance(
        self,
        top_n: int = 20,
        save_path: Optional[str] = None
    ):
        """
        Plot feature importance from XGBoost model.
        
        Args:
            top_n: Number of top features to display (default: 20)
            save_path: Path to save plot (optional)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_with_cv() first.")
        
        logger.info(f"Plotting top {top_n} feature importances...")
        
        # Get feature importances
        importances = self.model.feature_importances_
        feature_names = self.selected_features
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.barplot(
            data=importance_df.head(top_n),
            x='importance',
            y='feature',
            palette='viridis'
        )
        plt.title(f'Top {top_n} Feature Importances (XGBoost)', fontsize=14, fontweight='bold')
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"  Saved to {save_path}")
        
        plt.show()
    
    
    def plot_shap_summary(
        self,
        X: pd.DataFrame,
        plot_type: str = 'dot',
        top_n: int = 20,
        save_path: Optional[str] = None
    ):
        """
        Plot SHAP summary showing feature importance and impact.
        
        Args:
            X: Feature matrix used for SHAP computation
            plot_type: 'dot' or 'bar' (default: 'dot')
            top_n: Number of features to display (default: 20)
            save_path: Path to save plot (optional)
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call compute_shap_values() first.")
        
        logger.info(f"Plotting SHAP summary ({plot_type})...")
        
        plt.figure(figsize=(10, 8))
        
        if plot_type == 'dot':
            shap.summary_plot(
                self.shap_values,
                X.iloc[:len(self.shap_values)],
                max_display=top_n,
                show=False
            )
        elif plot_type == 'bar':
            shap.summary_plot(
                self.shap_values,
                X.iloc[:len(self.shap_values)],
                plot_type='bar',
                max_display=top_n,
                show=False
            )
        
        plt.title(f'SHAP Feature Importance (Top {top_n})', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"  Saved to {save_path}")
        
        plt.show()
    
    
    def plot_shap_by_modality(
        self,
        X: pd.DataFrame,
        save_path: Optional[str] = None
    ):
        """
        Plot SHAP importance grouped by data modality (Clinical, Mutation, CNV, RNA).
        
        This visualization supports domain-aware SHAP hypothesis testing.
        
        Args:
            X: Feature matrix
            save_path: Path to save plot (optional)
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call compute_shap_values() first.")
        
        logger.info("Plotting SHAP importance by modality...")
        
        # Compute mean absolute SHAP values per feature
        mean_shap = np.abs(self.shap_values).mean(axis=0)
        
        # Create DataFrame with modality labels
        shap_df = pd.DataFrame({
            'feature': self.selected_features,
            'mean_shap': mean_shap
        })
        
        # Assign modality labels
        def assign_modality(feature):
            if feature in self.feature_groups['Clinical']:
                return 'Clinical'
            elif feature in self.feature_groups['Mutation']:
                return 'Mutation'
            elif feature in self.feature_groups['CNV']:
                return 'CNV'
            elif feature in self.feature_groups['RNA']:
                return 'RNA-seq'
            elif feature in self.feature_groups['Engineered']:
                return 'Engineered'
            else:
                return 'Other'
        
        shap_df['modality'] = shap_df['feature'].apply(assign_modality)
        
        # Aggregate by modality
        modality_importance = shap_df.groupby('modality')['mean_shap'].agg(['sum', 'mean', 'count'])
        modality_importance = modality_importance.sort_values('sum', ascending=False)
        
        logger.info("\nSHAP Importance by Modality:")
        logger.info("=" * 60)
        for modality in modality_importance.index:
            total = modality_importance.loc[modality, 'sum']
            mean = modality_importance.loc[modality, 'mean']
            count = modality_importance.loc[modality, 'count']
            logger.info(f"{modality:12s}: Total={total:.4f}, Mean={mean:.6f}, N={int(count)}")
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Total importance by modality
        modality_importance['sum'].plot(kind='barh', ax=axes[0], color='steelblue')
        axes[0].set_title('Total SHAP Importance by Modality', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Total |SHAP|', fontsize=11)
        axes[0].set_ylabel('Modality', fontsize=11)
        
        # Plot 2: Top features colored by modality
        top_features = shap_df.nlargest(20, 'mean_shap')
        colors = {'Clinical': 'blue', 'Mutation': 'red', 'CNV': 'green', 
                 'RNA-seq': 'purple', 'Engineered': 'orange'}
        bar_colors = [colors.get(m, 'gray') for m in top_features['modality']]
        
        axes[1].barh(range(len(top_features)), top_features['mean_shap'], color=bar_colors)
        axes[1].set_yticks(range(len(top_features)))
        axes[1].set_yticklabels(top_features['feature'], fontsize=9)
        axes[1].set_title('Top 20 Features (Colored by Modality)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Mean |SHAP|', fontsize=11)
        axes[1].invert_yaxis()
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=modality) 
                          for modality, color in colors.items()]
        axes[1].legend(handles=legend_elements, loc='lower right', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"  Saved to {save_path}")
        
        plt.show()
    
    
    def plot_confusion_matrix(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        save_path: Optional[str] = None
    ):
        """
        Plot confusion matrix for model predictions.
        
        Args:
            X: Feature matrix
            y: True labels
            save_path: Path to save plot (optional)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_with_cv() first.")
        
        logger.info("Plotting confusion matrix...")
        
        # Get predictions
        y_pred = self.model.predict(X)
        
        # Compute confusion matrix
        cm = confusion_matrix(y, y_pred)
        
        # Plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_,
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"  Saved to {save_path}")
        
        plt.show()
        
        # Print classification report
        logger.info("\nClassification Report:")
        logger.info("=" * 60)
        print(classification_report(
            y, y_pred,
            target_names=self.label_encoder.classes_,
            digits=4
        ))
    
    
    def save_results(
        self,
        output_dir: str,
        save_model: bool = True
    ):
        """
        Save model, results, and feature importance to files.
        
        Args:
            output_dir: Directory to save outputs
            save_model: Whether to save trained model (default: True)
        """
        logger.info(f"Saving results to {output_dir}...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        if save_model and self.model is not None:
            model_path = output_path / 'xgboost_model.json'
            self.model.save_model(str(model_path))
            logger.info(f"  Model saved to {model_path}")
        
        # Save selected features
        if self.selected_features is not None:
            features_path = output_path / 'selected_features.txt'
            with open(features_path, 'w') as f:
                f.write('\n'.join(self.selected_features))
            logger.info(f"  Features saved to {features_path}")
        
        # Save feature groups
        if self.feature_groups:
            for modality, features in self.feature_groups.items():
                group_path = output_path / f'features_{modality.lower()}.txt'
                with open(group_path, 'w') as f:
                    f.write('\n'.join(features))
            logger.info(f"  Feature groups saved")
        
        # Save CV results
        if self.cv_results:
            import json
            cv_path = output_path / 'cv_results.json'
            
            # Convert numpy arrays to lists for JSON serialization
            cv_results_serializable = {}
            for key, value in self.cv_results.items():
                if isinstance(value, dict):
                    cv_results_serializable[key] = {
                        k: v.tolist() if isinstance(v, np.ndarray) else v
                        for k, v in value.items()
                    }
                elif isinstance(value, np.ndarray):
                    cv_results_serializable[key] = value.tolist()
                else:
                    cv_results_serializable[key] = value
            
            with open(cv_path, 'w') as f:
                json.dump(cv_results_serializable, f, indent=2)
            logger.info(f"  CV results saved to {cv_path}")
        
        # Save feature importance
        if self.model is not None and self.selected_features is not None:
            importance_df = pd.DataFrame({
                'feature': self.selected_features,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            importance_path = output_path / 'feature_importance.csv'
            importance_df.to_csv(importance_path, index=False)
            logger.info(f"  Feature importance saved to {importance_path}")
        
        logger.info("Results saved successfully!")
    
    
    def run_complete_pipeline(
        self,
        output_dir: str = '/mnt/user-data/outputs/tcga_survival_results',
        n_rna_genes: int = 150,
        xgb_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run the complete end-to-end pipeline.
        
        Steps:
        1. Load and aggregate data to patient-level
        2. Engineer features
        3. Handle missing data
        4. Select features
        5. Encode categorical variables
        6. Prepare target variable
        7. Train with cross-validation
        8. Compute SHAP values
        9. Generate visualizations
        10. Save results
        
        Args:
            output_dir: Directory for saving outputs
            n_rna_genes: Number of RNA genes to select
            xgb_params: XGBoost hyperparameters (optional)
        
        Returns:
            Dictionary containing all results
        """
        logger.info("="*70)
        logger.info("STARTING COMPLETE TCGA SURVIVAL PREDICTION PIPELINE")
        logger.info("="*70)
        
        # Step 1: Load and aggregate data
        logger.info("\n[STEP 1/10] Loading and aggregating data...")
        patient_df = self.load_and_aggregate_data()
        
        # Step 2: Engineer features
        logger.info("\n[STEP 2/10] Engineering features...")
        patient_df = self.engineer_features(patient_df)
        
        # Step 3: Handle missing data
        logger.info("\n[STEP 3/10] Handling missing data...")
        patient_df = self.handle_missing_data(patient_df)
        
        # Step 4: Prepare target variable
        logger.info("\n[STEP 4/10] Preparing target variable...")
        y, label_mapping = self.prepare_target(patient_df)
        
        # Step 5: Separate features from target
        X = patient_df.drop(columns=[self.target_column, 'patient_id'])
        
        # Step 6: Select features
        logger.info("\n[STEP 5/10] Selecting features...")
        X_selected, selected_features = self.select_features(X, y, n_rna_genes=n_rna_genes)
        
        # Step 7: Encode categorical variables
        logger.info("\n[STEP 6/10] Encoding categorical features...")
        X_encoded = self.encode_categorical_features(X_selected)
        
        # Step 8: Train with cross-validation
        logger.info("\n[STEP 7/10] Training model with cross-validation...")
        cv_results = self.train_with_cv(X_encoded, y, xgb_params=xgb_params)
        
        # Step 9: Compute SHAP values
        logger.info("\n[STEP 8/10] Computing SHAP values...")
        shap_values = self.compute_shap_values(X_encoded, sample_size=500)
        
        # Step 10: Generate visualizations
        logger.info("\n[STEP 9/10] Generating visualizations...")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Plot feature importance
        self.plot_feature_importance(
            top_n=20,
            save_path=output_path / 'feature_importance.png'
        )
        
        # Plot SHAP summary
        self.plot_shap_summary(
            X_encoded,
            plot_type='dot',
            top_n=20,
            save_path=output_path / 'shap_summary_dot.png'
        )
        
        self.plot_shap_summary(
            X_encoded,
            plot_type='bar',
            top_n=20,
            save_path=output_path / 'shap_summary_bar.png'
        )
        
        # Plot SHAP by modality (for hypothesis testing)
        self.plot_shap_by_modality(
            X_encoded,
            save_path=output_path / 'shap_by_modality.png'
        )
        
        # Plot confusion matrix
        self.plot_confusion_matrix(
            X_encoded,
            y,
            save_path=output_path / 'confusion_matrix.png'
        )
        
        # Step 11: Save results
        logger.info("\n[STEP 10/10] Saving results...")
        self.save_results(output_dir, save_model=True)
        
        # Compile final results
        results = {
            'n_patients': len(patient_df),
            'n_features_total': X.shape[1],
            'n_features_selected': len(selected_features),
            'feature_groups': {k: len(v) for k, v in self.feature_groups.items()},
            'label_mapping': label_mapping,
            'class_distribution': y.value_counts().to_dict(),
            'cv_results': cv_results,
            'model': self.model,
            'selected_features': selected_features,
            'shap_values': shap_values
        }
        
        logger.info("\n" + "="*70)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*70)
        logger.info(f"\nFinal Model Performance:")
        logger.info(f"  ROC-AUC: {cv_results['mean_test_auc']:.4f} ± {cv_results['std_test_auc']:.4f}")
        logger.info(f"  F1-Score: {cv_results['mean_test_f1']:.4f} ± {cv_results['std_test_f1']:.4f}")
        logger.info(f"\nResults saved to: {output_dir}")
        
        return results


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
