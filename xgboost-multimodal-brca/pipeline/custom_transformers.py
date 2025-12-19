import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical features - fits on training data only."""
    def __init__(self):
        self.encoders_ = {}

    def fit(self, X, y=None):
        X = X.copy()
        cat_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            encoder = LabelEncoder()
            # Convert to string to handle mixed types/NaNs
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
    """Select features by modality with variance filtering on RNA genes."""
    def __init__(self, n_rna_genes: int = 150):
        self.n_rna_genes = n_rna_genes
        self.selected_features_ = None
        self.feature_groups_ = {}
        self.biomarkers = [
            'ESR1', 'PGR', 'ERBB2', 'TP63', 'EGFR', 'KRT5', 'KRT14',
            'KRT17', 'MKI67', 'TOP2A', 'CCND1', 'VIM', 'FN1', 'CD274', 'PDCD1'
        ]

    def fit(self, X, y=None):
        # Identify feature groups
        clinical = [col for col in X.columns if any(kw in col.lower() for kw in ['age', 'stage', 'node', 'tumor', 'gender', 'race', 'ethnicity', 'therapy', 'days_'])]
        mutation = [col for col in X.columns if any(kw in col for kw in ['mutation', '_mutated', 'vaf', 'unique_genes', 'high_impact'])]
        cnv = [col for col in X.columns if any(kw in col for kw in ['cnv_', 'amplification', 'deletion', 'segment', 'probes'])]
        engineered = [col for col in X.columns if any(kw in col for kw in ['ratio', 'score', 'burden', 'instability'])]
        
        core = set(clinical + mutation + cnv + engineered)
        rna_genes = [col for col in X.columns if col not in core]

        # Variance filter on RNA genes
        if len(rna_genes) > 0:
            selector = VarianceThreshold(threshold=0.01)
            X_rna = X[rna_genes]
            selector.fit(X_rna)
            rna_filtered = [rna_genes[i] for i in range(len(rna_genes)) if selector.get_support()[i]]
        else:
            rna_filtered = []

        # Select top N variable genes
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
            'Clinical': clinical, 'Mutation': mutation,
            'CNV': cnv, 'Engineered': engineered, 'RNA': selected_rna
        }
        return self

    def transform(self, X):
        return X[self.selected_features_]
