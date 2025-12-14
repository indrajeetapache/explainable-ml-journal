"""
Execute TCGA Survival Prediction (CORRECTED - NO DATA LEAKAGE)
===============================================================
Uses sklearn Pipeline to ensure all preprocessing happens inside CV folds.
"""

from tcga_pipeline_corrected import TCGASurvivalPredictor

def main():
    # Configuration
    DATA_PATH = "/content/drive/MyDrive/PHD_dataset_shap/SHAP_TCGA_dataset/Combination_ofall_Dataset/RNA_mutation_dataset.parquet"
    OUTPUT_DIR = "/mnt/user-data/outputs/tcga_results_corrected"
    
    # XGBoost params (optional - will use defaults if None)
    # xgb_params = {
    #     'max_depth': 5,
    #     'learning_rate': 0.05,
    #     'n_estimators': 200,
    #     'subsample': 0.8,
    #     'colsample_bytree': 0.8,
    #     'min_child_weight': 3,
    #     'gamma': 0.1,
    # }
    
    # Initialize
    # predictor = TCGASurvivalPredictor(
    #     data_path=DATA_PATH,
    #     target_column='vital_status',
    #     n_folds=5,
    #     n_rna_genes=150,
    #     random_state=42
    # )

    predictor = TCGASurvivalPredictor(
    # data_path=None,  # Not needed
    # target_column='vital_status',
    # n_folds=5,
    # n_rna_genes=150,
    # random_state=42
    # )
        
    # # Run pipeline (NO DATA LEAKAGE)
    # results = predictor.run_complete_pipeline(
    #     output_dir=OUTPUT_DIR,
    #     xgb_params=xgb_params
    # )
    
    # print("\n" + "="*70)
    # print("RESULTS (NO DATA LEAKAGE):")
    # print("="*70)
    # print(f"ROC-AUC: {results['cv_results']['mean_test_auc']:.4f} ± {results['cv_results']['std_test_auc']:.4f}")
    # print(f"F1:      {results['cv_results']['mean_test_f1']:.4f} ± {results['cv_results']['std_test_f1']:.4f}")
    # print(f"\nOutputs: {OUTPUT_DIR}")
    
    # return results

# if __name__ == "__main__":
#     main()
