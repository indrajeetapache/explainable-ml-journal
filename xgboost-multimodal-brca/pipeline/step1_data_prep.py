import dask.dataframe as dd
import pandas as pd
import numpy as np
import argparse
import sys

class DataAggregator:
    def __init__(self, target_column: str = 'vital_status'):
        self.target_column = target_column
        self.key_genes = ['TP53', 'PIK3CA', 'BRCA1', 'BRCA2', 'PTEN', 'AKT1', 'CDH1']

    def process(self, input_path: str, output_path: str):
        print("\n" + "="*60)
        print(f"🚀  STARTING STEP 1: DATA AGGREGATION")
        print("="*60)
        print(f"📂  Input File:  {input_path}")
        print(f"💾  Output File: {output_path}")
        print("-" * 60)

        # 1. Load Data
        print(f"⏳  Reading Dask DataFrame (Lazy Loading)...")
        if input_path.endswith('.parquet'):
            df = dd.read_parquet(input_path)
        else:
            df = dd.read_csv(input_path, blocksize="256MB")

        # 2. Setup Aggregations
        print(f"⚙️   Configuring aggregation rules...")
        agg_dict = {self.target_column: 'first'}
        cols = df.columns
        
        if 'mutation_count' in cols: agg_dict['mutation_count'] = 'sum'
        if 'tumor_vaf' in cols: agg_dict['tumor_vaf'] = 'mean'
        if 'num_probes' in cols: agg_dict['num_probes'] = 'sum'
        
        clinical_cols = ['years_to_birth', 'gender', 'race', 'ethnicity',
                        'tumor_tissue_site', 'histological_type', 'pathologic_stage',
                        'pathology_t_stage', 'pathology_n_stage', 'pathology_m_stage',
                        'number_of_lymph_nodes', 'radiation_therapy']
        for col in clinical_cols:
            if col in cols: agg_dict[col] = 'first'

        # 3. Compute Base Table
        print(f"🔥  Executing Dask Compute (Aggregating 11GB -> Patient Level)...")
        print(f"    (This is the heavy step, please wait...)")
        
        patient_df = df.groupby('patient_id').agg(agg_dict).compute()
        patient_df = patient_df.reset_index()
        
        if 'num_probes' in patient_df.columns:
            patient_df = patient_df.rename(columns={'num_probes': 'total_cnv_probes'})
            
        print(f"✅  Base aggregation complete. Patients found: {len(patient_df)}")

        # 4. Complex Features
        print(f"🧬  Computing specific gene mutations and CNV stats...")
        
        def compute_and_map(filtered_dask_df):
            counts = filtered_dask_df.groupby('patient_id').size().compute()
            return patient_df['patient_id'].map(counts).fillna(0)

        for gene in self.key_genes:
            if 'hugo_symbol' in cols:
                print(f"    Processing {gene}...", end='\r')
                mask = df['hugo_symbol'] == gene
                patient_df[f'{gene}_mutated'] = (compute_and_map(df[mask]) > 0).astype(int)
        print(f"    Processing Genes... Done!          ")

        if 'segment_mean' in cols:
            print(f"📊  Computing CNV statistics (Amplifications/Deletions)...")
            patient_df['amplification_count'] = compute_and_map(df[df['segment_mean'] > 0.2])
            patient_df['deletion_count'] = compute_and_map(df[df['segment_mean'] < -0.2])
            
            cnv_stats = df.groupby('patient_id')['segment_mean'].agg(['mean', 'std']).compute()
            patient_df['cnv_mean'] = patient_df['patient_id'].map(cnv_stats['mean'])
            patient_df['cnv_std'] = patient_df['patient_id'].map(cnv_stats['std'])

        # 5. Feature Engineering
        print(f"🛠️   Engineering final derived features...")
        if 'years_to_birth' in patient_df.columns:
            patient_df['age_at_diagnosis'] = -patient_df['years_to_birth']
            
        if 'mutation_count' in patient_df.columns and 'age_at_diagnosis' in patient_df.columns:
            patient_df['mutation_burden_per_age'] = patient_df['mutation_count'] / (patient_df['age_at_diagnosis'] + 1)

        if 'amplification_count' in patient_df.columns and 'deletion_count' in patient_df.columns:
            patient_df['amp_del_ratio'] = (patient_df['amplification_count'] + 1) / (patient_df['deletion_count'] + 1)

        # Save
        print(f"💾  Saving processed file to disk...")
        patient_df.to_parquet(output_path, index=False)
        print("="*60)
        print(f"✅  STEP 1 COMPLETE. File saved at: {output_path}")
        print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    
    aggregator = DataAggregator(target_column='vital_status')
    aggregator.process(args.input, args.output)
