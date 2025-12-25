import dask.dataframe as dd
import pandas as pd
import numpy as np

class DataAggregator:
    def __init__(self, target_column: str = 'vital_status'):
        self.target_column = target_column
        self.key_genes = ['TP53', 'PIK3CA', 'BRCA1', 'BRCA2', 'PTEN', 'AKT1', 'CDH1']

    def process_from_file(self, input_path: str, output_path: str):
        """Load from file, process, and save to file"""
        print("\n" + "="*60)
        print(f"🚀  STARTING STEP 1: DATA AGGREGATION")
        print("="*60)
        print(f"📂  Input File:  {input_path}")
        print(f"💾  Output File: {output_path}")
        print("-" * 60)

        # Load data
        print(f"⏳  Reading Dask DataFrame...")
        if input_path.endswith('.parquet'):
            df = dd.read_parquet(input_path)
        else:
            df = dd.read_csv(input_path, blocksize="256MB")

        # Process
        patient_df = self._process_dataframe(df)

        # Save
        print(f"💾  Saving processed file to disk...")
        patient_df.to_parquet(output_path, index=False)
        print("="*60)
        print(f"✅  STEP 1 COMPLETE. File saved at: {output_path}")
        print("="*60 + "\n")
        
        return patient_df

    def process_from_dataframe(self, dask_df):
        """Process existing Dask DataFrame (for notebook use)"""
        print("\n" + "="*60)
        print(f"🚀  STARTING DATA AGGREGATION FROM DATAFRAME")
        print("="*60)
        
        patient_df = self._process_dataframe(dask_df)
        
        print("="*60)
        print(f"✅  PROCESSING COMPLETE")
        print("="*60 + "\n")
        
        return patient_df

    def _process_dataframe(self, df):
        """Core processing logic (internal method)"""
        # Setup Aggregations
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

        # Compute Base Table
        print(f"🔥  Executing Dask Compute (Aggregating to Patient Level)...")
        patient_df = df.groupby('patient_id').agg(agg_dict).compute()
        patient_df = patient_df.reset_index()

        if 'num_probes' in patient_df.columns:
            patient_df = patient_df.rename(columns={'num_probes': 'total_cnv_probes'})

        print(f"✅  Base aggregation complete. Patients found: {len(patient_df)}")

        # Gene Mutations
        print(f"🧬  Computing specific gene mutations...")
        def compute_and_map(filtered_dask_df):
            counts = filtered_dask_df.groupby('patient_id').size().compute()
            return patient_df['patient_id'].map(counts).fillna(0)

        for gene in self.key_genes:
            if 'hugo_symbol' in cols:
                print(f"    Processing {gene}...", end='\r')
                mask = df['hugo_symbol'] == gene
                patient_df[f'{gene}_mutated'] = (compute_and_map(df[mask]) > 0).astype(int)
        print(f"    Processing Genes... Done!          ")

        # CNV Stats
        if 'segment_mean' in cols:
            print(f"📊  Computing CNV statistics...")
            patient_df['amplification_count'] = compute_and_map(df[df['segment_mean'] > 0.2])
            patient_df['deletion_count'] = compute_and_map(df[df['segment_mean'] < -0.2])

            cnv_stats = df.groupby('patient_id')['segment_mean'].agg(['mean', 'std']).compute()
            patient_df['cnv_mean'] = patient_df['patient_id'].map(cnv_stats['mean'])
            patient_df['cnv_std'] = patient_df['patient_id'].map(cnv_stats['std'])

        # Feature Engineering
        print(f"🛠️   Engineering final derived features...")
        if 'years_to_birth' in patient_df.columns:
            patient_df['age_at_diagnosis'] = -patient_df['years_to_birth']

        if 'mutation_count' in patient_df.columns and 'age_at_diagnosis' in patient_df.columns:
            patient_df['mutation_burden_per_age'] = patient_df['mutation_count'] / (patient_df['age_at_diagnosis'] + 1)

        if 'amplification_count' in patient_df.columns and 'deletion_count' in patient_df.columns:
            patient_df['amp_del_ratio'] = (patient_df['amplification_count'] + 1) / (patient_df['deletion_count'] + 1)

        return patient_df
