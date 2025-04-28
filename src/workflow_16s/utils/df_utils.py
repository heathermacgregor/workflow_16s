# ===================================== IMPORTS ====================================== #

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Union

import re
import numpy as np
import pandas as pd
from pandarallel import pandarallel

from Bio import SeqIO

from biom import load_table
from biom import Table as BiomTable

from scipy import sparse
from scipy.spatial.distance import cdist

from tabulate import tabulate

import matplotlib.pyplot as plt

import logging
logger = logging.getLogger('workflow_16s')

# ================================== LOCAL IMPORTS =================================== #

from workflow_16s.time import timer

# ================================= DEFAULT VALUES =================================== #

# Metadata column standardization
COLUMN_ORDER = [
    'dataset_id', 'dataset_type', 'ena_project_accession', 'ena_project_description',
    'instrument_platform', 'instrument_model', 'library_layout', 'target_subfragment',
    'sequence_length_bp', 'pcr_primer_fwd', 'pcr_primer_rev', 'pcr_primer_fwd_seq',
    'pcr_primer_rev_seq', 'publication_url', 'principal_investigator', 'dna_extraction_method',
    'sequencing_center', 'nuclear_contamination_status', 'nuclear_contamination_level',
    'nuclear_contamination_source', 'nuclear_contamination_source_type',
    'distance_from_nuclear_contamination_source_m', 'run_accession', 'sample_accession',
    'experiment_accession', 'submission_accession', 'secondary_study_accession',
    'secondary_sample_accession', 'sample_alias', 'run_alias', 'sample_internal_id',
    'sample_description', 'replicate', 'collection_date', 'treatment', 'city',
    'state_or_province', 'country', 'sample_sub_area', 'sample_area', 'sample_site',
    'latitude_deg', 'longitude_deg', 'elevation_m', 'altitude_m', 'depth_m', 'mass_g',
    'temperature_c', 'env_biome', 'env_feature', 'env_material', 'ph'
]

# Required columns
#MANUAL_METADATA_REQUIRED_COLUMNS = ['dataset_id']
#ENA_METADATA_REQUIRED_COLUMNS = ['run_accession']

ENA_METADATA_UNNECCESSARY_COLUMNS = [
    'sra_bytes', 'sra_aspera', 'sra_galaxy', 'sra_md5', 'sra_ftp', 
    'fastq_bytes', 'fastq_aspera', 'fastq_galaxy', 'fastq_md5',
    'collection_date_start', 'collection_date_end',
    'location_start', 'location_end',
    'ncbi_reporting_standard',
    'datahub',
    'tax_lineage', 'tax_id', 'scientific_name', 'isolation_source',
    'first_created', 'first_public', 'last_updated', 'status'
]
ENA_METADATA_COLUMNS_TO_RENAME = {
    'lat': 'latitude_deg',
    'lon': 'longitude_deg'
}

# ==================================== FUNCTIONS ===================================== #

def combine_ena_and_manual_metadata(
    ena_meta: pd.DataFrame,
    manual_meta: pd.DataFrame
):
    # Standardize column names
    ena_meta.columns = ena_meta.columns.astype(str).str.lower().str.strip()
    manual_meta.columns = manual_meta.columns.astype(str).str.lower().str.strip()
    
    # Check that both dataframes have the 'run_accession' column
    required_columns = ['run_accession']
    ena_missing_cols = [col for col in required_columns if col not in ena_meta.columns] 
    if ena_missing_cols:
        raise ValueError(f"ENA metadata for dataset '{dataset}' missing required columns: {', '.join(ena_missing_cols)}")
    manual_missing_cols = [col for col in required_columns if col not in manual_meta.columns]
    if manual_missing_cols:
        raise ValueError(f"ENA metadata for dataset '{dataset}' missing required columns: {', '.join(manual_missing_cols)}")
    
    def _resolve_column_conflicts(manual_meta, ena_meta):
        # Identify common columns between the two dataframes
        common_cols = set(ena_meta.columns) & set(manual_meta.columns) - {'run_accession'}
        
        # Create a copy of the second dataframe to avoid modifying the original
        ena_meta_processed = ena_meta.copy()

        for col in common_cols:
            # Check if the columns in both dataframes are duplicates
            if manual_meta[col].equals(ena_meta_processed[col]):
                # Drop the column from the second dataframe
                ena_meta_processed = ena_meta_processed.drop(columns=col)
            else:
                # Rename the column in the second dataframe with '_ena' suffix
                new_col_name = f"{col}_ena"
                ena_meta_processed = ena_meta_processed.rename(columns={col: new_col_name})

        return manual_meta, ena_meta_processed
    
    # Resolve column conflicts
    manual_meta, ena_meta = _resolve_column_conflicts(manual_meta, ena_meta)
    
    # Drop unneccessary columns from ENA metadata
    ena_meta = ena_meta.drop(columns=ena_meta.columns.intersection(ENA_METADATA_UNNECCESSARY_COLUMNS))
    
    # Rename columns from ENA metadata
    ena_meta = ena_meta.rename(columns={
        col: ENA_METADATA_COLUMNS_TO_RENAME[col] for col in ena_meta.columns.intersection(ENA_METADATA_COLUMNS_TO_RENAME.keys())
    })
    
    # Merge the ENA and manual metadata, only keeping samples that were retained in the manual metadata
    meta = manual_meta.merge(ena_meta, on='run_accession', how='left')
    
    # Create a 'dataset_id' column if it does not already exist
    if 'dataset_id' not in meta.columns:
        meta['dataset_id'] = f"ENA_{dataset}"
        
    return meta
    

def combine_metadata(
    ena_meta: pd.DataFrame,
    manual_meta: pd.DataFrame,
    dataset_id: str
) -> pd.DataFrame:
    """Merge ENA and manual metadata with comprehensive error handling.
    
    Args:
        ena_meta: DataFrame from European Nucleotide Archive
        manual_meta: DataFrame from local user input
        dataset_id: Unique project identifier
        
    Returns:
        Integrated DataFrame with standardized columns
        
    Notes:
        - Column name standardization
        - Missing column handling
        - Conflict resolution (manual data priority)
        - Type validation
    """
    # Standardize column names
    ena_meta.columns = ena_meta.columns.astype(str).str.lower().str.strip()
    manual_meta.columns = manual_meta.columns.astype(str).str.lower().str.strip()

    # Add dataset identifier
    for df in [ena_meta, manual_meta]:
        if not df.empty:
            df.insert(0, 'dataset_id', str(dataset_id))

    # Ensure required columns exist
    required_columns = set(COLUMN_ORDER)
    for df in [ena_meta, manual_meta]:
        missing_cols = required_columns - set(df.columns)
        for col in missing_cols:
            df[col] = pd.NA  # Add missing columns with empty values

    # Merge datasets using outer join
    merged = pd.merge(
        ena_meta,
        manual_meta,
        on='run_accession',
        how='outer',
        suffixes=('_ena', '_manual'),
        indicator=True
    )

    # Resolve column conflicts
    conflict_cols = set(ena_meta.columns) & set(manual_meta.columns) - {'run_accession'}
    for col in conflict_cols:
        merged[col] = merged[f"{col}_manual"].combine_first(merged[f"{col}_ena"])
        merged.drop(columns=[f"{col}_ena", f"{col}_manual"], inplace=True)

    # Final column validation
    missing_final = required_columns - set(merged.columns)
    if missing_final:
        logger.warning(f"Adding missing final columns: {missing_final}")
        for col in missing_final:
            merged[col] = pd.NA

    # Type enforcement and cleanup
    merged['run_accession'] = merged['run_accession'].astype(str)
    merged['dataset_id'] = merged['dataset_id'].astype(str)
    
    return merged[COLUMN_ORDER].dropna(axis=1, how='all')

def check_matching_index(
    metadata: pd.DataFrame, 
    features: pd.DataFrame
) -> bool:
    """Checks if there are any matching values between metadata's index 
    and either features's index or features's columns.
    
    Args:
        metadata:
        features:

    Returns:
        bool: True if there are matches, False otherwise.
    """
    samples = set(metadata.index)
    
    # Check if there's any overlap with df2's index or columns
    index_match = not samples.isdisjoint(features.index)
    columns_match = samples.isdisjoint(features.columns)

    return index_match and columns_match

def match_samples(
    metadata: pd.DataFrame, 
    features: pd.DataFrame
):
    matching_values = metadata.index.intersection(features.columns)

    metadata = metadata.loc[matching_values]
    features = features[matching_values]

    return metadata, features


def classify_feature_format(columns) -> Dict[str, int]:
    """Classifies column names into taxonomic formats, QIIME 2-style hashes (MD5/SHA256),
    IUPAC nucleotide sequences, or unknown patterns.

    Parameters:
    columns (Iterable[str]): An iterable of column names to classify.

    Returns:
    Dict[str, int]: A dictionary with counts for each category.

    Examples:
    >>> classify_column_format(['d__Bacteria;p__Firmicutes', 
    ...                         '1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p',
    ...                         'ACTGNYR'])
    {'taxonomic': 1, 'hashes': 1, 'raw_sequences': 1, 'unknown': 0}
    """
    # Pre-compiled regex patterns for efficiency
    TAXONOMIC_PATTERN = re.compile(
        r'^d__[\w]+(;p__[\w]+)?(;c__[\w]+)?(;o__[\w]+)?(;f__[\w]+)?(;g__[\w]+)?(;s__[\w]+)?$'
    )
    QIIME_HASH_PATTERN = re.compile(
        r'^[a-f0-9]{32}$|^[a-f0-9]{64}$'  # MD5 (32) or SHA256 (64)
    )
    IUPAC_SEQUENCE_PATTERN = re.compile(
        r'^[ACGTRYSWKMBDHVN]+$',  # All IUPAC nucleotide codes
        re.IGNORECASE
    )

    counts = {
        "taxonomic": 0,
        "hashes": 0,
        "raw_sequences": 0,
        "unknown": 0
    }
    
    for col in columns:
        # Normalize input and handle empty strings
        col_str = str(col).strip()
        if not col_str:
            counts["unknown"] += 1
            continue

        if TAXONOMIC_PATTERN.match(col_str):
            counts["taxonomic"] += 1
        elif QIIME_HASH_PATTERN.match(col_str):
            counts["hashes"] += 1
        elif IUPAC_SEQUENCE_PATTERN.match(col_str):
            counts["raw_sequences"] += 1
        else:
            counts["unknown"] += 1
            
    column_format = max(counts, key=counts.get)
    confidence = round(max(counts.values()) / sum(counts.values()), 2)
    print(f"Columns are {column_format} ({confidence} confidence)")
    return column_format 


@timer
def trim_and_merge_asvs(
    asv_table: pd.DataFrame,
    asv_sequences: List[str],
    trim_length: int = 250
) -> pd.DataFrame:
    """
    BIOM-optimized ASV merging with correct matrix orientation
    and duplicate handling.
    """
    # Initialize parallel processing
    pandarallel.initialize(progress_bar=False)

    # Validate inputs
    if len(asv_sequences) != asv_table.shape[0]:
        raise ValueError("ASV sequences count must match table rows")
    if trim_length < 1:
        raise ValueError("Trim length must be ≥1")

    # Trim sequences in parallel
    trimmed_seqs = (
        pd.Series(asv_sequences)
        .parallel_apply(lambda x: x[:trim_length])
    )
    trimmed_seqs.index = asv_sequences
    #print(trimmed_seqs)

    # Create unique temporary IDs for initial table
    unique_obs_ids = [f"TMP_FEATURE_{i}" for i in range(asv_table.shape[0])]

    # Create BIOM table with CORRECT ORIENTATION
    biom_table = BiomTable(
        data=asv_table.values.astype(np.uint32),  # Remove .T for correct orientation
        observation_ids=unique_obs_ids,
        sample_ids=asv_table.columns.astype(str).tolist(),
        observation_metadata=[{'trimmed_seq': s} for s in trimmed_seqs]
    )

    # Collapse by trimmed sequences
    merged_biom = biom_table.collapse(
        lambda id_, md: md['trimmed_seq'],
        axis='observation',
        norm=False,
        min_group_size=1,
        include_collapsed_metadata=False
    )

    # Convert to DataFrame
    merged_df = merged_biom.to_dataframe(dense=True).astype(np.uint32)
    
    print(f"Feature reduction: {len(asv_sequences)} → {merged_df.shape[0]} "
          f"({merged_df.shape[0]/len(asv_sequences):.1%})")
    
    return merged_df, trimmed_seqs, unique_obs_ids


@timer
def filter_features(
    df: pd.DataFrame,
    min_rel_abundance: float = 1,
    min_samples: int = 10
) -> pd.DataFrame:
    """Filter for columns (samples) where at least one row (features) has a relative 
    abundance of at least X%. Filter for rows (features) that are present in at least 
    Y columns (samples)."""
    features_0 = df.shape[0]
    df = df.loc[:, df.max(axis=0) >= min_rel_abundance / 100]           
    df = df.loc[(df > 0).sum(axis=1) > min_samples, :]   

    print(f"Feature reduction: {features_0} → {df.shape[0]} "
          f"({df.shape[0]/features_0:.1%})")
    return df

@timer
def filter_samples(
    df: pd.DataFrame, 
    min_counts: int = 1000
) -> pd.DataFrame:                  
    """Filter for columns (samples) that have at least 1000 counts total"""
    df = df.loc[:, (df.sum(axis=0) > min_counts)]    
    return df
