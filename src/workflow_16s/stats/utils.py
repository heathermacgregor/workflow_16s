# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Third-Party Imports
import numpy as np
import pandas as pd
from biom import Table as BiomTable
from scipy import stats
from skbio.stats.composition import clr

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger("workflow_16s")

# ================================= DEFAULT VALUES =================================== #

DEFAULT_MIN_REL_ABUNDANCE = 1
DEFAULT_MIN_SAMPLES = 10
DEFAULT_MIN_COUNTS = 1000
DEFAULT_PSEUDOCOUNT = 1e-5

# ================================ CORE FUNCTIONALITY ================================ #

def table_to_dataframe(table: Union[Dict, BiomTable, pd.DataFrame]) -> pd.DataFrame:
    """
    Convert BIOM Table/dict to samples × features DataFrame.
    """
    if isinstance(table, pd.DataFrame):  # samples  × features
        return table
    if isinstance(table, Table):  # features × samples
        return table.to_dataframe(dense=True).T
    if isinstance(table, dict):  # samples  × features
        return pd.DataFrame(table)
    raise TypeError("Input must be BIOM Table, dict, or DataFrame.")


def merge_table_with_metadata(
    table: pd.DataFrame,
    metadata: pd.DataFrame,
    group_column: str,
    metadata_id_column: Optional[str] = '#sampleid',
    verbose: bool = False
) -> pd.DataFrame:
    """
    Merge abundance table with metadata column using direct ID matching.
    Automatically handles orientation, ID matching, and duplicate detection.
    
    Args:
        table:              Feature table (Samples × features) or 
                            (features × Samples).
        metadata:           Metadata table.
        group_column:       Metadata column to add.
        metadata_id_column: Column in metadata containing sample IDs.
        verbose:            Enable debug output.
        
    Returns:
        Table with added group_column (Samples × features+1).
        
    Raises:
        ValueError for common data issues.
    """
    # Identify sample IDs in metadata
    if metadata_id_column:
        if verbose:
            print(f"Using metadata column '{metadata_id_column}' for sample IDs")
        
        if metadata_id_column not in metadata.columns:
            raise ValueError(f"Column '{metadata_id_column}' not found in metadata")
        
        # Extract metadata sample IDs
        meta_ids = metadata[metadata_id_column].astype(str).str.strip().str.lower()
    else:
        if verbose:
            print("Using metadata index for sample IDs")
        meta_ids = metadata.index.astype(str).str.strip().str.lower()

    # Check for duplicates in normalized metadata IDs
    duplicate_mask = meta_ids.duplicated(keep=False)
    if duplicate_mask.any():
        duplicates = meta_ids[duplicate_mask].unique()
        n_duplicates = len(duplicates)
        example_duplicates = duplicates[:5]
        
        # Find original values for duplicates
        if metadata_id_column:
            original_values = metadata.loc[duplicate_mask, metadata_id_column].unique()
        else:
            original_values = metadata.index[duplicate_mask].unique()
        
        example_originals = original_values[:5]
        
        raise ValueError(
            f"Found {n_duplicates} duplicate sample IDs in metadata after normalization\n"
            f"Duplicate normalized IDs: {example_duplicates}\n"
            f"Original values: {example_originals}\n"
            "Please resolve duplicate entries in metadata"
        )
    
    # Assume samples are rows (standard orientation)
    table_ids = table.index.astype(str).str.strip().str.lower()
    # Check intersection
    shared_ids = set(table_ids) & set(meta_ids)
    
    # If no overlap, transpose table (features as rows)
    if not shared_ids:
        if verbose:
            print("No shared IDs found - transposing table")
        table = table.T
        table_ids = table.index.astype(str).str.strip().str.lower()
        shared_ids = set(table_ids) & set(meta_ids)
        
        # If still no matches, raise error
        if not shared_ids:
            table_examples = sorted(table_ids)[:5]
            meta_examples = sorted(meta_ids)[:5]
            raise ValueError(
                "No common sample IDs found after transposition\n"
                f"Table IDs: {table_examples}\n"
                f"Metadata IDs: {meta_examples}"
            )
    
    if verbose:
        print(f"Found {len(shared_ids)} shared sample IDs")
    
    # Create normalized ID to group mapping
    if metadata_id_column:
        group_map = (
            metadata
            .assign(norm_id=meta_ids)
            .set_index("norm_id")[group_column]
        )
    else:
        # Use normalized index directly
        group_map = metadata.set_index(meta_ids)[group_column]
    
    # Create normalized table index
    table_normalized_index = table.index.astype(str).str.strip().str.lower()
    # Map group values using normalized IDs
    table[group_column] = table_normalized_index.map(group_map)
    
    # Validate mapping
    if table[group_column].isna().any():
        missing_count = table[group_column].isna().sum()
        missing_samples = table.index[table[group_column].isna()][:5].tolist()
        raise ValueError(
            f"{missing_count} samples missing '{group_column}' values\n"
            f"First 5: {missing_samples}"
        )
    
    return table


def to_biom_table(table: Union[dict, BiomTable, pd.DataFrame]) -> BiomTable:
    """Robust conversion to BIOM Table with orientation handling"""
    if isinstance(table, BiomTable):
        return table
    elif isinstance(table, dict):
        return BiomTable.from_json(table)
    elif isinstance(table, pd.DataFrame):
        # Ensure features x samples orientation
        return BiomTable(
            table.values,
            observation_ids=table.index.tolist(),
            sample_ids=table.columns.tolist(),
            observation_metadata=None,
            sample_metadata=None
        )
    else:
        raise ValueError(f"Unsupported table type: {type(table)}")
        

def filter_table(
    table: Union[dict, BiomTable, pd.DataFrame],
    min_rel_abundance: float = DEFAULT_MIN_REL_ABUNDANCE,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    min_counts: int = DEFAULT_MIN_COUNTS,
) -> BiomTable:
    """Filter features and samples with strict type enforcement"""
    biom_table = to_biom_table(table)
    biom_table = filter_features(biom_table, min_rel_abundance, min_samples)
    biom_table = filter_samples(biom_table, min_counts)
    return biom_table
    

def filter_features(
    table: BiomTable, 
    min_rel_abundance: float, 
    min_samples: int
) -> BiomTable:
    """Filter features using BIOM-native methods"""
    min_abs_abundance = min_rel_abundance / 100
    
    # Convert to DataFrame for vectorized operations
    df = table.to_dataframe().astype(float)
    
    # Calculate filtering criteria
    max_per_feature = df.max(axis=1)
    non_zero_per_feature = (df > 0).sum(axis=1)
    
    # Create feature mask
    feature_mask = (max_per_feature >= min_abs_abundance) & (non_zero_per_feature >= min_samples)
    
    # Apply filtering
    feature_ids = table.ids(axis='observation')
    ids_to_keep = [fid for fid, keep in zip(feature_ids, feature_mask) if keep]
    
    return table.filter(ids_to_keep, axis='observation')
    

def filter_samples(table: BiomTable, min_counts: int) -> BiomTable:
    """Filter samples using BIOM-native methods"""
    # Convert to DataFrame for vectorized operations
    df = table.to_dataframe().astype(float)
    
    # Calculate total counts per sample
    total_per_sample = df.sum(axis=0)
    
    # Create sample mask
    sample_mask = total_per_sample >= min_counts
    
    # Apply filtering
    sample_ids = table.ids(axis='sample')
    ids_to_keep = [sid for sid, keep in zip(sample_ids, sample_mask) if keep]
    
    return table.filter(ids_to_keep, axis='sample')
    

def normalize_table(
    table: Union[dict, BiomTable, pd.DataFrame], 
    axis: int = 1
) -> BiomTable:
    """Normalize with strict type enforcement"""
    biom_table = to_biom_table(table)
    
    if axis == 1:  # Sample-wise normalization
        return biom_table.norm(axis='sample')
    elif axis == 0:  # Feature-wise normalization
        return biom_table.norm(axis='observation')
    else:
        raise ValueError("axis must be 0 (features) or 1 (samples)")
        

def clr_transform_table(
    table: Union[dict, BiomTable, pd.DataFrame], 
    pseudocount: float = DEFAULT_PSEUDOCOUNT
) -> BiomTable:
    """CLR transformation with strict type enforcement"""
    biom_table = to_biom_table(table)
    
    # Convert to dense array (samples x features)
    dense_data = biom_table.matrix_data.toarray().T
    
    # Apply CLR transformation
    clr_data = clr(dense_data + pseudocount)
    
    # Transpose back to features x samples
    clr_data = clr_data.T
    
    # Create new BIOM Table with original metadata
    return BiomTable(
        data=clr_data,
        observation_ids=biom_table.ids(axis='observation'),
        sample_ids=biom_table.ids(axis='sample'),
        observation_metadata=biom_table.metadata(axis='observation'),
        sample_metadata=biom_table.metadata(axis='sample')
    )
