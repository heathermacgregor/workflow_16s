# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Third-Party Imports
import numpy as np
import pandas as pd
from biom import Table
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


def table_to_dataframe(table: Union[Dict, Table, pd.DataFrame]) -> pd.DataFrame:
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
    Automatically handles orientation and ID matching.
    
    Args:
        table: Feature table (Samples × features) or (features × Samples)
        metadata: Metadata table
        group_column: Metadata column to add
        metadata_id_column: Column in metadata containing sample IDs
        verbose: Enable debug output
        
    Returns:
        Table with added group_column (Samples × features+1)
        
    Raises:
        ValueError for common data issues
    """
    # =====================================================================
    # 1. Identify sample IDs in metadata
    # =====================================================================
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

    # =====================================================================
    # 2. Identify sample IDs in table
    # =====================================================================
    # First try: assume samples are rows (standard orientation)
    table_ids = table.index.astype(str).str.strip().str.lower()
    
    # Check intersection
    shared_ids = set(table_ids) & set(meta_ids)
    
    # Second try: if no overlap, transpose table (features as rows)
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
    
    # =====================================================================
    # 3. Prepare metadata mapping
    # =====================================================================
    # Create normalized ID to group mapping
    if metadata_id_column:
        # Create mapping from normalized IDs to group values
        group_map = (
            metadata
            .assign(norm_id=meta_ids)
            .set_index("norm_id")[group_column]
        )
    else:
        # Use normalized index directly
        group_map = metadata.set_index(meta_ids)[group_column]
    
    # =====================================================================
    # 4. Merge group column into table
    # =====================================================================
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


def filter_table(
    table: Union[Dict, Table, pd.DataFrame],
    min_rel_abundance: float = DEFAULT_MIN_REL_ABUNDANCE,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    min_counts: int = DEFAULT_MIN_COUNTS,
) -> pd.DataFrame:
    """
    Filter features and samples based on abundance thresholds.

    samples × features
    """
    df = table_to_dataframe(table)
    df = filter_features(df, min_rel_abundance, min_samples)
    df = filter_samples(df, min_counts)
    return df


def filter_features(
    table: pd.DataFrame, min_rel_abundance: float, min_samples: int
) -> pd.DataFrame:
    """
    Filter features by relative abundance and sample presence.
    """
    min_abs_abundance = min_rel_abundance / 100
    feature_mask = (table.max(axis=0) >= min_abs_abundance) & (
        table.astype(bool).sum(axis=0) >= min_samples
    )
    table = table.loc[:, feature_mask]
    return table


def filter_samples(table: pd.DataFrame, min_counts: int) -> pd.DataFrame:
    """
    Filter samples by total counts.
    """
    sample_mask = table.sum(axis=1) >= min_counts
    table = table.loc[sample_mask]
    return table


def normalize_table(
    table: Union[Dict, Table, pd.DataFrame], axis: int = 1
) -> pd.DataFrame:
    """
    Convert to relative abundances along specified axis.
    """
    df = table_to_dataframe(table)
    df = df.div(df.sum(axis=axis), axis=1 - axis)
    return df


def clr_transform_table(
    table: Union[Dict, Table, pd.DataFrame], pseudocount: float = DEFAULT_PSEUDOCOUNT
) -> pd.DataFrame:
    """
    Apply centered log-ratio transformation.
    """
    df = table_to_dataframe(table)
    df = pd.DataFrame(clr(df + pseudocount), index=df.index, columns=df.columns)
    return df
