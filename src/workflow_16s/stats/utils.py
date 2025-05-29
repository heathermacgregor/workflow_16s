# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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


from typing import Optional

def merge_table_with_metadata(
    table: pd.DataFrame,
    metadata: pd.DataFrame,
    group_column: str,
    metadata_id_column: Optional[str] = None  # NEW: Column containing sample IDs
) -> pd.DataFrame:
    """
    Merge abundance table with metadata column after index sanitization.
    
    Args:
        table: Samples × features abundance table
        metadata: Metadata table with sample information
        group_column: Metadata column to merge
        metadata_id_column: Optional column in metadata containing sample IDs
        
    Returns:
        Merged DataFrame with samples × features
        
    Raises:
        ValueError: If no common IDs found or missing group_column values
    """
    # Preserve original index names
    table_index_name = table.index.name or "index"
    
    # NEW: Diagnostic logging
    print(f"Table index type: {type(table.index[0]) if len(table.index) > 0 else 'empty'}")
    print(f"Metadata index type: {type(metadata.index[0]) if len(metadata.index) > 0 else 'empty'}")

    # NEW: Handle metadata sample ID column
    if metadata_id_column:
        print(f"Using metadata column '{metadata_id_column}' for sample IDs")
        if metadata_id_column not in metadata.columns:
            raise ValueError(f"Column '{metadata_id_column}' not found in metadata")
        
        # Create temp DF with sample IDs
        metadata = metadata.reset_index()
        metadata_for_merge = metadata[[metadata_id_column, group_column]].copy()
        meta_index_name = metadata_id_column
    else:
        print("Using metadata index for sample IDs")
        meta_index_name = metadata.index.name or "index"
        metadata_for_merge = metadata[[group_column]].copy()
        metadata_for_merge = metadata_for_merge.reset_index()

    # Reset table index
    table = table.reset_index().rename(columns={table_index_name: "temp_index"})
    
    # Reset metadata index
    metadata_for_merge = metadata_for_merge.rename(columns={meta_index_name: "temp_index"})

    # Sanitize IDs
    table["temp_index"] = table["temp_index"].astype(str).str.strip().str.lower()
    metadata_for_merge["temp_index"] = metadata_for_merge["temp_index"].astype(str).str.strip().str.lower()

    # NEW: Print diagnostic samples
    print("Table sample IDs:", table["temp_index"].head(5).tolist())
    print("Metadata sample IDs:", metadata_for_merge["temp_index"].head(5).tolist())
    
    # NEW: Check for duplicates
    if metadata_for_merge["temp_index"].duplicated().any():
        duplicates = metadata_for_merge["temp_index"].duplicated().sum()
        raise ValueError(f"{duplicates} duplicate sample IDs found in metadata")

    # Perform merge
    merged = pd.merge(
        table,
        metadata_for_merge,
        on="temp_index",
        how="inner"
    ).set_index("temp_index")

    # Restore original index name
    merged.index.name = table_index_name

    # Validate merge
    if merged.empty:
        table_samples = table["temp_index"].unique()[:5]
        meta_samples = metadata_for_merge["temp_index"].unique()[:5]
        raise ValueError(
            "No samples remaining after merge. Possible causes:\n"
            "- Metadata doesn't contain sample IDs from table\n"
            "- Sample ID column not properly specified\n\n"
            f"Table sample IDs: {table_samples}\n"
            f"Metadata sample IDs: {meta_samples}"
        )

    if merged[group_column].isna().any():
        missing = merged[group_column].isna().sum()
        missing_samples = merged[merged[group_column].isna()].index.tolist()[:5]
        raise ValueError(
            f"{missing} samples have NaN in '{group_column}' after merge. "
            f"First 5 affected samples: {missing_samples}"
        )

    print(f"Successfully merged {len(merged)} samples")
    return merged


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
