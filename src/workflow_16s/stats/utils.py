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


def merge_table_with_metadata(
    table: pd.DataFrame, metadata: pd.DataFrame, group_column: str
) -> pd.DataFrame:
    """
    Merge abundance table with metadata column after index sanitization.
    """
    # Preserve original index names
    table_index_name = table.index.name or "index"
    meta_index_name = metadata.index.name or "index"

    # Reset indexes for merging
    table = table.reset_index().rename(columns={table_index_name: "temp_index"})
    metadata = metadata.reset_index().rename(columns={meta_index_name: "temp_index"})

    # Sanitize IDs
    table["temp_index"] = table["temp_index"].astype(str).str.strip().str.lower()
    metadata["temp_index"] = metadata["temp_index"].astype(str).str.strip().str.lower()

    # Perform merge
    merged = pd.merge(
        table, metadata[[group_column, "temp_index"]], on="temp_index", how="inner"
    ).set_index("temp_index")

    # Restore original index name
    merged.index.name = table_index_name

    # Validate merge
    if merged[group_column].isna().any():
        missing = merged[group_column].isna().sum()
        raise ValueError(
            f"{missing} samples have NaN in '{group_column}' after merge. "
            "Check metadata completeness."
        )

    return merged  # samples × features


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
