# ===================================== IMPORTS ====================================== #
"""
Reorganized data utilities module providing backward compatibility.

This module has been split into focused sub-modules for better maintainability:
- table_conversion.py:     Table format conversion utilities (table_to_df, to_biom)
- table_filtering.py:      Table filtering operations (filter, filter_features, filter_samples, presence_absence)
- table_processing.py:     Table processing operations (normalize, clr)
- feature_classification.py: Feature ID classification (classify_feature_format)

For new code, consider importing directly from the specific modules.
"""

# Standard Library Imports
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

# Third-Party Imports
import numpy as np
import pandas as pd
from biom import Table as Table
from pandarallel import pandarallel
from scipy.sparse import issparse
from skbio.stats.composition import clr as CLR

# Local Imports
from workflow_16s import constants
from workflow_16s.utils.progress import get_progress_bar, _format_task_desc

# ================================== LOCAL IMPORTS =================================== #

# Import all functions from the new modules for backward compatibility
from workflow_16s.utils.table_conversion import (
    table_to_df,
    to_biom
)

from workflow_16s.utils.table_filtering import (
    filter,
    filter_features,
    filter_samples,
    presence_absence,
    filter_presence_absence
)

from workflow_16s.utils.table_processing import (
    normalize,
    clr
)

from workflow_16s.utils.feature_classification import (
    classify_feature_format,
    FEATURE_PATTERNS
)

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger("workflow_16s")

# ================================= LEGACY FUNCTIONS ================================== #

# Keep any remaining functions that don't fit cleanly into the new modules
# and haven't been moved yet. The rest are now imported from above.

def merge_table_with_meta(
    table: pd.DataFrame, 
    metadata: pd.DataFrame, 
    column: str, 
    verbose: bool = False
) -> pd.DataFrame:
    """Merge feature table with metadata on a specific column.
    
    Args:
        table:    Features DataFrame (samples × features).
        metadata: Metadata DataFrame with sample information.
        column:   Column name to merge on.
        verbose:  Enable detailed logging.
        
    Returns:
        Merged DataFrame with metadata column added.
        
    Raises:
        KeyError: If merge column is missing from either DataFrame.
    """
    # Ensure consistent indexing
    table = table.copy()
    metadata = metadata.copy()
    
    # Get sample IDs (table index should be sample IDs)
    table_samples = set(table.index.astype(str))
    
    # Find appropriate metadata join column
    meta_sample_cols = ['#SampleID', 'run_accession', 'sample_id']
    join_col = None
    
    for col in meta_sample_cols:
        if col in metadata.columns:
            metadata[col] = metadata[col].astype(str)
            meta_samples = set(metadata[col])
            overlap = len(table_samples.intersection(meta_samples))
            if overlap > 0:
                join_col = col
                break
    
    if join_col is None:
        raise KeyError("No matching sample ID column found in metadata")
    
    # Set metadata index for joining
    metadata = metadata.set_index(join_col)
    
    # Find common samples
    common_samples = table_samples.intersection(set(metadata.index))
    
    if len(common_samples) == 0:
        raise ValueError("No common samples found between table and metadata")
    
    # Filter to common samples
    table_filtered = table.loc[list(common_samples)]
    metadata_filtered = metadata.loc[list(common_samples)]
    
    # Add the specified column to the table
    if column not in metadata_filtered.columns:
        raise KeyError(f"Column '{column}' not found in metadata")
    
    result = table_filtered.copy()
    result[column] = metadata_filtered[column]
    
    if verbose:
        logger.info(f"Merged {len(common_samples)} samples with column '{column}'")
    
    return result


def update_table_and_meta(
    table: pd.DataFrame, 
    metadata: pd.DataFrame, 
    new_column_name: str, 
    new_column_data: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Add new column to both table and metadata.
    
    Args:
        table:            Features DataFrame.
        metadata:         Metadata DataFrame.
        new_column_name:  Name for the new column.
        new_column_data:  Series with new data (index = sample IDs).
        
    Returns:
        Tuple of (updated table, updated metadata).
    """
    table_updated = table.copy()
    metadata_updated = metadata.copy()
    
    # Add to table
    table_updated[new_column_name] = new_column_data
    
    # Add to metadata (match by sample ID)
    meta_index_col = metadata.columns[0] if '#SampleID' not in metadata.columns else '#SampleID'
    metadata_updated = metadata_updated.set_index(meta_index_col)
    metadata_updated[new_column_name] = new_column_data
    metadata_updated = metadata_updated.reset_index()
    
    return table_updated, metadata_updated


def _normalize_metadata(
    metadata_df: pd.DataFrame, 
    sample_column: str
) -> pd.DataFrame:
    """Normalize metadata for table joining."""
    if sample_column not in metadata_df.columns:
        raise ValueError(f"Sample column '{sample_column}' not found in metadata")
    
    normalized = metadata_df.copy()
    normalized = normalized.set_index(sample_column)
    normalized.index = normalized.index.astype(str)
    
    return normalized


def _create_biom_id_mapping(table: Table) -> Dict[str, str]:
    """Create mapping between normalized and original BIOM sample IDs."""
    mapping = {}
    for orig_id in table.ids(axis='sample'):
        lower_id = str(orig_id).lower()
        mapping[lower_id] = orig_id
    return mapping


# TODO: Integrate into workflow - these functions are marked as TODO in original
def trim_and_merge_asvs(
    asv_table: pd.DataFrame,
    asv_seqs: List[str],
    trim_len: int = 250,
    n_workers: int = 8,
    verbose: bool = False,
    progress: bool = True
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Trim and merge ASVs based on sequence similarity.
    
    This function is marked as TODO in the original codebase.
    Implementation would require additional dependencies and testing.
    """
    # Placeholder implementation
    logger.warning("trim_and_merge_asvs is not yet implemented")
    return asv_table, pd.Series(), asv_seqs


def collapse_taxa(
    table: Union[pd.DataFrame, Table], 
    target_level: str, 
    progress=None, 
    task_id=None,
    verbose: bool = False
) -> Table:
    """Collapse features to specific taxonomic level.
    
    This function requires taxonomic annotation and would need
    integration with the taxonomy utilities.
    """
    # Placeholder implementation
    logger.warning("collapse_taxa requires taxonomy integration")
    return to_biom(table)


# ==================================== USAGE RECOMMENDATIONS =================================== #

"""
For new code, consider importing directly from the specific modules for better clarity:

    from workflow_16s.utils.table_conversion import table_to_df, to_biom
    from workflow_16s.utils.table_filtering import filter, presence_absence
    from workflow_16s.utils.table_processing import normalize, clr
    from workflow_16s.utils.feature_classification import classify_feature_format

This makes dependencies more explicit and improves code organization.
"""