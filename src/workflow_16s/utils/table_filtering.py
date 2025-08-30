# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import logging
from typing import Union

# Third-Party Imports
import pandas as pd
from biom import Table
from typing import List

# ================================== LOCAL IMPORTS =================================== #

from workflow_16s import constants
from workflow_16s.utils.table_conversion import to_biom

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger("workflow_16s")

# ================================ TABLE FILTERING =================================== #

def filter(
    table: Union[dict, Table, pd.DataFrame],
    min_rel_abundance: float = constants.DEFAULT_MIN_REL_ABUNDANCE,
    min_samples: int = constants.DEFAULT_MIN_SAMPLES,
    min_counts: int = constants.DEFAULT_MIN_COUNTS,
) -> Table:
    """Filter features and samples with strict type enforcement.
    
    Applies two-step filtering:
    1. Feature filtering (min_rel_abundance and min_samples)
    2. Sample filtering (min_counts)
    
    Args:
        table: Input table
        min_rel_abundance: Minimum relative abundance (%) for feature retention.
        min_samples:       Minimum samples where feature must appear.
        min_counts:        Minimum total counts per sample.
        
    Returns:
        Filtered BIOM Table.
    """
    table = to_biom(table)
    table = filter_features(table, min_rel_abundance, min_samples)
    table = filter_samples(table, min_counts)
    return table
    

def filter_features(
    table: Table, 
    min_rel_abundance: float, 
    min_samples: int
) -> Table:
    """Filter features based on prevalence and abundance.
    
    Args:
        table:             BIOM Table to filter.
        min_rel_abundance: Minimum relative abundance (%).
        min_samples:       Minimum samples where feature must appear.
        
    Returns:
        Filtered BIOM Table.
    """
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
    

def filter_samples(
    table: Table, 
    min_counts: int
) -> Table:
    """Filter samples based on minimum total counts.
    
    Args:
        table:      BIOM Table to filter.
        min_counts: Minimum total counts per sample.
        
    Returns:
        Filtered BIOM Table.
    """
    sample_sums = table.sum(axis='sample')
    ids_to_keep = [sid for sid, total in zip(table.ids(axis='sample'), sample_sums) 
                   if total >= min_counts]
    
    return table.filter(ids_to_keep, axis='sample')


def presence_absence(table: Union[Table, pd.DataFrame]) -> Table:
    """Convert abundance table to presence/absence (binary).
    
    Args:
        table: Input abundance table.
        
    Returns:
        Binary BIOM Table (0/1).
    """
    table = to_biom(table)
    
    # Convert to binary DataFrame
    df = table.to_dataframe()
    binary_df = (df > 0).astype(int)
    
    # Convert back to BIOM
    return Table(binary_df.values, 
                observation_ids=table.ids(axis='observation'),
                sample_ids=table.ids(axis='sample'))


def filter_presence_absence(
    table: Table, 
    metadata: pd.DataFrame, 
    group_column: str = constants.DEFAULT_GROUP_COLUMN,
    group_column_values: List[Union[bool, int, str]] = constants.DEFAULT_GROUP_COLUMN_VALUES,
    min_percentage: float = constants.DEFAULT_MIN_PERCENTAGE,
    invert: bool = False,
    verbose: bool = False
) -> Table:
    """Filter features by presence/absence criteria in groups.
    
    Args:
        table:                Input BIOM table.
        metadata:             Sample metadata DataFrame.
        group_column:         Metadata column for grouping.
        group_column_values:  Group values to compare.
        min_percentage:       Minimum presence percentage in groups.
        invert:               Invert filtering logic.
        verbose:              Enable detailed logging.
        
    Returns:
        Filtered BIOM Table.
    """
    # Convert to presence/absence
    pa_table = presence_absence(table)
    pa_df = pa_table.to_dataframe().T  # Transpose to samples × features
    
    # Merge with metadata
    merged = metadata.set_index('#SampleID').join(pa_df, how='inner')
    
    features_to_keep = []
    
    for feature in pa_df.columns:
        # Calculate presence percentages per group
        group_stats = {}
        for group_value in group_column_values:
            group_mask = merged[group_column] == group_value
            group_data = merged.loc[group_mask, feature]
            
            if len(group_data) > 0:
                presence_pct = (group_data > 0).mean() * 100
                group_stats[group_value] = presence_pct
            else:
                group_stats[group_value] = 0
        
        # Apply filtering criteria
        if not invert:
            # Keep if ANY group meets threshold
            if any(pct >= min_percentage for pct in group_stats.values()):
                features_to_keep.append(feature)
        else:
            # Keep if ALL groups meet threshold
            if all(pct >= min_percentage for pct in group_stats.values()):
                features_to_keep.append(feature)
        
        if verbose:
            logger.debug(f"{feature}: {group_stats}")
    
    # Filter original table (not presence/absence)
    return table.filter(features_to_keep, axis='observation')