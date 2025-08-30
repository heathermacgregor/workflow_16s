# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import logging
from functools import reduce
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-Party Imports
import h5py
import pandas as pd
from biom import load_table
from biom.table import Table

# ================================== LOCAL IMPORTS =================================== #

from workflow_16s.utils.progress import get_progress_bar
from workflow_16s.stats.utils import table_to_dataframe

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger('workflow_16s')

# ================================= DEFAULT VALUES =================================== #

DEFAULT_PROGRESS_TEXT_N: int = 65

# ==================================== FUNCTIONS ===================================== #

def import_table_biom(
    biom_path: Union[str, Path], 
    as_type: str = 'table'
) -> Union[Table, pd.DataFrame]:
    """
    Load a BIOM table from file.
    
    Args:
        biom_path: Path to .biom file.
        as_type:   Output format ('table' or 'dataframe').
    
    Returns:
        BIOM Table object or pandas DataFrame.
    
    Raises:
        ValueError: For invalid as_type values.
    """
    try:
        with h5py.File(biom_path) as f:
            table = Table.from_hdf5(f)
    except:
        table = load_table(biom_path)
        
    if as_type == 'table':
        return table
    elif as_type == 'dataframe':
        return table_to_dataframe(table)
    else:
        raise ValueError(
            f"Invalid output type: {as_type}. Use 'table' or 'dataframe'"
        )


def import_merged_table_biom(
    biom_paths: List[Union[str, Path]], 
    as_type: str = 'table',
    verbose: bool = False
) -> Union[Table, pd.DataFrame]:
    """
    Merge multiple BIOM tables into a single unified table.
    
    Args:
        biom_paths: List of paths to .biom files.
        as_type:    Output format ('table' or 'dataframe').
        verbose:    Enable detailed logging during loading.
    
    Returns:
        Merged BIOM Table or DataFrame.
    
    Raises:
        ValueError: If no valid tables are loaded.
    """
    tables: List[Table] = []

    if verbose:
        for path in biom_paths:
            try:
                table = import_table_biom(path, 'table')
                tables.append(table)
                logger.info(f"Loaded {Path(path).name} with {table.shape[1]} samples")
            except Exception as e:
                logger.error(f"BIOM load failed for {path}: {str(e)}")
    else:
        with get_progress_bar() as progress:
            task = progress.add_task(
                "Loading BIOM tables".ljust(DEFAULT_PROGRESS_TEXT_N), 
                total=len(biom_paths))
            for path in biom_paths:
                try:
                    tables.append(import_table_biom(path, 'table'))
                except Exception as e:
                    logger.error(f"BIOM load failed for {path}: {str(e)}")
                finally:
                    progress.update(task, advance=1)

    if not tables:
        raise ValueError("No valid BIOM tables loaded")

    merged_table = reduce(lambda t1, t2: t1.merge(t2), tables)
    return merged_table if as_type == 'table' else table_to_dataframe(merged_table)


def filter_and_reorder_biom_and_metadata(
    table: Table,
    metadata_df: pd.DataFrame,
    sample_column: str = '#sampleid'
) -> Tuple[Table, pd.DataFrame]:
    """
    Align BIOM table with metadata using sample IDs.
    
    Args:
        table:         BIOM feature table.
        metadata_df:   Sample metadata DataFrame.
        sample_column: Column containing sample identifiers.
    
    Returns:
        Tuple of (filtered BIOM table, filtered metadata DataFrame).
    """
    normalized_metadata = _normalize_metadata(metadata_df, sample_column)
    biom_mapping = _create_biom_id_mapping(table)
    
    # Find samples present in both
    metadata_ids = set(normalized_metadata.index)
    biom_ids = set(biom_mapping.keys())
    common_ids = metadata_ids.intersection(biom_ids)
    
    if not common_ids:
        raise ValueError("No common sample IDs found between BIOM table and metadata")
    
    logger.info(f"Found {len(common_ids)} samples in both BIOM table and metadata")
    
    # Filter and reorder BIOM table
    biom_samples_to_keep = [biom_mapping[sample_id] for sample_id in common_ids]
    filtered_table = table.filter(biom_samples_to_keep, axis='sample', inplace=False)
    
    # Filter and reorder metadata
    filtered_metadata = normalized_metadata.loc[list(common_ids)]
    
    return filtered_table, filtered_metadata


def _normalize_metadata(
    metadata_df: pd.DataFrame, 
    sample_column: str
) -> pd.DataFrame:
    """
    Normalize metadata for BIOM alignment.
    
    Args:
        metadata_df:   Input metadata DataFrame.
        sample_column: Column containing sample identifiers.
    
    Returns:
        Normalized DataFrame with sample IDs as index.
    """
    if sample_column not in metadata_df.columns:
        raise ValueError(f"Sample column '{sample_column}' not found in metadata")
    
    normalized = metadata_df.copy()
    normalized = normalized.set_index(sample_column)
    normalized.index = normalized.index.astype(str)
    
    return normalized


def _create_biom_id_mapping(table: Table) -> Dict[str, str]:
    """
    Create mapping between normalized and original BIOM sample IDs.
    
    Args:
        table: BIOM table object.
    
    Returns:
        Dictionary mapping normalized IDs to original BIOM IDs.
    """
    mapping = {}
    for sample_id in table.ids(axis='sample'):
        normalized_id = str(sample_id)
        mapping[normalized_id] = sample_id
    
    return mapping