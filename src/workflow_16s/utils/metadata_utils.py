# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

# Third-Party Imports
import pandas as pd

# ================================== LOCAL IMPORTS =================================== #

from workflow_16s.utils.progress import get_progress_bar

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger('workflow_16s')

# ================================= DEFAULT VALUES =================================== #

DEFAULT_PROGRESS_TEXT_N: int = 65

# ==================================== FUNCTIONS ===================================== #

def import_metadata_tsv(
    tsv_path: Union[str, Path],
    column_renames: Optional[List[Tuple[str, str]]] = None
) -> pd.DataFrame:
    """
    Load and standardize a sample metadata TSV file.
    
    Args:
        tsv_path:       Path to metadata TSV file.
        column_renames: List of (old_name, new_name) tuples for column renaming.
    
    Returns:
        Standardized metadata DataFrame.
    
    Raises:
        FileNotFoundError: If specified path doesn't exist.
    """
    tsv_path = Path(tsv_path)
    if not tsv_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {tsv_path}")

    column_renames = column_renames or []
    df = pd.read_csv(tsv_path, sep='\t')
    df.columns = df.columns.str.lower()

    sample_id_col = next(
        (col for col in ['run_accession', '#sampleid', 'sample-id'] if col in df.columns),
        None
    )
    df['SAMPLE ID'] = (
        df[sample_id_col] 
        if sample_id_col 
        else [f"{tsv_path.parents[5].name}_x{i}" for i in range(1, len(df)+1)]
    )

    dataset_id_col = next(
        (col for col in ['project_accession', 'dataset_id', 'dataset_name'] if col in df.columns),
        None
    )
    df['DATASET ID'] = (
        df[dataset_id_col] 
        if dataset_id_col 
        else tsv_path.parents[5].name
    )

    if 'nuclear_contamination_status' not in df.columns:
        df['nuclear_contamination_status'] = False

    for old, new in column_renames:
        if old in df.columns:
            df.rename(columns={old: new}, inplace=True)

    return df


def import_merged_metadata_tsv(
    meta_paths: List[Union[str, Path]],
    column_renames: Optional[List[Tuple[str, str]]] = None,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Merge multiple metadata files into a single DataFrame.
    
    Args:
        meta_paths:     List of paths to metadata files.
        column_renames: List of (old_name, new_name) tuples for column renaming.
        verbose:        Enable detailed logging during loading.
    
    Returns:
        Concatenated metadata DataFrame.
    
    Raises:
        FileNotFoundError: If no valid metadata files could be loaded.
    """
    dfs: List[pd.DataFrame] = []

    if verbose:
        for path in meta_paths:
            try:
                df = import_metadata_tsv(path, column_renames)
                dfs.append(df)
                logger.info(f"Loaded {Path(path).name} with {len(df)} samples")
            except Exception as e:
                logger.error(f"Metadata load failed for {path}: {e!r}")
    else:
        with get_progress_bar() as progress:
            task = progress.add_task(
                "Loading metadata files".ljust(DEFAULT_PROGRESS_TEXT_N), 
                total=len(meta_paths)
            )
            for path in meta_paths:
                try:
                    dfs.append(import_metadata_tsv(path, column_renames))
                except Exception as e:
                    logger.error(f"Metadata load failed for {path}: {e!r}")
                finally:
                    progress.update(task, advance=1)

    if not dfs:
        raise FileNotFoundError(
            "No valid metadata files loaded. Check paths and file formats."
        )

    return pd.concat(dfs, ignore_index=True)


def write_metadata_tsv(
    df: pd.DataFrame, 
    tsv_path: Union[str, Path],
    verbose: bool = True
) -> None:
    """
    Write metadata DataFrame to standardized TSV format.
    
    Args:
        df:       Metadata DataFrame.
        tsv_path: Output file path.
        verbose:  Whether to log success message.
    """
    df = df.copy()
    if '#SampleID' not in df.columns and 'run_accession' in df.columns:
        df['#SampleID'] = df['run_accession']
    df.set_index('#SampleID', inplace=True)
    
    df.to_csv(tsv_path, sep='\t')
    
    if verbose:
        logger.info(f"Wrote metadata TSV to '{tsv_path}'")


def manual_meta(
    metadata_dir: Union[str, Path], 
    dataset: str
) -> pd.DataFrame:
    """
    Load manual metadata for a specific dataset.
    
    Args:
        metadata_dir: Directory containing manual metadata files.
        dataset:      Dataset identifier.
    
    Returns:
        Manual metadata DataFrame.
    """
    metadata_path = Path(metadata_dir) / f"{dataset}.tsv"
    return import_metadata_tsv(metadata_path)


def write_manifest_tsv(
    metadata_df: pd.DataFrame, 
    tsv_path: Union[str, Path], 
    verbose: bool = True
) -> None:
    """
    Generate and write a QIIME2 manifest file from metadata.
    
    Args:
        metadata_df: Metadata DataFrame containing sample information.
        tsv_path:    Output path for manifest file.
        verbose:     Whether to log success message.
    """
    manifest_df = metadata_df[['run_accession', 'fastq_ftp']].copy()
    manifest_df.columns = ['sample-id', 'absolute-filepath']
    manifest_df['direction'] = 'forward'
    
    manifest_df.to_csv(tsv_path, sep='\t', index=False)
    
    if verbose:
        logger.info(f"Wrote manifest TSV to '{tsv_path}'")