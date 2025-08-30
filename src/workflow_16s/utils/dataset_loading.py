# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import logging
from pathlib import Path
from typing import List, Union

# Third-Party Imports
import pandas as pd

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger('workflow_16s')

# ==================================== FUNCTIONS ===================================== #

def load_datasets_list(path: Union[str, Path]) -> List[str]:
    """
    Load dataset IDs from a text file.
    
    Parses a text file where each line contains a single dataset ID, 
    ignoring empty lines and lines containing only whitespace.
    
    Args:
        path: Path to the dataset list file.
    
    Returns:
        List of dataset ID strings.
    """
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def load_datasets_info(tsv_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load dataset metadata from TSV file.
    
    Args:
        tsv_path: Path to TSV file containing dataset metadata.
    
    Returns:
        DataFrame with dataset information.
    """
    tsv_path = Path(tsv_path)
    df = pd.read_csv(tsv_path, sep="\t", dtype={'ena_project_accession': str})
    return df.loc[:, ~df.columns.str.startswith('Unnamed')]


def fetch_first_match(
    dataset: str,
    dataset_info: pd.DataFrame
) -> pd.Series:
    """
    Find the best matching metadata record for a dataset.
    
    Args:
        dataset:      Dataset identifier to search for.
        dataset_info: DataFrame containing dataset metadata.
    
    Returns:
        First matching row as a pandas Series.
    
    Raises:
        ValueError: If no matches found for the dataset.
    """
    mask_ena_type = dataset_info['dataset_type'].str.lower().eq('ena')
    mask_manual_type = dataset_info['dataset_type'].str.lower().eq('manual')
    
    mask_ena = (
        dataset_info['ena_project_accession'].str.contains(dataset, case=False, regex=False) |
        dataset_info['dataset_id'].str.contains(dataset, case=False, regex=False)
    ) & mask_ena_type

    mask_manual = (
        dataset_info['dataset_id'].str.contains(dataset, case=False, regex=False)
    ) & mask_manual_type

    combined_mask = mask_ena | mask_manual
    matching_rows = dataset_info[combined_mask]

    if matching_rows.empty:
        raise ValueError(f"No metadata matches found for dataset: {dataset}")

    return matching_rows.sort_values(
        by='dataset_type', 
        key=lambda x: x.str.lower().map({'ena': 0, 'manual': 1})
    ).iloc[0]