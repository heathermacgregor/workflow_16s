# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import logging
from pathlib import Path
from typing import List, Union

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger('workflow_16s')

# ==================================== FUNCTIONS ===================================== #

def safe_delete(file_path: Union[str, Path]) -> None:
    """
    Safely delete a file if it exists, logging warnings on errors.
    
    Args:
        file_path: Path to the file to be deleted.
    """
    try:
        Path(file_path).unlink(missing_ok=True)
    except Exception as e:
        logger.warning(f"Error deleting {file_path}: {e}")


def missing_output_files(file_list: List[Union[str, Path]]) -> List[Path]:
    """
    Identify missing files from a list of expected paths.
    
    Args:
        file_list: List of file paths to check for existence.
    
    Returns:
        List of Path objects for files that don't exist.
    """
    return [Path(file) for file in file_list if not Path(file).exists()]