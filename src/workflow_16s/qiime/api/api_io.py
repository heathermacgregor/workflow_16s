# ===================================== IMPORTS ====================================== #

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import qiime2
from qiime2 import Artifact

# ================================= DEFAULT VALUES =================================== #

DEFAULT_N = 20

# ==================================== FUNCTIONS ===================================== #

def output_files_exist(
    file_dir: Union[str, Path], 
    prefixes: List[str], 
    extension: str = "qza"
) -> bool:
    """Check if all expected output files exist in a directory.
    
    Args:
        file_dir: Directory containing the files.
        prefixes: List of file prefixes to check.
        extension: File extension to check for. Defaults to "qza".

    Returns:
        True if all files exist, False otherwise.
    """
    file_dir = Path(file_dir)  # Ensure file_dir is a Path object
    missing_files = [
        file_dir / f"{prefix}.{extension}"
        for prefix in prefixes
        if not (file_dir / f"{prefix}.{extension}").exists()
    ]

    return not missing_files

# Print context when importing and exporting QIIME artifacts
def construct_file_path(
    file_dir: Union[str, Path], 
    prefix: str, 
    extension: str = "qza"
) -> Path:
    """Constructs a file path with a given prefix and extension."""
    return Path(file_dir) / f"{prefix}.{extension}"


def load_with_print(
    file_dir: Union[str, Path], 
    prefix: str, 
    suffix: str = "qza", 
    n: int = DEFAULT_N
) -> Artifact:
    """Constructs a file path and loads a QIIME 2 artifact from it."""
    file_path = str(construct_file_path(file_dir, prefix, suffix))
    artifact = Artifact.load(file_path)
    print(f"{'    Loaded from':{n}.{n}}: {file_path}")
    return artifact


def save_with_print(
    artifact: Artifact,
    file_dir: Union[str, Path],
    prefix: str,
    suffix: str = "qza",
    n: int = DEFAULT_N,
) -> None:
    """"""
    file_path = str(construct_file_path(file_dir, prefix, suffix))
    artifact.save(file_path)
    print(f"{'    Saved to':{n}.{n}}: {file_path}")


def export_with_print(
    artifact: Artifact,
    file_dir: Union[str, Path],
    prefix: str,
    suffix: str = "qza",
    n: int = DEFAULT_N,
) -> None:
    """"""
    file_path = construct_file_path(file_dir, prefix, suffix)
    dir_path = str(file_path).strip(suffix).strip(".")
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    artifact.export_data(str(dir_path))
    print(f"{'    Exported to':{n}.{n}}: {file_path}")


def save_and_export_with_print(
    artifact: Artifact,
    file_dir: Union[str, Path],
    prefix: str,
    suffix: str = "qza",
    n: int = DEFAULT_N,
) -> None:
    """"""
    save_with_print(artifact, file_dir, prefix, suffix, n)
    export_with_print(artifact, file_dir, prefix, suffix, n)
