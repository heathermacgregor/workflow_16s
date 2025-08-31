# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# ================================== LOCAL IMPORTS =================================== #

from workflow_16s.utils.dir_utils import SubDirs

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger('workflow_16s')

# ==================================== FUNCTIONS ===================================== #

def processed_dataset_files(
    dirs: SubDirs, 
    dataset: str, 
    params: Dict[str, Any], 
    cfg: Dict[str, Any]
) -> Dict[str, Path]:
    """
    Generate expected file paths for processed dataset outputs.
    
    Args:
        dirs:    Project directory structure object.
        dataset: Dataset identifier.
        params:  Processing parameters dictionary.
        cfg:     Configuration dictionary.
    
    Returns:
        Dictionary mapping file types to absolute paths.
    """
    classifier: str = cfg["classifier"]
    base_dir = (
        Path(dirs.qiime_data_per_dataset) / dataset / 
        params['instrument_platform'].lower() / 
        params['library_layout'].lower() / 
        params['target_subfragment'].lower() / 
        f"FWD_{params['pcr_primer_fwd_seq']}_REV_{params['pcr_primer_rev_seq']}"
    )
    base_dir.mkdir(parents=True, exist_ok=True)
    
    return {
        'metadata_tsv': Path(dirs.metadata_per_dataset) / dataset / 'metadata.tsv',
        'manifest_tsv': base_dir / 'manifest.tsv',
        'table_biom': base_dir / 'table' / 'feature-table.biom',
        'seqs_fasta': base_dir / 'rep-seqs' / 'dna-sequences.fasta',
        'taxonomy_tsv': base_dir / classifier / 'taxonomy' / 'taxonomy.tsv',
    }


def find_required_qiime_output_files(test: Dict[str, Path]) -> Optional[Dict[str, Path]]:
    """
    Check for required output files in QIIME directories.
    
    Args:
        test: Dictionary containing base paths for QIIME output directories.
    
    Returns:
        Dictionary of found file paths keyed by file type, or None if any 
        required file is missing.
    """
    targets: List[Tuple[str, Optional[str]]] = [
        ("feature-table.biom", "table"),
        ("feature-table.biom", "table_6"),
        ("dna-sequences.fasta", "rep-seqs"),
        ("taxonomy.tsv", "taxonomy"),
        ("sample-metadata.tsv", None)
    ]
    qiime_base: Optional[Path] = test.get('qiime')
    
    if not qiime_base:
        return None
        
    found_files = {}
    
    for filename, subdir in targets:
        if subdir:
            file_path = qiime_base / subdir / filename
        else:
            file_path = qiime_base / filename
            
        if file_path.exists():
            found_files[f"{subdir or 'base'}_{filename}"] = file_path
        else:
            logger.warning(f"Required file not found: {file_path}")
            return None
    
    return found_files