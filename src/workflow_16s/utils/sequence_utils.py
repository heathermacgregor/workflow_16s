# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import logging
from pathlib import Path
from typing import Dict, Union

# Third-Party Imports
from Bio import SeqIO

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger('workflow_16s')

# ==================================== FUNCTIONS ===================================== #

def import_seqs_fasta(fasta_path: Union[str, Path]) -> Dict[str, str]:
    """
    Load sequences from FASTA file into a dictionary.
    
    Args:
        fasta_path: Path to FASTA file.
    
    Returns:
        Dictionary mapping sequence IDs to sequences.
    """
    return {rec.id: str(rec.seq) for rec in SeqIO.parse(fasta_path, "fasta")}