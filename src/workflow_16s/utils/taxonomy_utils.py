# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import logging
from pathlib import Path
from typing import Optional, Union

# Third-Party Imports
import pandas as pd

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger('workflow_16s')

# ==================================== FUNCTIONS ===================================== #

def import_faprotax_tsv(tsv_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load FAPROTAX functional prediction results.
    
    Args:
        tsv_path: Path to FAPROTAX output TSV.
    
    Returns:
        Transposed DataFrame with samples as rows and functions as columns.
    """
    return pd.read_csv(tsv_path, sep="\t", index_col=0).T


# ================================== TAXONOMY CLASS ================================== #

class Taxonomy:
    """
    Handler for taxonomic classification data.
    
    Attributes:
        taxonomy (pd.DataFrame): Parsed taxonomy data with columns:
            - id:              Feature ID
            - taxonomy:        Raw taxonomy string
            - confidence:      Classification confidence score
            - taxstring:       Cleaned taxonomy string
            - [D/P/C/O/F/G/S]: Taxonomic levels (Domain to Species)
    """
    
    def __init__(self, tsv_path: Union[str, Path]) -> None:
        """
        Initialize Taxonomy object from TSV file.
        
        Args:
            tsv_path: Path to QIIME2 taxonomy TSV.
        """
        self.taxonomy: pd.DataFrame = self._import_taxonomy_tsv(tsv_path)
        
    def _import_taxonomy_tsv(self, tsv_path: Union[str, Path]) -> pd.DataFrame:
        """
        Parse taxonomy TSV into structured DataFrame.
        
        Args:
            tsv_path: Path to taxonomy TSV file.
        
        Returns:
            Structured taxonomy DataFrame.
        """
        tsv_path = Path(tsv_path)
        df = pd.read_csv(tsv_path, sep='\t')
        df = df.rename(columns={
            'Feature ID': 'id', 
            'Taxon': 'taxonomy', 
            'Consensus': 'confidence'
        }).set_index('id')
        
        df['taxstring'] = df['taxonomy'].str.replace(r' *[dpcofgs]__', '', regex=True)
        
        for level in ['d', 'p', 'c', 'o', 'f', 'g', 's']:
            df[level.upper()] = df['taxonomy'].apply(
                lambda x: self._extract_level(x, level))
            
        return df.rename(columns={
            'D': 'Domain', 'P': 'Phylum', 'C': 'Class',
            'O': 'Order', 'F': 'Family', 'G': 'Genus', 'S': 'Species'
        })
        
    def _extract_level(
        self, 
        taxonomy: str, 
        level: str
    ) -> Optional[str]:
        """
        Extract specific taxonomic level from taxonomy string.
        
        Args:
            taxonomy: Raw taxonomy string.
            level:    Taxonomic level prefix (d/p/c/o/f/g/s).
        
        Returns:
            Taxonomic name for specified level, or None if not found.
        """
        prefix = level + '__'
        if not taxonomy or taxonomy in ['Unassigned', 'Unclassified']:
            return 'Unclassified'
            
        start = taxonomy.find(prefix)
        if start == -1:
            return None
            
        end = taxonomy.find(';', start)
        return (
            taxonomy[start+len(prefix):end] 
            if end != -1 else 
            taxonomy[start+len(prefix):]
        )
        
    def get_taxstring_by_id(self, feature_id: str) -> Optional[str]:
        """
        Retrieve clean taxonomy string for a feature ID.
        
        Args:
            feature_id: Feature identifier.
        
        Returns:
            Cleaned taxonomy string, or None if feature not found.
        """
        return (
            self.taxonomy.loc[feature_id, 'taxstring'] 
            if feature_id in self.taxonomy.index else 
            None
        )