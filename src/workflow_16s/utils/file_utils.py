# ===================================== IMPORTS ====================================== #

from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import re
import pandas as pd
from Bio import SeqIO
from biom import load_table

import logging
logger = logging.getLogger('workflow_16s')

# ================================== LOCAL IMPORTS =================================== #

from workflow_16s.utils.dir_utils import SubDirs

# ==================================== FUNCTIONS ===================================== #

def load_datasets_list(path: Union[str, Path]) -> List[str]:
    """
    Load dataset IDs from configuration file
    """
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]
    
    
def load_datasets_info(tsv_path: Union[str, Path]) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t", dtype={'ena_project_accession': str})
    # Remove 'Unnamed' columns
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
    return df


def fetch_first_match(dataset_info: pd.DataFrame, dataset: str):
    # Case-insensitive masks
    mask_ena_type = dataset_info['dataset_type'].str.lower().eq('ena')
    mask_manual_type = dataset_info['dataset_type'].str.lower().eq('manual')

    # Check ENA: either ena_project_accession OR dataset_id contains the dataset
    mask_ena = ((
        dataset_info['ena_project_accession'].str.contains(dataset, case=False, regex=False)
        | dataset_info['dataset_id'].str.contains(dataset, case=False, regex=False)
    ) & mask_ena_type)

    # Check Manual: dataset_id contains the dataset
    mask_manual = (
        dataset_info['dataset_id'].str.contains(dataset, case=False, regex=False) 
        & mask_manual_type
    )

    combined_mask = mask_ena | mask_manual
    matching_rows = dataset_info[combined_mask]

    # Handle no matches
    if matching_rows.empty:
        raise ValueError(f"No matches found for dataset: {dataset}")
        # Or return None, depending on your use case

    # Prioritize ENA matches over manual ones
    matching_rows = matching_rows.sort_values(
         by='dataset_type', 
         key=lambda x: x.str.lower().map({'ena': 0, 'manual': 1})  # ENA first
    )

    return matching_rows.iloc[0]
    
def processed_dataset_files(dirs: SubDirs, dataset: str, params: Any, cfg: Any) -> Dict[str, Path]:
    """Generate expected file paths for processed dataset outputs.
    
    Args:
        dirs: Project directory structure
        dataset: Dataset identifier
        classifier: Taxonomic classifier name
    
    Returns:
        Dictionary mapping file types to their expected paths
    """
    classifier = cfg["Classifier"]
    base_dir = (
        dirs.qiime_data / dataset / params['instrument_platform'].lower() / 
        params['library_layout'].lower() / params['target_subfragment'].lower() / 
        f"FWD_{params['pcr_primer_fwd_seq']}_REV_{params['pcr_primer_rev_seq']}"
    )
    base_dir.mkdir(parents=True, exist_ok=True)
    return {
        'metadata_tsv': dirs.metadata / dataset / 'metadata.tsv',
        'manifest_tsv': base_dir / 'manifest.tsv',
        'table_biom': base_dir / 'table' / 'feature-table.biom',  # BIOM format feature table
        'seqs_fasta': base_dir / 'rep-seqs' / 'dna-sequences.fasta',  # Representative sequences
        'taxonomy_tsv': base_dir / classifier / 'taxonomy' / 'taxonomy.tsv',  # Taxonomic assignments
    }

def missing_output_files(file_list: List[Union[str, Path]]) -> List[Path]:
    """Identify missing output files from a list of expected paths.
    
    Args:
        file_list: List of file paths to check
    
    Returns:
        List of paths that don't exist on the filesystem
    """
    return [Path(file) for file in file_list if not Path(file).exists()]

def import_metadata_tsv(tsv_path: Union[str, Path], index_col: str = '#SampleID') -> pd.DataFrame:
    """"""
    return pd.read_csv(tsv_path, sep="\t", encoding="utf8", low_memory=False, index_col=index_col).sort_index()


def import_features_biom(biom_path: Union[str, Path]) -> pd.DataFrame:
    """"""
    table = load_table(biom_path)
    
    feature_ids = table.ids(axis='observation')
    sample_ids = table.ids(axis='sample')
    
    data = table.matrix_data.toarray()
    return pd.DataFrame(data, index=feature_ids, columns=sample_ids)
    

def load_and_merge_biom_tables(biom_paths) -> Table:
    """Load and merge multiple BIOM tables with validation."""
    tables = []
    for path in biom_paths:
        try:
            tables.append(load_table(path))
        except Exception as e:
            logger.warning(f"Failed to load {path}: {str(e)}")
            continue

    if not tables:
        raise ValueError("No valid BIOM tables loaded")
    
    merged_table = tables[0]
    for table in tables[1:]:
        merged_table = merged_table.merge(table)
    logger.info(f"Merged table dimensions: {merged_table.shape}")
    return merged_table
    

def import_seqs_fasta(fasta_path: Union[str, Path]) -> pd.DataFrame:
    """"""
    seqs = dict(zip((record.id for record in SeqIO.parse(fasta_path, "fasta")), (str(record.seq) for record in SeqIO.parse(fasta_path, "fasta"))))
    return seqs



def import_faprotax_tsv(tsv_path: Union[str, Path]) -> pd.DataFrame:
    """"""
    df = pd.read_csv(tsv_path, sep="\t", encoding="utf8", low_memory=False).sort_index().T
    df.columns = df.iloc[0]
    df = df.iloc[1:]
    return df



class Taxonomy:
    """"""
    def __init__(self, tsv_path: Union[str, Path]):
       self.taxonomy = self.import_taxonomy_tsv(tsv_path)
        
    def import_taxonomy_tsv(self, tsv_path: Union[str, Path]) -> pd.DataFrame:
        def extract_taxonomic_level(taxonomy, level):
            level_prefix = level + '__'
            if taxonomy == 'Unassigned' or taxonomy == 'Unclassified':
                return 'Unclassified'
            start_index = taxonomy.find(level_prefix)
            if start_index != -1:
                end_index = taxonomy.find(';', start_index)
                if end_index != -1:
                    return taxonomy[start_index + len(level_prefix):end_index]
                else:
                    return taxonomy[start_index + len(level_prefix):]
            else:
                return None
        taxonomy = pd.read_csv(tsv_path, sep='\t', encoding='UTF8', skiprows=0)
        taxonomy = taxonomy.rename(columns={
            'Feature ID': 'id', 
            'Taxon'     : 'taxonomy', 
            'Consensus' : 'confidence'
        })
        # Index by Feature ID
        taxonomy = taxonomy.set_index('id')                                                                                       
        taxonomy['taxstring'] = [re.sub(' *[dpcofgs]__', '', s) for s in taxonomy['taxonomy'].values]
        # Create new columns for each taxonomic level
        for level in ['d', 'p', 'c', 'o', 'f', 'g', 's']:
            taxonomy[level.capitalize()] = taxonomy['taxonomy'].apply(lambda x: extract_taxonomic_level(x, level))                
    
        column_mapping = {
            'D': 'Domain', 
            'P': 'Phylum', 
            'C': 'Class', 
            'O': 'Order', 
            'F': 'Family', 
            'G': 'Genus', 
            'S': 'Species'
        }
        #  Rename the new columns
        taxonomy.rename(columns=column_mapping, inplace=True)                                                                        
    
        taxonomy.reset_index(inplace=True)
        # Reset the index to assign unique numbers to the 'id' column and store them in a new column 'n'
        taxonomy.rename(columns={'index': 'n'}, inplace=True)                                                                        
        taxonomy.set_index('id', inplace=True)
        return taxonomy
        
    def get_taxstring_by_id(self, id):
        """"""
        row = self.taxonomy[self.taxonomy['id'] == id]
        if not row.empty:
            return row.iloc[0]['taxstring']
        else:
            return None

    def get_taxstring_by_asv(self, asv):
        """"""
        row = self.taxonomy[self.taxonomy['trimmed_asv'] == id]
        if not row.empty:
            return row.iloc[0]['taxstring']
        else:
            return None


def write_metadata_tsv(df: pd.DataFrame, tsv_path: str) -> None:
    """"""
    df = df.copy()
    # Create #SampleID column from 'run_accession' if it doesn't exist
    if '#SampleID' not in df.columns:
        df['#SampleID'] = df['run_accession']
    
    df.set_index('#SampleID', inplace=True, drop=True)
    df.to_csv(tsv_path, sep='\t', index=True)
    #logger.info(f"Saved sample metadata to: {tsv_path}")


def write_manifest_tsv(results: Dict, tsv_path: str) -> None:
    """"""
    rows = []
    
    for run_accession, file_paths in results.items():
        if type(file_paths) != list:
            file_paths = str(file_paths).split(';')
        num_files = len(file_paths)
        
        if num_files == 1:
            rows.append({
                    'sample-id'        : run_accession,
                    'absolute-filepath': file_paths[0]
                })
            
        elif num_files == 2:
            rows.append({
                    'sample-id'                : run_accession,
                    'forward-absolute-filepath': file_paths[0],
                    'reverse-absolute-filepath': file_paths[1]
                })
            
        else:
            logger.debug(f"Warning: Run accession {run_accession} has an invalid number of file paths ({num_files}).")
            break

    # Create DataFrame
    df = pd.DataFrame(rows)
    df.set_index('sample-id', inplace=True, drop=True)
    df.to_csv(tsv_path, sep='\t', index=True)
    #logger.info(f"Saved manifest file to: {tsv_path}")
    

def manual_meta(dataset: str, metadata_dir: Union[str, Path]):
    """
    Retrieves and processes manually-collected dataset metadata.
    
    Args:
        dataset: ENA Project Accession number (e.g., PRJEB1234)
    
    Returns:
        Dictionary containing metadata, run characteristics, and filtered run lists.
        Returns None if invalid dataset format or metadata retrieval fails.
    """
    manual_metadata_tsv = Path(metadata_dir) / dataset / 'manual-metadata.tsv'
    if manual_metadata_tsv.is_file():
        return pd.read_csv(manual_metadata_tsv, sep="\t", encoding="utf8", low_memory=False)
    else:
        return pd.DataFrame({})
    
