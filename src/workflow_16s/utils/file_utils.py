# ===================================== IMPORTS ====================================== #

from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import re
import pandas as pd
from Bio import SeqIO
from biom import load_table
from biom.table import Table

import logging

# ================================== LOCAL IMPORTS =================================== #

from workflow_16s.utils.dir_utils import SubDirs

logger = logging.getLogger('workflow_16s')

# ==================================== FUNCTIONS ===================================== #

def load_datasets_list(path: Union[str, Path]) -> List[str]:
    """Load dataset IDs from configuration file."""
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]
    
    
def load_datasets_info(tsv_path: Union[str, Path]) -> pd.DataFrame:
    """"""
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
        dataset_info['ena_project_accession'].str.contains(
            dataset, case=False, regex=False
        )
        | dataset_info['dataset_id'].str.contains(
            dataset, case=False, regex=False
        )
    ) & mask_ena_type)

    # Check Manual: dataset_id contains the dataset
    mask_manual = (
        dataset_info['dataset_id'].str.contains(
            dataset, case=False, regex=False
        ) 
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


def processed_dataset_files(
    dirs: SubDirs, 
    dataset: str, 
    params: Any, 
    cfg: Any
) -> Dict[str, Path]:
    """
    Generate expected file paths for processed dataset outputs.
    
    Args:
        dirs:       Project directory structure.
        dataset:    Dataset identifier.
        classifier: Taxonomic classifier name.
    
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
        # BIOM format feature table
        'table_biom': base_dir / 'table' / 'feature-table.biom',  
        # Representative sequences
        'seqs_fasta': base_dir / 'rep-seqs' / 'dna-sequences.fasta',  
        # Taxonomic assignments
        'taxonomy_tsv': base_dir / classifier / 'taxonomy' / 'taxonomy.tsv',  
    }


def missing_output_files(file_list: List[Union[str, Path]]) -> List[Path]:
    """
    Identify missing output files from a list of expected paths.
    
    Args:
        file_list: List of file paths to check.
    
    Returns:
        List of paths that don't exist on the filesystem
    """
    return [Path(file) for file in file_list if not Path(file).exists()]


def import_metadata_tsv(
    tsv_path: Union[str, Path], 
    index_col: str = '#SampleID'
) -> pd.DataFrame:
    """
    
    """
    return pd.read_csv(
        tsv_path, sep="\t", encoding="utf8", low_memory=False, index_col=index_col
    ).sort_index()


def get_first_existing_column(df, columns):
    # Loop through the list of columns
    for col in columns:
        if col in df.columns:
            return df[col]  # Return the first existing column's data
    return None  # Return None if none are found

class AmpliconData:
    def __init__(
        self, 
        project_dir: Union[str, Path] = "/usr2/people/macgregor/amplicon/test",
        mode: str = 'genus',
        verbose: bool = True
    ):
        self.project_dir = project_dir
        self.mode = mode
        self.verbose = verbose
        

        self.table = None
        self.meta = None
        self.taxa = None
        
        if mode == 'asv':
            self.BIOM_PATTERN = "data/per_dataset/qiime/*/*/*/*/FWD_*_REV_*/table/feature-table.biom"
            self.output_path = os.path.join(str(self.project_dir), 'data/merged/l6/feature-table.biom')

        elif mode == 'genus':
            self.BIOM_PATTERN = "data/per_dataset/qiime/*/*/*/*/FWD_*_REV_*/table_6/feature-table.biom"
            self.output_path = os.path.join(str(self.project_dir), 'data/merged/asv/feature-table.biom')

        self._get_biom_table()
        self._get_metadata()

    def _get_biom_paths(self):
        return glob.glob(os.path.join(
            str(self.project_dir), self.BIOM_PATTERN
        ), recursive=True)    

    def _get_meta_paths(self):
        meta_paths = []
        for biom_path in self._get_biome_paths():
            # Navigate upwards using .parents
            qiime_dir = Path(biom_path).parents[5]  # Equivalent to 6 levels up
            base_dir = Path(biom_path).parents[9]   # Equivalent to 10 levels up
            project_id = qiime_dir.name
            
            # Build metadata path using pathlib
            try:
                # Get relative path from base_dir to biom_path and take the first 5 components
                subdirs = biom_path.relative_to(base_dir).parts[:5]
            except ValueError:
                raise ValueError(f"{biom_path} is not relative to {base_dir}")
            
            meta_path = (
                base_dir / "data" / "per_dataset" / "metadata" 
                / Path(*subdirs) / "sample-metadata.tsv"
            )
            meta_paths.append(meta_path)
        return meta_paths
        
    def _get_biom_table(self):
        biom_paths = self._get_biome_paths()
        if not biom_paths:
            raise FileNotFoundError(f"No BIOM files found matching {self.BIOM_PATTERN}")   
        if self.verbose:
            logger.info(f"Found {len(biom_paths)} BIOM files")
        self.table = import_merged_table_biom(
            biom_paths, 
            'dataframe',
            self.output_path,
            self.verbose
        )
        
    def _get_metadata(self):
        def process_meta_path(csv_path, column_renames):
            if column_renames == None:
                column_renames = []
            if not os.path.exists(csv_path):
                raise FileNotFoundError()
            try:
                df = pd.read_csv(meta_path, sep='\t')#, index_col=0)
                
                # Convert column names to lowercase
                df.columns = df.columns.str.lower() 
                
                options = ['run_accession', '#sampleid', 'sample-id']
                sample_id_col = get_first_existing_column(df, options)
                if sample_id_col:
                    df['SAMPLE ID'] = sample_id_col
                else:
                    df['SAMPLE ID'] = [
                        f"{str(Path(csv_path).parents[5].name)}_x" 
                        for x in range(1, df.shape[0]+1)
                    ] 
                    
                options = ['project_accession', 'dataset_id', 'dataset_name']
                dataset_id_col = get_first_existing_column(df, options)
                if dataset_id_col:
                    df['DATASET ID'] = dataset_id_col
                else:
                    df['DATASET ID'] = str(Path(csv_path).parents[5].name) 
                
                # Add nuclear_contamination_status if missing
                if 'nuclear_contamination_status' not in df.columns:
                    df['nuclear_contamination_status'] = False

                # Rename columns 
                df = df.rename(columns=column_renames)  

                # Remove duplicate columns
                duplicate_cols = df.columns[df.columns.duplicated()]
                if len(duplicate_cols) > 0:
                    df = df.loc[:, ~df.columns.duplicated()]
                    
                if self.verbose:
                    logger.info(
                        f"Loaded {str(Path(csv_path).parents[5].name)} "
                        f"with {df.shape[0]} samples."
                    )
            
            except Exception as e:
                logger.warning(f"Error loading {meta_path}: {str(e)}") 
                
        meta_dfs = []
        for meta_path in self._get_meta_paths():  
            meta_df = process_meta_path(meta_path)
            meta_dfs.append(meta_df)
        merged_df = pd.concat(meta_dfs, ignore_index=True)
        #merged_df = pd.merge(
        #    processed_metadata_df,
        #    manual_metadata_df,
        #    on='unified_sample_id',
        #    how='outer',
        #    suffixes=('_auto', '_manual')
        #)
                
                
        
# ======================================= BIOM ======================================= #

def import_table_biom(
    biom_path: Union[str, Path], as_type: str = 'table'
) -> Union[Table, pd.DataFrame]:
    """
    Import a BIOM feature table as a Table or pandas DataFrame.

    Args:
        biom_path: Path where the .biom file is located.
        as_type:   Type to return the table as.
        
    Returns:
        Table or Pandas DataFrame
    """
    try:
        with biom_open(biom_path) as f:
            table = Table.from_hdf5(f)
    except:
        table = load_table(biom_path)
        
    if as_type == 'table':
        return table
    elif as_type == 'dataframe':
        feature_ids = table.ids(axis='observation')
        sample_ids = table.ids(axis='sample')
        data = table.matrix_data.toarray()
        df = pd.DataFrame(data, index=feature_ids, columns=sample_ids)
        return df
    else:
        raise ValueError(
            f"Output type '{as_type}' not recognized. "
            f"Expected 'table' or 'dataframe'."
        )    

def import_merged_table_biom(
    biom_paths: List[Union[str, Path]], 
    as_type: str = 'table',
    output_path: Union[str, Path] = '/usr2/people/macgregor/amplicon/test/data/merged/l6/feature-table.biom',
    verbose: bool = False
) -> Union[Table, pd.DataFrame]:
    """
    Import and merge a list of BIOM feature tables.

    Args:
        biom_paths:
        as_type:
        output_dir:

    Returns:
        merged_table:
    """    
    if verbose:
        logger.info(f"Found {len(biom_paths)} BIOM files")
    
    tables = []
    for biom_path in biom_paths:
        try:
            table = import_table_biom(biom_path, 'table')
            tables.append(table)
            if verbose:
                biom_path_parts = os.path.normpath(biom_path).split(os.sep)
                biom_path_report = os.sep.join(parts[-3:])
                if verbose:
                    logger.info(
                        f"Loaded {biom_path_report} with "
                        f"{len(table.ids(axis='sample'))} samples."
                    )
            
        except Exception as e:
            logger.error(f"Failed to load {biom_path}: {str(e)}")
            continue
    
    if not tables:
        raise ValueError("No valid BIOM tables loaded")
    
    # Merge tables
    merged_table = tables[0]
    for table in tables[1:]:
        merged_table = merged_table.merge(table)

    if verbose:
        logger.info(f"Merged table contains {merged_table.shape[1]} samples.")

    if output_path:
        # Create output directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output_path, 'w') as f:
            merged_table.to_hdf5(
                f, 
                generated_by="/usr2/people/macgregor/amplicon/workflow_16s/run.sh"
            )
        if verbose:
            logger.info(f"Wrote merged table to '{output_path}'")

    if as_type == 'table':
        return merged_table
    elif as_type == 'dataframe':
        feature_ids = merged_table.ids(axis='observation')
        sample_ids = merged_table.ids(axis='sample')
        data = merged_table.matrix_data.toarray()
        df = pd.DataFrame(data, index=feature_ids, columns=sample_ids)
        return df
    else:
        raise ValueError(
            f"Output type '{as_type}' not recognized. "
            f"Expected 'table' or 'dataframe'."
        )    

# ====================================== FASTA ======================================= #

def import_seqs_fasta(fasta_path: Union[str, Path]) -> pd.DataFrame:
    """
    
    """
    seqs = dict(zip(
        (record.id for record in SeqIO.parse(fasta_path, "fasta")), 
        (str(record.seq) for record in SeqIO.parse(fasta_path, "fasta"))
    ))
    return seqs


def import_faprotax_tsv(tsv_path: Union[str, Path]) -> pd.DataFrame:
    """
    
    """
    df = pd.read_csv(
        tsv_path, sep="\t", encoding="utf8", low_memory=False
    ).sort_index().T
    df.columns = df.iloc[0]
    df = df.iloc[1:]
    return df

# ==================================== CLASSES ====================================== #

class Taxonomy:
    """
    
    """
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
        taxonomy['taxstring'] = [re.sub(' *[dpcofgs]__', '', s) 
                                 for s in taxonomy['taxonomy'].values]
        # Create new columns for each taxonomic level
        for level in ['d', 'p', 'c', 'o', 'f', 'g', 's']:
            taxonomy[level.capitalize()] = taxonomy['taxonomy'].apply(
                lambda x: extract_taxonomic_level(x, level)
            )                
    
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
        # Reset the index to assign unique numbers to the 'id' column and store 
        # them in a new column 'n'
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
    """
    
    """
    df = df.copy()
    # Create #SampleID column from 'run_accession' if it doesn't exist
    if '#SampleID' not in df.columns:
        df['#SampleID'] = df['run_accession']
    
    df.set_index('#SampleID', inplace=True, drop=True)
    df.to_csv(tsv_path, sep='\t', index=True)


def write_manifest_tsv(results: Dict, tsv_path: str) -> None:
    """
    
    """
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
            logger.debug(
                f"Warning: Run accession {run_accession} has an "
                f"invalid number of file paths ({num_files})."
            )
            break

    # Create DataFrame
    df = pd.DataFrame(rows)
    df.set_index('sample-id', inplace=True, drop=True)
    df.to_csv(tsv_path, sep='\t', index=True)
    

def manual_meta(dataset: str, metadata_dir: Union[str, Path]):
    """
    Retrieves and processes manually-collected dataset metadata.
    
    Args:
        dataset: ENA Project Accession number (e.g., PRJEB1234).
    
    Returns:
        Dictionary containing metadata, run characteristics, and filtered 
        run lists. Returns None if invalid dataset format or metadata 
        retrieval fails.
    """
    manual_metadata_tsv = Path(metadata_dir) / dataset / 'manual-metadata.tsv'
    if manual_metadata_tsv.is_file():
        return pd.read_csv(
            manual_metadata_tsv, sep="\t", encoding="utf8", low_memory=False
        )
    else:
        return pd.DataFrame({})
    
