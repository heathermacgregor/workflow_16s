# ===================================== IMPORTS ====================================== #

from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import re
import os
import glob
import h5py
import pandas as pd
import numpy as np
from Bio import SeqIO
from biom import load_table
from biom.table import Table
import logging

# ================================== LOCAL IMPORTS =================================== #

from workflow_16s.utils.dir_utils import SubDirs
from workflow_16s.stats.utils import t_test

logger = logging.getLogger('workflow_16s')

# ==================================== FUNCTIONS ===================================== #

def load_datasets_list(path: Union[str, Path]) -> List[str]:
    """Load dataset IDs from configuration file.
    
    Args:
        path: Path to text file containing dataset IDs (one per line)
    
    Returns:
        List of dataset ID strings
    """
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]
    
    
def load_datasets_info(tsv_path: Union[str, Path]) -> pd.DataFrame:
    """Load dataset metadata from TSV file.
    
    Args:
        tsv_path: Path to TSV file containing dataset metadata
    
    Returns:
        DataFrame with dataset information, cleaned of unnamed columns
    """
    df = pd.read_csv(tsv_path, sep="\t", dtype={'ena_project_accession': str})
    # Remove 'Unnamed' columns
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
    return df


def fetch_first_match(dataset_info: pd.DataFrame, dataset: str) -> pd.Series:
    """Find matching dataset information from metadata DataFrame.
    
    Args:
        dataset_info: DataFrame containing dataset metadata
        dataset: Dataset identifier to search for
    
    Returns:
        First matching row from dataset_info as a pandas Series
    
    Raises:
        ValueError: If no matches found for the dataset
    """
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
    """Generate expected file paths for processed dataset outputs.
    
    Args:
        dirs: Project directory structure
        dataset: Dataset identifier
        params: Processing parameters dictionary
        cfg: Configuration dictionary
    
    Returns:
        Dictionary mapping file types to their expected paths
    """
    classifier = cfg["Classifier"]
    base_dir = (
        Path(dirs.qiime_data_per_dataset) / dataset / params['instrument_platform'].lower() / 
        params['library_layout'].lower() / params['target_subfragment'].lower() / 
        f"FWD_{params['pcr_primer_fwd_seq']}_REV_{params['pcr_primer_rev_seq']}"
    )
    base_dir.mkdir(parents=True, exist_ok=True)
    return {
        'metadata_tsv': Path(dirs.metadata_per_dataset) / dataset / 'metadata.tsv',
        'manifest_tsv': base_dir / 'manifest.tsv',
        'table_biom': base_dir / 'table' / 'feature-table.biom',  # BIOM feature table
        'seqs_fasta': base_dir / 'rep-seqs' / 'dna-sequences.fasta',  # Representative seqs
        'taxonomy_tsv': base_dir / classifier / 'taxonomy' / 'taxonomy.tsv',  # Taxonomy
    }


def missing_output_files(file_list: List[Union[str, Path]]) -> List[Path]:
    """Identify missing output files from a list of expected paths.
    
    Args:
        file_list: List of file paths to check
    
    Returns:
        List of Path objects for files that don't exist
    """
    return [Path(file) for file in file_list if not Path(file).exists()]

# ==================================== FUNCTIONS ===================================== #

def convert_to_biom(table: pd.DataFrame) -> Table:
    """Convert pandas DataFrame to BIOM Table.
    
    Args:
        table: Input DataFrame containing feature counts
    
    Returns:
        BIOM Table representation of the DataFrame
    """
    if not isinstance(table, pd.DataFrame):
        return table
    
    observation_ids = table.index.astype(str).tolist()
    sample_ids = table.columns.astype(str).tolist()
    data = table.values
    
    return Table(
        data=data,
        observation_ids=observation_ids,
        sample_ids=sample_ids,
        type="OTU table"
    )
    

def collapse_taxa(
    table: Union[pd.DataFrame, Table], 
    target_level: str, 
    output_dir: Union[str, Path],
    verbose: bool = True
) -> Table:
    """Collapse feature table to specified taxonomic level.
    
    Args:
        table: Input BIOM Table or DataFrame
        target_level: Taxonomic level to collapse to (phylum/class/order/family)
        output_dir: Directory to save collapsed table
    
    Returns:
        Collapsed BIOM Table
    
    Raises:
        ValueError: For invalid target_level
    """
    if not isinstance(table, Table):
        table = convert_to_biom(table)
        
    levels = {
        'phylum': 1, 'class': 2, 'order': 3, 'family': 4
    }

    if target_level not in levels:
        raise ValueError(
            f"Invalid `target_level`: {target_level}. "
            f"Expected one of {list(levels.keys())}")

    level_idx = levels[target_level]

    # Create taxonomy mapping
    id_map = {}
    for taxon in table.ids(axis='observation').astype(str):
        parts = taxon.split(';')
        truncated = ';'.join(parts[:level_idx + 1]) if len(parts) >= level_idx + 1 else 'Unclassified'
        id_map[taxon] = truncated

    # Collapse table
    collapsed_table = table.collapse(
        lambda id, _: id_map.get(id, 'Unclassified'),
        norm=False,
        axis='observation',
        include_collapsed_metadata=False
    ).remove_empty()

    # Save output
    output_biom_path = Path(output_dir) / f'l{level_idx + 1}' / "feature-table.biom"
    output_biom_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_biom_path, 'w') as f:
        collapsed_table.to_hdf5(f, generated_by=f"Collapsed to {target_level}")
    if verbose:
        print(table)
        n_features, n_samples = table.shape
        # Format into [x, y] string
        shape_str = f"[{n_features}, {n_samples}]"
        logger.info(
            f"Wrote table {shape_str} collapsed to {target_level} to '{output_biom_path}'"
        )
    
    return collapsed_table

def presence_absence(
    table: Union[Table, pd.DataFrame], 
    target_level: str, 
    output_dir: Union[str, Path],
    verbose: bool = True
) -> Table:
    """
    Convert table to presence/absence format and filter by abundance.
    
    Args:
        table: Input BIOM Table or DataFrame.
        target_level: Taxonomic level for output naming.
        output_dir: Directory to save output.
    
    Returns:
        Presence/absence BIOM Table filtered by abundance.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if not isinstance(table, Table):
        table = convert_to_biom(table)

    levels = {'phylum': 1, 'class': 2, 'order': 3, 'family': 4, 'genus': 5}
    
    # Filter by abundance
    feature_sums = np.array(table.sum(axis='observation')).flatten()
    sorted_idx = np.argsort(feature_sums)[::-1]
    cumulative = np.cumsum(feature_sums[sorted_idx]) / feature_sums.sum()
    stop_idx = np.searchsorted(cumulative, 0.99) + 1
    keep_ids = [table.ids(axis='observation')[i] for i in sorted_idx[:stop_idx]]
    
    # Convert to presence/absence
    pa_table = table.pa(inplace=False)
    pa_table_filtered = pa_table.filter(keep_ids, axis='observation')
    pa_df_filtered = pa_table_filtered.to_dataframe(dense=True)

    # Save output
    pa_table = Table(
        pa_df_filtered.values,
        pa_df_filtered.index,
        pa_df_filtered.columns,
        table_id='Presence Absence BIOM Table'
    )
    output_biom_path = Path(output_dir) / f'l{levels[target_level]+1}' / "feature-table_pa.biom"
    output_biom_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_biom_path, 'w') as f:
        pa_table.to_hdf5(f, generated_by=f"Collapsed to {target_level}")
    if verbose:
        n_features, n_samples = pa_table.shape
        # Format into [x, y] string
        shape_str = f"[{n_features}, {n_samples}]"
        logger.info(
            f"Wrote presence-absence table {shape_str} to '{output_biom_path}'"
        )
    
    return pa_table


def filter_presence_absence(
    table: Table, 
    metadata: pd.DataFrame, 
    col: str = 'nuclear_contamination_status', 
    prevalence_threshold: float = 0.05, 
    group_threshold: float = 0.05
) -> Table:
    """
    Filter presence/absence table based on prevalence and group differences.
    
    Args:
        table: Input BIOM Table
        metadata: Sample metadata DataFrame
        col: Metadata column to group by
        prevalence_threshold: Minimum prevalence across all samples
        group_threshold: Minimum prevalence difference between groups
    
    Returns:
        Filtered BIOM Table
    """
    df = table.to_dataframe(dense=True).T
    metadata = metadata.set_index("run_accession.1")
    df_with_meta = df.join(metadata[[col]], how='inner')

    # Apply prevalence filter
    if prevalence_threshold:
        species_data = df_with_meta.drop(columns=[col])
        prev = species_data.mean(axis=0)
        filtered_species = prev[prev >= prevalence_threshold].index
        df_with_meta = df_with_meta[filtered_species.union(pd.Index([col]))]

    # Apply group filter
    if group_threshold:
        groups = df_with_meta.groupby(col)
        if True not in groups.groups or False not in groups.groups:
            raise ValueError(f"Metadata column '{col}' must have True/False groups")
        sum_per_group = groups.sum(numeric_only=True)
        n_samples = groups.size()
        percentages = sum_per_group.div(n_samples, axis=0)
        mask = (percentages.loc[True] >= group_threshold) & (percentages.loc[False] >= group_threshold)
        selected_species = mask[mask].index
        df_with_meta = df_with_meta[selected_species.union(pd.Index([col]))]

    return Table(
        df_with_meta.drop(columns=[col]).values.T,
        df_with_meta.columns.tolist(),
        df_with_meta.index.tolist(),
        table_id='Filtered Presence/Absence Table'
    )
    
    
class AmpliconData:
    """
    Main class for handling amplicon sequencing data analysis.
    
    Attributes:
        project_dir: Root directory for project data
        mode: Analysis mode ('asv' or 'genus')
        verbose: Enable verbose logging
        table: Main feature table
        meta: Metadata DataFrame
        taxa: Taxonomy information
        tables: Dictionary of collapsed tables by level
        presence_absence_tables: PA tables by taxonomic level
    """
    def __init__(
        self, 
        cfg,
        project_dir: Union[str, Path] = "/usr2/people/macgregor/amplicon/test",
        mode: str = 'genus',
        verbose: bool = True
    ):
        self.cfg = cfg
        self.project_dir = project_dir
        self.mode = mode
        self.verbose = verbose
        self.table = None
        self.meta = None
        self.taxa = None

        self.tables = {}
        self.presence_absence_tables = {}
        
        if self.mode == 'asv':
            table_dir = 'table'
            output_dir = 'asv'
        elif self.mode == 'genus':
            table_dir = 'table_6'
            output_dir = 'l6'

        self.BIOM_PATTERN = '/'.join([
            'data', 'per_dataset', 'qiime', '*', '*', '*', '*', 
            'FWD_*_REV_*', table_dir, 'feature-table.biom'
        ])
        self.output_path = Path(project_dir) / 'data' / 'merged' / output_dir / 'feature-table.biom'
        self._get_metadata()
        self._get_biom_table()
        
        # Collapse tables
        
        if self.mode == 'genus':
            for level in ['phylum', 'class', 'order', 'family']:
                collapsed_table = collapse_taxa(
                    self.table, 
                    level, 
                    Path(project_dir) / 'data' / 'merged',
                    self.verbose
                )
                self.tables[level] = collapsed_table
            self.tables['genus'] = self.table   

            if self.cfg['presence_absence']:
                for level in self.tables:
                    pa = presence_absence(
                        self.tables[level], 
                        level, 
                        Path(project_dir) / 'data' / 'merged',
                        self.verbose
                    )
                    self.presence_absence_tables[level] = pa

        
        elif self.mode == 'asv':
            logger.info("ASV mode is not yet supported!")

        self.stats = {}
        self.stats['raw'] = {}
        if self.cfg['stats']['raw']['t_test']:
            self.stats['raw']['t_test'] = {}
            for level in self.tables:
                logger.info(f"Running t-test for {level}...")
                results = t_test(
                    table=self.tables[level], 
                    metadata=self.meta,
                    col='nuclear_contamination_status',
                    col_values=[True, False]
                )
                self.stats['raw']['t_test'][level] = results

        self.stats['presence_absence'] = {}
        if self.cfg['stats']['presence_absence']['t_test']:
            self.stats['presence_absence']['t_test'] = {}
            for level in self.presence_absence_tables:
                logger.info(f"Running t-test for {level}...")
                results = t_test(
                    table=self.presence_absence_tables[level], 
                    metadata=self.meta,
                    col='nuclear_contamination_status',
                    col_values=[True, False]
                )
                self.stats['presence_absence']['t_test'][level] = results
            

    def _get_biom_paths(self) -> List[str]:
        """Get paths to BIOM files matching pattern."""
        return glob.glob(str(Path(self.project_dir) / self.BIOM_PATTERN), recursive=True)    

    def _get_meta_paths(self) -> List[Path]:
        """Generate metadata paths corresponding to BIOM files."""
        meta_paths = []
        for biom_path in self._get_biom_paths():
            biom_path = Path(biom_path)
            
            # Find the 'data' directory
            current = biom_path
            data_dir = None
            while current != current.root:
                if current.name == 'data':
                    data_dir = current
                    break
                current = current.parent
            if not data_dir:
                raise ValueError(f"{biom_path} is not under a 'data' directory")
            
            biom_dir = data_dir / "per_dataset" / "qiime"
            try:
                rel_path = biom_path.parent.relative_to(biom_dir)
            except ValueError:
                raise ValueError(f"{biom_path} is not under {biom_dir}")
            
            # Slice to remove last 2 directory levels
            sliced_parts = rel_path.parts[:-1]  # Removes 'FWD...' and 'table_6'
            
            meta_path = (
                data_dir / "per_dataset" / "metadata"
                / Path(*sliced_parts)  # Use sliced parts
                / "sample-metadata.tsv"
            )
            meta_paths.append(meta_path)
        return meta_paths
        
    def _get_biom_table(self):
        """Load and merge BIOM tables from discovered paths."""
        biom_paths = self._get_biom_paths()
        if not biom_paths:
            raise FileNotFoundError(f"No BIOM files found matching {self.BIOM_PATTERN}")   
        self.table = import_merged_table_biom(
            biom_paths, 
            'dataframe',
            self.output_path,
            self.verbose
        )
        
    def _get_metadata(self):
        """Load and merge metadata from all datasets."""
        meta_dfs = []
        for meta_path in self._get_meta_paths():  
            meta_df = self._process_meta_path(meta_path, [])
            meta_dfs.append(meta_df)
        self.meta = pd.concat(meta_dfs, ignore_index=True)

    def _process_meta_path(
        self, 
        csv_path: Path, 
        column_renames: List[Tuple[str, str]]
    ) -> pd.DataFrame:
        """Process individual metadata file.
        
        Args:
            csv_path: Path to metadata file
            column_renames: List of (old_name, new_name) tuples
        
        Returns:
            Processed metadata DataFrame
        """
        if not csv_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {csv_path}")
            
        df = pd.read_csv(csv_path, sep='\t')
        df.columns = df.columns.str.lower()
        
        # Handle sample IDs
        sample_id_col = next((col for col in ['run_accession', '#sampleid', 'sample-id'] if col in df.columns), None)
        if sample_id_col:
            df['SAMPLE ID'] = df[sample_id_col]
        else:
            df['SAMPLE ID'] = [f"{Path(csv_path).parents[5].name}_x{i}" for i in range(1, len(df)+1)]
            
        # Handle dataset IDs
        dataset_id_col = next((col for col in ['project_accession', 'dataset_id', 'dataset_name'] if col in df.columns), None)
        if dataset_id_col:
            df['DATASET ID'] = df[dataset_id_col]
        else:
            df['DATASET ID'] = Path(csv_path).parents[5].name
            
        # Add contamination status if missing
        if 'nuclear_contamination_status' not in df.columns:
            df['nuclear_contamination_status'] = False

        # Apply column renames
        for old, new in column_renames:
            if old in df.columns:
                df.rename(columns={old: new}, inplace=True)
                
        return df


def import_metadata_tsv(
    tsv_path: Union[str, Path], 
    index_col: str = '#SampleID'
) -> pd.DataFrame:
    """Import metadata from TSV file.
    
    Args:
        tsv_path: Path to metadata TSV
        index_col: Column to use as index
    
    Returns:
        Metadata DataFrame
    """
    return pd.read_csv(
        tsv_path, sep="\t", encoding="utf8", low_memory=False, index_col=index_col
    ).sort_index()

   
# ======================================= BIOM ======================================= #

def import_table_biom(
    biom_path: Union[str, Path], 
    as_type: str = 'table'
) -> Union[Table, pd.DataFrame]:
    """Import BIOM table from file.
    
    Args:
        biom_path: Path to .biom file
        as_type: Return type ('table' or 'dataframe')
    
    Returns:
        BIOM Table or DataFrame
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
        return pd.DataFrame(data, index=feature_ids, columns=sample_ids)
    else:
        raise ValueError(f"Invalid as_type: {as_type}. Use 'table' or 'dataframe'")

def import_merged_table_biom(
    biom_paths: List[Union[str, Path]], 
    as_type: str = 'table',
    output_path: Union[str, Path] = None,
    verbose: bool = False
) -> Union[Table, pd.DataFrame]:
    """Merge multiple BIOM tables into one.
    
    Args:
        biom_paths: List of paths to BIOM files
        as_type: Return type ('table' or 'dataframe')
        output_path: Optional path to save merged table
        verbose: Enable progress logging
    
    Returns:
        Merged BIOM Table or DataFrame
    """
    tables = []
    for path in biom_paths:
        try:
            table = import_table_biom(path, 'table')
            tables.append(table)
            if verbose:
                logger.info(f"Loaded {Path(path).name} with {len(table.ids('sample'))} samples")
        except Exception as e:
            logger.error(f"Failed to load {path}: {str(e)}")
    
    if not tables:
        raise ValueError("No valid BIOM tables loaded")
    
    merged_table = tables[0]
    for table in tables[1:]:
        merged_table = merged_table.merge(table)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output_path, 'w') as f:
            merged_table.to_hdf5(f, generated_by="workflow_16s")
        if verbose:
            n_features, n_samples = table.shape
            # Format into [x, y] string
            shape_str = f"[{n_features}, {n_samples}]"
            logger.info(f"Wrote table {shape_str} to {output_path}")

    return merged_table if as_type == 'table' else merged_table.to_dataframe()

# ====================================== FASTA ======================================= #

def import_seqs_fasta(fasta_path: Union[str, Path]) -> Dict[str, str]:
    """Import sequences from FASTA file.
    
    Args:
        fasta_path: Path to FASTA file
    
    Returns:
        Dictionary mapping sequence IDs to sequences
    """
    return {
        record.id: str(record.seq)
        for record in SeqIO.parse(fasta_path, "fasta")
    }

def import_faprotax_tsv(tsv_path: Union[str, Path]) -> pd.DataFrame:
    """Import FAPROTAX results from TSV.
    
    Args:
        tsv_path: Path to FAPROTAX TSV output
    
    Returns:
        Transposed DataFrame with samples as rows and functions as columns
    """
    df = pd.read_csv(tsv_path, sep="\t", index_col=0)
    return df.T

# ==================================== CLASSES ====================================== #

class Taxonomy:
    """Class for handling taxonomic classification data.
    
    Attributes:
        taxonomy: DataFrame containing taxonomy information
    """
    def __init__(self, tsv_path: Union[str, Path]):
        self.taxonomy = self.import_taxonomy_tsv(tsv_path)
        
    def import_taxonomy_tsv(self, tsv_path: Union[str, Path]) -> pd.DataFrame:
        """Process taxonomy TSV file into structured DataFrame.
        
        Args:
            tsv_path: Path to QIIME2 taxonomy TSV
        
        Returns:
            DataFrame with separate columns for each taxonomic level
        """
        def extract_level(taxonomy: str, level: str) -> str:
            prefix = level + '__'
            if not taxonomy or taxonomy in ['Unassigned', 'Unclassified']:
                return 'Unclassified'
            start = taxonomy.find(prefix)
            if start == -1:
                return None
            end = taxonomy.find(';', start)
            return taxonomy[start+len(prefix):end] if end != -1 else taxonomy[start+len(prefix):]
        
        df = pd.read_csv(tsv_path, sep='\t')
        df = df.rename(columns={
            'Feature ID': 'id', 
            'Taxon': 'taxonomy', 
            'Consensus': 'confidence'
        }).set_index('id')
        
        # Clean taxonomy strings
        df['taxstring'] = df['taxonomy'].str.replace(r' *[dpcofgs]__', '', regex=True)
        
        # Add taxonomic levels
        for level in ['d', 'p', 'c', 'o', 'f', 'g', 's']:
            df[level.upper()] = df['taxonomy'].apply(lambda x: extract_level(x, level))
            
        return df.rename(columns={
            'D': 'Domain', 'P': 'Phylum', 'C': 'Class',
            'O': 'Order', 'F': 'Family', 'G': 'Genus', 'S': 'Species'
        })
        
    def get_taxstring_by_id(self, feature_id: str) -> Optional[str]:
        """Get taxonomy string for a feature ID.
        
        Args:
            feature_id: Feature ID to look up
        
        Returns:
            Taxonomy string or None if not found
        """
        return self.taxonomy.loc[feature_id, 'taxstring'] if feature_id in self.taxonomy.index else None


def write_metadata_tsv(df: pd.DataFrame, tsv_path: str) -> None:
    """Write metadata DataFrame to TSV format.
    
    Args:
        df: Metadata DataFrame
        tsv_path: Output path for TSV file
    """
    df = df.copy()
    if '#SampleID' not in df.columns and 'run_accession' in df.columns:
        df['#SampleID'] = df['run_accession']
    df.set_index('#SampleID', inplace=True)
    df.to_csv(tsv_path, sep='\t', index=True)


def write_manifest_tsv(results: Dict, tsv_path: str) -> None:
    """Write manifest file for QIIME2 import.
    
    Args:
        results: Dictionary mapping run accessions to file paths
        tsv_path: Output path for manifest file
    """
    rows = []
    for run_id, paths in results.items():
        if len(paths) == 1:
            rows.append({'sample-id': run_id, 'absolute-filepath': paths[0]})
        elif len(paths) == 2:
            rows.append({
                'sample-id': run_id,
                'forward-absolute-filepath': paths[0],
                'reverse-absolute-filepath': paths[1]
            })
    pd.DataFrame(rows).set_index('sample-id').to_csv(tsv_path, sep='\t', index=True)
    

def manual_meta(dataset: str, metadata_dir: Union[str, Path]) -> pd.DataFrame:
    """Load manually curated metadata for a dataset.
    
    Args:
        dataset: Dataset identifier
        metadata_dir: Base directory containing metadata files
    
    Returns:
        DataFrame with manual metadata, empty if not found
    """
    path = Path(metadata_dir) / dataset / 'manual-metadata.tsv'
    return pd.read_csv(path, sep="\t") if path.exists() else pd.DataFrame()
