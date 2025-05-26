# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import glob
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from functools import reduce

# Third-Party Imports
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio import SeqIO
from biom import load_table
from biom.table import Table
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    track,
    TaskID
)

# ================================== LOCAL IMPORTS =================================== #

from workflow_16s.utils.biom_utils import (
    collapse_taxa, presence_absence, filter_presence_absence
)
from workflow_16s.utils import df_utils
from workflow_16s.utils.dir_utils import SubDirs
from workflow_16s.stats import beta_diversity 
from workflow_16s.stats.utils import (
    preprocess_table, mwu_bonferroni, kruskal_bonferroni, t_test
)
from workflow_16s.figures.html_report import HTMLReport
from workflow_16s.figures.merged.merged import (
    sample_map_categorical, pcoa, pca, mds
)

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger('workflow_16s')

# ================================= DEFAULT VALUES =================================== #

DEFAULT_PROGRESS_TEXT_N = 50

DEFAULT_GROUP_COLUMN = 'nuclear_contamination_status'
DEFAULT_GROUP_COLUMN_VALUES = [True, False]

# ==================================== FUNCTIONS ===================================== #

def load_datasets_list(path: Union[str, Path]) -> List[str]:
    """
    Load dataset IDs from configuration file.
    
    Args:
        path: Path to text file containing dataset IDs (one per line).
    
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
        DataFrame with dataset information, cleaned of unnamed columns.
    """
    df = pd.read_csv(tsv_path, sep="\t", dtype={'ena_project_accession': str})
    # Remove 'Unnamed' columns
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
    return df


def fetch_first_match(dataset_info: pd.DataFrame, dataset: str) -> pd.Series:
    """
    Find matching dataset information from metadata DataFrame.
    
    Args:
        dataset_info: DataFrame containing dataset metadata.
        dataset:      Dataset identifier to search for.
    
    Returns:
        First matching row from dataset_info as a pandas Series.
    
    Raises:
        ValueError: If no matches found for the dataset.
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
    """
    Generate expected file paths for processed dataset outputs.
    
    Args:
        dirs:    Project directory structure.
        dataset: Dataset identifier.
        params:  Processing parameters dictionary.
        cfg:     Configuration dictionary.
    
    Returns:
        Dictionary mapping file types to their expected paths.
    """
    classifier = cfg["classifier"]
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
        # BIOM feature table
        'table_biom': base_dir / 'table' / 'feature-table.biom',  
        # Representative seqs
        'seqs_fasta': base_dir / 'rep-seqs' / 'dna-sequences.fasta',  
        # Taxonomy
        'taxonomy_tsv': base_dir / classifier / 'taxonomy' / 'taxonomy.tsv',  
    }


def missing_output_files(file_list: List[Union[str, Path]]) -> List[Path]:
    """
    Identify missing output files from a list of expected paths.
    
    Args:
        file_list: List of file paths to check.
    
    Returns:
        List of Path objects for files that don't exist.
    """
    return [Path(file) for file in file_list if not Path(file).exists()]

# ==================================== FUNCTIONS ===================================== #   

def create_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        expand=False
    )
    
class AmpliconData:
    def __init__(
        self, 
        cfg,
        project_dir,
        mode: str = 'genus',
        verbose: bool = False
    ):
        self.cfg = cfg
        self.project_dir = project_dir
        self.mode = mode
        self.verbose = verbose
        
        self.table, self.meta, self.taxa = None, None, None

        self.tables = {}
        if self.cfg['presence_absence']:
            self.presence_absence_tables = {} 
        
        self.figures = {}
        self.color_maps = {}

        self.stats = {}
        
        mode_map = {
            'asv': ('table', 'asv'),
            'genus': ('table_6', 'l6')
        }
        
        self.table_dir, output_dir = mode_map.get(mode, (None, None))

        
        self.table_output_path = (
            Path(self.project_dir.data) / 'merged' / output_dir / 
            'feature-table.biom'
        )
        self.meta_output_path = (
            Path(self.project_dir.data) / 'merged' / 'metadata' / 
            'sample-metadata.tsv'
        )

        self._get_meta_df()
        self._get_biom_table()

        mode_funcs = {
            'asv': self._asv_mode,
            'genus': self._genus_mode,
        }
        mode_funcs[mode]()  
        # Run statistical analyses
        self._run_statistical_analyses('raw')
        self._top_features('raw')
        #if self.cfg['presence_absence']:   
        #    self._run_statistical_analyses('presence_absence') 
        #    self._top_features('presence_absence') -

    def _get_biom_paths(self) -> List:
        """Get feature table BIOM paths from a pattern."""
        BIOM_PATTERN = '/'.join(
            ['*', '*', '*', '*', 'FWD_*_REV_*', self.table_dir, 'feature-table.biom']
        )
        biom_paths = glob.glob(
            str(Path(self.project_dir.qiime_data_per_dataset) / BIOM_PATTERN), 
            recursive=True
        )
        if self.verbose:
            logger.info(f"Found {len(biom_paths)} unique feature tables.")
        return biom_paths
        
    def _get_meta_paths(self) -> List:
        """"""
        meta_paths = []
        for biom_path in self._get_biom_paths():
            biom_path = Path(biom_path)
            if biom_path.is_file() or biom_path.suffix:
                biom_path = biom_path.parent
            parts = biom_path.parts
            meta_path = str(
                Path(self.project_dir.metadata_per_dataset) / 
                '/'.join(list(parts[-6:-1]))
            ) + '/sample-metadata.tsv'        
            meta_paths.append(meta_path)
        if self.verbose:
            logger.info(f"Found {len(meta_paths)} unique metadata files.")
        return meta_paths

    def _get_biom_table(self):
        biom_paths = self._get_biom_paths()
        if not biom_paths:
            error_text = f"No BIOM files found matching {self.BIOM_PATTERN}"
            logger.error(error_text)
            raise FileNotFoundError(error_text)   
            
        self.table = import_merged_table_biom(
            biom_paths, 
            'dataframe',
            self.table_output_path,
            self.verbose
        )

    def _get_meta_df(self):
        meta_paths = []
        for biom_path in self._get_biom_paths():
            p = Path(biom_path)
            # if it's a file *or* has a suffix, work in its parent dir
            dataset_dir = p.parent if p.is_file() or p.suffix else p
        
            # grab the last 6→1 parts of that directory
            tail_parts   = dataset_dir.parts[-6:-1]
        
            # build the full metadata path and cast to str
            meta_file = (
                Path(self.project_dir.metadata_per_dataset)
                .joinpath(*tail_parts, "sample-metadata.tsv")
            )
            meta_paths.append(str(meta_file))
        self.meta = import_merged_meta_tsv(
            meta_paths, 
            self.meta_output_path, 
            None,
            self.verbose
        )
        
    def _genus_mode(self):
        tax_levels = ['phylum', 'class', 'order', 'family', 'genus']
        table_dir = Path(self.project_dir.tables) / 'merged'
    
        if self.verbose:
            self.tables["raw"] = {}
            for level in tax_levels:
                biom_table = collapse_taxa(
                    self.table,
                    level,
                    table_dir,
                    self.verbose
                )
                self.tables["raw"][level] = biom_table
                logger.info(f"Collapsed to {level} level")
    
            if self.cfg['presence_absence']:
                self.tables["presence_absence"] = {}
                for level in self.tables["raw"]:
                    pa_table = presence_absence(
                        self.tables[level],
                        level,
                        table_dir,
                        self.verbose
                    )
                    self.tables["presence_absence"][level] = pa_table
                    logger.info(f"Converted {level} table to presence/absence")

            filter_table = True
            normalize_table = True
            clr_table = True
            if filter_table:
                self.tables["filtered"] = {}
                for level in tax_levels:
                    table = preprocess_table(
                        table=self.tables["raw"][level],
                        apply_filter=True,
                        normalize=False,
                        clr_transform=False,
                    )
                    self.tables["filtered"][level] = table
                    logger.info(f"Filtered {level} table")

            if filter_table and normalize_table:
                self.tables["normalized"] = {}
                for level in tax_levels:
                    table = preprocess_table(
                        table=self.tables["filtered"][level],
                        apply_filter=False,
                        normalize=True,
                        clr_transform=False,
                    )
                    self.tables["normalized"][level] = table
                    logger.info(f"Normalized {level} table")

            if filter_table and normalize_table and clr_table:
                self.tables["clr"] = {}
                for level in tax_levels:
                    table = preprocess_table(
                        table=self.tables["normalized"][level],
                        apply_filter=False,
                        normalize=False,
                        clr_transform=True,
                    )
                    self.tables["clr"][level] = table
                    logger.info(f"CLR-transformed {level} table")
        else:
            with create_progress() as progress:
                collapse_task = progress.add_task(
                    "[white]Collapsing taxonomy...".ljust(DEFAULT_PROGRESS_TEXT_N), 
                    total=len(tax_levels)
                )
                self.tables["raw"] = {}
                for level in tax_levels:
                    biom_table = collapse_taxa(
                        self.table,
                        level,
                        table_dir,
                        self.verbose
                    )
                    self.tables["raw"][level] = biom_table
                    progress.update(collapse_task, advance=1)
    
                if self.cfg['presence_absence']:
                    self.tables["presence_absence"] = {}
                    pa_task = progress.add_task(
                        "[white]Generating presence/absence tables...".ljust(DEFAULT_PROGRESS_TEXT_N), 
                        total=len(self.tables)
                    )
                    for level in self.tables["raw"]:
                        pa_table = presence_absence(
                            self.tables["raw"][level],
                            level,
                            table_dir,
                            self.verbose
                        )
                        self.tables["presence_absence"][level] = pa_table
                        progress.update(pa_task, advance=1)

                filter_table = True
                normalize_table = True
                clr_table = True
                if filter_table:
                    self.tables["filtered"] = {}
                    filter_task = progress.add_task(
                        "[white]Filtering tables...".ljust(DEFAULT_PROGRESS_TEXT_N), 
                        total=len(self.tables)
                    )
                    for level in self.tables["raw"]:
                        table = preprocess_table(
                            table=self.tables["raw"][level],
                            apply_filter=True,
                            normalize=False,
                            clr_transform=False,
                        )
                        self.tables["filtered"][level] = table
                        progress.update(filter_task, advance=1)
    
                if filter_table and normalize_table:
                    self.tables["normalized"] = {}
                    n_task = progress.add_task(
                        "[white]Normalizing tables...".ljust(DEFAULT_PROGRESS_TEXT_N), 
                        total=len(self.tables)
                    )
                    for level in self.tables["filtered"]:
                        table = preprocess_table(
                            table=self.tables["filtered"][level],
                            apply_filter=False,
                            normalize=True,
                            clr_transform=False,
                        )
                        self.tables["normalized"][level] = table
                        progress.update(n_task, advance=1)
    
                if filter_table and normalize_table and clr_table:
                    self.tables["clr"] = {}
                    clr_task = progress.add_task(
                        "[white]CLR-transforming tables...".ljust(DEFAULT_PROGRESS_TEXT_N), 
                        total=len(self.tables)
                    )
                    for level in self.tables["normalized"]:
                        table = preprocess_table(
                            table=self.tables["normalized"][level],
                            apply_filter=False,
                            normalize=False,
                            clr_transform=True,
                        )
                        self.tables["clr"][level] = table
                        progress.update(clr_task, advance=1)

                
    def _asv_mode(self):
        logger.info("ASV mode is not yet supported!")

    def _fetch_tables(
        self, 
        table_type: str = 'presence_absence'
    ):
        if table_type == 'raw':
            tables = self.tables
        elif table_type == 'presence_absence':
            tables = self.presence_absence_tables
        else:
            logger.error(
                f"Unknown table type '{table_type}.'"
                f"Expected either 'raw' or 'presence_absence'."
            )
        return tables

    def _run_statistical_analyses(self, table_type: str = 'presence_absence'):
        self.stats[table_type] = {}
        tables = self._fetch_tables(table_type)
        
        enabled_tests = [
            test for test in [
                't_test', 'mwu_bonferroni', 'kruskal_bonferroni', 'pca', 'tsne'
            ] if self.cfg['stats'][table_type].get(test, False)
        ]

        with create_progress() as progress:
            main_task = progress.add_task(
                f"[white]Running {table_type} analyses",
                total=len(enabled_tests)*len(tables)
            )
            # T-Test
            if 't_test' in enabled_tests:
                self.stats[table_type]['t-test'] = {}
                ttest_task = progress.add_task(
                    "[white]T-Test...".ljust(DEFAULT_PROGRESS_TEXT_N), 
                    total=len(tables)
                )
                for level in tables:
                    self.stats[table_type]['t-test'][level] = t_test(
                        table=tables[level],
                        metadata=self.meta,
                        group_column='nuclear_contamination_status',
                        groups=[True, False],
                        progress=progress,
                        parent_task_id=main_task,
                        level=level,
                    )
                    progress.update(ttest_task, advance=1)
                progress.stop_task(ttest_task)
                progress.update(ttest_task, visible=False)
            """
            # Mann-Whitney U with Bonferroni
            if 'mwu_bonferroni' in enabled_tests:
                self.stats[table_type]['mwu_bonferroni'] = self._run_test_for_all_levels(
                    progress=progress,
                    parent_task_id=main_task,
                    test_name="Mann-Whitney U",
                    test_func=mwu_bonferroni,
                    tables=tables
                )

            # Kruskal-Wallis with Bonferroni
            if 'kruskal_bonferroni' in enabled_tests:
                self.stats[table_type]['kruskal_bonferroni'] = self._run_test_for_all_levels(
                    progress=progress,
                    parent_task_id=main_task,
                    test_name="Kruskal-Wallis",
                    test_func=kruskal_bonferroni,
                    tables=tables
                )

            # Visualization analyses (PCA/t-SNE)
            if 'pca' in enabled_tests or 'tsne' in enabled_tests:
                self._run_visual_analyses(
                    progress=progress,
                    parent_task_id=main_task,
                    table_type=table_type,
                    tables=tables,
                    enabled_tests=enabled_tests
                )
            """

    def _run_test_for_all_levels(self, progress, parent_task_id, test_name, test_func, tables):
        results = {}
        for level in tables:
            task_desc = f"[white]{test_name} [dim]({level})".ljust(DEFAULT_PROGRESS_TEXT_N)
            features = tables[level].shape[0]
            task_id = progress.add_task(
                description=task_desc,
                total=features,
                parent=parent_task_id
            )
            
            with progress:
                results[level] = test_func(
                    table=tables[level],
                    metadata=self.meta,
                    group_column='nuclear_contamination_status',
                )
                progress.update(parent_task_id, advance=1)
        # stop and hide progress bar for this step when done
        progress.stop_task(task_id)
        progress.update(task_id, visible=False)
        return results

    def _run_visual_analyses(self, progress, parent_task_id, table_type, tables, enabled_tests):
        vis_task = progress.add_task(
            "[bold magenta]Visual Analyses",
            parent=parent_task_id,
            total=len(enabled_tests)*len(tables)
        )
        
        levels = {'phylum': 1, 'class': 2, 'order': 3, 'family': 4, 'genus': 5}
        
        if 'pca' in enabled_tests:
            self.stats[table_type]['pca'] = {}
            for level in tables:
                progress.update(vis_task, description=f"[white]PCA {level}")
                meta, table, _ = df_utils.match_indices_or_transpose(self.meta, tables[level])
                pca_results = beta_diversity.pca(table=table, n_components=3)
                self.stats[table_type]['pca'][level] = pca_results
                progress.advance(vis_task)

        if 'tsne' in enabled_tests:
            self.stats[table_type]['tsne'] = {}
            for level in tables:
                progress.update(vis_task, description=f"[white]t-SNE {level}")
                meta, table, _ = df_utils.match_indices_or_transpose(self.meta, tables[level])
                tsne_results = beta_diversity.tsne(table=table, n_components=3)
                self.stats[table_type]['tsne'][level] = tsne_results
                progress.advance(vis_task)
        """
        # Save statistical results (same as before)
        stats_dir = Path(self.project_dir.tables) / 'stats' / table_type
        stats_dir.mkdir(parents=True, exist_ok=True)
        
        for test_type in self.stats[table_type]:
            test_dir = stats_dir / test_type
            test_dir.mkdir(exist_ok=True)
            for level in self.stats[table_type][test_type]:
                df = self.stats[table_type][test_type][level]
                output_path = test_dir / f"{level}_results.tsv"
                df.to_csv(output_path, sep='\t', index=True)
                logger.info(
                    f"Saved statistical analysis results to {str(output_path)}"
                )
        """

    def _top_features(
        self, 
        table_type: str = 'presence_absence'    
    ):
        # Find top features
        contaminated_features = []
        pristine_features = []
        
        tables = self._fetch_tables(table_type)
        
        # Ensure the stats for the table_type exist
        if table_type not in self.stats:
            logger.warning(f"No statistical results found for {table_type}.")
            return
        for level in ['phylum', 'class', 'order', 'family', 'genus']:
            dfs = []
            for test_type in self.stats[table_type]:    
                dfs.append(self.stats[table_type][test_type][level])
            # Merge list of DataFrames on 'feature' using outer join
            df = reduce(
                lambda left, right: pd.merge(
                    left, right, on='feature', how='outer'
                ), 
                dfs
            )
            #logger.info(df.head())
            
            """
                if 'q_value' not in df.columns:
                    logger.warning(f"'q_value' column missing in {test_type} results for {level}.")
                    continue
                sig_df = df[df['q_value'] < 0.05]
                for feature, row in sig_df.iterrows():
                    effect = row.get('mean_diff', row.get('effect_size', 0))
                    if effect > 0:
                        contaminated_features.append({
                            'feature': feature,
                            'level': level,
                            'test': test_type,
                            'effect': effect,
                            'q_value': row['q_value']
                        })
                    else:
                        pristine_features.append({
                            'feature': feature,
                            'level': level,
                            'test': test_type,
                            'effect': abs(effect),
                            'q_value': row['q_value']
                        })
                """
    
        # Process and save top features
        top_dir = Path(self.project_dir.tables) / 'stats' / 'top_features'
        top_dir.mkdir(parents=True, exist_ok=True)
        
        def _process_features(feature_list):
            if not feature_list:
                return pd.DataFrame()
            df = pd.DataFrame(feature_list)
            return df.sort_values(['effect', 'q_value'], ascending=[False, True]).head(20)
    
        top_contam = _process_features(contaminated_features)
        top_pristine = _process_features(pristine_features)
    
        top_contam.to_csv(top_dir / 'top20_contaminated.tsv', sep='\t', index=False)
        logger.info(f"Saved top 20 features associated with contaminated environments to {str(top_dir / 'top20_contaminated.tsv')}")
        top_pristine.to_csv(top_dir / 'top20_pristine.tsv', sep='\t', index=False)
        logger.info(f"Saved top 20 features associated with pristine environments to {str(top_dir / 'top20_pristine.tsv')}")

    def _plot_stuff(
        self, 
        table_type: str = 'presence_absence',
        figure_type: str = 'pca'
    ):
        levels = {'phylum': 1, 'class': 2, 'order': 3, 'family': 4, 'genus': 5}
        color_cols=['dataset_name']
        symbol_col='nuclear_contamination_status'
        
        tables = self._fetch_tables(table_type) 
        self.figures[table_type] = {}
        self.figures[table_type][figure_type] = []
        
        for level in tables:
            results = self.stats[table_type][figure_type][level]

            logger.info(f"Plotting {figure_type.upper()}...")
            for color_col in color_cols:
                if figure_type == 'pca':
                    plot, _ = pca(
                        components = results['components'], 
                        proportion_explained = results['exp_var_ratio'], 
                        metadata=self.meta,
                        color_col=color_col, 
                        color_map=self.color_maps[color_col],
                        symbol_col=symbol_col,
                        show=False,
                        output_dir=Path(self.project_dir.figures) / 'pca' / f'l{levels[level]+1}', 
                        x=1, 
                        y=2
                    )
                elif figure_type == 'tsne':
                    plot, _ = mds(
                        df=results, 
                        metadata=self.meta,
                        group_col=color_col, 
                        symbol_col=symbol_col,
                        show=False,
                        output_dir=Path(self.project_dir.figures) / 'tsne' / f'l{levels[level]+1}',
                        mode='TSNE',
                        x=1, 
                        y=2
                    )
                
                self.figures[figure_type].append({
                    'title': f'{figure_type.upper()} - {level}',
                    'level': level,
                    'color_col': color_col,
                    'symbol_col': symbol_col,
                    'figure': plot
                })
        
def import_meta_tsv(
    tsv_path: Union[str, Path],
    column_renames: Optional[List[Tuple[str, str]]] = None
) -> pd.DataFrame:
    """
    Load a single sample-metadata.tsv, rename columns, and ensure required fields exist.
    """
    tsv_path = Path(tsv_path)
    if not tsv_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {tsv_path}")

    # ensure we can iterate even if None was passed
    if column_renames is None:
        column_renames = []

    df = pd.read_csv(tsv_path, sep='\t')
    df.columns = df.columns.str.lower()

    # determine sample ID
    sample_id_col = next(
        (col for col in ['run_accession', '#sampleid', 'sample-id'] if col in df.columns),
        None
    )
    if sample_id_col:
        df['SAMPLE ID'] = df[sample_id_col]
    else:
        df['SAMPLE ID'] = [
            f"{Path(tsv_path).parents[5].name}_x{i}"
            for i in range(1, len(df) + 1)
        ]

    # determine dataset ID
    dataset_id_col = next(
        (col for col in ['project_accession', 'dataset_id', 'dataset_name'] if col in df.columns),
        None
    )
    if dataset_id_col:
        df['DATASET ID'] = df[dataset_id_col]
    else:
        df['DATASET ID'] = Path(tsv_path).parents[5].name

    # ensure nuclear_contamination_status exists
    if 'nuclear_contamination_status' not in df.columns:
        df['nuclear_contamination_status'] = False

    # apply any requested renames
    for old, new in column_renames:
        if old in df.columns:
            df.rename(columns={old: new}, inplace=True)

    return df


def import_merged_meta_tsv(
    meta_paths: List[Union[str, Path]],
    output_path: Union[str, Path] = None,
    column_renames: Optional[List[Tuple[str, str]]] = None,
    verbose: bool = False
) -> pd.DataFrame:
    """Load and merge multiple metadata TSV files."""
    dfs = []

    if verbose:
        for path in meta_paths:
            try:
                df = import_meta_tsv(path, column_renames)
                dfs.append(df)
                logger.info(f"Loaded {Path(path).name} with {df.shape[0]} samples")
            except Exception as e:
                logger.error(f"Failed to load {path}: {e!r}")
    else:
        with create_progress() as progress:
            task = progress.add_task(
                "[white]Loading metadata files...".ljust(DEFAULT_PROGRESS_TEXT_N), 
                total=len(meta_paths)
            )
            for path in meta_paths:
                try:
                    df = import_meta_tsv(path, column_renames)
                    dfs.append(df)
                except Exception as e:
                    logger.error(f"Failed to load {path}: {e!r}")
                finally:
                    progress.update(task, advance=1)

    if not dfs:
        raise FileNotFoundError(
            "None of the metadata files could be loaded; "
            "check that your sample-metadata.tsv files exist and are readable."
        )

    df_merged = pd.concat(dfs, ignore_index=True)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_merged.to_csv(output_path, sep='\t', index=True)
        if verbose:
            n_samples, n_features = df_merged.shape
            logger.info(f"Wrote merged metadata [{n_samples}, {n_features}] to {output_path}")

    return df_merged

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
    """Merge multiple BIOM tables into one."""
    tables = []

    if verbose:
        for path in biom_paths:
            try:
                table = import_table_biom(path, 'table')
                tables.append(table)
                logger.info(f"Loaded {Path(path).name} with {len(table.ids('sample'))} samples")
            except Exception as e:
                logger.error(f"Failed to load {path}: {str(e)}")
    else:
        with create_progress() as progress:
            task = progress.add_task(
                "[white]Loading BIOM files...".ljust(DEFAULT_PROGRESS_TEXT_N), 
                total=len(biom_paths)
            )
            for path in biom_paths:
                try:
                    table = import_table_biom(path, 'table')
                    tables.append(table)
                except Exception as e:
                    logger.error(f"Failed to load {path}: {str(e)}")
                finally:
                    progress.update(task, advance=1)

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
            n_features, n_samples = merged_table.shape
            logger.info(f"Wrote table [{n_features}, {n_samples}] to {output_path}")

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
