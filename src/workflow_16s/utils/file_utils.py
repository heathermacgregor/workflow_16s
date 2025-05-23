# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import glob
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-Party Imports
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio import SeqIO
from biom import load_table
from biom.table import Table

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

class AmpliconData:
    def __init__(
        self, 
        cfg,
        project_dir,
        mode: str = 'genus',
        verbose: bool = True
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

        
        self.output_path = (
            Path(self.project_dir.data) / 'merged' / output_dir / 
            'feature-table.biom'
        )

        self._get_metadata()
        self._get_biom_table()

        mode_funcs = {
            'genus': self._genus_mode,
            'asv': self._asv_mode,
        }
        mode_funcs[mode]()  
        
        # Run statistical analyses
        self._run_statistical_analyses('raw')
        self._top_features('raw')
        if self.cfg['presence_absence']:   
            self._run_statistical_analyses('presence_absence') 
            self._top_features('presence_absence') 

    def _get_biom_paths(self) -> List:
        BIOM_PATTERN = '/'.join(
            ['*', '*', '*', '*', 'FWD_*_REV_*', self.table_dir, 'feature-table.biom']
        )
        biom_paths = glob.glob(
            str(Path(self.project_dir.qiime_data_per_dataset) / BIOM_PATTERN), 
            recursive=True
        )
        logger.info(f"Found {len(biom_paths)} unique feature tables.")
        return biom_paths
        
    def _get_meta_paths(self) -> List:
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
        logger.info(f"Found {len(meta_paths)} unique metadata files.")
        return meta_paths

    def _get_biom_table(self):
        biom_paths = self._get_biom_paths()
        if not biom_paths:
            raise FileNotFoundError(
                f"No BIOM files found matching {self.BIOM_PATTERN}"
            )   
        self.table = import_merged_table_biom(
            biom_paths, 
            'dataframe',
            self.output_path,
            self.verbose
        )

    def _process_meta_path(
        self, 
        csv_path: Union[str, Path], 
        column_renames: List[Tuple[str, str]]
    ) -> pd.DataFrame:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {csv_path}")
            
        df = pd.read_csv(csv_path, sep='\t')
        df.columns = df.columns.str.lower()
        
        sample_id_col = next(
            (col for col in ['run_accession', '#sampleid', 'sample-id'] 
             if col in df.columns), 
            None
        )
        if sample_id_col:
            df['SAMPLE ID'] = df[sample_id_col]
        else:
            df['SAMPLE ID'] = [f"{Path(csv_path).parents[5].name}_x{i}" 
                               for i in range(1, len(df)+1)]
            
        dataset_id_col = next(
            (col for col in ['project_accession', 'dataset_id', 'dataset_name'] 
             if col in df.columns), 
            None
        )
        if dataset_id_col:
            df['DATASET ID'] = df[dataset_id_col]
        else:
            df['DATASET ID'] = Path(csv_path).parents[5].name
            
        if 'nuclear_contamination_status' not in df.columns:
            df['nuclear_contamination_status'] = False

        for old, new in column_renames:
            if old in df.columns:
                df.rename(columns={old: new}, inplace=True)
                
        return df
    
    def _get_metadata(self):
        meta_dfs = []
        for meta_path in self._get_meta_paths():  
            meta_df = self._process_meta_path(meta_path, []).set_index('#sampleid')
            meta_dfs.append(meta_df)
        self.meta = pd.concat(meta_dfs)
        
    def _genus_mode(self):
    for level in ['phylum', 'class', 'order', 'family']:
        self.tables[level] = collapse_taxa(
            self.table, 
            level, 
            Path(self.project_dir.tables) / 'merged',
            self.verbose
        )
        # Transpose genus table to match samples-as-rows format
        self.tables['genus'] = self.table.T  # Samples become rows
    
        if self.cfg['presence_absence']:
            for level in self.tables:
                self.presence_absence_tables[level] = presence_absence(
                    self.tables[level], 
                    level, 
                    Path(self.project_dir.tables) / 'merged',
                    self.verbose
                )
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

    def _run_statistical_analyses(
        self, 
        table_type: str = 'presence_absence'
    ):
        self.stats[table_type] = {}
        
        tables = self._fetch_tables(table_type)   
        if self.cfg['stats'][table_type].get('t_test', False):
            self.stats[table_type]['t_test'] = {}
            for level in tables:
                logger.info(f"Running t-test for {level}...")
                results = t_test(
                    table=tables[level], 
                    metadata=self.meta,
                    col='nuclear_contamination_status',
                    col_values=[True, False]
                )
                self.stats[table_type]['t_test'][level] = results
        
        if self.cfg['stats'][table_type].get('mwu_bonferroni', False):
            self.stats[table_type]['mwu_bonferroni'] = {}
            for level in tables:
                logger.info(f"Running Mann-Whitney U with Bonferroni for {level}...")
                results = mwu_bonferroni(
                    table=tables[level],
                    metadata=self.meta,
                    col='nuclear_contamination_status',
                    col_values=[True, False]
                )
                self.stats[table_type]['mwu_bonferroni'][level] = results
        
        if self.cfg['stats'][table_type].get('kruskal_bonferroni', False):
            self.stats[table_type]['kruskal_bonferroni'] = {}
            for level in tables:
                logger.info(f"Running Kruskal-Wallis with Bonferroni for {level}...")
                results = kruskal_bonferroni(
                    table=tables[level],
                    metadata=self.meta,
                    col='nuclear_contamination_status'
                )
                self.stats[table_type]['kruskal_bonferroni'][level] = results

        # Save statistical results
        stats_dir = Path(self.project_dir.tables) / 'stats' / table_type
        stats_dir.mkdir(parents=True, exist_ok=True)
        
        for test_type in self.stats[table_type]:
            test_dir = stats_dir / test_type
            test_dir.mkdir(exist_ok=True)
            for level in self.stats[table_type][test_type]:
                df = self.stats[table_type][test_type][level]
                output_path = test_dir / f"{level}_results.tsv"
                df.to_csv(output_path, sep='\t', index=True)
                print(f"Saved results to {str(output_path)}")
                

        if self.cfg['stats'][table_type].get('pca', False):
            self.stats[table_type]['pca'] = {}
            for level in tables:
                logger.info(f"Calculating PCA ({level})...")
            
                meta, table, _ = df_utils.match_indices_or_transpose(
                    self.meta, 
                    tables[level]
                )
                    
                pca_results = beta_diversity.pca(
                    table=table,
                    n_components=3
                )
                self.stats[table_type]['pca'][level] = pca_results
                
        if self.cfg['stats'][table_type].get('tsne', False):
            self.stats[table_type]['tsne'] = {}
            for level in tables:
                logger.info(f"Calculating TSNE ({level})...")
        
                meta, table, _ = df_utils.match_indices_or_transpose(
                    self.meta, 
                    tables[level]
                )
                tsne_results = beta_diversity.tsne(
                    table=table,
                    n_components=3
                )
                self.stats[table_type]['tsne'][level] = tsne_results

    def _top_features(
        self, 
        table_type: str = 'presence_absence'    
    ):
        # Find top features
        contaminated_features = []
        pristine_features = []
        
        self.stats[table_type] = {}
        tables = self._fetch_tables(table_type)
        
        for test_type in self.stats[table_type]:
            for level in self.stats[table_type][test_type]:
                df = self.stats[table_type][test_type][level]
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
        



"""
class AmpliconData:
    '''
    Main class for handling amplicon sequencing data analysis.
    
    Attributes:
        project_dir: Subdirs object with directory path attributes
        mode: Analysis mode ('asv' or 'genus')
        verbose: Enable verbose logging
        table: Main feature table
        meta: Metadata DataFrame
        taxa: Taxonomy information
        tables: Dictionary of collapsed tables by level
        presence_absence_tables: PA tables by taxonomic level
    '''
    def __init__(
        self, 
        cfg,
        project_dir,
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
        self.figures = {}
        self.color_maps = {}
        
        if self.mode == 'asv':
            table_dir = 'table'
            output_dir = 'asv'
        elif self.mode == 'genus':
            table_dir = 'table_6'
            output_dir = 'l6'

        table_paths, meta_path = get_amplicon_data_paths(project_dir, table_dir)
        
        self.output_path = Path(self.project_dir.main) / 'data' / 'merged' / output_dir / 'feature-table.biom'
        self._get_metadata()

        self._plot_sample_map()
        self._get_biom_table()

        if self.mode == 'genus':
            for level in ['phylum', 'class', 'order', 'family']:
                collapsed_table = collapse_taxa(
                    self.table, 
                    level, 
                    Path(self.project_dir.main) / 'data' / 'merged',
                    self.verbose
                )
                self.tables[level] = collapsed_table
            self.tables['genus'] = self.table   

            if self.cfg['presence_absence']:
                for level in self.tables:
                    pa = presence_absence(
                        self.tables[level], 
                        level, 
                        Path(self.project_dir.main) / 'data' / 'merged',
                        self.verbose
                    )
                    self.presence_absence_tables[level] = pa

        elif self.mode == 'asv':
            logger.info("ASV mode is not yet supported!")

        self._plot_pca()
        self._plot_tsne()
        

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
        
        if self.cfg['stats']['presence_absence'].get('mwu_bonferroni', False):
            self.stats['presence_absence']['mwu_bonferroni'] = {}
            for level in self.presence_absence_tables:
                logger.info(f"Running Mann-Whitney U with Bonferroni for {level}...")
                results = mwu_bonferroni(
                    table=self.presence_absence_tables[level],
                    metadata=self.meta,
                    col='nuclear_contamination_status',
                    col_values=[True, False]
                )
                self.stats['presence_absence']['mwu_bonferroni'][level] = results
        
        if self.cfg['stats']['presence_absence'].get('kruskal_bonferroni', False):
            self.stats['presence_absence']['kruskal_bonferroni'] = {}
            for level in self.presence_absence_tables:
                logger.info(f"Running Kruskal-Wallis with Bonferroni for {level}...")
                results = kruskal_bonferroni(
                    table=self.presence_absence_tables[level],
                    metadata=self.meta,
                    col='nuclear_contamination_status'
                )
                self.stats['presence_absence']['kruskal_bonferroni'][level] = results

        # Save statistical results
        stats_dir = Path(self.project_dir.main) / 'final' / 'tables' / 'stats' / 'presence_absence'
        stats_dir.mkdir(parents=True, exist_ok=True)
        
        for test_type in self.stats['presence_absence']:
            test_dir = stats_dir / test_type
            test_dir.mkdir(exist_ok=True)
            for level in self.stats['presence_absence'][test_type]:
                df = self.stats['presence_absence'][test_type][level]
                output_path = test_dir / f"{level}_results.tsv"
                df.to_csv(output_path, sep='\t', index=True)

        # Find top features
        contaminated_features = []
        pristine_features = []
        
        for test_type in self.stats['presence_absence']:
            for level in self.stats['presence_absence'][test_type]:
                df = self.stats['presence_absence'][test_type][level]
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

        # Process and save top features
        top_dir = Path(self.project_dir.main) / 'final' / 'tables' / 'stats' / 'top_features'
        top_dir.mkdir(parents=True, exist_ok=True)
        
        def _process_features(feature_list):
            if not feature_list:
                return pd.DataFrame()
            df = pd.DataFrame(feature_list)
            return df.sort_values(['effect', 'q_value'], ascending=[False, True]).head(20)

        top_contam = _process_features(contaminated_features)
        top_pristine = _process_features(pristine_features)

        top_contam.to_csv(top_dir / 'top20_contaminated.tsv', sep='\t', index=False)
        top_pristine.to_csv(top_dir / 'top20_pristine.tsv', sep='\t', index=False)

    def _plot_sample_map(self) -> None:
        self.figures["sample_map"] = []
        logger.info("Creating sample map...")
        
        for col in ['dataset_name', 'nuclear_contamination_status']:
            fig, map = sample_map_categorical(
                metadata=self.meta, 
                show=False, 
                output_dir=Path(self.project_dir.main) / 'final' / 'figures' / 'sample_maps', 
                color_col=col,
            )
            self.figures["sample_map"].append({
                'title': f'Sample Map - {col}',
                'color_col': col,
                'figure': fig
            })
            self.color_maps[col] = map

    def _plot_pca(self):
        self.figures["pca"] = []
        
        levels = {'phylum': 1, 'class': 2, 'order': 3, 'family': 4, 'genus': 5}
        color_col='dataset_name'
        symbol_col='nuclear_contamination_status'
        
        tables = self.presence_absence_tables if self.cfg['presence_absence'] else self.tables
            
        for level in tables:
            logger.info(f"Calculating PCA ({level})...")
            
            meta, table, _ = df_utils.match_indices_or_transpose(
                self.meta, 
                tables[level]
            )
                
            pca_results = beta_diversity.pca(
                table=table,
                n_components=3
            )
            
            logger.info("Plotting PCA...")
            
            pca_plot, _ = pca(
                components = pca_results['components'], 
                proportion_explained = pca_results['exp_var_ratio'], 
                metadata=meta,
                color_col=color_col, 
                color_map=self.color_maps[color_col],
                symbol_col=symbol_col,
                show=False,
                output_dir=Path(self.project_dir.main) / 'final' / 'figures' / 'pca' / f'l{levels[level]+1}', 
                x=1, 
                y=2
            )
            
            self.figures["pca"].append({
                'title': f'PCA - {level}',
                'level': level,
                'color_col': color_col,
                'symbol_col': symbol_col,
                'figure': pca_plot
            })

    def _plot_tsne(self) -> None:
        self.figures["tsne"] = []
        levels = {'phylum': 1, 'class': 2, 'order': 3, 'family': 4, 'genus': 5}
        color_col='dataset_name'
        symbol_col='nuclear_contamination_status'
        
        tables = self.presence_absence_tables if self.cfg['presence_absence'] else self.tables
            
        for level in tables:
            logger.info(f"Calculating TSNE ({level})...")
    
            meta, table, _ = df_utils.match_indices_or_transpose(
                self.meta, 
                tables[level]
            )
            tsne_results = beta_diversity.tsne(
                    table=table,
                    n_components=3
            )
                
            logger.info("Plotting TSNE...")
                
            tsne_plot, _ = mds(
                df=tsne_results, 
                metadata=meta,
                group_col=color_col, 
                symbol_col=symbol_col,
                show=False,
                output_dir=Path(self.project_dir.main) / 'final' / 'figures' / 'tsne' / f'l{levels[level]+1}',
                mode='TSNE',
                x=1, 
                y=2
            )
        
            self.figures["tsne"].append({
                'title': f'TSNE - {level}',
                'level': level,
                'color_col': color_col,
                'symbol_col': symbol_col,
                'figure': tsne_plot
            })
            
    def _get_biom_paths(self) -> List[str]:
        return glob.glob(str(Path(self.project_dir.main) / 'data' / 'per_dataset' / self.BIOM_PATTERN), recursive=True)    

    def _get_meta_paths(self) -> List[Path]:
        meta_paths = []
        for biom_path in self._get_biom_paths():
            biom_path = Path(biom_path)
            current = biom_path
            data_dir = None
            while current != current.root:
                if current.name == 'data':
                    data_dir = current
                    break
                current = current.parent
            if not data_dir:
                continue
            
            biom_dir = data_dir / "per_dataset" / "qiime"
            try:
                rel_path = biom_path.parent.relative_to(biom_dir)
            except ValueError:
                continue
            
            sliced_parts = rel_path.parts[:-1]
            
            meta_path = (
                data_dir / "per_dataset" / "metadata"
                / Path(*sliced_parts)
                / "sample-metadata.tsv"
            )
            meta_paths.append(meta_path)
        return meta_paths
        
    def _get_biom_table(self):
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
        if not csv_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {csv_path}")
            
        df = pd.read_csv(csv_path, sep='\t')
        df.columns = df.columns.str.lower()
        
        sample_id_col = next((col for col in ['run_accession', '#sampleid', 'sample-id'] if col in df.columns), None)
        if sample_id_col:
            df['SAMPLE ID'] = df[sample_id_col]
        else:
            df['SAMPLE ID'] = [f"{Path(csv_path).parents[5].name}_x{i}" for i in range(1, len(df)+1)]
            
        dataset_id_col = next((col for col in ['project_accession', 'dataset_id', 'dataset_name'] if col in df.columns), None)
        if dataset_id_col:
            df['DATASET ID'] = df[dataset_id_col]
        else:
            df['DATASET ID'] = Path(csv_path).parents[5].name
            
        if 'nuclear_contamination_status' not in df.columns:
            df['nuclear_contamination_status'] = False

        for old, new in column_renames:
            if old in df.columns:
                df.rename(columns={old: new}, inplace=True)
                
        return df


"""
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
            n_features, n_samples = merged_table.shape
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
