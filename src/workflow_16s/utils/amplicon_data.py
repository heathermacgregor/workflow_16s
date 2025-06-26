# ===================================== IMPORTS ====================================== #
# Standard Library
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Third-Party
import pandas as pd
from biom.table import Table
from rich.progress import Progress, TaskID

# Local - Specific imports from submodules
from workflow_16s.utils.biom import collapse_taxa, export_h5py, presence_absence
from workflow_16s.utils.progress import create_progress
from workflow_16s.utils.file_utils import import_merged_table_biom, import_merged_meta_tsv, filter_and_reorder_biom_and_metadata
from workflow_16s.stats.utils import clr_transform_table, filter_table, normalize_table, table_to_dataframe
from workflow_16s.stats.tests import fisher_exact_bonferroni, kruskal_bonferroni, mwu_bonferroni, ttest
from workflow_16s.stats.beta_diversity import pcoa, pca, tsne, umap
from workflow_16s.figures.merged.merged import mds, pca as plot_pca, pcoa as plot_pcoa, sample_map_categorical
from workflow_16s.function.faprotax import get_faprotax_parsed, faprotax_functions_for_taxon
from workflow_16s.models.feature_selection import catboost_feature_selection
# ================================== CONFIGURATION =================================== #
logger = logging.getLogger('workflow_16s')
DEFAULT_GROUP_COLUMN = 'nuclear_contamination_status'
DEFAULT_GROUP_COLUMN_VALUES = [True, False]

def print_structure(obj, indent=0, _key='root'):
    """
    Recursively generate the structural outline of a Python object as a block of text.
    
    Parameters
    ----------
    obj : Any
        The object to inspect (dict, list, or anything else).
    indent : int, optional
        Current indentation level (used internally by the function).
    _key : str, optional
        The name of the current branch (used internally).
    
    Returns
    -------
    str
        A string representing the structural outline of the object.
    """
    spacer = ' ' * indent
    type_name = type(obj).__name__
    
    # Create line for the current level
    if indent == 0:
        line = f'{_key} ({type_name})'
    else:
        line = f'{spacer}|-- {_key} ({type_name})'
    
    # Initialize the result with the current line
    result = line
    
    # Recurse into dictionaries
    if isinstance(obj, dict):
        for k, v in obj.items():
            child_result = print_structure(v, indent + 4, k)
            result += '\n' + child_result
    
    # For non-empty lists, show the first element as representative
    elif isinstance(obj, list) and obj:
        child_result = print_structure(obj[0], indent + 4, '[0]')
        result += '\n' + child_result
    
    return result

# =============================== STATISTICAL ANALYZER ================================ #
class StatisticalAnalyzer:
    """Handles statistical analyses with standardized configuration"""
    
    TEST_MAP = {
        'fisher': {
            'func': fisher_exact_bonferroni,
            'name': 'Fisher test (w/ Bonferroni)',
            'effect_col': 'proportion_diff',
            'alt_effect_col': 'odds_ratio'
        },
        'ttest': {
            'func': ttest,
            'name': 't-test',
            'effect_col': 'mean_difference',
            'alt_effect_col': 'cohens_d'
        },
        'mwu_bonferroni': {
            'func': mwu_bonferroni,
            'name': 'Mann-Whitney U test (w/ Bonferroni)',
            'effect_col': 'effect_size_r',
            'alt_effect_col': 'median_difference'
        },
        'kruskal_bonferroni': {
            'func': kruskal_bonferroni,
            'name': 'Kruskal-Wallis test (w/ Bonferroni)',
            'effect_col': 'epsilon_squared',
            'alt_effect_col': None
        }
    }
    
    def __init__(self, cfg: Dict, verbose: bool = False):
        self.cfg = cfg
        self.verbose = verbose
    
    def run_tests(self, table: Table, metadata: pd.DataFrame, group_column: str, 
                 group_values: List[str], enabled_tests: List[str], 
                 progress: Optional[Progress] = None, task_id: Optional[TaskID] = None) -> Dict[str, Any]:
        """Run configured statistical tests"""
        results = {}
        for test_name in enabled_tests:
            if test_name not in self.TEST_MAP:
                continue
                
            config = self.TEST_MAP[test_name]
            if self.verbose:
                logger.info(f"Running {config['name']}...")
                
            results[test_name] = config['func'](
                table=table,
                metadata=metadata,
                group_column=group_column,
                group_column_values=group_values,
            )
            
            if progress and task_id:
                progress.update(task_id, advance=1)
                
        return results

    def get_effect_size(self, test_name: str, result_row: pd.Series) -> Optional[float]:
        """Extract effect size from test results"""
        if test_name not in self.TEST_MAP:
            return None
            
        config = self.TEST_MAP[test_name]
        for col in [config['effect_col'], config['alt_effect_col']]:
            if col and col in result_row:
                return result_row[col]
        return None

# ================================= ORDINATION HANDLER ================================ #
class OrdinationHandler:
    """Manages ordination analyses and visualization"""
    
    METHOD_MAP = {
        'pca': {
            'func': pca,
            'plotter': plot_pca,
            'name': 'Principal Components Analysis'
        },
        'pcoa': {
            'func': pcoa,
            'plotter': plot_pcoa,
            'name': 'Principal Coordinates Analysis',
        },
        'tsne': {
            'func': tsne,
            'plotter': mds,
            'name': 't-SNE',
            'plot_kwargs': {'mode': 'TSNE'}
        },
        'umap': {
            'func': umap,
            'plotter': mds,
            'name': 'UMAP',
            'plot_kwargs': {'mode': 'UMAP'}
        }
    }
    
    def __init__(self, output_dir: Path, verbose: bool = False):
        self.output_dir = output_dir
        self.verbose = verbose
        
    def run_analyses(self, table: Table, metadata: pd.DataFrame, color_col: str, symbol_col: str,
                    transformation: str, methods: List[str], progress: Optional[Progress] = None,
                    task_id: Optional[TaskID] = None) -> Tuple[Dict, Dict]:
        """Run ordination methods and generate plots"""
        results, figures = {}, {}
        table, metadata = filter_and_reorder_biom_and_metadata(table, metadata)
        
        for method in methods:
            if method not in self.METHOD_MAP:
                continue
                
            config = self.METHOD_MAP[method]
            try:
                # Compute ordination
                ord_result = config['func'](table=table)
                results[method] = ord_result
                
                # Generate plot
                fig = self._generate_plot(
                    config, ord_result, metadata, color_col, 
                    symbol_col, transformation, method
                )
                figures[method] = fig
            except Exception as e:
                if self.verbose:
                    logger.error(f"Ordination failed for {method}: {str(e)}")
                figures[method] = None
            finally:
                if progress and task_id:
                    progress.update(task_id, advance=1)
                    
        return results, figures

    def _generate_plot(self, config: Dict, result: Any, metadata: pd.DataFrame, 
                      color_col: str, symbol_col: str, transformation: str, method: str):
        """Generate visualization for ordination result"""
        plot_kwargs = {
            'metadata': metadata,
            'color_col': color_col,
            'symbol_col': symbol_col,
            'transformation': transformation,
            'output_dir': self.output_dir,
            **config.get('plot_kwargs', {})
        }
        
        if method == 'pca':
            plot_kwargs.update({
                'components': result['components'],
                'proportion_explained': result['exp_var_ratio']
            })
        elif method == 'pcoa':
            plot_kwargs.update({
                'components': result.samples,
                'proportion_explained': result.proportion_explained
            })
        else:  # t-SNE/UMAP
            plot_kwargs['df'] = result
            
        return config['plotter'](**plot_kwargs)[0]

# =============================== TOP FEATURES ANALYZER =============================== #
class TopFeaturesAnalyzer:
    """Identifies significant features associated with groups"""
    def __init__(self, cfg: Dict, verbose: bool = False):
        self.cfg = cfg
        self.verbose = verbose
    
    def analyze(self, stats_results: Dict, group_column: str) -> Tuple[List, List]:
        """Identify top features from statistical results"""
        contaminated, pristine = [], []
        analyzer = StatisticalAnalyzer(self.cfg, self.verbose)
        
        for level in ['phylum', 'class', 'order', 'family', 'genus']:
            level_features = self._process_level_features(
                stats_results, level, analyzer
            )
            self._classify_features(level, level_features, contaminated, pristine)
            
        return self._sort_features(contaminated), self._sort_features(pristine)

    def _process_level_features(self, stats_results, level, analyzer):
        """Process features at a specific taxonomic level"""
        features = {}
        for table_type, tests in stats_results.items():
            for test_name, test_results in tests.items():
                if level not in test_results:
                    continue
                    
                for _, row in test_results[level].iterrows():
                    if not self._is_significant(row):
                        continue
                        
                    feature = row['feature']
                    effect = analyzer.get_effect_size(test_name, row)
                    if effect is None:
                        continue
                        
                    current = features.get(feature)
                    if not current or row['p_value'] < current['p_value']:
                        features[feature] = {
                            'p_value': row['p_value'],
                            'effect': effect,
                            'table_type': table_type,
                            'test': test_name
                        }
        return features

    def _is_significant(self, row):
        return not (pd.isna(row['p_value']) or row['p_value'] > 0.05)

    def _classify_features(self, level, features, contaminated, pristine):
        """Classify features based on effect direction"""
        for feature, data in features.items():
            feature_data = {
                'feature': feature,
                'level': level,
                **data
            }
            if data['effect'] > 0:
                contaminated.append(feature_data)
            else:
                pristine.append(feature_data)

    def _sort_features(self, features):
        return sorted(features, key=lambda x: (-abs(x['effect']), x['p_value']))

# ================================== DATA PROCESSOR ================================== #
class DataProcessor:
    """Handles data processing pipeline"""
    def __init__(self, cfg: Dict, verbose: bool = False):
        self.cfg = cfg
        self.verbose = verbose
        self.tables = {}
        
    def process(self, table: Table) -> Dict:
        """Execute full processing pipeline"""
        self.tables["raw"] = table
        self._apply_preprocessing()
        self._collapse_taxa()
        self._create_presence_absence()
        return self.tables
    
    def _apply_preprocessing(self):
        """Apply filtering, normalization, and CLR transformation"""
        current_table = self.tables["raw"]
        
        if self.cfg['features']['filter']:
            current_table = self._process_step(
                current_table, "Filtering", filter_table
            )
            self.tables["filtered"] = current_table
            
        if self.cfg['features']['normalize']:
            current_table = self._process_step(
                current_table, "Normalizing", 
                lambda t: normalize_table(t, axis=1)
            )
            self.tables["normalized"] = current_table
            
        if self.cfg['features']['clr_transform']:
            current_table = self._process_step(
                current_table, "CLR Transforming", clr_transform_table
            )
            self.tables["clr_transformed"] = current_table
            
    def _process_step(self, table: Table, name: str, func: Callable) -> Table:
        """Execute a processing step with progress tracking"""
        if self.verbose:
            logger.info(f"{name} table...")
        return func(table)
    
    def _collapse_taxa(self):
        """Collapse tables to different taxonomic levels"""
        tax_levels = ['phylum', 'class', 'order', 'family', 'genus']
        for table_type in list(self.tables.keys()):
            self.tables[table_type] = {
                level: collapse_taxa(self.tables[table_type], level)
                for level in tax_levels
            }
    
    def _create_presence_absence(self):
        """Create presence/absence tables if configured"""
        if not self.cfg['features']['presence_absence']:
            return
            
        self.tables["presence_absence"] = {
            level: presence_absence(self.tables["raw"][level])
            for level in self.tables["raw"]
        }

# ================================== AMPLICON ANALYZER ================================== #
class AmpliconData:
    """Main analysis controller"""
    MODES = {
        'asv': ('table', 'asv'),
        'genus': ('table_6', 'l6')
    }
    
    def __init__(self, cfg: Dict, project_dir: Path, mode: str = 'genus', verbose: bool = False):
        self.cfg = cfg
        self.project_dir = project_dir
        self.mode = mode
        self.verbose = verbose
        self.meta = None
        self.table = None
        self.tables = {}
        self.results = {
            'stats': {},
            'ordination': {},
            'figures': {},
            'top_features': {'contaminated': [], 'pristine': []}
        }
        
        if mode not in self.MODES:
            raise ValueError(f"Invalid mode: {mode}")
        
        table_dir, output_dir = self.MODES[mode]
        self.paths = {
            'figures': project_dir.figures,
            'table': project_dir.data / 'merged' / 'table' / output_dir / 'feature-table.biom',
            'metadata': project_dir.data / 'merged' / 'metadata' / 'sample-metadata.tsv'
        }
        
        if cfg.get("faprotax", False):
            self.fdb = get_faprotax_parsed()
        
        self._load_data()
        self._process_data()
        self._run_analyses()
        
    def _load_data(self):
        """Load and align metadata and feature tables"""
        self.meta = self._load_metadata()
        self.table = self._load_biom_table()
        self.table, self.meta = filter_and_reorder_biom_and_metadata(
            self.table, self.meta, "#sampleid"
        )
        
    def _load_metadata(self) -> pd.DataFrame:
        """Load metadata from project directory"""
        meta_paths = [
            self.project_dir / 'metadata_per_dataset' / p.parts[-6:-1] / "sample-metadata.tsv"
            for p in self._get_biom_paths()
        ]
        return import_merged_meta_tsv(meta_paths, self.verbose)
    
    def _load_biom_table(self) -> Table:
        """Load BIOM table from project directory"""
        biom_paths = self._get_biom_paths()
        if not biom_paths:
            raise FileNotFoundError("No BIOM files found")
        return import_merged_table_biom(biom_paths, 'table', self.verbose)
    
    def _get_biom_paths(self) -> List[Path]:
        """Discover BIOM file paths"""
        pattern = self.project_dir / 'qiime_data_per_dataset' / '*' / '*' / '*' / '*' / 'FWD_*_REV_*' / self.MODES[self.mode][0] / 'feature-table.biom'
        return list(pattern.parent.glob('feature-table.biom'))
    
    def _process_data(self):
        """Process data through transformation pipeline"""
        processor = DataProcessor(self.cfg, self.verbose)
        self.tables = processor.process(self.table)
        self._save_tables()
        
    def _save_tables(self):
        """Save processed tables to appropriate locations"""
        base_dir = self.project_dir / 'data' / 'merged' / 'table'
        for table_type, level_data in self.tables.items():
            for level, table in level_data.items():
                output_path = base_dir / table_type / level / "feature-table.biom"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                export_h5py(table, output_path)
    
    def _run_analyses(self):
        """Execute statistical and ordination analyses"""
        logger.info(print_structure(self.tables))
        
        self._run_statistical_tests()
        logger.info(print_structure(self.results))
        self._run_ordination()
        logger.info(print_structure(self.results))
        self._identify_top_features()
        logger.info(print_structure(self.results))
        
        if self.cfg.get("run_ml", False):
            self._run_ml_feature_selection()
            logger.info(print_structure(self.results))
    
    def _run_statistical_tests(self):
        """Run configured statistical tests"""
        group_col = self.cfg.get('group_column', DEFAULT_GROUP_COLUMN)
        group_vals = self.cfg.get('group_values', DEFAULT_GROUP_COLUMN_VALUES)
        tests = self.cfg['stats'].get('tests', ['fisher', 'ttest'])
        analyzer = StatisticalAnalyzer(self.cfg, self.verbose)
        
        for table_type, level_data in self.tables.items():
            self.results['stats'][table_type] = {}
            for level, table in level_data.items():
                table, meta = filter_and_reorder_biom_and_metadata(table, self.meta)
                self.results['stats'][table_type][level] = analyzer.run_tests(
                    table, meta, group_col, group_vals, tests
                )
    
    def _run_ordination(self):
        """Run ordination analyses"""
        methods = ['pca', 'pcoa', 'tsne', 'umap']
        ord_handler = OrdinationHandler(self.paths['figures'], self.verbose)
        
        for table_type, level_data in self.tables.items():
            self.results['ordination'][table_type] = {}
            self.results['figures'][table_type] = {}
            for level, table in level_data.items():
                results, figures = ord_handler.run_analyses(
                    table, self.meta, 'dataset_name', 
                    'nuclear_contamination_status', f"{table_type}_{level}", 
                    methods
                )
                self.results['ordination'][table_type][level] = results
                self.results['figures'][table_type][level] = figures
    
    def _identify_top_features(self):
        """Identify significant features"""
        analyzer = TopFeaturesAnalyzer(self.cfg, self.verbose)
        contaminated, pristine = analyzer.analyze(
            self.results['stats'], DEFAULT_GROUP_COLUMN
        )
        self.results['top_features']['contaminated'] = contaminated
        self.results['top_features']['pristine'] = pristine
    
    def _run_ml_feature_selection(self):
        """Execute machine learning feature selection"""
        for table_type, level_data in self.tables.items():
            for level, table in level_data.items():
                X = table_to_dataframe(table)
                y = self.meta.set_index('#sampleid')[[DEFAULT_GROUP_COLUMN]]
                
                # Align data
                common_idx = X.index.intersection(y.index)
                X, y = X.loc[common_idx], y.loc[common_idx]
                
                output_dir = self.project_dir / 'final' / 'ml' / level / table_type
                catboost_feature_selection(
                    metadata=y,
                    features=X,
                    output_dir=output_dir,
                    contamination_status_col=DEFAULT_GROUP_COLUMN,
                    method=self.cfg['ml'].get('method', 'rfe')
                )
