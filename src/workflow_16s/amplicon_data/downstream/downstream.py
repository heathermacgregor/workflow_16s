# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import glob
import logging
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Third-Party Imports
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from skbio.stats.ordination import OrdinationResults
from statsmodels.stats.multitest import multipletests
from biom.table import Table

# Visualization imports (moved to separate section)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from scipy import stats
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
import umap

# Local Imports
from workflow_16s import constants
from workflow_16s.amplicon_data.statistical_analyses import (
    run_statistical_tests_for_group, TopFeaturesAnalyzer
)
from workflow_16s.amplicon_data.top_features import top_features_plots
from workflow_16s.function.faprotax import (
    faprotax_functions_for_taxon, get_faprotax_parsed
)
from workflow_16s.utils.progress import get_progress_bar, _format_task_desc
from workflow_16s.amplicon_data.downstream.alpha import AlphaDiversity
from workflow_16s.amplicon_data.downstream.beta import Ordination
from workflow_16s.amplicon_data.downstream.feature_selection import FeatureSelection
from workflow_16s.amplicon_data.downstream.input import DownstreamDataLoader as InputData
from workflow_16s.amplicon_data.downstream.maps import Maps
from workflow_16s.amplicon_data.downstream.tables import PrepData
from workflow_16s.amplicon_data.downstream.stats import run_statistical_analysis_with_loading

# ================================= CONFIGURATION & CONSTANTS ========================= #

logger = logging.getLogger("workflow_16s")
umap_lock = threading.Lock()  # Global lock for UMAP operations

# Default analysis parameters
DEFAULT_TOP_N_FEATURES = 20
DEFAULT_NETWORK_CORRELATION_THRESHOLD = 0.3
DEFAULT_P_VALUE_THRESHOLD = 0.05
DEFAULT_EFFECT_SIZE_THRESHOLD = 0.5

# ================================= UTILITY CLASSES ================================== #

class AnalysisConfig:
    """Configuration container for downstream analysis parameters."""
    
    def __init__(self, config: Dict):
        self.config = config
        
    def is_enabled(self, module: str) -> bool:
        """Check if a specific analysis module is enabled."""
        return self.config.get(module, {}).get('enabled', False)
    
    def get_parameter(self, module: str, parameter: str, default: Any = None) -> Any:
        """Get a specific parameter for an analysis module."""
        return self.config.get(module, {}).get(parameter, default)

class ResultsContainer:
    """Container for organizing analysis results."""
    
    def __init__(self):
        self.metadata: Dict[str, Any] = {}
        self.tables: Dict[str, Any] = {}
        self.maps: Optional[Dict[str, Any]] = {}
        self.stats: Optional[Dict[str, Any]] = {}
        self.alpha_diversity: Optional[Dict[str, Any]] = {}
        self.ordination: Optional[Dict[str, Any]] = {}
        self.top_features: Optional[Dict[str, Any]] = {}
        self.models: Optional[Dict[str, Any]] = {}
        self.analysis_statistics: Dict[str, Any] = {}

# ================================= FUNCTIONAL ANNOTATION ========================== #

class FunctionalAnnotation:
    """Handles FAPROTAX functional annotations for features."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.db = None
        self._faprotax_cache: Dict[str, Any] = {}
        
        if self.config.get("faprotax", {}).get('enabled', False):
            self.db = get_faprotax_parsed()

    def _get_cached_faprotax(self, taxon: str) -> List[str]:
        """Get FAPROTAX functions for a taxon with caching."""
        if taxon not in self._faprotax_cache:
            self._faprotax_cache[taxon] = faprotax_functions_for_taxon(
                taxon=taxon, 
                faprotax_db=self.db, 
                include_references=False
            )
        return self._faprotax_cache[taxon]
    
    def annotate_features(self, features: List[str]) -> Dict[str, List[str]]:
        """Annotate a list of features with functional information."""
        if not self.db:
            logger.warning("FAPROTAX database not loaded")
            return {feature: [] for feature in features}
            
        results = {}
        
        with ThreadPoolExecutor() as executor:
            future_to_feature = {
                executor.submit(self._get_cached_faprotax, feature): feature 
                for feature in features
            }
            
            with get_progress_bar() as progress:
                task_desc = "Annotating features with functional information"
                task_desc_fmt = _format_task_desc(task_desc)
                task = progress.add_task(description=task_desc_fmt, total=len(features))
                
                for future in as_completed(future_to_feature):
                    feature = future_to_feature[future]
                    try:
                        results[feature] = future.result()
                    except Exception as e:
                        logger.error(f"Error annotating feature {feature}: {e}")
                        results[feature] = []
                    progress.update(task, advance=1)
        
        return results

# ================================= MAIN DOWNSTREAM CLASS ========================== #

class Downstream:
    """Main class for orchestrating 16S amplicon data analysis pipeline."""
    
    MODE_CONFIG = {
        "asv": ("table", "asv"), 
        "genus": ("table_6", "l6")
    }
    
    def __init__(
        self, 
        config: Dict, 
        project_dir: Any, 
        existing_subsets: Optional[Dict[str, Dict[str, Path]]] = None,
        verbose: bool = False,
        # Result loading parameters
        load_existing_results: bool = True,
        max_result_age_hours: Optional[float] = None,
        force_recalculate_stats: List[str] = None,
        invalidate_results_patterns: List[str] = None
    ):
        # Initialize core attributes
        self.config = AnalysisConfig(config)
        self.verbose = verbose
        self.project_dir = project_dir
        self.output_dir = self.project_dir.final
        self.existing_subsets = existing_subsets

        # Result loading configuration
        self.load_existing_results = load_existing_results
        self.max_result_age_hours = max_result_age_hours
        self.force_recalculate_stats = force_recalculate_stats or []
        self.invalidate_results_patterns = invalidate_results_patterns or []

        # Initialize analysis settings
        self._setup_analysis_mode()
        self.group_columns = config.get("group_columns", [])
        
        # Initialize result containers
        self.results = ResultsContainer()
        
        # Initialize analysis components
        self.functional_annotation = FunctionalAnnotation(config)
        
        # Execute pipeline
        self._execute_pipeline()
    
    def _setup_analysis_mode(self) -> None:
        """Setup analysis mode based on configuration."""
        default_mode = self.config.config.get("target_subfragment_mode", 
                                            constants.DEFAULT_MODE)
        self.mode = 'genus' if default_mode == 'any' else 'asv'
        
        if self.mode not in self.MODE_CONFIG:
            raise ValueError(f"Invalid mode: {self.mode}. Must be one of {list(self.MODE_CONFIG.keys())}")
    
    def _execute_pipeline(self) -> None:
        """Execute the complete analysis pipeline."""
        logger.info("Starting downstream analysis pipeline...")
        
        try:
            # Data loading and preparation
            self._load_and_prepare_data()
            
            # Run analyses based on configuration
            self._run_enabled_analyses()
            
            # Generate summary
            self._log_analysis_summary()
            
            logger.info("Downstream analysis pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise

    def _load_and_prepare_data(self) -> None:
        """Load and prepare data for analysis."""
        logger.info("Loading and preparing data...")
        
        # Load data
        data_loader = InputData(
            self.config.config, self.mode, self.project_dir, self.existing_subsets
        )
        self.results.metadata = data_loader.metadata
        self.results.tables = data_loader.tables
        self.nfc_facilities = data_loader.nfc_facilities
        
        # Prepare data for analysis
        data_prep = PrepData(
            self.config.config, 
            self.results.tables, 
            self.results.metadata, 
            self.mode, 
            self.project_dir
        )
        self.results.metadata = data_prep.metadata
        self.results.tables = data_prep.tables

    def _run_enabled_analyses(self) -> None:
        """Run all enabled analysis modules."""
        analysis_modules = [
            ('maps', self._run_sample_maps),
            ('stats', self._run_statistical_analysis),
            ('alpha_diversity', self._run_alpha_diversity),
            ('ordination', self._run_beta_diversity),
            ('ml', self._run_ml_feature_selection),
            ('top_features', self._run_top_features_analysis),
        ]
        
        for module_name, analysis_method in analysis_modules:
            if self.config.is_enabled(module_name):
                logger.info(f"Running {module_name} analysis...")
                try:
                    analysis_method()
                except Exception as e:
                    logger.error(f"Error in {module_name} analysis: {e}")
                    if self.verbose:
                        raise
            else:
                logger.info(f"{module_name} analysis disabled in configuration")

    def _run_sample_maps(self) -> None:
        """Generate sample maps if enabled."""
        maps = Maps(
            self.config.config, 
            self.results.metadata, 
            Path(self.output_dir) / 'sample_maps', 
            self.verbose
        )
        maps.generate_sample_maps(nfc_facility_data=self.nfc_facilities)
        self.results.maps = maps.maps

    def _run_statistical_analysis(self) -> None:
        """Run statistical analysis with result loading."""
        logger.info("Statistical analysis configuration:")
        logger.info(f"  - Load existing results: {self.load_existing_results}")
        logger.info(f"  - Max file age: {self.max_result_age_hours} hours" 
                   if self.max_result_age_hours else "  - No age limit")
        logger.info(f"  - Force recalculate patterns: {self.force_recalculate_stats}")

        with run_statistical_analysis_with_loading(
            config=self.config.config,
            tables=self.results.tables,
            metadata=self.results.metadata,
            mode=self.mode,
            group_columns=self.group_columns,
            project_dir=self.project_dir,
            load_existing=self.load_existing_results,
            max_file_age_hours=self.max_result_age_hours,
            force_recalculate=self.force_recalculate_stats
        ) as stats:
            # Validate configuration
            self._validate_statistical_configuration(stats)
            
            # Get and log analysis information
            self._log_statistical_analysis_info(stats)
            
            # Store results
            self.stats_obj = stats
            self.results.stats = self._compile_statistical_results(stats)

    def _validate_statistical_configuration(self, stats) -> None:
        """Validate statistical analysis configuration."""
        issues = stats.validate_configuration()
        
        if issues['errors']:
            logger.error("Configuration errors:")
            for error in issues['errors']:
                logger.error(f"  - {error}")
            raise ValueError("Statistical analysis configuration validation failed")
        
        if issues['warnings']:
            logger.warning("Configuration warnings:")
            for warning in issues['warnings']:
                logger.warning(f"  - {warning}")

    def _log_statistical_analysis_info(self, stats) -> None:
        """Log statistical analysis information."""
        recommendations = stats.get_analysis_recommendations()
        logger.info(recommendations)
        
        summary = stats.get_summary_statistics()
        logger.info(f"Statistical Analysis Summary:")
        logger.info(f"  - Total tests run: {summary['total_tests_run']}")
        logger.info(f"  - Group columns analyzed: {len(summary['group_columns_analyzed'])}")
        
        # Log loading performance
        load_stats = summary.get('performance_metrics', {}).get('load_statistics', {})
        if load_stats:
            self._log_loading_performance(load_stats)

    def _log_loading_performance(self, load_stats: Dict) -> None:
        """Log result loading performance statistics."""
        total_tasks = load_stats.get('total_tasks', 0)
        loaded_tasks = load_stats.get('loaded_from_files', 0)
        calculated_tasks = load_stats.get('calculated_fresh', 0)
        
        if total_tasks > 0:
            load_percentage = (loaded_tasks / total_tasks) * 100
            logger.info(f"  - Results loaded from files: {loaded_tasks}/{total_tasks} ({load_percentage:.1f}%)")
            logger.info(f"  - Results calculated fresh: {calculated_tasks}/{total_tasks} ({100-load_percentage:.1f}%)")

    def _compile_statistical_results(self, stats) -> Dict[str, Any]:
        """Compile statistical analysis results."""
        top_features = stats.get_top_features_across_tests()
        summary = stats.get_summary_statistics()
        
        return {
            'test_results': stats.results,
            'top_features': top_features,
            'summary': summary,
            'load_statistics': stats.get_load_report()
        }

    def _run_alpha_diversity(self) -> None:
        """Run alpha diversity analysis."""
        alpha = AlphaDiversity(self.config.config, self.results.metadata, self.results.tables)
        alpha.run(output_dir=self.output_dir)
        self.results.alpha_diversity = alpha.results

    def _run_beta_diversity(self) -> None:
        """Run beta diversity (ordination) analysis."""
        results = {}
        for group_column in self.group_columns:
            beta = Ordination(
                self.config.config, 
                self.results.metadata, 
                self.results.tables, 
                group_column['name'], 
                self.verbose
            )
            beta.run(output_dir=self.output_dir)
            results[group_column['name']] = beta.results
            
        self.results.ordination = results

    def _run_ml_feature_selection(self) -> None:
        """Run machine learning feature selection."""
        results = {}
        for group_column in self.group_columns:
            if group_column.get('type') == 'bool':
                fs = FeatureSelection(
                    self.config.config, 
                    self.results.metadata, 
                    self.results.tables, 
                    group_column['name'], 
                    self.verbose
                )
                fs.run(output_dir=self.output_dir)
                results[group_column['name']] = fs.models
                
        self.results.models = results

    def _run_top_features_analysis(self) -> None:
        """Run top features analysis."""
        self.results.top_features = {
            "stats": {},
            "models": {}
        }
        
        # Process statistical top features
        if self.config.is_enabled('stats') and self.results.stats:
            for group_column in self.group_columns:
                self._process_statistical_top_features(group_column)
        
        # Process ML top features
        if self.config.is_enabled('ml') and self.results.models:
            for group_column in self.group_columns:
                self._process_ml_top_features(group_column)

    def _process_statistical_top_features(self, group_column: Dict) -> None:
        """Process top features from statistical analysis."""
        n_features = self.config.get_parameter('top_features', 'n', DEFAULT_TOP_N_FEATURES)
        
        if not self._validate_group_column_for_top_features(group_column):
            return
            
        if not self.results.stats['test_results'].get(group_column['name']):
            logger.warning(f"No statistics calculated for group '{group_column['name']}'")
            return
            
        self.results.top_features["stats"][group_column['name']] = {}
        
        # Extract and rank features
        all_features = self._extract_statistical_features(group_column)
        if not all_features:
            return
            
        # Split by effect direction and rank
        positive_features = [f for f in all_features if f["effect"] > 0]
        negative_features = [f for f in all_features if f["effect"] < 0]
        
        positive_features.sort(key=lambda d: (-d["effect"], d["p_value"]))
        negative_features.sort(key=lambda d: (d["effect"], d["p_value"]))
        
        # Store results
        values = group_column.get('values', [True, False])
        self.results.top_features["stats"][group_column['name']][values[0]] = positive_features[:n_features]
        self.results.top_features["stats"][group_column['name']][values[1]] = negative_features[:n_features]
        
        logger.info(f"Top features for {group_column['name']}: "
                   f"{values[0]} ({len(positive_features)}), {values[1]} ({len(negative_features)})")

    def _validate_group_column_for_top_features(self, group_column: Dict) -> bool:
        """Validate group column for top features analysis."""
        if not group_column.get('values'):
            if group_column.get('type') == 'bool':
                group_column['values'] = [True, False]
            else:
                logger.warning(f"Group column values not found for {group_column.get('name')}")
                return False
        
        if len(group_column['values']) != 2:
            logger.warning(f"Group column must have exactly 2 values, got {len(group_column['values'])}")
            return False
            
        return True

    def _extract_statistical_features(self, group_column: Dict) -> List[Dict]:
        """Extract significant features from statistical tests."""
        all_features = []
        
        with self.stats_obj as stats:
            test_results = self.results.stats['test_results'][group_column['name']]
            
            for table_type, levels in test_results.items():
                for level, tests in levels.items():
                    for test_name, df in tests.items():
                        if df is None or not isinstance(df, pd.DataFrame) or "p_value" not in df.columns:
                            continue
                            
                        # Get significant features
                        sig_df = df[df["p_value"] < DEFAULT_P_VALUE_THRESHOLD].copy()
                        if sig_df.empty:
                            continue
                            
                        # Calculate effect sizes
                        sig_df["effect"] = sig_df.apply(
                            lambda row: stats.get_effect_size(test_name, row), axis=1
                        )
                        sig_df = sig_df.dropna(subset=["effect"])

                        # Add features to list
                        for _, row in sig_df.iterrows():
                            all_features.append({
                                "feature": row["feature"],
                                "column": group_column['name'],
                                "table_type": table_type,
                                "level": level,
                                "method": "statistical_test",
                                "test": test_name,
                                "effect": row["effect"],
                                "p_value": row["p_value"],
                                "effect_dir": "positive" if row["effect"] > 0 else "negative",
                            })
        
        return all_features

    def _process_ml_top_features(self, group_column: Dict) -> None:
        """Process top features from ML models."""
        n_features = self.config.get_parameter('top_features', 'n', DEFAULT_TOP_N_FEATURES)
        
        if not self.results.models.get(group_column['name']):
            logger.warning(f"No ML models for group '{group_column['name']}'")
            return
        
        features_summary = []
        models_data = self.results.models[group_column['name']]
        
        for table_type, levels in models_data.items():
            for level, methods in levels.items():
                for method, result in methods.items():
                    if not self._validate_ml_result(result, group_column['name'], table_type, level, method):
                        continue
                    
                    # Extract feature importance
                    feat_imp = result.get("feature_importances", {})
                    top_features = result.get("top_features", [])
                    
                    for i, feat in enumerate(top_features[:n_features], 1):
                        importance = feat_imp.get(feat, 0)
                        features_summary.append({
                            "Column": group_column['name'],
                            "Table Type": table_type,
                            "Level": level,
                            "Method": method,
                            "Rank": i,
                            "Feature": feat,
                            "Importance": f"{importance:.4f}" if isinstance(importance, (int, float)) else "N/A"
                        })
        
        features_df = pd.DataFrame(features_summary) if features_summary else pd.DataFrame()
        self.results.top_features["models"][group_column['name']] = features_df

    def _validate_ml_result(self, result: Any, group_name: str, table_type: str, 
                           level: str, method: str) -> bool:
        """Validate ML model result structure."""
        if not result or not isinstance(result, dict):
            logger.warning(f"Invalid result for {group_name}/{table_type}/{level}/{method}")
            return False
            
        if "top_features" not in result:
            logger.error(f"Missing 'top_features' in {group_name}/{table_type}/{level}/{method}")
            return False
            
        return True

    def _log_analysis_summary(self) -> None:
        """Log comprehensive analysis summary."""
        logger.info("=" * 60)
        logger.info("DOWNSTREAM ANALYSIS SUMMARY")
        logger.info("=" * 60)
        
        # Log enabled/disabled modules
        self._log_module_status()
        
        # Log data dimensions
        self._log_data_dimensions()
        
        # Log result loading statistics
        if hasattr(self, 'results') and self.results.stats:
            self._log_result_loading_stats()
        
        logger.info("=" * 60)

    def _log_module_status(self) -> None:
        """Log which analysis modules were enabled/disabled."""
        modules = [
            ('stats', 'Statistical Analysis'),
            ('alpha_diversity', 'Alpha Diversity'),
            ('ordination', 'Beta Diversity/Ordination'),
            ('ml', 'Machine Learning Feature Selection'),
            ('maps', 'Sample Maps'),
            ('faprotax', 'Functional Annotation')
        ]
        
        enabled = [name for key, name in modules if self.config.is_enabled(key)]
        disabled = [name for key, name in modules if not self.config.is_enabled(key)]
        
        logger.info(f"Enabled modules: {', '.join(enabled)}")
        if disabled:
            logger.info(f"Disabled modules: {', '.join(disabled)}")

    def _log_data_dimensions(self) -> None:
        """Log data dimensions summary."""
        if not (self.results.tables and self.results.metadata):
            return
            
        logger.info("Data Summary:")
        for table_type in self.results.tables:
            for level in self.results.tables[table_type]:
                table = self.results.tables[table_type][level]
                metadata = self.results.metadata[table_type][level]
                logger.info(f"  - {table_type}/{level}: {table.shape[1]} samples, {table.shape[0]} features")

    def _log_result_loading_stats(self) -> None:
        """Log result loading performance statistics."""
        if 'load_statistics' not in self.results.stats:
            return
            
        load_stats = self.results.stats['load_statistics']
        summary_info = self.results.stats.get('summary', {})
        
        if summary_info:
            logger.info("Statistical Analysis Performance:")
            logger.info(f"  - Total tasks: {summary_info.get('total_tasks', 'N/A')}")
            logger.info(f"  - Loaded from cache: {summary_info.get('loaded_from_files', 'N/A')}")
            logger.info(f"  - Calculated fresh: {summary_info.get('calculated_fresh', 'N/A')}")

    # ========================== RESULT MANAGEMENT METHODS ========================= #

    def get_analysis_report(self) -> Dict[str, Any]:
        """Get comprehensive analysis report."""
        return {
            'config': self.config.config,
            'mode': self.mode,
            'group_columns': self.group_columns,
            'load_settings': {
                'load_existing_results': self.load_existing_results,
                'max_result_age_hours': self.max_result_age_hours,
                'force_recalculate_stats': self.force_recalculate_stats,
                'invalidate_results_patterns': self.invalidate_results_patterns
            },
            'results_summary': self._generate_results_summary()
        }

    def _generate_results_summary(self) -> Dict[str, Any]:
        """Generate summary of analysis results."""
        summary = {}
        
        result_modules = [
            ('stats', self.results.stats),
            ('alpha_diversity', self.results.alpha_diversity),
            ('ordination', self.results.ordination),
            ('models', self.results.models)
        ]
        
        for module_name, module_results in result_modules:
            if module_results:
                summary[module_name] = {'enabled': True}
                if module_name == 'stats' and 'load_statistics' in module_results:
                    summary[module_name]['load_statistics'] = module_results['load_statistics']
        
        return summary

    def invalidate_and_rerun_stats(self, patterns: List[str]) -> None:
        """Invalidate specific results and rerun statistical analysis."""
        logger.info(f"Invalidating and recalculating statistical results for patterns: {patterns}")
        
        # Invalidate existing results
        stats_dir = self.project_dir.final / 'stats'
        total_deleted = 0
        for pattern in patterns:
            deleted_count = self._delete_matching_results(stats_dir, pattern)
            total_deleted += deleted_count
        
        logger.info(f"Invalidated {total_deleted} result files")
        
        # Rerun statistical analysis
        if self.config.is_enabled('stats'):
            original_patterns = self.force_recalculate_stats.copy()
            self.force_recalculate_stats.extend(patterns)
            
            try:
                self._run_statistical_analysis()
                logger.info("Statistical analysis completed successfully")
            finally:
                self.force_recalculate_stats = original_patterns

    def _delete_matching_results(self, stats_dir: Path, pattern: str) -> int:
        """Delete result files matching a specific pattern."""
        if not stats_dir.exists():
            return 0
            
        deleted_count = 0
        
        for group_dir in stats_dir.iterdir():
            if not group_dir.is_dir():
                continue
                
            for table_dir in group_dir.iterdir():
                if not table_dir.is_dir():
                    continue
                    
                for level_dir in table_dir.iterdir():
                    if not level_dir.is_dir():
                        continue
                        
                    for result_file in level_dir.glob('*.tsv'):
                        full_path = f"{group_dir.name}_{table_dir.name}_{level_dir.name}_{result_file.stem}"
                        
                        if self._pattern_matches(pattern, full_path, result_file, table_dir, level_dir, group_dir):
                            result_file.unlink()
                            deleted_count += 1
                            
                            # Also delete correlation matrices for network analysis
                            if result_file.stem == 'network_analysis':
                                corr_file = level_dir / f"{result_file.stem}_correlation_matrix.tsv"
                                if corr_file.exists():
                                    corr_file.unlink()
                                    deleted_count += 1
        
        return deleted_count

    def _pattern_matches(self, pattern: str, full_path: str, result_file: Path, 
                        table_dir: Path, level_dir: Path, group_dir: Path) -> bool:
        """Check if a file matches the deletion pattern."""
        return (pattern in full_path or 
                pattern == result_file.stem or
                pattern == f"{table_dir.name}_{level_dir.name}" or
                pattern == group_dir.name or
                pattern == table_dir.name)


# ================================= RESULTS ANALYZER CLASS ========================= #

class DownstreamResultsAnalyzer:
    """Comprehensive analysis framework for integrating downstream pipeline results."""
    
    def __init__(self, downstream_results: Downstream, config: Dict, verbose: bool = True):
        """Initialize analyzer with downstream pipeline results."""
        self.downstream = downstream_results
        self.config = config
        self.verbose = verbose
        
        # Extract data components
        self.metadata = downstream_results.results.metadata
        self.tables = downstream_results.results.tables
        self.stats = downstream_results.results.stats or {}
        self.alpha_diversity = downstream_results.results.alpha_diversity or {}
        self.ordination = downstream_results.results.ordination or {}
        self.models = downstream_results.results.models or {}
        self.top_features = downstream_results.results.top_features or {}
        
        # Initialize analysis containers
        self.integrated_results: Dict[str, Any] = {}
        self.consensus_features: Optional[pd.DataFrame] = None
        self.environmental_thresholds: Dict[str, Any] = {}
        self.networks: Dict[str, Any] = {}
        self.functional_analysis: Dict[str, Any] = {}
        
        if self.verbose:
            print("DownstreamResultsAnalyzer initialized successfully")

    # ========================== FEATURE IMPORTANCE SYNTHESIS ======================== #
    
    def synthesize_feature_importance(self, top_n: int = 50) -> pd.DataFrame:
        """Combine feature importance from multiple analysis approaches."""
        if self.verbose:
            print("Synthesizing feature importance across modules...")
        
        importance_scores = {}
        
        # Extract scores from different analysis types
        extractors = [
            ('statistical', self._extract_statistical_importance),
            ('ml_importance', self._extract_ml_importance),
            ('alpha_association', self._extract_alpha_associations),
            ('beta_loading', self._extract_beta_loadings)
        ]
        
        for score_type, extractor in extractors:
            try:
                scores = extractor()
                for feature, score in scores.items():
                    importance_scores.setdefault(feature, {})[score_type] = score
            except Exception as e:
                logger.warning(f"Error extracting {score_type} scores: {e}")
        
        if not importance_scores:
            logger.warning("No feature importance scores found")
            return pd.DataFrame()
        
        # Create consensus DataFrame
        consensus_df = pd.DataFrame.from_dict(importance_scores, orient='index').fillna(0)
        
        # Calculate weighted consensus score
        weights = {
            'statistical': 0.3,
            'ml_importance': 0.3,
            'alpha_association': 0.2,
            'beta_loading': 0.2
        }
        
        consensus_df['consensus_score'] = 0
        for col, weight in weights.items():
            if col in consensus_df.columns:
                consensus_df['consensus_score'] += consensus_df[col] * weight
        
        # Sort and return top features
        consensus_df = consensus_df.sort_values('consensus_score', ascending=False)
        self.consensus_features = consensus_df.head(top_n)
        
        if self.verbose:
            print(f"Generated consensus ranking for {len(consensus_df)} features")
        
        return self.consensus_features
    
    def _extract_statistical_importance(self) -> Dict[str, float]:
        """Extract feature importance scores from statistical tests."""
        importance_dict = {}
        
        if not self.top_features or 'stats' not in self.top_features:
            return importance_dict
            
        for group_col, group_data in self.top_features['stats'].items():
            if not isinstance(group_data, dict):
                continue
                
            for condition, features in group_data.items():
                for feature_info in features:
                    if isinstance(feature_info, dict) and 'feature' in feature_info:
                        feature = feature_info['feature']
                        p_value = feature_info.get('p_value', 1.0)
                        # Use negative log p-value as importance score
                        importance_dict[feature] = -np.log10(max(p_value, 1e-10))
        
        return importance_dict
    
    def _extract_ml_importance(self) -> Dict[str, float]:
        """Extract feature importance from ML models."""
        importance_dict = {}
        
        if not self.top_features or 'models' not in self.top_features:
            return importance_dict
            
        for group_col, df in self.top_features['models'].items():
            if isinstance(df, pd.DataFrame) and 'Feature' in df.columns and 'Importance' in df.columns:
                for _, row in df.iterrows():
                    feature = row['Feature']
                    importance_str = row['Importance']
                    
                    # Convert importance to float
                    try:
                        importance = float(importance_str) if importance_str != "N/A" else 0
                        if feature in importance_dict:
                            importance_dict[feature] += importance
                        else:
                            importance_dict[feature] = importance
                    except (ValueError, TypeError):
                        continue
        
        # Normalize by number of models if needed
        if importance_dict:
            max_importance = max(importance_dict.values())
            if max_importance > 0:
                importance_dict = {k: v/max_importance for k, v in importance_dict.items()}
        
        return importance_dict
    
    def _extract_alpha_associations(self) -> Dict[str, float]:
        """Extract features associated with alpha diversity metrics."""
        associations = {}
        
        if not self.alpha_diversity:
            return associations
            
        # Look for correlation results with features
        for metric, results in self.alpha_diversity.items():
            if isinstance(results, dict) and 'correlations' in results:
                for feature, corr_data in results['correlations'].items():
                    if isinstance(corr_data, dict) and 'correlation' in corr_data:
                        associations[feature] = abs(corr_data['correlation'])
        
        return associations
    
    def _extract_beta_loadings(self) -> Dict[str, float]:
        """Extract feature loadings from ordination analysis."""
        loadings = {}
        
        if not self.ordination:
            return loadings
            
        for group_col, group_results in self.ordination.items():
            if not isinstance(group_results, dict):
                continue
                
            for level, level_results in group_results.items():
                if not isinstance(level_results, dict):
                    continue
                    
                for method, results in level_results.items():
                    if method in ['pca', 'umap', 'tsne'] and isinstance(results, dict):
                        if 'loadings' in results:
                            method_loadings = results['loadings']
                            if isinstance(method_loadings, dict):
                                for feature, loading_value in method_loadings.items():
                                    if isinstance(loading_value, (list, np.ndarray)):
                                        loadings[feature] = np.linalg.norm(loading_value)
                                    elif isinstance(loading_value, (int, float)):
                                        loadings[feature] = abs(loading_value)
                    elif method == 'pcoa' and isinstance(results, OrdinationResults):
                        if hasattr(results, 'features') and results.features is not None:
                            feature_loadings = results.features
                            for i, feature in enumerate(feature_loadings.index):
                                # Use L2 norm across first two axes
                                loading_norm = np.linalg.norm(feature_loadings.iloc[i, :2])
                                loadings[feature] = loading_norm
        
        return loadings

    # ========================== NETWORK ANALYSIS ========================== #
    
    def build_integrated_networks(self, method: str = 'spearman', 
                                 threshold: float = DEFAULT_NETWORK_CORRELATION_THRESHOLD) -> Dict[str, Any]:
        """Create networks connecting statistically significant features."""
        if self.verbose:
            print(f"Building integrated networks using {method}...")
        
        network_results = {}
        
        if self.consensus_features is None:
            logger.warning("No consensus features available for network analysis")
            return network_results
            
        top_features = self.consensus_features.index.tolist()[:30]
        
        # Extract abundance data for top features
        abundance_data = self._get_feature_abundance_matrix(top_features)
        
        if abundance_data is None:
            logger.warning("Could not extract abundance data for network analysis")
            return network_results
        
        # Calculate correlation matrix
        try:
            if method == 'spearman':
                corr_matrix = abundance_data.corr(method='spearman')
            elif method == 'pearson':
                corr_matrix = abundance_data.corr(method='pearson')
            else:
                raise ValueError(f"Unknown correlation method: {method}")
            
            # Create network from correlation matrix
            network = self._create_network_from_correlations(corr_matrix, threshold)
            
            # Calculate network properties
            network_properties = self._calculate_network_properties(network)
            
            network_results[method] = {
                'network': network,
                'correlation_matrix': corr_matrix,
                'properties': network_properties,
                'adjacency_matrix': nx.adjacency_matrix(network).todense()
            }
            
            if self.verbose:
                print(f"Network created: {network_properties.get('n_nodes', 0)} nodes, "
                     f"{network_properties.get('n_edges', 0)} edges")
                     
        except Exception as e:
            logger.error(f"Error creating {method} network: {e}")
        
        self.networks = network_results
        self.integrated_results['networks'] = network_results
        return network_results
    
    def _get_feature_abundance_matrix(self, features: List[str]) -> Optional[pd.DataFrame]:
        """Get abundance matrix for specified features."""
        community_data = self._extract_community_matrix()
        
        if community_data is not None:
            available_features = [f for f in features if f in community_data.columns]
            if available_features:
                return community_data[available_features].copy()
        
        return None
    
    def _extract_community_matrix(self) -> Optional[pd.DataFrame]:
        """Extract community abundance matrix from tables."""
        for mode in ['genus', 'asv']:
            if mode not in self.tables:
                continue
                
            tables = self.tables[mode]
            if not isinstance(tables, dict):
                continue
                
            for subset_name, table_data in tables.items():
                try:
                    if hasattr(table_data, 'to_dataframe'):
                        # BIOM table - transpose so samples are rows
                        community_data = table_data.to_dataframe().T
                        return community_data
                    elif isinstance(table_data, pd.DataFrame):
                        # Regular DataFrame - ensure samples as rows
                        if table_data.shape[0] < table_data.shape[1]:
                            return table_data.T
                        else:
                            return table_data
                except Exception as e:
                    logger.warning(f"Could not extract data from {mode}/{subset_name}: {e}")
                    continue
        
        return None
    
    def _create_network_from_correlations(self, corr_matrix: pd.DataFrame, 
                                        threshold: float) -> nx.Graph:
        """Create network graph from correlation matrix."""
        G = nx.Graph()
        
        # Add nodes
        G.add_nodes_from(corr_matrix.index)
        
        # Add edges for correlations above threshold
        for i in range(len(corr_matrix)):
            for j in range(i+1, len(corr_matrix)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val >= threshold and not np.isnan(corr_val):
                    G.add_edge(
                        corr_matrix.index[i], 
                        corr_matrix.index[j], 
                        weight=corr_val
                    )
        
        return G
    
    def _calculate_network_properties(self, network: nx.Graph) -> Dict[str, Any]:
        """Calculate network topology properties."""
        properties = {
            'n_nodes': network.number_of_nodes(),
            'n_edges': network.number_of_edges(),
            'density': 0.0,
            'clustering_coefficient': 0.0,
            'n_components': 0
        }
        
        if properties['n_nodes'] == 0:
            return properties
            
        properties['density'] = nx.density(network)
        properties['n_components'] = nx.number_connected_components(network)
        properties['clustering_coefficient'] = nx.average_clustering(network)
        
        # Calculate centrality measures only if we have edges
        if properties['n_edges'] > 0:
            try:
                properties['degree_centrality'] = nx.degree_centrality(network)
                properties['betweenness_centrality'] = nx.betweenness_centrality(network)
                properties['closeness_centrality'] = nx.closeness_centrality(network)
                # Eigenvector centrality can fail for disconnected graphs
                if nx.is_connected(network):
                    properties['eigenvector_centrality'] = nx.eigenvector_centrality(network)
            except Exception as e:
                logger.warning(f"Error calculating network centralities: {e}")
        
        return properties

    # ========================== ENVIRONMENTAL ANALYSIS ========================== #
    
    def analyze_environmental_gradients(self, continuous_vars: List[str] = None) -> Dict[str, Any]:
        """Analyze environmental gradients using canonical correspondence analysis."""
        if self.verbose:
            print("Analyzing environmental gradients...")
        
        if continuous_vars is None:
            continuous_vars = ['ph', 'facility_distance_km']
        
        gradient_results = {}
        
        # Extract data
        env_data = self._extract_environmental_data(continuous_vars)
        community_data = self._extract_community_matrix()
        
        if env_data is None or community_data is None:
            logger.warning("Could not extract environmental or community data")
            return gradient_results
        
        # Align samples
        common_samples = list(set(env_data.index) & set(community_data.index))
        if len(common_samples) < 10:
            logger.warning(f"Insufficient overlapping samples: {len(common_samples)}")
            return gradient_results
        
        env_aligned = env_data.loc[common_samples]
        comm_aligned = community_data.loc[common_samples]
        
        # Remove any remaining NaN values
        env_aligned = env_aligned.dropna()
        comm_aligned = comm_aligned.loc[env_aligned.index]
        
        if len(env_aligned) < 10:
            logger.warning("Insufficient samples after removing NaN values")
            return gradient_results
        
        try:
            # Perform canonical correspondence analysis approximation
            from sklearn.cross_decomposition import CCA
            
            n_components = min(len(continuous_vars), 3, len(env_aligned.columns))
            cca = CCA(n_components=n_components)
            env_scores, comm_scores = cca.fit_transform(env_aligned, comm_aligned)
            
            gradient_results['cca_results'] = {
                'environmental_scores': pd.DataFrame(
                    env_scores, 
                    index=env_aligned.index, 
                    columns=[f'CCA{i+1}_env' for i in range(env_scores.shape[1])]
                ),
                'community_scores': pd.DataFrame(
                    comm_scores,
                    index=env_aligned.index,
                    columns=[f'CCA{i+1}_comm' for i in range(comm_scores.shape[1])]
                ),
                'explained_variance': cca.score(env_aligned, comm_aligned)
            }
            
            # Calculate feature loadings
            feature_loadings = self._calculate_cca_loadings(
                comm_aligned, comm_scores, continuous_vars
            )
            gradient_results['feature_loadings'] = feature_loadings
            
            if self.verbose:
                print(f"CCA analysis completed with {n_components} components")
                
        except Exception as e:
            logger.error(f"Error in CCA analysis: {e}")
        
        self.integrated_results['environmental_gradients'] = gradient_results
        return gradient_results
    
    def _extract_environmental_data(self, variables: List[str] = None) -> Optional[pd.DataFrame]:
        """Extract environmental variables from metadata."""
        if variables is None:
            variables = ['ph', 'facility_distance_km']
        
        # Try to extract from different metadata levels
        for mode in ['genus', 'asv']:
            if mode not in self.metadata:
                continue
                
            metadata = self.metadata[mode]
            if isinstance(metadata, dict):
                for subset_name, subset_data in metadata.items():
                    if isinstance(subset_data, pd.DataFrame):
                        available_vars = [v for v in variables if v in subset_data.columns]
                        if available_vars:
                            return subset_data[available_vars].copy()
            elif isinstance(metadata, pd.DataFrame):
                available_vars = [v for v in variables if v in metadata.columns]
                if available_vars:
                    return metadata[available_vars].copy()
        
        return None
    
    def _calculate_cca_loadings(self, community_data: pd.DataFrame, 
                              comm_scores: np.ndarray, env_vars: List[str]) -> Dict[str, Dict]:
        """Calculate feature loadings on CCA axes."""
        loadings = {}
        
        for i in range(comm_scores.shape[1]):
            axis_name = f'CCA{i+1}'
            axis_loadings = {}
            
            for feature in community_data.columns:
                try:
                    corr, p_val = spearmanr(community_data[feature], comm_scores[:, i])
                    axis_loadings[feature] = {
                        'loading': corr if not np.isnan(corr) else 0.0,
                        'p_value': p_val if not np.isnan(p_val) else 1.0
                    }
                except Exception:
                    axis_loadings[feature] = {'loading': 0.0, 'p_value': 1.0}
            
            loadings[axis_name] = axis_loadings
        
        return loadings

    # ========================== COMPREHENSIVE ANALYSIS RUNNER ======================== #
    
    def run_comprehensive_analysis(self, output_dir: str = 'integrated_analysis_output') -> Dict[str, Any]:
        """Run comprehensive integrated analysis pipeline."""
        if self.verbose:
            print("=" * 60)
            print("RUNNING COMPREHENSIVE DOWNSTREAM ANALYSIS")
            print("=" * 60)
        
        results = {}
        
        try:
            # 1. Feature importance synthesis
            consensus_features = self.synthesize_feature_importance()
            results['consensus_features'] = consensus_features
            
            # 2. Network analysis
            if consensus_features is not None and not consensus_features.empty:
                network_results = self.build_integrated_networks()
                results['networks'] = network_results
            
            # 3. Environmental gradient analysis
            gradient_results = self.analyze_environmental_gradients()
            results['environmental_gradients'] = gradient_results
            
            # 4. Create output directory and save results
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            # Save consensus features
            if consensus_features is not None and not consensus_features.empty:
                consensus_features.to_csv(f"{output_dir}/consensus_features.csv")
                if self.verbose:
                    print(f"Saved consensus features to {output_dir}/consensus_features.csv")
            
            # Generate summary report
            summary_report = self._generate_analysis_summary(results)
            with open(f"{output_dir}/analysis_summary.txt", 'w') as f:
                f.write(summary_report)
            
            if self.verbose:
                print(f"Analysis completed! Results saved to: {output_dir}")
                print("=" * 60)
            
            return results
            
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
            if self.verbose:
                raise
            return {}
    
    def _generate_analysis_summary(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive analysis summary report."""
        report_lines = [
            "INTEGRATED DOWNSTREAM ANALYSIS SUMMARY",
            "=" * 50,
            ""
        ]
        
        # Consensus features summary
        if 'consensus_features' in results and results['consensus_features'] is not None:
            cf = results['consensus_features']
            report_lines.extend([
                "CONSENSUS FEATURES ANALYSIS",
                "-" * 30,
                f"Total features analyzed: {len(cf)}",
                f"Top 5 features by consensus score:"
            ])
            
            for i, (feature, row) in enumerate(cf.head().iterrows(), 1):
                score = row.get('consensus_score', 0)
                report_lines.append(f"  {i}. {feature} (score: {score:.3f})")
            
            report_lines.append("")
        
        # Network analysis summary
        if 'networks' in results and results['networks']:
            report_lines.extend([
                "NETWORK ANALYSIS",
                "-" * 20
            ])
            
            for method, network_data in results['networks'].items():
                props = network_data.get('properties', {})
                n_nodes = props.get('n_nodes', 0)
                n_edges = props.get('n_edges', 0)
                density = props.get('density', 0)
                
                report_lines.append(
                    f"{method.capitalize()} network: {n_nodes} nodes, "
                    f"{n_edges} edges (density: {density:.3f})"
                )
            
            report_lines.append("")
        
        # Environmental gradients summary
        if 'environmental_gradients' in results and results['environmental_gradients']:
            eg = results['environmental_gradients']
            report_lines.extend([
                "ENVIRONMENTAL GRADIENTS",
                "-" * 25
            ])
            
            if 'cca_results' in eg:
                explained_var = eg['cca_results'].get('explained_variance', 0)
                report_lines.append(f"CCA explained variance: {explained_var:.3f}")
                
                env_scores = eg['cca_results'].get('environmental_scores')
                if env_scores is not None:
                    n_components = env_scores.shape[1]
                    n_samples = len(env_scores)
                    report_lines.append(f"Components: {n_components}, Samples: {n_samples}")
            
            report_lines.append("")
        
        # Analysis modules status
        modules_status = []
        if self.stats:
            modules_status.append("Statistical Analysis")
        if self.alpha_diversity:
            modules_status.append("Alpha Diversity")
        if self.ordination:
            modules_status.append("Beta Diversity")
        if self.models:
            modules_status.append("Machine Learning")
        
        if modules_status:
            report_lines.extend([
                "COMPLETED ANALYSIS MODULES",
                "-" * 30,
                ", ".join(modules_status),
                ""
            ])
        
        # Data summary
        if self.metadata and self.tables:
            report_lines.extend([
                "DATA SUMMARY",
                "-" * 15,
                "Available data modes:"
            ])
            
            for mode in ['genus', 'asv']:
                if mode in self.tables:
                    report_lines.append(f"  - {mode}: available")
            
            report_lines.append("")
        
        report_lines.extend([
            "=" * 50,
            "Analysis completed successfully"
        ])
        
        return "\n".join(report_lines)


# ================================= MAIN EXECUTION EXAMPLE ========================== #

def run_integrated_downstream_analysis(config: Dict, project_dir: Any, **kwargs) -> Tuple[Downstream, DownstreamResultsAnalyzer]:
    """
    Run the complete integrated downstream analysis pipeline.
    
    Args:
        config: Analysis configuration dictionary
        project_dir: Project directory object
        **kwargs: Additional parameters for Downstream class
    
    Returns:
        Tuple of (Downstream results object, Results analyzer object)
    """
    
    # Run main downstream analysis
    logger.info("Starting integrated downstream analysis...")
    
    downstream = Downstream(
        config=config,
        project_dir=project_dir,
        **kwargs
    )
    
    # Create results analyzer
    analyzer = DownstreamResultsAnalyzer(
        downstream_results=downstream,
        config=config,
        verbose=kwargs.get('verbose', False)
    )
    
    # Run comprehensive analysis
    integrated_results = analyzer.run_comprehensive_analysis()
    
    logger.info("Integrated downstream analysis completed successfully")
    
    return downstream, analyzer
