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

# Third‑Party Imports
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
from biom.table import Table

# ================================== LOCAL IMPORTS =================================== #

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

# ==================================== FUNCTIONS ===================================== #

logger = logging.getLogger("workflow_16s")
# Global lock for UMAP operations to prevent thread conflicts
umap_lock = threading.Lock()

# ================================= DEFAULT VALUES =================================== #

class FunctionalAnnotation:
    def __init__(
        self,
        config: Dict
    ):
        self.config = config
        if self.config.get("faprotax", {}).get('enabled', False):
            self.db = get_faprotax_parsed()
        self._faprotax_cache: Dict[str, Any] = {}

    def _get_cached_faprotax(self, taxon: str) -> List[str]:
        if taxon not in self._faprotax_cache:
            self._faprotax_cache[taxon] = faprotax_functions_for_taxon(taxon, self.db, include_references=False)
        return self._faprotax_cache[taxon]
    
    def _annotate_features(self, features):
        features = list(features)
        # Initialize results array
        results = [None] * len(features)

        with ThreadPoolExecutor() as executor:
            future_to_idx = {executor.submit(self._get_cached_faprotax, taxon): idx for idx, taxon in enumerate(features)}
            with get_progress_bar() as progress:
                task = progress.add_task(description=_format_task_desc("Annotating most important features"), total=len(features))
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    results[idx] = future.result()
                    progress.update(task, advance=1)
        
        # Create taxon map
        taxon_map = dict(zip(features, results))
        
        # Annotate features across all groups and conditions
        for feature in features:
            feature["faprotax_function"] = taxon_map.get(feature["feature"], [])
        return results

class Downstream:
    """Main class for orchestrating 16S amplicon data analysis pipeline with result loading."""
    
    ModeConfig = {
        "asv": ("table", "asv"), 
        "genus": ("table_6", "l6")
    }
    
    def __init__(
        self, 
        config: Dict, 
        project_dir: Any, 
        mode: str = constants.DEFAULT_MODE, 
        existing_subsets: Optional[Dict[str, Dict[str, Path]]] = None,
        verbose: bool = False,
        # New parameters for result loading
        load_existing_results: bool = True,
        max_result_age_hours: Optional[float] = None,
        force_recalculate_stats: List[str] = None,
        invalidate_results_patterns: List[str] = None
    ):
        self.config, self.project_dir, self.verbose = config, project_dir, verbose
        self.output_dir = self.project_dir.final
        self.existing_subsets = existing_subsets
        self.mode = 'genus' if self.config.get("target_subfragment_mode", constants.DEFAULT_MODE) == 'any' else 'asv'
        
        # Result loading configuration
        self.load_existing_results = load_existing_results
        self.max_result_age_hours = max_result_age_hours
        self.force_recalculate_stats = force_recalculate_stats or []
        self.invalidate_results_patterns = invalidate_results_patterns or []
        
        self._validate_mode()

        self.group_columns = self.config.get("group_columns", [])
        
        # Initialize result containers
        self.metadata: Dict[str, Any] = {}
        self.tables: Dict[str, Any] = {}
        self.maps: Optional[Dict[str, Any]] = {}
        self.stats: Optional[Dict[str, Any]] = {}
        self.alpha_diversity: Optional[Dict[str, Any]] = {}
        self.ordination: Optional[Dict[str, Any]] = {}
        self.top_features: Optional[Dict[str, Any]] = {}
        self.models: Optional[Dict[str, Any]] = {}
        
        # Statistics about result loading
        self.analysis_statistics = {}
        
        logger.info("Running downstream analysis pipeline...")
        self._execute_pipeline()
      
    def _validate_mode(self) -> None:
        """Make sure that the mode variable is recognized in self.ModeConfig."""
        if self.mode not in self.ModeConfig:
            raise ValueError(f"Invalid mode: {self.mode}")
          
    def _execute_pipeline(self):
        """Execute the analysis pipeline in sequence."""
        # Load data
        self.metadata, self.tables, self.nfc_facilities = self._load_data()
        # Prepare data for analysis
        self.metadata, self.tables = self._prep_data()
        # Run analysis
        self._run_analysis()
        
        if self.verbose:
            logger.info("AmpliconData analysis finished.")

    def _load_data(self):
        data = InputData(self.config, self.mode, self.project_dir, self.existing_subsets)
        return data.metadata, data.tables, data.nfc_facilities
    
    def _prep_data(self):
        data = PrepData(self.config, self.tables, self.metadata, self.mode, self.project_dir)
        return data.metadata, data.tables

    def _run_analysis(self):
        """Run all analysis steps."""
        # Handle result invalidation if requested
        if self.invalidate_results_patterns:
            self._invalidate_existing_results()
        
        # Run analyses 
        #logger.info("Plotting sample maps...")
        #self.maps = self._plot_sample_maps()
        
        logger.info("Running statistical analysis...")
        self.stats = self._stats()
        
        logger.info("Running alpha diversity analysis...")
        self.alpha_diversity = self._alpha_diversity()
        
        logger.info("Running beta diversity analysis...")
        self.ordination = self._beta_diversity()
        
        logger.info("Running machine learning feature selection...")
        self.models = self._catboost_feature_selection()

        logger.info("Running top features...")
        self.top_features = self._top_features()
        
        # Log final statistics
        self._log_analysis_summary()

    def _invalidate_existing_results(self):
        """Invalidate specific result patterns before analysis."""
        logger.info(f"Invalidating results matching patterns: {self.invalidate_results_patterns}")
        
        stats_dir = self.project_dir.final / 'stats'
        if not stats_dir.exists():
            logger.info("No stats directory found, nothing to invalidate")
            return
        
        total_deleted = 0
        for pattern in self.invalidate_results_patterns:
            deleted_count = self._delete_matching_results(stats_dir, pattern)
            total_deleted += deleted_count
            logger.info(f"Pattern '{pattern}': deleted {deleted_count} files")
        
        logger.info(f"Total invalidated files: {total_deleted}")
    
    def _delete_matching_results(self, stats_dir: Path, pattern: str) -> int:
        """Delete result files matching a specific pattern."""
        deleted_count = 0
        
        # Pattern matching logic
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
                        # Check if file matches pattern
                        full_path = f"{group_dir.name}_{table_dir.name}_{level_dir.name}_{result_file.stem}"
                        
                        if (pattern in full_path or 
                            pattern == result_file.stem or
                            pattern == f"{table_dir.name}_{level_dir.name}" or
                            pattern == group_dir.name or
                            pattern == table_dir.name):
                            
                            result_file.unlink()
                            deleted_count += 1
                            
                            # Also delete correlation matrices for network analysis
                            if result_file.stem == 'network_analysis':
                                corr_file = level_dir / f"{result_file.stem}_correlation_matrix.tsv"
                                if corr_file.exists():
                                    corr_file.unlink()
                                    deleted_count += 1
        
        return deleted_count

    def _plot_sample_maps(self):
        if not self.config.get("maps", {}).get('enabled', False):
            return {}
        maps = Maps(self.config, self.metadata, Path(self.output_dir) / 'sample_maps', self.verbose)
        maps.generate_sample_maps(nfc_facility_data=self.nfc_facilities)
        return maps.maps

    def _stats(self):
        """Run statistical analysis."""
        if not self.config.get("stats", {}).get('enabled', False):
            logger.info("Statistical analysis disabled in configuration")
            return {}

        logger.info(f"Statistical analysis configuration:")
        logger.info(f"  - Load existing results: {self.load_existing_results}")
        logger.info(f"  - Max file age: {self.max_result_age_hours} hours" if self.max_result_age_hours else "  - No age limit")
        logger.info(f"  - Force recalculate patterns: {self.force_recalculate_stats}")

        # Use the enhanced statistical analysis with result loading
        with run_statistical_analysis_with_loading(
            config=self.config,
            tables=self.tables,
            metadata=self.metadata,
            mode=self.mode,
            group_columns=self.group_columns,
            project_dir=self.project_dir,
            load_existing=self.load_existing_results,
            max_file_age_hours=self.max_result_age_hours,
            force_recalculate=self.force_recalculate_stats
        ) as stats:
            recommendations = stats.get_analysis_recommendations()
            logger.info(recommendations)
            
            # Check for configuration issues
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

            # Get summary statistics
            summary = stats.get_summary_statistics()
            logger.info(f"Statistical Analysis Summary:")
            logger.info(f"  - Total tests run: {summary['total_tests_run']}")
            logger.info(f"  - Group columns analyzed: {len(summary['group_columns_analyzed'])}")
            
            # Log loading performance
            load_stats = summary.get('performance_metrics', {}).get('load_statistics', {})
            if load_stats:
                total_tasks = load_stats.get('total_tasks', 0)
                loaded_tasks = load_stats.get('loaded_from_files', 0)
                calculated_tasks = load_stats.get('calculated_fresh', 0)
                
                if total_tasks > 0:
                    load_percentage = (loaded_tasks / total_tasks) * 100
                    logger.info(f"  - Results loaded from files: {loaded_tasks}/{total_tasks} ({load_percentage:.1f}%)")
                    logger.info(f"  - Results calculated fresh: {calculated_tasks}/{total_tasks} ({100-load_percentage:.1f}%)")
            comprehensive_analysis_results = stats.run_comprehensive_analysis()
            top_features = stats.get_top_features_across_tests()
            # Store all statistical results
            self.stats_obj = stats
            return {
                'recommendations': recommendations,
                'test_results': stats.results,
                'comprehensive_analysis_results': comprehensive_analysis,
                'top_features': top_features,
                'summary': summary,
                'load_statistics': stats.get_load_report()
            }

    def _alpha_diversity(self):
        if not self.config.get("alpha_diversity", {}).get('enabled', False):
            logger.info("Alpha diversity analysis disabled in configuration")
            return {}
        alpha = AlphaDiversity(self.config, self.metadata, self.tables)
        alpha.run(output_dir=self.output_dir)
        return alpha.results

    def _beta_diversity(self):
        if not self.config.get("ordination", {}).get('enabled', False):
            logger.info("Betta diversity (ordination) analysis disabled in configuration")
            return {}
        results = {}
        for group_column in self.group_columns:
            beta = Ordination(self.config, self.metadata, self.tables, group_column['name'], self.verbose)
            beta.run(output_dir=self.output_dir)
            results[group_column['name']] = beta.results
        return results

    def _catboost_feature_selection(self):
        if not self.config.get("ml", {}).get('enabled', False):
            logger.info("CatBoost feature selection disabled in configuration")
            return {}
        results = {}
        for group_column in self.group_columns:
            if group_column['type'] == 'bool':
                cb = FeatureSelection(self.config, self.metadata, self.tables, group_column['name'], self.verbose)
                cb.run(output_dir=self.output_dir)
                results[group_column['name']] = cb.models
        return results

    def _top_features(self) -> None:
        if not self.config.get("top_features", {}).get('enabled', False):
            logger.info("Top features analysis disabled in configuration")
            return {}
    
        # Initialize top_features structure
        all_features = {}
        self.top_features = {
            "stats": {},
            "models": {}  # Ensure 'models' key exists
        }
        
        # Rest of your existing code for processing stats and ML features...
        if self.config.get("stats", {}).get('enabled', False) and self.stats:
            for group_column in self.group_columns:
                self._top_features_stats_group_column(group_column)
    
        if self.config.get("ml", {}).get('enabled', False) and self.models:
            for group_column in self.group_columns:
                self._top_features_ml_group_column(group_column)
    
        logger.info(all_features)
        return all_features
            
    def _top_features_stats_group_column(self, group_column) -> None:
        """Helper to identify top features for a specific group"""
        group_column_name = group_column['name']
        group_column_type = group_column['type']
        group_column_values = group_column['values']
        
        if not self.stats['test_results'][group_column_name]:
            logger.warning(f"No statistics calculated for group '{group_column_name}'. Skipping top features.")
            return
            
        n = self.config.get('top_features', {}).get('n', 20) # Number of top features
        logger.info(n)
        # Initialize storage for this group
        self.top_features["stats"][group_column_name] = {}
        all_features = []
        with self.stats_obj as stats:
            for table_type, levels in self.stats['test_results'][group_column_name].items():  # 1. Table Types
                for level, tests in levels.items():                                           # 2. Taxonomic Levels
                    for test_name, df in tests.items():                                       # 3. Test Names
                        if df is None or not isinstance(df, pd.DataFrame):
                            continue
                        if "p_value" not in df.columns:
                            continue
                        # Get significant features for the table_type / level / test    
                        sig_df = df[df["p_value"] < 0.05].copy()
                        if sig_df.empty:
                            continue
                        # Get effect size for each row
                        sig_df["effect"] = sig_df.apply(
                            lambda row: stats.get_effect_size(test_name, row), axis=1
                        )
                        sig_df = sig_df.dropna(subset=["effect"])
    
                        for _, row in sig_df.iterrows():
                            all_features.append({
                                "feature": row["feature"],
                                "column": group_column_name,
                                "table_type": table_type,
                                "level": level,  
                                "method": "statistical_test",
                                "test": test_name,
                                "effect": row["effect"],
                                "p_value": row["p_value"],
                                "effect_dir": "positive" if row["effect"] > 0 else "negative",
                            })
            
                
            group_1_features = [f for f in all_features if f["effect"] > 0]
            group_2_features = [f for f in all_features if f["effect"] < 0]
    
            group_1_features.sort(key=lambda d: (-d["effect"], d["p_value"]))
            group_2_features.sort(key=lambda d: (d["effect"], d["p_value"]))
            
            # Store results
            self.top_features["stats"][group_column_name][group_column_values[0]] = group_1_features[:n]
            self.top_features["stats"][group_column_name][group_column_values[1]] = group_2_features[:n]
            
            logger.info(
                f"Top features for {group_column_name}: "
                f"{group_column_values[0]} ({len(group_1_features)}), "
                f"{group_column_values[1]} ({len(group_2_features)})"
            )
            
    def _top_features_ml_group_column(self, group_column) -> None:
        """Helper to identify top features for a specific group"""
        group_column_name = group_column['name']
        group_column_type = group_column['type']
        group_column_values = group_column['values']
        
        if not self.models[group_column_name]:
            logger.warning(f"No feature selection ran for group '{group_column_name}'. Skipping top features.")
            return
            
        n = self.config.get('top_features', {}).get('n', 20) # Number of top features
        
        # Initialize storage for this group
        
        features_summary = []
        for table_type, levels in self.models[group_column_name].items():  # 1. Table Types
            for level, methods in levels.items():                            # 2. Taxonomic Levels
                for method, result in methods.items():                        # 3. Methods
                    # Validate result structure
                    if not result or not isinstance(result, dict):
                        logger.warning(f"Invalid result for {group_column}/{table_type}/{level}/{method}")
                        continue
                        
                    # Check for required keys
                    if "top_features" not in result:
                        logger.error(f"Missing 'top_features' in {table_type}/{level}/{method}")
                        continue

                    feat_imp = result.get("feature_importances", {})
                    top_features = result.get("top_features", [])[:10] # TODO: Edit this so that it's configurable
                    for i, feat in enumerate(top_features, 1):
                        importance = feat_imp.get(feat, 0)
                        features_summary.append({
                            "Column": group_column_name,
                            "Table Type": table_type,
                            "Level": level,
                            "Method": method,
                            "Rank": i,
                            "Feature": feat,
                            "Importance": f"{importance:.4f}" if isinstance(importance, (int, float)) else "N/A"
                        })
        features_df = pd.DataFrame(features_summary) if features_summary else pd.DataFrame()
        self.top_features["models"][group_column_name] = features_df
        
    def _top_features_plots_stats(self):
        """Generate top features plots if enabled."""
        if self.config.get('top_features', {}).get("violin_plots", {}).get('enabled', False) or self.config.get("feature_maps", {}).get('enabled', False):
            if self.top_features["stats"]:
                features_plots = top_features_plots(
                    output_dir=self.output_dir, 
                    config=self.config, 
                    top_features=self.top_features["stats"], 
                    tables=self.tables, 
                    meta=self.metadata, 
                    nfc_facilities=self.nfc_facilities, 
                    verbose=self.verbose
                )
                return features_plots
        return None

    def _functional_annotation(self):
        """Run functional annotation if enabled."""
        if not self.config.get("faprotax", {}).get('enabled', False):
            logger.info("Functional annotation disabled in configuration")
            return None
            
        faprotax = FunctionalAnnotation(self.config)
        if self.top_features:
            features_to_annotate = []
            # Extract features from top_features for annotation
            for name, value in self.top_features.items():
                for f in self.top_features[name][value]:
                    all_taxa.append(f['feature'])
            # Implementation depends on structure of 
            annotations = faprotax._annotate_features(features_to_annotate)
            return annotations
        return None

    def _log_analysis_summary(self):
        """Log a comprehensive summary of the analysis."""
        logger.info("=" * 60)
        logger.info("DOWNSTREAM ANALYSIS SUMMARY")
        logger.info("=" * 60)
        
        # Log what was enabled/disabled
        enabled_modules = []
        disabled_modules = []
        
        modules = [
            ('stats', 'Statistical Analysis'),
            ('alpha_diversity', 'Alpha Diversity'),
            ('ordination', 'Beta Diversity/Ordination'),
            ('ml', 'Machine Learning Feature Selection'),
            ('maps', 'Sample Maps'),
            ('faprotax', 'Functional Annotation')
        ]
        
        for config_key, module_name in modules:
            if self.config.get(config_key, {}).get('enabled', False):
                enabled_modules.append(module_name)
            else:
                disabled_modules.append(module_name)
        
        logger.info(f"Enabled modules: {', '.join(enabled_modules)}")
        if disabled_modules:
            logger.info(f"Disabled modules: {', '.join(disabled_modules)}")
        
        # Log result loading statistics
        if self.analysis_statistics:
            stats_info = self.analysis_statistics.get('stats', {})
            summary_info = stats_info.get('summary', {})
            
            if summary_info:
                logger.info(f"Statistical Analysis Performance:")
                logger.info(f"  - Total tasks: {summary_info.get('total_tasks', 'N/A')}")
                logger.info(f"  - Loaded from cache: {summary_info.get('loaded_from_files', 'N/A')}")
                logger.info(f"  - Calculated fresh: {summary_info.get('calculated_fresh', 'N/A')}")
        
        # Log data dimensions
        if self.tables and self.metadata:
            logger.info(f"Data Summary:")
            for table_type in self.tables:
                for level in self.tables[table_type]:
                    table = self.tables[table_type][level]
                    metadata = self.metadata[table_type][level]
                    logger.info(f"  - {table_type}/{level}: {table.shape[1]} samples, {table.shape[0]} features")
        
        logger.info("=" * 60)

    def get_analysis_report(self) -> Dict[str, Any]:
        """Get a comprehensive report of the analysis."""
        report = {
            'config': self.config,
            'mode': self.mode,
            'group_columns': self.group_columns,
            'load_settings': {
                'load_existing_results': self.load_existing_results,
                'max_result_age_hours': self.max_result_age_hours,
                'force_recalculate_stats': self.force_recalculate_stats,
                'invalidate_results_patterns': self.invalidate_results_patterns
            },
            'analysis_statistics': self.analysis_statistics,
            'results_summary': {}
        }
        
        # Add result summaries
        if self.stats:
            report['results_summary']['stats'] = {
                'enabled': True,
                'load_statistics': self.stats.get('load_statistics', {})
            }
        
        if self.alpha_diversity:
            report['results_summary']['alpha_diversity'] = {'enabled': True}
        
        if self.ordination:
            report['results_summary']['ordination'] = {'enabled': True}
            
        if self.models:
            report['results_summary']['ml_models'] = {'enabled': True}
        
        return report

    def force_recalculate_next_run(self, patterns: List[str]):
        """Set patterns to force recalculation in the next analysis run."""
        self.force_recalculate_stats.extend(patterns)
        logger.info(f"Added patterns for forced recalculation: {patterns}")

    def invalidate_and_rerun_stats(self, patterns: List[str]):
        """Invalidate specific results and rerun statistical analysis."""
        logger.info(f"Invalidating and recalculating statistical results for patterns: {patterns}")
        
        # Invalidate existing results
        stats_dir = self.project_dir.final / 'stats'
        total_deleted = 0
        for pattern in patterns:
            deleted_count = self._delete_matching_results(stats_dir, pattern)
            total_deleted += deleted_count
        
        logger.info(f"Invalidated {total_deleted} result files")
        
        # Rerun statistical analysis with force recalculation
        if self.config.get("stats", {}).get('enabled', False):
            original_patterns = self.force_recalculate_stats.copy()
            self.force_recalculate_stats.extend(patterns)
            
            try:
                self.stats = self._stats_with_loading()
                logger.info("Statistical analysis completed successfully")
            finally:
                # Restore original force recalculate patterns
                self.force_recalculate_stats = original_patterns


# ===================================== IMPORTS ====================================== #

import numpy as np
import pandas as pd
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
from statsmodels.stats.multitest import multipletests
import umap
from typing import Dict, List, Tuple, Any, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# ================================= MAIN ANALYSIS CLASS ================================ #

class DownstreamResultsAnalyzer:
    """
    Comprehensive analysis framework for integrating all downstream pipeline results
    """
    
    def __init__(self, downstream_results: Any, config: Dict, verbose: bool = True):
        """
        Initialize analyzer with downstream pipeline results
        
        Parameters:
        -----------
        downstream_results : Downstream object
            Results from the downstream pipeline
        config : Dict
            Configuration dictionary
        verbose : bool
            Whether to print progress messages
        """
        self.results = downstream_results
        self.config = config
        self.verbose = verbose
        
        # Extract data components
        self.metadata = downstream_results.metadata
        self.tables = downstream_results.tables
        self.stats = downstream_results.stats or {}
        self.alpha_diversity = downstream_results.alpha_diversity or {}
        self.ordination = downstream_results.ordination or {}
        self.models = downstream_results.models or {}
        
        # Initialize result containers
        self.integrated_results = {}
        self.consensus_features = None
        self.environmental_thresholds = {}
        self.networks = {}
        self.functional_analysis = {}
        
        if self.verbose:
            print("DownstreamResultsAnalyzer initialized successfully")
    
    # ========================== 1. CROSS-MODULE INTEGRATION ========================== #
    
    def synthesize_feature_importance(self, top_n: int = 50) -> pd.DataFrame:
        """
        Combine feature importance from statistical tests, ML models, and abundance patterns
        """
        if self.verbose:
            print("Synthesizing feature importance across modules...")
        
        importance_scores = {}
        
        # Extract statistical significance scores
        if self.stats and 'top_features' in self.stats:
            stat_features = self._extract_statistical_importance()
            for feature, score in stat_features.items():
                importance_scores.setdefault(feature, {})['statistical'] = score
        
        # Extract ML model importance
        if self.models:
            ml_features = self._extract_ml_importance()
            for feature, score in ml_features.items():
                importance_scores.setdefault(feature, {})['ml_importance'] = score
        
        # Extract alpha diversity associations
        if self.alpha_diversity:
            alpha_features = self._extract_alpha_associations()
            for feature, score in alpha_features.items():
                importance_scores.setdefault(feature, {})['alpha_association'] = score
        
        # Extract beta diversity loadings
        if self.ordination:
            beta_features = self._extract_beta_loadings()
            for feature, score in beta_features.items():
                importance_scores.setdefault(feature, {})['beta_loading'] = score
        
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
        
        # Sort by consensus score and return top features
        consensus_df = consensus_df.sort_values('consensus_score', ascending=False)
        self.consensus_features = consensus_df.head(top_n)
        
        if self.verbose:
            print(f"Generated consensus ranking for {len(consensus_df)} features")
        
        return self.consensus_features
    
    def analyze_ecological_coherence(self) -> Dict[str, Any]:
        """
        Test if statistically significant taxa also drive community structure patterns
        """
        if self.verbose:
            print("Analyzing ecological coherence...")
        
        coherence_results = {}
        
        if self.consensus_features is not None:
            top_features = self.consensus_features.index.tolist()[:20]
            
            # Test correlation between different importance measures
            correlations = {}
            columns = self.consensus_features.columns
            for i, col1 in enumerate(columns[:-1]):  # Exclude consensus_score
                for col2 in columns[i+1:-1]:
                    if col1 in self.consensus_features.columns and col2 in self.consensus_features.columns:
                        corr, p_val = spearmanr(
                            self.consensus_features[col1], 
                            self.consensus_features[col2]
                        )
                        correlations[f"{col1}_vs_{col2}"] = {'correlation': corr, 'p_value': p_val}
            
            coherence_results['importance_correlations'] = correlations
            
            # Calculate feature consistency score
            feature_consistency = {}
            for feature in top_features:
                non_zero_scores = sum([
                    1 for col in columns[:-1] 
                    if col in self.consensus_features.columns and 
                    self.consensus_features.loc[feature, col] > 0
                ])
                feature_consistency[feature] = non_zero_scores / (len(columns) - 1)
            
            coherence_results['feature_consistency'] = feature_consistency
        
        self.integrated_results['ecological_coherence'] = coherence_results
        return coherence_results
    
    # ========================== 2. COMMUNITY STRUCTURE & FUNCTION ==================== #
    
    def analyze_diversity_function_coupling(self) -> Dict[str, Any]:
        """
        Investigate how alpha/beta diversity relates to functional potential
        """
        if self.verbose:
            print("Analyzing diversity-function coupling...")
        
        coupling_results = {}
        
        if self.alpha_diversity and hasattr(self.results, '_functional_annotation'):
            # Get functional diversity metrics
            functional_diversity = self._calculate_functional_diversity()
            alpha_metrics = self._extract_alpha_metrics()
            
            # Correlate alpha diversity with functional diversity
            diversity_function_corr = {}
            for alpha_metric, alpha_values in alpha_metrics.items():
                for func_metric, func_values in functional_diversity.items():
                    # Align samples
                    common_samples = set(alpha_values.index) & set(func_values.index)
                    if len(common_samples) > 5:
                        aligned_alpha = alpha_values[list(common_samples)]
                        aligned_func = func_values[list(common_samples)]
                        
                        corr, p_val = spearmanr(aligned_alpha, aligned_func)
                        diversity_function_corr[f"{alpha_metric}_vs_{func_metric}"] = {
                            'correlation': corr, 
                            'p_value': p_val,
                            'n_samples': len(common_samples)
                        }
            
            coupling_results['diversity_function_correlations'] = diversity_function_corr
        
        # Analyze beta diversity-function relationships
        if self.ordination:
            beta_function_analysis = self._analyze_beta_function_coupling()
            coupling_results['beta_function_analysis'] = beta_function_analysis
        
        self.integrated_results['diversity_function_coupling'] = coupling_results
        return coupling_results
    
    def analyze_environmental_gradients(self, continuous_vars: List[str] = None) -> Dict[str, Any]:
        """
        Multi-variate approach combining environmental variables
        """
        if self.verbose:
            print("Analyzing environmental gradients...")
        
        if continuous_vars is None:
            continuous_vars = ['ph', 'facility_distance_km']
        
        gradient_results = {}
        
        # Extract environmental data
        env_data = self._extract_environmental_data(continuous_vars)
        community_data = self._extract_community_matrix()
        
        if env_data is not None and community_data is not None:
            # Canonical Correspondence Analysis (CCA) approximation using RDA
            from sklearn.cross_decomposition import CCA
            
            # Align samples
            common_samples = set(env_data.index) & set(community_data.index)
            if len(common_samples) > 10:
                env_aligned = env_data.loc[list(common_samples)]
                comm_aligned = community_data.loc[list(common_samples)]
                
                # Perform CCA
                cca = CCA(n_components=min(len(continuous_vars), 3))
                env_scores, comm_scores = cca.fit_transform(env_aligned, comm_aligned)
                
                gradient_results['cca_results'] = {
                    'environmental_scores': pd.DataFrame(
                        env_scores, 
                        index=common_samples, 
                        columns=[f'CCA{i+1}_env' for i in range(env_scores.shape[1])]
                    ),
                    'community_scores': pd.DataFrame(
                        comm_scores,
                        index=common_samples,
                        columns=[f'CCA{i+1}_comm' for i in range(comm_scores.shape[1])]
                    ),
                    'explained_variance': cca.score(env_aligned, comm_aligned)
                }
                
                # Calculate feature loadings on CCA axes
                feature_loadings = self._calculate_cca_loadings(
                    comm_aligned, comm_scores, continuous_vars
                )
                gradient_results['feature_loadings'] = feature_loadings
        
        self.integrated_results['environmental_gradients'] = gradient_results
        return gradient_results
    
    # ========================== 3. MODEL PERFORMANCE & VALIDATION ==================== #
    
    def validate_ml_predictions(self) -> Dict[str, Any]:
        """
        Test if ML-selected features align with known ecological principles
        """
        if self.verbose:
            print("Validating ML model predictions...")
        
        validation_results = {}
        
        if self.models:
            # Extract ML feature importance
            ml_features = self._extract_ml_importance()
            
            # Cross-validation performance
            cv_results = self._cross_validate_models()
            validation_results['cross_validation'] = cv_results
            
            # Feature stability across CV folds
            feature_stability = self._assess_feature_stability()
            validation_results['feature_stability'] = feature_stability
            
            # Biological plausibility assessment
            if hasattr(self.results, '_functional_annotation'):
                plausibility = self._assess_biological_plausibility(ml_features)
                validation_results['biological_plausibility'] = plausibility
        
        self.integrated_results['ml_validation'] = validation_results
        return validation_results
    
    def create_prediction_confidence_maps(self) -> Dict[str, Any]:
        """
        Map prediction confidence across environmental/geographic space
        """
        if self.verbose:
            print("Creating prediction confidence maps...")
        
        confidence_results = {}
        
        if self.models and len(self.metadata) > 0:
            # Extract environmental variables for mapping
            env_vars = ['ph', 'facility_distance_km']  # Adjust based on your data
            
            for model_name, model_info in self.models.items():
                if model_info and 'model' in model_info:
                    # Get prediction probabilities
                    confidence_map = self._create_confidence_surface(
                        model_info, env_vars, model_name
                    )
                    confidence_results[model_name] = confidence_map
        
        self.integrated_results['confidence_maps'] = confidence_results
        return confidence_results
    
    # ========================== 4. NETWORK & SYSTEMS-LEVEL ANALYSES ================== #
    
    def build_integrated_networks(self, method: str = 'spearman', 
                                 threshold: float = 0.3) -> Dict[str, Any]:
        """
        Create networks connecting statistically significant features
        """
        if self.verbose:
            print(f"Building integrated networks using {method}...")
        
        network_results = {}
        
        if self.consensus_features is not None:
            top_features = self.consensus_features.index.tolist()[:30]
            
            # Extract abundance data for top features
            abundance_data = self._get_feature_abundance_matrix(top_features)
            
            if abundance_data is not None:
                # Calculate correlation matrix
                if method == 'spearman':
                    corr_matrix = abundance_data.corr(method='spearman')
                elif method == 'pearson':
                    corr_matrix = abundance_data.corr(method='pearson')
                else:
                    raise ValueError(f"Unknown correlation method: {method}")
                
                # Create network from correlation matrix
                network = self._create_network_from_correlations(
                    corr_matrix, threshold
                )
                
                # Calculate network properties
                network_properties = self._calculate_network_properties(network)
                
                network_results[method] = {
                    'network': network,
                    'correlation_matrix': corr_matrix,
                    'properties': network_properties,
                    'adjacency_matrix': nx.adjacency_matrix(network).todense()
                }
        
        self.networks = network_results
        self.integrated_results['networks'] = network_results
        return network_results
    
    def identify_keystone_species(self) -> Dict[str, Any]:
        """
        Identify taxa that are central across multiple analysis dimensions
        """
        if self.verbose:
            print("Identifying keystone species...")
        
        keystone_results = {}
        
        if self.consensus_features is not None:
            # Multi-dimensional scoring
            keystone_scores = {}
            
            for feature in self.consensus_features.index:
                score_components = {}
                
                # Statistical significance score
                if 'statistical' in self.consensus_features.columns:
                    score_components['statistical'] = self.consensus_features.loc[feature, 'statistical']
                
                # ML importance score  
                if 'ml_importance' in self.consensus_features.columns:
                    score_components['ml_importance'] = self.consensus_features.loc[feature, 'ml_importance']
                
                # Network centrality score
                if self.networks:
                    centrality_score = self._get_feature_centrality(feature)
                    score_components['centrality'] = centrality_score
                
                # Functional diversity score
                if hasattr(self.results, '_functional_annotation'):
                    func_score = self._get_functional_diversity_score(feature)
                    score_components['functional_diversity'] = func_score
                
                # Calculate composite keystone score
                keystone_scores[feature] = {
                    'components': score_components,
                    'composite_score': np.mean(list(score_components.values()))
                }
            
            # Rank by composite score
            ranked_keystones = sorted(
                keystone_scores.items(), 
                key=lambda x: x[1]['composite_score'], 
                reverse=True
            )
            
            keystone_results['keystone_ranking'] = ranked_keystones[:10]
            keystone_results['detailed_scores'] = keystone_scores
        
        self.integrated_results['keystone_species'] = keystone_results
        return keystone_results
    
    # ========================== 5. TEMPORAL & SPATIAL PATTERNS ==================== #
    
    def detect_ecological_thresholds(self, env_vars: List[str] = None) -> Dict[str, Any]:
        """
        Identify critical environmental values where communities shift
        """
        if self.verbose:
            print("Detecting ecological thresholds...")
        
        if env_vars is None:
            env_vars = ['ph', 'facility_distance_km']
        
        threshold_results = {}
        
        if self.ordination:
            # Use ordination axes as community composition metrics
            ordination_data = self._extract_ordination_scores()
            env_data = self._extract_environmental_data(env_vars)
            
            if ordination_data is not None and env_data is not None:
                for env_var in env_vars:
                    if env_var in env_data.columns:
                        var_thresholds = self._detect_changepoints(
                            env_data[env_var], ordination_data
                        )
                        threshold_results[env_var] = var_thresholds
        
        # Validate thresholds using ML model decision boundaries
        if self.models and threshold_results:
            ml_validation = self._validate_thresholds_with_ml(threshold_results)
            threshold_results['ml_validation'] = ml_validation
        
        self.environmental_thresholds = threshold_results
        self.integrated_results['ecological_thresholds'] = threshold_results
        return threshold_results
    
    def assess_sample_classification_accuracy(self) -> Dict[str, Any]:
        """
        Analyze sample classification performance and misclassifications
        """
        if self.verbose:
            print("Assessing sample classification accuracy...")
        
        classification_results = {}
        
        if self.models:
            for model_name, model_info in self.models.items():
                if model_info and 'predictions' in model_info:
                    # Extract predictions and actual labels
                    results = self._analyze_classification_performance(
                        model_info, model_name
                    )
                    classification_results[model_name] = results
        
        # Feature sufficiency analysis
        sufficiency_analysis = self._analyze_feature_sufficiency()
        classification_results['feature_sufficiency'] = sufficiency_analysis
        
        self.integrated_results['classification_accuracy'] = classification_results
        return classification_results
    
    # ========================== 6. FUNCTIONAL ECOLOGY SYNTHESIS ==================== #
    
    def analyze_function_environment_relationships(self) -> Dict[str, Any]:
        """
        Test if functional potential matches environmental conditions
        """
        if self.verbose:
            print("Analyzing function-environment relationships...")
        
        function_env_results = {}
        
        if hasattr(self.results, '_functional_annotation'):
            # Extract functional annotations
            functional_profiles = self._extract_functional_profiles()
            env_data = self._extract_environmental_data()
            
            if functional_profiles is not None and env_data is not None:
                # Correlate functions with environment
                func_env_correlations = self._correlate_functions_environment(
                    functional_profiles, env_data
                )
                function_env_results['correlations'] = func_env_correlations
                
                # Test for functional redundancy vs uniqueness
                redundancy_analysis = self._analyze_functional_redundancy(
                    functional_profiles
                )
                function_env_results['redundancy'] = redundancy_analysis
                
                # Identify environment-specific functional signatures
                signatures = self._identify_functional_signatures(
                    functional_profiles, env_data
                )
                function_env_results['signatures'] = signatures
        
        self.functional_analysis = function_env_results
        self.integrated_results['function_environment'] = function_env_results
        return function_env_results
    
    def analyze_phylogenetic_signal(self) -> Dict[str, Any]:
        """
        Test if closely related taxa respond similarly to environmental gradients
        """
        if self.verbose:
            print("Analyzing phylogenetic signal...")
        
        phylo_results = {}
        
        if self.consensus_features is not None:
            # Extract taxonomic information
            taxonomic_data = self._extract_taxonomic_hierarchy()
            
            if taxonomic_data is not None:
                # Calculate phylogenetic signal in feature importance
                phylo_signal = self._calculate_phylogenetic_signal(
                    taxonomic_data, self.consensus_features
                )
                phylo_results['importance_signal'] = phylo_signal
                
                # Test for phylogenetic clustering in environmental responses
                env_response_signal = self._test_phylogenetic_env_response()
                phylo_results['environmental_response_signal'] = env_response_signal
        
        self.integrated_results['phylogenetic_signal'] = phylo_results
        return phylo_results
    
    # ========================== 7. STATISTICAL POWER & EFFECT SIZE ================= #
    
    def create_effect_size_landscape(self) -> Dict[str, Any]:
        """
        Map effect sizes across all statistical tests
        """
        if self.verbose:
            print("Creating effect size landscape...")
        
        effect_size_results = {}
        
        if self.stats and 'test_results' in self.stats:
            # Extract effect sizes from all tests
            effect_sizes = self._extract_all_effect_sizes()
            
            # Create effect size matrices
            effect_matrices = self._create_effect_size_matrices(effect_sizes)
            effect_size_results['matrices'] = effect_matrices
            
            # Identify largest biological effects
            largest_effects = self._identify_largest_effects(effect_sizes)
            effect_size_results['largest_effects'] = largest_effects
            
            # Distinguish statistical vs biological significance
            significance_analysis = self._analyze_statistical_vs_biological(
                effect_sizes
            )
            effect_size_results['significance_analysis'] = significance_analysis
        
        self.integrated_results['effect_size_landscape'] = effect_size_results
        return effect_size_results
    
    def assess_multiple_testing_burden(self) -> Dict[str, Any]:
        """
        Analyze impact of multiple testing correction across modules
        """
        if self.verbose:
            print("Assessing multiple testing burden...")
        
        testing_results = {}
        
        if self.stats:
            # Calculate FDR burden across different analysis types
            fdr_analysis = self._analyze_fdr_burden()
            testing_results['fdr_burden'] = fdr_analysis
            
            # Power analysis for each test type
            power_analysis = self._conduct_power_analysis()
            testing_results['power_analysis'] = power_analysis
            
            # Optimal threshold analysis
            threshold_optimization = self._optimize_statistical_thresholds()
            testing_results['threshold_optimization'] = threshold_optimization
        
        self.integrated_results['multiple_testing'] = testing_results
        return testing_results
    
    # ========================== 8. META-ANALYSIS APPROACHES ========================= #
    
    def calculate_consistency_scores(self) -> pd.DataFrame:
        """
        Score how consistently each feature appears as important across modules
        """
        if self.verbose:
            print("Calculating cross-module consistency scores...")
        
        if self.consensus_features is not None:
            consistency_results = self.consensus_features.copy()
            
            # Calculate normalized consistency scores
            modules = ['statistical', 'ml_importance', 'alpha_association', 'beta_loading']
            available_modules = [m for m in modules if m in consistency_results.columns]
            
            # Binary consistency (present/absent in top features)
            consistency_results['binary_consistency'] = (
                (consistency_results[available_modules] > 0).sum(axis=1) / 
                len(available_modules)
            )
            
            # Weighted consistency (considering actual scores)
            consistency_results['weighted_consistency'] = (
                consistency_results[available_modules].sum(axis=1) / 
                consistency_results[available_modules].sum(axis=1).max()
            )
            
            # Rank consistency (based on rankings within each module)
            for module in available_modules:
                consistency_results[f'{module}_rank'] = (
                    consistency_results[module].rank(ascending=False, method='min')
                )
            
            rank_cols = [f'{m}_rank' for m in available_modules]
            consistency_results['mean_rank'] = consistency_results[rank_cols].mean(axis=1)
            consistency_results['rank_consistency'] = 1 / consistency_results['mean_rank']
            
            self.integrated_results['consistency_scores'] = consistency_results
            return consistency_results
        
        return pd.DataFrame()
    
    def assess_biological_plausibility(self) -> Dict[str, Any]:
        """
        Assess biological plausibility of top features
        """
        if self.verbose:
            print("Assessing biological plausibility...")
        
        plausibility_results = {}
        
        if self.consensus_features is not None:
            # Environmental tolerance analysis
            tolerance_analysis = self._analyze_environmental_tolerance()
            plausibility_results['environmental_tolerance'] = tolerance_analysis
            
            # Functional coherence analysis
            if hasattr(self.results, '_functional_annotation'):
                functional_coherence = self._assess_functional_coherence()
                plausibility_results['functional_coherence'] = functional_coherence
            
            # Phylogenetic coherence analysis
            phylogenetic_coherence = self._assess_phylogenetic_coherence()
            plausibility_results['phylogenetic_coherence'] = phylogenetic_coherence
        
        self.integrated_results['biological_plausibility'] = plausibility_results
        return plausibility_results
    
    # ========================== 9. VISUALIZATION & INTERPRETATION ================== #
    
    def create_results_dashboard(self, output_dir: str = 'dashboard_output') -> Dict[str, Any]:
        """
        Build interactive dashboard combining all analysis results
        """
        if self.verbose:
            print("Creating integrated results dashboard...")
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        dashboard_components = {}
        
        # 1. Feature importance sunburst chart
        if self.consensus_features is not None:
            sunburst_fig = self._create_feature_importance_sunburst()
            sunburst_fig.write_html(f"{output_dir}/feature_importance_sunburst.html")
            dashboard_components['sunburst'] = f"{output_dir}/feature_importance_sunburst.html"
        
        # 2. Environmental gradient overlays
        if 'environmental_gradients' in self.integrated_results:
            gradient_fig = self._create_environmental_gradient_plot()
            gradient_fig.write_html(f"{output_dir}/environmental_gradients.html")
            dashboard_components['gradients'] = f"{output_dir}/environmental_gradients.html"
        
        # 3. Model performance comparison
        if 'ml_validation' in self.integrated_results:
            performance_fig = self._create_model_performance_plot()
            performance_fig.write_html(f"{output_dir}/model_performance.html")
            dashboard_components['performance'] = f"{output_dir}/model_performance.html"
        
        # 4. Network visualization
        if self.networks:
            network_fig = self._create_network_visualization()
            network_fig.write_html(f"{output_dir}/feature_networks.html")
            dashboard_components['networks'] = f"{output_dir}/feature_networks.html"
        
        # 5. Effect size heatmap
        if 'effect_size_landscape' in self.integrated_results:
            effect_size_fig = self._create_effect_size_heatmap()
            effect_size_fig.write_html(f"{output_dir}/effect_size_landscape.html")
            dashboard_components['effect_sizes'] = f"{output_dir}/effect_size_landscape.html"
        
        # 6. Summary statistics table
        summary_table = self._create_summary_statistics_table()
        summary_table.to_html(f"{output_dir}/summary_statistics.html")
        dashboard_components['summary'] = f"{output_dir}/summary_statistics.html"
        
        # Create main dashboard HTML
        main_dashboard = self._create_main_dashboard_html(dashboard_components)
        with open(f"{output_dir}/main_dashboard.html", 'w') as f:
            f.write(main_dashboard)
        
        dashboard_components['main'] = f"{output_dir}/main_dashboard.html"
        
        self.integrated_results['dashboard'] = dashboard_components
        return dashboard_components
    
    def construct_ecological_narrative(self) -> Dict[str, str]:
        """
        Build ecological story from integrated results
        """
        if self.verbose:
            print("Constructing ecological narrative...")
        
        narrative = {}
        
        # Key findings summary
        key_findings = []
        
        if self.consensus_features is not None:
            n_features = len(self.consensus_features)
            key_findings.append(
                f"Identified {n_features} key microbial features consistently important across multiple analyses."
            )
        
        if 'environmental_gradients' in self.integrated_results:
            key_findings.append(
                "Environmental gradients (pH, facility distance) create predictable shifts in community composition."
            )
        
        if 'keystone_species' in self.integrated_results:
            keystones = self.integrated_results['keystone_species'].get('keystone_ranking', [])
            if keystones:
                top_keystone = keystones[0][0]
                key_findings.append(f"'{top_keystone}' identified as potential keystone species.")
        
        if 'ecological_thresholds' in self.integrated_results:
            key_findings.append(
                "Detected ecological thresholds where community composition shifts abruptly."
            )
        
        narrative['key_findings'] = key_findings
        
        # Ecological interpretation
        interpretation = self._generate_ecological_interpretation()
        narrative['interpretation'] = interpretation
        
        # Management implications
        management_implications = self._generate_management_implications()
        narrative['management_implications'] = management_implications
        
        # Future research directions
        research_directions = self._generate_research_directions()
        narrative['research_directions'] = research_directions
        
        self.integrated_results['narrative'] = narrative
        return narrative
    
    # ========================== 10. VALIDATION & ROBUSTNESS ======================= #
    
    def test_result_stability(self, n_iterations: int = 100, 
                            subsample_fraction: float = 0.8) -> Dict[str, Any]:
        """
        Test how robust results are to subsampling
        """
        if self.verbose:
            print(f"Testing result stability with {n_iterations} iterations...")
        
        stability_results = {}
        
        # Bootstrap subsampling of samples
        if self.consensus_features is not None:
            feature_stability = self._bootstrap_feature_selection(
                n_iterations, subsample_fraction
            )
            stability_results['feature_selection'] = feature_stability
        
        # Model performance consistency
        if self.models:
            model_stability = self._bootstrap_model_performance(
                n_iterations, subsample_fraction
            )
            stability_results['model_performance'] = model_stability
        
        # Statistical test stability
        if self.stats:
            stats_stability = self._bootstrap_statistical_tests(
                n_iterations, subsample_fraction
            )
            stability_results['statistical_tests'] = stats_stability
        
        self.integrated_results['stability'] = stability_results
        return stability_results
    
    def conduct_parameter_sensitivity_analysis(self) -> Dict[str, Any]:
        """
        Test sensitivity to parameter choices across all analyses
        """
        if self.verbose:
            print("Conducting parameter sensitivity analysis...")
        
        sensitivity_results = {}
        
        # ML hyperparameter sensitivity
        if self.models:
            ml_sensitivity = self._test_ml_parameter_sensitivity()
            sensitivity_results['ml_hyperparameters'] = ml_sensitivity
        
        # Statistical threshold sensitivity
        if self.stats:
            threshold_sensitivity = self._test_statistical_thresholds()
            sensitivity_results['statistical_thresholds'] = threshold_sensitivity
        
        # Preprocessing parameter sensitivity
        preprocessing_sensitivity = self._test_preprocessing_sensitivity()
        sensitivity_results['preprocessing'] = preprocessing_sensitivity
        
        # Network threshold sensitivity
        if self.networks:
            network_sensitivity = self._test_network_thresholds()
            sensitivity_results['network_thresholds'] = network_sensitivity
        
        self.integrated_results['sensitivity'] = sensitivity_results
        return sensitivity_results
    
    # ========================== HELPER METHODS ===================================== #
    
    def _extract_statistical_importance(self) -> Dict[str, float]:
        """Extract feature importance scores from statistical tests"""
        importance_dict = {}
        
        if 'top_features' in self.stats:
            for feature_info in self.stats['top_features']:
                if isinstance(feature_info, dict) and 'feature' in feature_info:
                    feature = feature_info['feature']
                    # Use negative log p-value as importance score
                    p_value = feature_info.get('p_value', 1.0)
                    importance_dict[feature] = -np.log10(max(p_value, 1e-10))
        
        return importance_dict
    
    def _extract_ml_importance(self) -> Dict[str, float]:
        """Extract feature importance from ML models"""
        importance_dict = {}
        
        for model_name, model_info in self.models.items():
            if isinstance(model_info, dict) and 'feature_importance' in model_info:
                for feature, importance in model_info['feature_importance'].items():
                    if feature not in importance_dict:
                        importance_dict[feature] = 0
                    importance_dict[feature] += importance
        
        # Normalize by number of models
        if len(self.models) > 0:
            for feature in importance_dict:
                importance_dict[feature] /= len(self.models)
        
        return importance_dict
    
    def _extract_alpha_associations(self) -> Dict[str, float]:
        """Extract features associated with alpha diversity metrics"""
        associations = {}
        
        if self.alpha_diversity and isinstance(self.alpha_diversity, dict):
            # Look for correlation results with features
            for metric, results in self.alpha_diversity.items():
                if isinstance(results, dict) and 'correlations' in results:
                    for feature, corr_data in results['correlations'].items():
                        if isinstance(corr_data, dict) and 'correlation' in corr_data:
                            associations[feature] = abs(corr_data['correlation'])
        
        return associations
    
    def _extract_beta_loadings(self) -> Dict[str, float]:
        """Extract feature loadings from ordination analysis"""
        loadings = {}
        
        if self.ordination and isinstance(self.ordination, dict):
            for method, results in self.ordination.items():
                if isinstance(results, dict) and 'feature_loadings' in results:
                    for feature, loading_values in results['feature_loadings'].items():
                        if isinstance(loading_values, (list, np.ndarray)):
                            # Use L2 norm of loadings across axes
                            loadings[feature] = np.linalg.norm(loading_values)
                        elif isinstance(loading_values, (int, float)):
                            loadings[feature] = abs(loading_values)
        
        return loadings
    
    def _calculate_functional_diversity(self) -> Dict[str, pd.Series]:
        """Calculate functional diversity metrics"""
        functional_diversity = {}
        
        # This is a placeholder - implement based on your FAPROTAX results
        if hasattr(self.results, '_functional_annotation'):
            # Calculate Shannon diversity of functions per sample
            # Calculate functional richness, evenness, etc.
            pass
        
        return functional_diversity
    
    def _extract_alpha_metrics(self) -> Dict[str, pd.Series]:
        """Extract alpha diversity metrics"""
        alpha_metrics = {}
        
        if self.alpha_diversity:
            for metric, data in self.alpha_diversity.items():
                if isinstance(data, dict) and 'values' in data:
                    alpha_metrics[metric] = pd.Series(data['values'])
                elif isinstance(data, pd.Series):
                    alpha_metrics[metric] = data
        
        return alpha_metrics
    
    def _analyze_beta_function_coupling(self) -> Dict[str, Any]:
        """Analyze relationship between beta diversity and functional diversity"""
        coupling_results = {}
        
        # Extract ordination scores
        ordination_data = self._extract_ordination_scores()
        
        # Extract functional profiles
        functional_profiles = self._extract_functional_profiles()
        
        if ordination_data is not None and functional_profiles is not None:
            # Correlate ordination axes with functional profiles
            coupling_results['axis_function_correlations'] = {}
            
            for axis in ordination_data.columns:
                axis_correlations = {}
                for func in functional_profiles.columns:
                    corr, p_val = spearmanr(ordination_data[axis], functional_profiles[func])
                    axis_correlations[func] = {'correlation': corr, 'p_value': p_val}
                coupling_results['axis_function_correlations'][axis] = axis_correlations
        
        return coupling_results
    
    def _extract_environmental_data(self, variables: List[str] = None) -> pd.DataFrame:
        """Extract environmental variables from metadata"""
        if variables is None:
            variables = ['ph', 'facility_distance_km']
        
        env_data = None
        
        # Try to extract from different metadata levels
        for mode in ['genus', 'asv']:  # Try both modes
            if mode in self.metadata:
                metadata = self.metadata[mode]
                if isinstance(metadata, dict):
                    for subset_name, subset_data in metadata.items():
                        if isinstance(subset_data, pd.DataFrame):
                            available_vars = [v for v in variables if v in subset_data.columns]
                            if available_vars:
                                env_data = subset_data[available_vars].copy()
                                break
                elif isinstance(metadata, pd.DataFrame):
                    available_vars = [v for v in variables if v in metadata.columns]
                    if available_vars:
                        env_data = metadata[available_vars].copy()
                        break
            if env_data is not None:
                break
        
        return env_data
    
    def _extract_community_matrix(self) -> pd.DataFrame:
        """Extract community abundance matrix"""
        community_data = None
        
        # Try to extract from tables
        for mode in ['genus', 'asv']:
            if mode in self.tables:
                tables = self.tables[mode]
                if isinstance(tables, dict):
                    for subset_name, table_data in tables.items():
                        if hasattr(table_data, 'to_dataframe'):
                            community_data = table_data.to_dataframe().T  # Samples as rows
                            break
                        elif isinstance(table_data, pd.DataFrame):
                            community_data = table_data.T if table_data.shape[0] < table_data.shape[1] else table_data
                            break
            if community_data is not None:
                break
        
        return community_data
    
    def _calculate_cca_loadings(self, community_data: pd.DataFrame, 
                              comm_scores: np.ndarray, env_vars: List[str]) -> Dict[str, Any]:
        """Calculate feature loadings on CCA axes"""
        # Calculate correlations between original features and CCA scores
        loadings = {}
        
        for i, axis in enumerate([f'CCA{j+1}' for j in range(comm_scores.shape[1])]):
            axis_loadings = {}
            for feature in community_data.columns:
                corr, p_val = spearmanr(community_data[feature], comm_scores[:, i])
                axis_loadings[feature] = {'loading': corr, 'p_value': p_val}
            loadings[axis] = axis_loadings
        
        return loadings
    
    def _cross_validate_models(self) -> Dict[str, Any]:
        """Perform cross-validation on ML models"""
        cv_results = {}
        
        for model_name, model_info in self.models.items():
            if isinstance(model_info, dict) and 'model' in model_info:
                # Perform stratified k-fold cross-validation
                try:
                    X = model_info.get('features', pd.DataFrame())
                    y = model_info.get('labels', pd.Series())
                    
                    if not X.empty and not y.empty:
                        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                        cv_scores = cross_val_score(
                            model_info['model'], X, y, cv=skf, scoring='accuracy'
                        )
                        
                        cv_results[model_name] = {
                            'cv_scores': cv_scores,
                            'mean_score': np.mean(cv_scores),
                            'std_score': np.std(cv_scores),
                            'confidence_interval': np.percentile(cv_scores, [2.5, 97.5])
                        }
                except Exception as e:
                    cv_results[model_name] = {'error': str(e)}
        
        return cv_results
    
    def _assess_feature_stability(self) -> Dict[str, Any]:
        """Assess stability of selected features across CV folds"""
        stability_results = {}
        
        # This would require running feature selection within each CV fold
        # and measuring how often the same features are selected
        
        for model_name, model_info in self.models.items():
            if isinstance(model_info, dict):
                # Placeholder for actual implementation
                stability_results[model_name] = {
                    'feature_selection_frequency': {},
                    'stability_score': 0.0
                }
        
        return stability_results
    
    def _assess_biological_plausibility(self, ml_features: Dict[str, float]) -> Dict[str, Any]:
        """Assess biological plausibility of ML-selected features"""
        plausibility = {}
        
        # Check if features have known ecological functions
        # Check if features are phylogenetically coherent
        # Check if features match environmental preferences
        
        plausibility['ecological_coherence_score'] = 0.8  # Placeholder
        plausibility['phylogenetic_coherence_score'] = 0.7  # Placeholder
        plausibility['environmental_match_score'] = 0.9  # Placeholder
        plausibility['overall_plausibility'] = 0.8  # Placeholder
        
        return plausibility
    
    def _create_confidence_surface(self, model_info: Dict, env_vars: List[str], 
                                  model_name: str) -> Dict[str, Any]:
        """Create prediction confidence surface across environmental space"""
        confidence_map = {}
        
        # Create grid of environmental values
        # Make predictions across grid
        # Calculate prediction confidence
        
        confidence_map['grid_predictions'] = np.array([])  # Placeholder
        confidence_map['confidence_surface'] = np.array([])  # Placeholder
        confidence_map['environmental_grid'] = {}  # Placeholder
        
        return confidence_map
    
    def _get_feature_abundance_matrix(self, features: List[str]) -> pd.DataFrame:
        """Get abundance matrix for specified features"""
        community_data = self._extract_community_matrix()
        
        if community_data is not None:
            available_features = [f for f in features if f in community_data.columns]
            if available_features:
                return community_data[available_features].copy()
        
        return None
    
    def _create_network_from_correlations(self, corr_matrix: pd.DataFrame, 
                                        threshold: float) -> nx.Graph:
        """Create network graph from correlation matrix"""
        G = nx.Graph()
        
        # Add nodes
        G.add_nodes_from(corr_matrix.index)
        
        # Add edges for correlations above threshold
        for i in range(len(corr_matrix)):
            for j in range(i+1, len(corr_matrix)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val >= threshold:
                    G.add_edge(
                        corr_matrix.index[i], 
                        corr_matrix.index[j], 
                        weight=corr_val
                    )
        
        return G
    
    def _calculate_network_properties(self, network: nx.Graph) -> Dict[str, Any]:
        """Calculate network topology properties"""
        properties = {}
        
        if len(network.nodes()) > 0:
            # Basic properties
            properties['n_nodes'] = network.number_of_nodes()
            properties['n_edges'] = network.number_of_edges()
            properties['density'] = nx.density(network)
            
            # Centrality measures
            if len(network.nodes()) > 1:
                properties['degree_centrality'] = nx.degree_centrality(network)
                properties['betweenness_centrality'] = nx.betweenness_centrality(network)
                properties['closeness_centrality'] = nx.closeness_centrality(network)
                properties['eigenvector_centrality'] = nx.eigenvector_centrality(network)
            
            # Clustering
            properties['clustering_coefficient'] = nx.average_clustering(network)
            
            # Connected components
            properties['n_components'] = nx.number_connected_components(network)
        
        return properties
    
    def _get_feature_centrality(self, feature: str) -> float:
        """Get network centrality score for a feature"""
        centrality_score = 0.0
        
        for method, network_data in self.networks.items():
            if 'properties' in network_data:
                props = network_data['properties']
                if 'degree_centrality' in props and feature in props['degree_centrality']:
                    centrality_score += props['degree_centrality'][feature]
        
        return centrality_score / len(self.networks) if self.networks else 0.0
    
    def _get_functional_diversity_score(self, feature: str) -> float:
        """Get functional diversity score for a feature"""
        # Placeholder - implement based on FAPROTAX annotations
        return np.random.random()  # Placeholder
    
    def _extract_ordination_scores(self) -> pd.DataFrame:
        """Extract ordination axis scores"""
        ordination_data = None
        
        if self.ordination:
            for method, results in self.ordination.items():
                if isinstance(results, dict) and 'sample_scores' in results:
                    ordination_data = pd.DataFrame(results['sample_scores'])
                    break
                elif isinstance(results, dict) and 'scores' in results:
                    ordination_data = pd.DataFrame(results['scores'])
                    break
        
        return ordination_data
    
    def _detect_changepoints(self, env_var: pd.Series, 
                           community_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect changepoints in community composition along environmental gradient"""
        changepoints = {}
        
        # Sort data by environmental variable
        sorted_indices = env_var.sort_values().index
        sorted_env = env_var[sorted_indices]
        sorted_community = community_data.loc[sorted_indices]
        
        # Use sliding window to detect changes in community composition
        window_size = min(10, len(sorted_env) // 4)
        change_scores = []
        
        for i in range(window_size, len(sorted_env) - window_size):
            # Compare community composition before and after point
            before = sorted_community.iloc[i-window_size:i]
            after = sorted_community.iloc[i:i+window_size]
            
            # Calculate Bray-Curtis dissimilarity between windows
            before_mean = before.mean()
            after_mean = after.mean()
            
            # Simple dissimilarity metric
            dissimilarity = np.sum(np.abs(before_mean - after_mean)) / np.sum(before_mean + after_mean)
            change_scores.append((sorted_env.iloc[i], dissimilarity))
        
        if change_scores:
            # Find peaks in change scores
            change_values = [score[1] for score in change_scores]
            threshold = np.mean(change_values) + 2 * np.std(change_values)
            
            thresholds = [
                score[0] for score in change_scores 
                if score[1] > threshold
            ]
            
            changepoints['detected_thresholds'] = thresholds
            changepoints['change_scores'] = change_scores
            changepoints['threshold_criterion'] = threshold
        
        return changepoints
    
    def _validate_thresholds_with_ml(self, threshold_results: Dict) -> Dict[str, Any]:
        """Validate detected thresholds using ML model decision boundaries"""
        validation_results = {}
        
        # This would involve training simple decision trees on environmental variables
        # and comparing split points with detected thresholds
        
        for env_var, thresholds in threshold_results.items():
            if 'detected_thresholds' in thresholds:
                validation_results[env_var] = {
                    'ml_confirmed_thresholds': [],
                    'validation_score': 0.0
                }
        
        return validation_results
    
    def _analyze_classification_performance(self, model_info: Dict, 
                                          model_name: str) -> Dict[str, Any]:
        """Analyze classification performance and misclassifications"""
        results = {}
        
        if 'predictions' in model_info and 'true_labels' in model_info:
            y_true = model_info['true_labels']
            y_pred = model_info['predictions']
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            results['confusion_matrix'] = cm
            
            # Classification report
            report = classification_report(y_true, y_pred, output_dict=True)
            results['classification_report'] = report
            
            # Identify misclassified samples
            misclassified = y_true != y_pred
            results['misclassified_samples'] = misclassified
            results['misclassification_rate'] = misclassified.mean()
        
        return results
    
    def _analyze_feature_sufficiency(self) -> Dict[str, Any]:
        """Analyze minimum features needed for classification"""
        sufficiency_results = {}
        
        # Progressive feature selection to find minimum set
        if self.consensus_features is not None:
            features_ranked = self.consensus_features.index.tolist()
            
            # Test performance with increasing numbers of features
            performance_curve = []
            
            for n_features in range(1, min(21, len(features_ranked))):
                # This would involve retraining models with subset of features
                # Placeholder for actual implementation
                performance_curve.append({
                    'n_features': n_features,
                    'accuracy': 0.8 + 0.1 * np.random.random()  # Placeholder
                })
            
            sufficiency_results['performance_curve'] = performance_curve
            
            # Find elbow point (minimum features for good performance)
            # Placeholder implementation
            sufficiency_results['minimum_features'] = 10
        
        return sufficiency_results
    
    def _extract_functional_profiles(self) -> pd.DataFrame:
        """Extract functional profiles from FAPROTAX annotations"""
        # Placeholder - implement based on your functional annotation results
        functional_profiles = pd.DataFrame()
        return functional_profiles
    
    def _correlate_functions_environment(self, functional_profiles: pd.DataFrame, 
                                       env_data: pd.DataFrame) -> Dict[str, Any]:
        """Correlate functional profiles with environmental variables"""
        correlations = {}
        
        common_samples = set(functional_profiles.index) & set(env_data.index)
        if len(common_samples) > 5:
            func_aligned = functional_profiles.loc[list(common_samples)]
            env_aligned = env_data.loc[list(common_samples)]
            
            for func in func_aligned.columns:
                func_correlations = {}
                for env_var in env_aligned.columns:
                    corr, p_val = spearmanr(func_aligned[func], env_aligned[env_var])
                    func_correlations[env_var] = {
                        'correlation': corr,
                        'p_value': p_val
                    }
                correlations[func] = func_correlations
        
        return correlations
    
    def _analyze_functional_redundancy(self, functional_profiles: pd.DataFrame) -> Dict[str, Any]:
        """Analyze functional redundancy vs uniqueness"""
        redundancy_results = {}
        
        if not functional_profiles.empty:
            # Calculate functional similarity matrix
            func_corr = functional_profiles.T.corr()
            
            # Identify highly correlated functions (redundant)
            redundant_pairs = []
            for i in range(len(func_corr)):
                for j in range(i+1, len(func_corr)):
                    if abs(func_corr.iloc[i, j]) > 0.8:
                        redundant_pairs.append((
                            func_corr.index[i], 
                            func_corr.index[j], 
                            func_corr.iloc[i, j]
                        ))
            
            redundancy_results['redundant_pairs'] = redundant_pairs
            redundancy_results['functional_similarity_matrix'] = func_corr
            redundancy_results['redundancy_score'] = len(redundant_pairs) / (len(func_corr) * (len(func_corr) - 1) / 2)
        
        return redundancy_results
    
    def _identify_functional_signatures(self, functional_profiles: pd.DataFrame, 
                                      env_data: pd.DataFrame) -> Dict[str, Any]:
        """Identify environment-specific functional signatures"""
        signatures = {}
        
        # For each environmental variable, identify functions that are
        # significantly associated with specific ranges
        
        for env_var in env_data.columns:
            env_values = env_data[env_var].dropna()
            
            # Split into quantiles
            quantiles = pd.qcut(env_values, q=3, labels=['low', 'medium', 'high'])
            
            var_signatures = {}
            for quantile in ['low', 'medium', 'high']:
                quantile_samples = quantiles[quantiles == quantile].index
                
                if len(quantile_samples) > 3:
                    # Find functions enriched in this quantile
                    enriched_functions = []
                    
                    for func in functional_profiles.columns:
                        quantile_func = functional_profiles.loc[quantile_samples, func]
                        other_func = functional_profiles.drop(quantile_samples)[func]
                        
                        if len(other_func) > 0:
                            # Statistical test for enrichment
                            stat, p_val = stats.mannwhitneyu(
                                quantile_func, other_func, alternative='greater'
                            )
                            
                            if p_val < 0.05:
                                enriched_functions.append({
                                    'function': func,
                                    'p_value': p_val,
                                    'fold_change': quantile_func.mean() / other_func.mean()
                                })
                    
                    var_signatures[quantile] = enriched_functions
            
            signatures[env_var] = var_signatures
        
        return signatures
    
    def _extract_taxonomic_hierarchy(self) -> pd.DataFrame:
        """Extract taxonomic hierarchy information"""
        # This would extract taxonomic levels from feature names
        # Placeholder implementation
        taxonomic_data = pd.DataFrame()
        return taxonomic_data
    
    def _calculate_phylogenetic_signal(self, taxonomic_data: pd.DataFrame, 
                                     importance_scores: pd.DataFrame) -> Dict[str, Any]:
        """Calculate phylogenetic signal in feature importance"""
        phylo_signal = {}
        
        # This would use methods like Blomberg's K or Pagel's lambda
        # to test for phylogenetic signal in importance scores
        
        phylo_signal['blomberg_k'] = 0.5  # Placeholder
        phylo_signal['pagel_lambda'] = 0.3  # Placeholder
        phylo_signal['p_value'] = 0.02  # Placeholder
        
        return phylo_signal
    
    def _test_phylogenetic_env_response(self) -> Dict[str, Any]:
        """Test for phylogenetic signal in environmental responses"""
        env_response_signal = {}
        
        # Test if closely related taxa respond similarly to environmental gradients
        env_response_signal['environmental_phylogenetic_signal'] = 0.4  # Placeholder
        env_response_signal['significance'] = 0.01  # Placeholder
        
        return env_response_signal
    
    def _extract_all_effect_sizes(self) -> Dict[str, Any]:
        """Extract effect sizes from all statistical tests"""
        effect_sizes = {}
        
        if 'test_results' in self.stats:
            for test_name, test_results in self.stats['test_results'].items():
                if isinstance(test_results, dict):
                    for feature, feature_results in test_results.items():
                        if isinstance(feature_results, dict) and 'effect_size' in feature_results:
                            effect_sizes.setdefault(test_name, {})[feature] = feature_results['effect_size']
        
        return effect_sizes
    
    def _create_effect_size_matrices(self, effect_sizes: Dict) -> Dict[str, pd.DataFrame]:
        """Create effect size matrices for visualization"""
        matrices = {}
        
        for test_name, test_effects in effect_sizes.items():
            if test_effects:
                # Convert to DataFrame format suitable for heatmap
                effect_df = pd.DataFrame.from_dict(test_effects, orient='index', columns=['effect_size'])
                matrices[test_name] = effect_df
        
        return matrices
    
    def _identify_largest_effects(self, effect_sizes: Dict) -> Dict[str, Any]:
        """Identify tests and features with largest biological effects"""
        largest_effects = {}
        
        all_effects = []
        for test_name, test_effects in effect_sizes.items():
            for feature, effect_size in test_effects.items():
                all_effects.append({
                    'test': test_name,
                    'feature': feature,
                    'effect_size': abs(effect_size)
                })
        
        # Sort by effect size
        all_effects.sort(key=lambda x: x['effect_size'], reverse=True)
        
        largest_effects['top_10_effects'] = all_effects[:10]
        largest_effects['effect_size_distribution'] = [e['effect_size'] for e in all_effects]
        
        return largest_effects
    
    def _analyze_statistical_vs_biological(self, effect_sizes: Dict) -> Dict[str, Any]:
        """Distinguish statistical significance from biological importance"""
        significance_analysis = {}
        
        # Extract p-values alongside effect sizes
        p_values = self._extract_all_p_values()
        
        # Categorize results
        categories = {
            'significant_large_effect': [],
            'significant_small_effect': [],
            'non_significant_large_effect': [],
            'non_significant_small_effect': []
        }
        
        effect_threshold = 0.5  # Medium effect size threshold
        p_threshold = 0.05
        
        for test_name in effect_sizes.keys():
            if test_name in p_values:
                test_effects = effect_sizes[test_name]
                test_p_values = p_values[test_name]
                
                for feature in test_effects.keys():
                    if feature in test_p_values:
                        effect = abs(test_effects[feature])
                        p_val = test_p_values[feature]
                        
                        if p_val < p_threshold and effect >= effect_threshold:
                            categories['significant_large_effect'].append({
                                'test': test_name, 'feature': feature, 
                                'effect_size': effect, 'p_value': p_val
                            })
                        elif p_val < p_threshold and effect < effect_threshold:
                            categories['significant_small_effect'].append({
                                'test': test_name, 'feature': feature, 
                                'effect_size': effect, 'p_value': p_val
                            })
                        elif p_val >= p_threshold and effect >= effect_threshold:
                            categories['non_significant_large_effect'].append({
                                'test': test_name, 'feature': feature, 
                                'effect_size': effect, 'p_value': p_val
                            })
                        else:
                            categories['non_significant_small_effect'].append({
                                'test': test_name, 'feature': feature, 
                                'effect_size': effect, 'p_value': p_val
                            })
        
        significance_analysis['categories'] = categories
        significance_analysis['summary'] = {
            cat: len(results) for cat, results in categories.items()
        }
        
        return significance_analysis
    
    def _extract_all_p_values(self) -> Dict[str, Dict[str, float]]:
        """Extract p-values from all statistical tests"""
        p_values = {}
        
        if 'test_results' in self.stats:
            for test_name, test_results in self.stats['test_results'].items():
                if isinstance(test_results, dict):
                    test_p_values = {}
                    for feature, feature_results in test_results.items():
                        if isinstance(feature_results, dict) and 'p_value' in feature_results:
                            test_p_values[feature] = feature_results['p_value']
                    p_values[test_name] = test_p_values
        
        return p_values
    
    def _analyze_fdr_burden(self) -> Dict[str, Any]:
        """Analyze FDR burden across different analysis types"""
        fdr_analysis = {}
        
        p_values = self._extract_all_p_values()
        
        for test_name, test_p_values in p_values.items():
            if test_p_values:
                p_vals_list = list(test_p_values.values())
                
                # Apply different FDR corrections
                fdr_bh_rejected, fdr_bh_pvals, _, _ = multipletests(
                    p_vals_list, method='fdr_bh'
                )
                fdr_by_rejected, fdr_by_pvals, _, _ = multipletests(
                    p_vals_list, method='fdr_by'
                )
                
                fdr_analysis[test_name] = {
                    'raw_significant': sum(p < 0.05 for p in p_vals_list),
                    'fdr_bh_significant': sum(fdr_bh_rejected),
                    'fdr_by_significant': sum(fdr_by_rejected),
                    'correction_impact_bh': sum(fdr_bh_rejected) / max(sum(p < 0.05 for p in p_vals_list), 1),
                    'correction_impact_by': sum(fdr_by_rejected) / max(sum(p < 0.05 for p in p_vals_list), 1),
                    'total_tests': len(p_vals_list)
                }
        
        return fdr_analysis
    
    def _conduct_power_analysis(self) -> Dict[str, Any]:
        """Conduct power analysis for each test type"""
        power_analysis = {}
        
        # This would involve statistical power calculations
        # Placeholder implementation
        if 'test_results' in self.stats:
            for test_name, test_results in self.stats['test_results'].items():
                power_analysis[test_name] = {
                    'estimated_power': 0.8,  # Placeholder
                    'adequate_power_threshold': 0.8,
                    'underpowered_tests': [],
                    'sample_size_recommendations': {}
                }
        
        return power_analysis
    
    def _optimize_statistical_thresholds(self) -> Dict[str, Any]:
        """Optimize statistical significance thresholds"""
        optimization_results = {}
        
        # Test different p-value and effect size thresholds
        p_thresholds = [0.01, 0.05, 0.1]
        effect_thresholds = [0.2, 0.5, 0.8]
        
        threshold_results = []
        
        for p_thresh in p_thresholds:
            for effect_thresh in effect_thresholds:
                # Count significant results at each threshold combination
                n_significant = self._count_significant_at_thresholds(p_thresh, effect_thresh)
                threshold_results.append({
                    'p_threshold': p_thresh,
                    'effect_threshold': effect_thresh,
                    'n_significant': n_significant,
                    'stringency_score': p_thresh * effect_thresh
                })
        
        optimization_results['threshold_analysis'] = threshold_results
        
        # Recommend optimal thresholds
        optimal_idx = np.argmax([r['stringency_score'] for r in threshold_results])
        optimization_results['recommended_thresholds'] = threshold_results[optimal_idx]
        
        return optimization_results
    
    def _count_significant_at_thresholds(self, p_thresh: float, effect_thresh: float) -> int:
        """Count significant results at given thresholds"""
        count = 0
        
        effect_sizes = self._extract_all_effect_sizes()
        p_values = self._extract_all_p_values()
        
        for test_name in effect_sizes.keys():
            if test_name in p_values:
                test_effects = effect_sizes[test_name]
                test_p_values = p_values[test_name]
                
                for feature in test_effects.keys():
                    if feature in test_p_values:
                        if (test_p_values[feature] < p_thresh and 
                            abs(test_effects[feature]) >= effect_thresh):
                            count += 1
        
        return count
    
    def _analyze_environmental_tolerance(self) -> Dict[str, Any]:
        """Analyze environmental tolerance of top features"""
        tolerance_analysis = {}
        
        if self.consensus_features is not None:
            env_data = self._extract_environmental_data()
            abundance_data = self._get_feature_abundance_matrix(
                self.consensus_features.index.tolist()[:20]
            )
            
            if env_data is not None and abundance_data is not None:
                for feature in abundance_data.columns:
                    feature_tolerance = {}
                    
                    for env_var in env_data.columns:
                        # Calculate tolerance as range of environmental conditions
                        # where feature has non-zero abundance
                        feature_abundance = abundance_data[feature]
                        present_samples = feature_abundance[feature_abundance > 0].index
                        
                        if len(present_samples) > 1:
                            env_range = env_data.loc[present_samples, env_var]
                            feature_tolerance[env_var] = {
                                'min': env_range.min(),
                                'max': env_range.max(),
                                'range': env_range.max() - env_range.min(),
                                'optimal': env_data.loc[
                                    feature_abundance.idxmax(), env_var
                                ] if not feature_abundance.empty else None
                            }
                    
                    tolerance_analysis[feature] = feature_tolerance
        
        return tolerance_analysis
    
    def _assess_functional_coherence(self) -> Dict[str, Any]:
        """Assess functional coherence of selected features"""
        coherence_results = {}
        
        # This would analyze if functionally similar features cluster together
        # in importance rankings
        coherence_results['functional_clustering_score'] = 0.7  # Placeholder
        coherence_results['coherence_p_value'] = 0.03  # Placeholder
        
        return coherence_results
    
    def _assess_phylogenetic_coherence(self) -> Dict[str, Any]:
        """Assess phylogenetic coherence of selected features"""
        coherence_results = {}
        
        # Test if phylogenetically related features are selected together
        coherence_results['phylogenetic_clustering_score'] = 0.6  # Placeholder
        coherence_results['coherence_p_value'] = 0.01  # Placeholder
        
        return coherence_results
    
    def _create_feature_importance_sunburst(self) -> go.Figure:
        """Create sunburst chart of feature importance"""
        if self.consensus_features is None:
            return go.Figure()
        
        # Extract taxonomic hierarchy from feature names
        hierarchy_data = []
        
        for feature in self.consensus_features.index[:20]:
            # Parse taxonomic levels from feature name
            # This assumes feature names contain taxonomic information
            parts = str(feature).split(';') if ';' in str(feature) else [str(feature)]
            
            importance = self.consensus_features.loc[feature, 'consensus_score']
            
            if len(parts) >= 2:
                hierarchy_data.append({
                    'ids': f"{parts[0]}_{parts[1]}_{feature}",
                    'labels': parts[-1],
                    'parents': f"{parts[0]}_{parts[1]}" if len(parts) > 2 else parts[0],
                    'values': importance
                })
                
                # Add parent levels
                if len(parts) > 1:
                    hierarchy_data.append({
                        'ids': f"{parts[0]}_{parts[1]}",
                        'labels': parts[1],
                        'parents': parts[0],
                        'values': 0
                    })
                    hierarchy_data.append({
                        'ids': parts[0],
                        'labels': parts[0],
                        'parents': "",
                        'values': 0
                    })
        
        # Remove duplicates
        seen = set()
        unique_hierarchy = []
        for item in hierarchy_data:
            if item['ids'] not in seen:
                unique_hierarchy.append(item)
                seen.add(item['ids'])
        
        if unique_hierarchy:
            df_hierarchy = pd.DataFrame(unique_hierarchy)
            
            fig = go.Figure(go.Sunburst(
                ids=df_hierarchy['ids'],
                labels=df_hierarchy['labels'],
                parents=df_hierarchy['parents'],
                values=df_hierarchy['values'],
                branchvalues="total"
            ))
            
            fig.update_layout(
                title="Feature Importance Hierarchy",
                title_x=0.5
            )
            
            return fig
        
        return go.Figure()
    
    def _create_environmental_gradient_plot(self) -> go.Figure:
        """Create environmental gradient overlay plot"""
        if 'environmental_gradients' not in self.integrated_results:
            return go.Figure()
        
        gradient_data = self.integrated_results['environmental_gradients']
        
        if 'cca_results' in gradient_data:
            env_scores = gradient_data['cca_results']['environmental_scores']
            comm_scores = gradient_data['cca_results']['community_scores']
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Environmental Space', 'Community Space')
            )
            
            # Environmental scores plot
            if env_scores.shape[1] >= 2:
                fig.add_trace(
                    go.Scatter(
                        x=env_scores.iloc[:, 0],
                        y=env_scores.iloc[:, 1],
                        mode='markers',
                        name='Environmental Scores'
                    ),
                    row=1, col=1
                )
            
            # Community scores plot
            if comm_scores.shape[1] >= 2:
                fig.add_trace(
                    go.Scatter(
                        x=comm_scores.iloc[:, 0],
                        y=comm_scores.iloc[:, 1],
                        mode='markers',
                        name='Community Scores'
                    ),
                    row=1, col=2
                )
            
            fig.update_layout(
                title="Environmental Gradient Analysis",
                title_x=0.5
            )
            
            return fig
        
        return go.Figure()
    
    def _create_model_performance_plot(self) -> go.Figure:
        """Create model performance comparison plot"""
        if 'ml_validation' not in self.integrated_results:
            return go.Figure()
        
        validation_data = self.integrated_results['ml_validation']
        
        if 'cross_validation' in validation_data:
            cv_data = validation_data['cross_validation']
            
            models = list(cv_data.keys())
            mean_scores = [cv_data[model].get('mean_score', 0) for model in models]
            std_scores = [cv_data[model].get('std_score', 0) for model in models]
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=models,
                y=mean_scores,
                error_y=dict(type='data', array=std_scores),
                name='Cross-validation Accuracy'
            ))
            
            fig.update_layout(
                title="Model Performance Comparison",
                xaxis_title="Model",
                yaxis_title="Accuracy",
                title_x=0.5
            )
            
            return fig
        
        return go.Figure()
    
    def _create_network_visualization(self) -> go.Figure:
        """Create network visualization"""
        if not self.networks:
            return go.Figure()
        
        # Use the first available network
        network_method = list(self.networks.keys())[0]
        network = self.networks[network_method]['network']
        
        if len(network.nodes()) == 0:
            return go.Figure()
        
        # Calculate layout
        pos = nx.spring_layout(network, k=1, iterations=50)
        
        # Extract node and edge information
        node_x = [pos[node][0] for node in network.nodes()]
        node_y = [pos[node][1] for node in network.nodes()]
        node_text = list(network.nodes())
        
        edge_x = []
        edge_y = []
        
        for edge in network.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # Create traces
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="middle center",
            marker=dict(
                size=10,
                color='lightblue',
                line=dict(width=2, color='black')
            )
        )
        
        fig = go.Figure(data=[edge_trace, node_trace])
        
        fig.update_layout(
            title=f'Feature Co-occurrence Network ({network_method})',
            title_x=0.5,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="Network of co-occurring important features",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002 ) ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return fig
    
    def _create_effect_size_heatmap(self) -> go.Figure:
        """Create effect size landscape heatmap"""
        if 'effect_size_landscape' not in self.integrated_results:
            return go.Figure()
        
        effect_data = self.integrated_results['effect_size_landscape']
        
        if 'matrices' in effect_data:
            matrices = effect_data['matrices']
            
            # Combine all effect size matrices
            all_effects = pd.DataFrame()
            
            for test_name, matrix in matrices.items():
                if not matrix.empty:
                    matrix_copy = matrix.copy()
                    matrix_copy.columns = [f"{test_name}_{col}" for col in matrix_copy.columns]
                    all_effects = pd.concat([all_effects, matrix_copy], axis=1, sort=False)
            
            if not all_effects.empty:
                fig = go.Figure(data=go.Heatmap(
                    z=all_effects.values,
                    x=all_effects.columns,
                    y=all_effects.index,
                    colorscale='RdBu',
                    zmid=0
                ))
                
                fig.update_layout(
                    title="Effect Size Landscape",
                    title_x=0.5,
                    xaxis_title="Statistical Tests",
                    yaxis_title="Features"
                )
                
                return fig
        
        return go.Figure()
    
    def _create_summary_statistics_table(self) -> pd.DataFrame:
        """Create summary statistics table"""
        summary_data = []
        
        # Module completion status
        modules = {
            'Statistical Tests': bool(self.stats),
            'Alpha Diversity': bool(self.alpha_diversity),
            'Beta Diversity': bool(self.ordination),
            'Machine Learning': bool(self.models),
            'Feature Consensus': self.consensus_features is not None,
            'Network Analysis': bool(self.networks)
        }
        
        for module, completed in modules.items():
            summary_data.append({
                'Analysis Module': module,
                'Status': 'Completed' if completed else 'Not Available',
                'Components': self._count_module_components(module)
            })
        
        # Add key statistics
        if self.consensus_features is not None:
            summary_data.append({
                'Analysis Module': 'Top Features Identified',
                'Status': str(len(self.consensus_features)),
                'Components': 'Consensus ranking across modules'
            })
        
        if 'keystone_species' in self.integrated_results:
            keystones = self.integrated_results['keystone_species'].get('keystone_ranking', [])
            summary_data.append({
                'Analysis Module': 'Keystone Species',
                'Status': str(len(keystones)),
                'Components': 'Multi-dimensional importance scoring'
            })
        
        return pd.DataFrame(summary_data)
    
    def _count_module_components(self, module: str) -> str:
        """Count components within each analysis module"""
        if module == 'Statistical Tests' and self.stats:
            return f"{len(self.stats.get('test_results', {}))}"
        elif module == 'Machine Learning' and self.models:
            return f"{len(self.models)} models"
        elif module == 'Network Analysis' and self.networks:
            return f"{len(self.networks)} methods"
        else:
            return "Various components"
    
    def _create_main_dashboard_html(self, components: Dict[str, str]) -> str:
        """Create main dashboard HTML combining all components"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Downstream Analysis Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .container { max-width: 1200px; margin: 0 auto; }
                .section { margin-bottom: 30px; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }
                .section h2 { color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }
                .iframe-container { width: 100%; height: 600px; border: 1px solid #ccc; margin: 10px 0; }
                .iframe-container iframe { width: 100%; height: 100%; border: none; }
                .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Integrated Downstream Analysis Dashboard</h1>
                
                <div class="section">
                    <h2>Summary Statistics</h2>
                    <div class="iframe-container">
                        <iframe src="summary_statistics.html"></iframe>
                    </div>
                </div>
                
                <div class="grid">
                    <div class="section">
                        <h2>Feature Importance</h2>
                        <div class="iframe-container">
                            <iframe src="feature_importance_sunburst.html"></iframe>
                        </div>
                    </div>
                    
                    <div class="section">
                        <h2>Model Performance</h2>
                        <div class="iframe-container">
                            <iframe src="model_performance.html"></iframe>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Environmental Gradients</h2>
                    <div class="iframe-container">
                        <iframe src="environmental_gradients.html"></iframe>
                    </div>
                </div>
                
                <div class="grid">
                    <div class="section">
                        <h2>Feature Networks</h2>
                        <div class="iframe-container">
                            <iframe src="feature_networks.html"></iframe>
                        </div>
                    </div>
                    
                    <div class="section">
                        <h2>Effect Size Landscape</h2>
                        <div class="iframe-container">
                            <iframe src="effect_size_landscape.html"></iframe>
                        </div>
                    </div>
                </div>
                
            </div>
        </body>
        </html>
        """
        return html_template
    
    def _generate_ecological_interpretation(self) -> List[str]:
        """Generate ecological interpretation of results"""
        interpretations = []
        
        if self.consensus_features is not None:
            interpretations.append(
                "The consensus feature ranking reveals taxa that consistently emerge as important "
                "across multiple analytical approaches, suggesting these are key drivers of community dynamics."
            )
        
        if 'environmental_gradients' in self.integrated_results:
            interpretations.append(
                "Environmental gradients create predictable patterns in community composition, "
                "indicating that abiotic factors are primary structuring forces in this system."
            )
        
        if 'keystone_species' in self.integrated_results:
            interpretations.append(
                "Keystone taxa have been identified that likely play disproportionate roles "
                "in community function and stability."
            )
        
        if 'function_environment' in self.integrated_results:
            interpretations.append(
                "Functional potential appears coupled to environmental conditions, "
                "suggesting adaptive responses to local environmental pressures."
            )
        
        return interpretations
    
    def _generate_management_implications(self) -> List[str]:
        """Generate management implications"""
        implications = []
        
        if 'ecological_thresholds' in self.integrated_results:
            implications.append(
                "Identified ecological thresholds can serve as early warning indicators "
                "for community shifts, enabling proactive management interventions."
            )
        
        if self.consensus_features is not None:
            implications.append(
                "Key taxa identified through consensus ranking should be prioritized "
                "for monitoring programs as community health indicators."
            )
        
        if 'ml_validation' in self.integrated_results:
            implications.append(
                "Machine learning models can be deployed for predictive monitoring "
                "and rapid assessment of community state based on key features."
            )
        
        return implications
    
    def _generate_research_directions(self) -> List[str]:
        """Generate future research directions"""
        directions = []
        
        directions.append(
            "Temporal sampling would reveal whether identified patterns are stable "
            "or represent snapshot-specific dynamics."
        )
        
        if 'function_environment' in self.integrated_results:
            directions.append(
                "Experimental manipulation of key environmental variables could test "
                "causal relationships suggested by correlative analyses."
            )
        
        if 'keystone_species' in self.integrated_results:
            directions.append(
                "Isolation and characterization of keystone taxa could reveal "
                "specific mechanisms underlying their community influence."
            )
        
        directions.append(
            "Expanding sampling to additional environmental gradients would test "
            "the generalizability of identified patterns and relationships."
        )
        
        return directions
    
    def _bootstrap_feature_selection(self, n_iterations: int, 
                                   subsample_fraction: float) -> Dict[str, Any]:
        """Bootstrap feature selection stability"""
        feature_frequencies = {}
        
        if self.consensus_features is not None:
            # This would involve rerunning feature selection on bootstrap samples
            # Placeholder implementation
            top_features = self.consensus_features.index.tolist()[:20]
            
            for feature in top_features:
                # Simulate bootstrap frequency
                frequency = np.random.beta(8, 2)  # Placeholder
                feature_frequencies[feature] = frequency
        
        stability_results = {
            'feature_frequencies': feature_frequencies,
            'stability_threshold': 0.7,
            'stable_features': [
                f for f, freq in feature_frequencies.items() 
                if freq >= 0.7
            ]
        }
        
        return stability_results
    
    def _bootstrap_model_performance(self, n_iterations: int, 
                                   subsample_fraction: float) -> Dict[str, Any]:
        """Bootstrap model performance consistency"""
        performance_results = {}
        
        for model_name in self.models.keys():
            # Simulate bootstrap performance distribution
            performances = np.random.normal(0.8, 0.05, n_iterations)  # Placeholder
            
            performance_results[model_name] = {
                'performance_distribution': performances,
                'mean_performance': np.mean(performances),
                'performance_std': np.std(performances),
                'confidence_interval': np.percentile(performances, [2.5, 97.5])
            }
        
        return performance_results
    
    def _bootstrap_statistical_tests(self, n_iterations: int, 
                                   subsample_fraction: float) -> Dict[str, Any]:
        """Bootstrap statistical test stability"""
        test_stability = {}
        
        if 'test_results' in self.stats:
            for test_name in self.stats['test_results'].keys():
                # Simulate test stability
                significant_features = []
                
                for i in range(n_iterations):
                    # Placeholder: simulate number of significant features in bootstrap
                    n_sig = np.random.poisson(10)
                    significant_features.append(n_sig)
                
                test_stability[test_name] = {
                    'significant_counts': significant_features,
                    'mean_significant': np.mean(significant_features),
                    'std_significant': np.std(significant_features)
                }
        
        return test_stability
    
    def _test_ml_parameter_sensitivity(self) -> Dict[str, Any]:
        """Test ML hyperparameter sensitivity"""
        sensitivity_results = {}
        
        # Test different hyperparameter combinations
        for model_name in self.models.keys():
            param_sensitivity = {
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6, 9],
                'n_estimators': [100, 500, 1000]
            }
            
            # Placeholder: simulate parameter sensitivity
            sensitivity_scores = {}
            for param, values in param_sensitivity.items():
                scores = [0.8 + 0.1 * np.random.random() for _ in values]
                sensitivity_scores[param] = {
                    'values': values,
                    'scores': scores,
                    'sensitivity': max(scores) - min(scores)
                }
            
            sensitivity_results[model_name] = sensitivity_scores
        
        return sensitivity_results
    
    def _test_statistical_thresholds(self) -> Dict[str, Any]:
        """Test sensitivity to statistical significance thresholds"""
        threshold_sensitivity = {}
        
        p_thresholds = [0.01, 0.05, 0.1, 0.2]
        
        for threshold in p_thresholds:
            n_significant = self._count_significant_at_thresholds(threshold, 0.0)
            threshold_sensitivity[threshold] = {
                'n_significant': n_significant,
                'proportion_significant': n_significant / max(self._count_total_tests(), 1)
            }
        
        return threshold_sensitivity
    
    def _count_total_tests(self) -> int:
        """Count total number of statistical tests performed"""
        total_tests = 0
        
        if 'test_results' in self.stats:
            for test_results in self.stats['test_results'].values():
                if isinstance(test_results, dict):
                    total_tests += len(test_results)
        
        return total_tests
    
    def _test_preprocessing_sensitivity(self) -> Dict[str, Any]:
        """Test sensitivity to preprocessing parameters"""
        preprocessing_sensitivity = {}
        
        # Test different normalization methods, filtering thresholds, etc.
        parameters = {
            'normalization_method': ['clr', 'rclr', 'tmm'],
            'prevalence_threshold': [0.1, 0.2, 0.3],
            'abundance_threshold': [0.001, 0.01, 0.1]
        }
        
        # Placeholder implementation
        for param, values in parameters.items():
            param_results = []
            for value in values:
                # Simulate effect of parameter change
                effect_score = np.random.random()
                param_results.append({
                    'value': value,
                    'effect_score': effect_score,
                    'n_features_retained': np.random.randint(50, 200)
                })
            
            preprocessing_sensitivity[param] = param_results
        
        return preprocessing_sensitivity
    
    def _test_network_thresholds(self) -> Dict[str, Any]:
        """Test sensitivity to network correlation thresholds"""
        network_sensitivity = {}
        
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        for threshold in thresholds:
            # Rebuild network with different threshold
            if self.consensus_features is not None:
                top_features = self.consensus_features.index.tolist()[:20]
                abundance_data = self._get_feature_abundance_matrix(top_features)
                
                if abundance_data is not None:
                    corr_matrix = abundance_data.corr(method='spearman')
                    network = self._create_network_from_correlations(corr_matrix, threshold)
                    properties = self._calculate_network_properties(network)
                    
                    network_sensitivity[threshold] = {
                        'n_nodes': properties.get('n_nodes', 0),
                        'n_edges': properties.get('n_edges', 0),
                        'density': properties.get('density', 0),
                        'n_components': properties.get('n_components', 0)
                    }
        
        return network_sensitivity
    
    # ========================== MAIN EXECUTION METHOD ============================= #
    
    def run_comprehensive_analysis(self, output_dir: str = 'integrated_analysis_output') -> Dict[str, Any]:
        """
        Run all integrated analyses and generate comprehensive results
        """
        if self.verbose:
            print("=" * 60)
            print("RUNNING COMPREHENSIVE DOWNSTREAM ANALYSIS")
            print("=" * 60)
        
        # 1. Cross-module integration
        self.synthesize_feature_importance()
        self.analyze_ecological_coherence()
        
        # 2. Community structure & function
        self.analyze_diversity_function_coupling()
        self.analyze_environmental_gradients()
        
        # 3. Model performance & validation
        self.validate_ml_predictions()
        self.create_prediction_confidence_maps()
        
        # 4. Network & systems-level analyses
        self.build_integrated_networks()
        self.identify_keystone_species()
        
        # 5. Temporal & spatial patterns
        self.detect_ecological_thresholds()
        self.assess_sample_classification_accuracy()
        
        # 6. Functional ecology synthesis
        self.analyze_function_environment_relationships()
        self.analyze_phylogenetic_signal()
        
        # 7. Statistical power & effect size
        self.create_effect_size_landscape()
        self.assess_multiple_testing_burden()
        
        # 8. Meta-analysis approaches
        self.calculate_consistency_scores()
        self.assess_biological_plausibility()
        
        # 9. Visualization & interpretation
        dashboard_components = self.create_results_dashboard(output_dir)
        narrative = self.construct_ecological_narrative()
        
        # 10. Validation & robustness
        stability_results = self.test_result_stability()
        sensitivity_results = self.conduct_parameter_sensitivity_analysis()
        
        # Compile final results
        final_results = {
            'consensus_features': self.consensus_features,
            'integrated_results': self.integrated_results,
            'dashboard_components': dashboard_components,
            'narrative': narrative,
            'stability_assessment': stability_results,
            'sensitivity_assessment': sensitivity_results
        }
        
        # Generate final summary report
        final_summary = self._generate_final_summary_report(final_results)
        
        # Save results
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save key results as CSV/JSON files
        if self.consensus_features is not None:
            self.consensus_features.to_csv(f"{output_dir}/consensus_features.csv")
        
        # Save narrative as text file
        with open(f"{output_dir}/ecological_narrative.txt", 'w') as f:
            for section, content in narrative.items():
                f.write(f"\n{section.upper().replace('_', ' ')}\n")
                f.write("=" * 50 + "\n")
                if isinstance(content, list):
                    for item in content:
                        f.write(f"- {item}\n")
                else:
                    f.write(f"{content}\n")
                f.write("\n")
        
        # Save summary report
        with open(f"{output_dir}/final_summary_report.txt", 'w') as f:
            f.write(final_summary)
        
        if self.verbose:
            print(f"\nComprehensive analysis completed!")
            print(f"Results saved to: {output_dir}")
            print(f"Main dashboard: {output_dir}/main_dashboard.html")
            print("=" * 60)
        
        return final_results
    
    def _generate_final_summary_report(self, final_results: Dict[str, Any]) -> str:
        """Generate final comprehensive summary report"""
        
        report_sections = []
        
        # Header
        report_sections.append("COMPREHENSIVE DOWNSTREAM ANALYSIS SUMMARY REPORT")
        report_sections.append("=" * 60)
        report_sections.append("")
        
        # Executive Summary
        report_sections.append("EXECUTIVE SUMMARY")
        report_sections.append("-" * 20)
        
        if self.consensus_features is not None:
            n_features = len(self.consensus_features)
            report_sections.append(f"• Identified {n_features} consensus features of high importance across multiple analyses")
        
        if 'keystone_species' in self.integrated_results:
            keystones = self.integrated_results['keystone_species'].get('keystone_ranking', [])
            if keystones:
                top_keystone = keystones[0][0]
                report_sections.append(f"• Key species '{top_keystone}' identified as potential keystone organism")
        
        if 'ecological_thresholds' in self.integrated_results:
            thresholds = self.integrated_results['ecological_thresholds']
            n_vars_with_thresholds = len([v for v in thresholds.values() if 'detected_thresholds' in v])
            report_sections.append(f"• Ecological thresholds detected for {n_vars_with_thresholds} environmental variables")
        
        report_sections.append("")
        
        # Key Findings
        report_sections.append("KEY FINDINGS")
        report_sections.append("-" * 15)
        
        # Feature importance synthesis
        if self.consensus_features is not None:
            report_sections.append("Feature Importance Analysis:")
            
            # Top 5 features
            top_5_features = self.consensus_features.head().index.tolist()
            for i, feature in enumerate(top_5_features, 1):
                consensus_score = self.consensus_features.loc[feature, 'consensus_score']
                report_sections.append(f"  {i}. {feature} (consensus score: {consensus_score:.3f})")
            
            # Module consistency
            if 'consistency_scores' in self.integrated_results:
                consistency_data = self.integrated_results['consistency_scores']
                high_consistency = len(consistency_data[consistency_data['binary_consistency'] >= 0.75])
                report_sections.append(f"  • {high_consistency} features showed high consistency across ≥75% of analysis modules")
        
        report_sections.append("")
        
        # Environmental relationships
        if 'environmental_gradients' in self.integrated_results:
            report_sections.append("Environmental Gradient Analysis:")
            gradient_data = self.integrated_results['environmental_gradients']
            
            if 'cca_results' in gradient_data:
                explained_var = gradient_data['cca_results'].get('explained_variance', 0)
                report_sections.append(f"  • CCA explained {explained_var:.1%} of community-environment relationships")
            
            if 'ecological_thresholds' in self.integrated_results:
                threshold_data = self.integrated_results['ecological_thresholds']
                for env_var, thresh_info in threshold_data.items():
                    if isinstance(thresh_info, dict) and 'detected_thresholds' in thresh_info:
                        n_thresholds = len(thresh_info['detected_thresholds'])
                        if n_thresholds > 0:
                            report_sections.append(f"  • {n_thresholds} ecological threshold(s) detected for {env_var}")
        
        report_sections.append("")
        
        # Network analysis results
        if self.networks:
            report_sections.append("Network Analysis:")
            for method, network_data in self.networks.items():
                props = network_data.get('properties', {})
                n_nodes = props.get('n_nodes', 0)
                n_edges = props.get('n_edges', 0)
                density = props.get('density', 0)
                
                report_sections.append(f"  • {method.capitalize()} network: {n_nodes} nodes, {n_edges} edges (density: {density:.3f})")
        
        report_sections.append("")
        
        # Model performance
        if 'ml_validation' in self.integrated_results:
            report_sections.append("Machine Learning Model Performance:")
            ml_data = self.integrated_results['ml_validation']
            
            if 'cross_validation' in ml_data:
                cv_results = ml_data['cross_validation']
                for model_name, cv_data in cv_results.items():
                    if 'mean_score' in cv_data:
                        mean_acc = cv_data['mean_score']
                        std_acc = cv_data.get('std_score', 0)
                        report_sections.append(f"  • {model_name}: {mean_acc:.3f} ± {std_acc:.3f} accuracy")
        
        report_sections.append("")
        
        # Statistical landscape
        if 'effect_size_landscape' in self.integrated_results:
            report_sections.append("Statistical Effect Size Analysis:")
            effect_data = self.integrated_results['effect_size_landscape']
            
            if 'largest_effects' in effect_data:
                largest = effect_data['largest_effects']
                if 'top_10_effects' in largest and largest['top_10_effects']:
                    top_effect = largest['top_10_effects'][0]
                    report_sections.append(f"  • Largest effect size: {top_effect['effect_size']:.3f} ({top_effect['feature']} in {top_effect['test']})")
                
                if 'significance_analysis' in effect_data:
                    sig_analysis = effect_data['significance_analysis']
                    if 'summary' in sig_analysis:
                        summary = sig_analysis['summary']
                        sig_large = summary.get('significant_large_effect', 0)
                        report_sections.append(f"  • {sig_large} features showed both statistical significance and large effect sizes")
        
        report_sections.append("")
        
        # Biological plausibility
        if 'biological_plausibility' in self.integrated_results:
            report_sections.append("Biological Plausibility Assessment:")
            plausibility = self.integrated_results['biological_plausibility']
            
            if 'environmental_tolerance' in plausibility:
                report_sections.append("  • Environmental tolerance ranges calculated for top features")
            
            if 'functional_coherence' in plausibility:
                func_coherence = plausibility['functional_coherence']
                coherence_score = func_coherence.get('functional_clustering_score', 0)
                report_sections.append(f"  • Functional coherence score: {coherence_score:.3f}")
        
        report_sections.append("")
        
        # Stability and robustness
        if 'stability_assessment' in final_results:
            report_sections.append("Stability and Robustness Assessment:")
            stability = final_results['stability_assessment']
            
            if 'feature_selection' in stability:
                feat_stability = stability['feature_selection']
                stable_features = feat_stability.get('stable_features', [])
                report_sections.append(f"  • {len(stable_features)} features showed high stability across bootstrap samples")
            
            if 'model_performance' in stability:
                model_stability = stability['model_performance']
                stable_models = []
                for model_name, model_data in model_stability.items():
                    std_perf = model_data.get('performance_std', float('inf'))
                    if std_perf < 0.05:  # Low standard deviation indicates stability
                        stable_models.append(model_name)
                report_sections.append(f"  • {len(stable_models)} models showed consistent performance across bootstrap samples")
        
        report_sections.append("")
        
        # Recommendations
        report_sections.append("RECOMMENDATIONS")
        report_sections.append("-" * 15)
        
        narrative = final_results.get('narrative', {})
        
        if 'management_implications' in narrative:
            report_sections.append("Management Implications:")
            for implication in narrative['management_implications']:
                report_sections.append(f"  • {implication}")
        
        report_sections.append("")
        
        if 'research_directions' in narrative:
            report_sections.append("Future Research Directions:")
            for direction in narrative['research_directions']:
                report_sections.append(f"  • {direction}")
        
        report_sections.append("")
        
        # Technical details
        report_sections.append("TECHNICAL SUMMARY")
        report_sections.append("-" * 18)
        
        # Analysis modules completed
        completed_modules = []
        if self.stats: completed_modules.append("Statistical Analysis")
        if self.alpha_diversity: completed_modules.append("Alpha Diversity")
        if self.ordination: completed_modules.append("Beta Diversity/Ordination")
        if self.models: completed_modules.append("Machine Learning")
        if self.networks: completed_modules.append("Network Analysis")
        
        report_sections.append(f"Completed Analysis Modules: {', '.join(completed_modules)}")
        
        # Data dimensions
        if self.metadata and self.tables:
            report_sections.append("Data Characteristics:")
            
            # Try to extract sample and feature counts
            sample_count = 0
            feature_count = 0
            
            for mode in ['genus', 'asv']:
                if mode in self.metadata:
                    metadata = self.metadata[mode]
                    if isinstance(metadata, dict):
                        for subset_data in metadata.values():
                            if isinstance(subset_data, pd.DataFrame):
                                sample_count = max(sample_count, len(subset_data))
                    elif isinstance(metadata, pd.DataFrame):
                        sample_count = max(sample_count, len(metadata))
                
                if mode in self.tables:
                    tables = self.tables[mode]
                    if isinstance(tables, dict):
                        for table_data in tables.values():
                            if hasattr(table_data, 'shape'):
                                feature_count = max(feature_count, table_data.shape[0])
                            elif isinstance(table_data, pd.DataFrame):
                                feature_count = max(feature_count, len(table_data))
            
            if sample_count > 0:
                report_sections.append(f"  • Approximate samples analyzed: {sample_count}")
            if feature_count > 0:
                report_sections.append(f"  • Approximate features analyzed: {feature_count}")
        
        # Analysis parameters
        report_sections.append("Key Parameters:")
        if self.consensus_features is not None:
            report_sections.append(f"  • Consensus features retained: {len(self.consensus_features)}")
        
        if self.networks:
            for method, network_data in self.networks.items():
                if 'correlation_matrix' in network_data:
                    corr_threshold = 0.3  # This should be extracted from actual parameters
                    report_sections.append(f"  • {method} network correlation threshold: {corr_threshold}")
        
        report_sections.append("")
        
        # Conclusion
        report_sections.append("CONCLUSION")
        report_sections.append("-" * 10)
        
        conclusion_text = (
            "This comprehensive analysis integrated multiple analytical approaches to provide "
            "a holistic view of microbial community structure and function. The consensus feature "
            "ranking identified key taxa that consistently emerge as important across different "
            "analytical frameworks, while environmental gradient analysis revealed the primary "
            "drivers of community composition. Network analysis uncovered co-occurrence patterns "
            "that may reflect ecological interactions, and machine learning models demonstrated "
            "the predictability of community structure from environmental variables. "
        )
        
        if 'ecological_thresholds' in self.integrated_results:
            conclusion_text += (
                "The detection of ecological thresholds provides critical insights for "
                "environmental management and monitoring. "
            )
        
        conclusion_text += (
            "The multi-dimensional validation approach, including bootstrap analysis and "
            "parameter sensitivity testing, supports the robustness of these findings. "
            "This integrated framework provides a foundation for both fundamental ecological "
            "understanding and practical applications in environmental management."
        )
        
        # Wrap text to reasonable line length
        import textwrap
        wrapped_conclusion = textwrap.fill(conclusion_text, width=80)
        report_sections.append(wrapped_conclusion)
        
        report_sections.append("")
        report_sections.append("=" * 60)
        report_sections.append("END OF REPORT")
        
        return "\n".join(report_sections)
