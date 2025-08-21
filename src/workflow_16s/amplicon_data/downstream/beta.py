# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import glob
import logging
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from functools import lru_cache

# Third‑Party Imports
import pandas as pd
import numpy as np
from biom.table import Table
import plotly.io as pio

# Local Imports
from workflow_16s import constants
from workflow_16s.amplicon_data.helpers import _init_dict_level
from workflow_16s.figures.merged import (
    pca as plot_pca,
    pcoa as plot_pcoa,
    mds as plot_mds
)
from workflow_16s.stats.beta_diversity import (
    pca as calculate_pca,
    pcoa as calculate_pcoa,
    tsne as calculate_tsne,
    umap as calculate_umap,
)
from workflow_16s.utils.data import table_to_df, update_table_and_meta
from workflow_16s.utils.progress import get_progress_bar, _format_task_desc

# ========================== INITIALISATION & CONFIGURATION ========================== #

logger = logging.getLogger("workflow_16s")
# Global lock for UMAP operations to prevent thread conflicts
umap_lock = threading.Lock()

# =================================== DATA CLASSES ================================== #

@dataclass(frozen=True)
class OrdinationTask:
    """Represents a single ordination task with all necessary parameters."""
    table_type: str
    level: str
    method: str
    
    def __str__(self):
        return f"{self.table_type}/{self.level}/{self.method}"

@dataclass  
class OrdinationConfig:
    """Configuration for ordination methods."""
    key: str
    func: Callable
    plot_func: Callable
    name: str
    plot_kwargs: Dict = None
    
    def __post_init__(self):
        if self.plot_kwargs is None:
            self.plot_kwargs = {}

# =================================== FUNCTIONS ====================================== #

class Ordination:
    """Performs ordination analyses (PCA, PCoA, t-SNE, UMAP) and stores figures."""
    
    # Class constants for better memory efficiency
    KNOWN_METHODS = frozenset(["pca", "pcoa", "tsne", "umap"])
    DEFAULT_METHODS = {
        "raw": ("pca",),
        "filtered": ("pca", "pcoa"),
        "normalized": ("pca", "pcoa", "tsne", "umap"),
        "clr_transformed": ("pca", "pcoa", "tsne", "umap"),
        "presence_absence": ("pcoa", "tsne", "umap")
    }
    
    DEFAULT_COLOR_COLUMNS = (
        constants.DEFAULT_DATASET_COLUMN,
        constants.DEFAULT_GROUP_COLUMN,
        "env_feature", 
        "env_material", 
        "country"
    )
    
    # Use dataclass for better structure
    TEST_CONFIG = {
        "pca": OrdinationConfig("pca", calculate_pca, plot_pca, "PCA"),
        "pcoa": OrdinationConfig("pcoa", calculate_pcoa, plot_pcoa, "PCoA"),
        "tsne": OrdinationConfig("tsne", calculate_tsne, plot_mds, "t‑SNE", {"mode": "TSNE"}),
        "umap": OrdinationConfig("umap", calculate_umap, plot_mds, "UMAP", {"mode": "UMAP"}),
    }

    def __init__(
        self, 
        config: Dict, 
        metadata: pd.DataFrame,
        tables: Dict[str, Dict[str, Table]],
        verbose: bool = False
    ):
        self.config = config
        self.verbose = verbose
        self.metadata = metadata
        self.tables = tables
        
        # Use tuple for immutable sequence (better memory)
        self.color_columns = tuple(config['maps'].get(
            "color_columns", 
            self.DEFAULT_COLOR_COLUMNS
        ))
        
        self.group_column = config.get("group_column", constants.DEFAULT_GROUP_COLUMN)  
        self.results = {}
        
        # Early return optimization
        ordination_config = self.config.get('ordination', {})
        if not ordination_config.get('enabled', False):
            logger.info("Beta diversity analysis (ordination) disabled")
            self.tasks = ()
            return
            
        self.tasks = self._get_enabled_tasks()          
        if not self.tasks:
            logger.info("No methods for beta diversity analysis (ordination) enabled")
        else:
            logger.info(f"Found {len(self.tasks)} ordination tasks to process")

    def _get_enabled_tasks(self) -> Tuple[OrdinationTask, ...]:
        """Get enabled tasks as immutable tuple for better performance."""
        logger.debug("Determining enabled ordination tasks")
        ordination_config = self.config.get('ordination', {})
        table_config = ordination_config.get('tables', {})
        tasks = []
        
        for table_type, levels in self.tables.items():
            table_type_config = table_config.get(table_type, {})
            if not table_type_config.get('enabled', False):
                logger.debug(f"Skipping table type {table_type}: disabled in config")
                continue
                
            # Use set intersection for better performance
            available_levels = set(levels.keys())
            enabled_levels = set(table_type_config.get('levels', available_levels))
            valid_levels = available_levels & enabled_levels
            
            # Use set intersection for methods too
            default_methods = set(self.DEFAULT_METHODS.get(table_type, ("pca",)))
            enabled_methods = set(table_type_config.get('methods', default_methods))
            valid_methods = self.KNOWN_METHODS & enabled_methods
            
            for level in valid_levels:
                for method in valid_methods:  
                    tasks.append(OrdinationTask(table_type, level, method))
                    logger.debug(f"Added task: {table_type}/{level}/{method}")
        
        logger.info(f"Total tasks identified: {len(tasks)}")
        return tuple(tasks)  # Return immutable tuple

    def _initialize_results(self) -> None:
        """Initialize results storage structure efficiently."""
        logger.debug("Initializing results structure")
        for task in self.tasks:
            if task.table_type not in self.results:
                self.results[task.table_type] = {}
            if task.level not in self.results[task.table_type]:
                self.results[task.table_type][task.level] = {'figures': {}}

    @lru_cache(maxsize=32)
    def _should_skip_existing(self, task: OrdinationTask, output_dir: Path) -> bool:
        """Check if we should skip calculation due to existing figures (cached)."""
        logger.info(f"Checking if we should skip calculation for task: {task}")
        load_existing_enabled = self.config.get('ordination', {}).get('load_existing', False)
        if not load_existing_enabled:
            logger.info(f"Skipping disabled in config: {load_existing_enabled}")
            return False
            
        # Check if color columns exist in metadata
        metadata = self.metadata[task.table_type][task.level]
        required_color_cols = [col for col in self.color_columns if col in metadata.columns]
        if not required_color_cols:
            logger.info(f"Skipping ordination {task}: no valid color columns")
            return True
        
        # Check if all required files exist
        for color_col in required_color_cols:
            fname = f"{task.method}.{task.table_type}.1-2.{color_col}.html"
            file_path = output_dir / fname
            logger.info(f"Checking if file exists: {file_path}")
            if not file_path.exists():
                logger.info(f"File not found: {file_path}")
                return False
                
        logger.info(f"Skipping ordination {task}: all figures exist")
        return True

    def _load_existing_figures(self, task: OrdinationTask, output_dir: Path) -> Dict[str, Any]:
        """Load existing figures from HTML files."""
        logger.info(f"Loading existing figures for {task}")
        figures = {}
        metadata = self.metadata[task.table_type][task.level]
        valid_color_cols = [col for col in self.color_columns if col in metadata.columns]
        
        # Check if plotly.io has read_html method
        if not hasattr(pio, 'read_html'):
            logger.error("Plotly version does not support read_html method. Cannot load existing figures.")
            return figures
        
        for color_col in valid_color_cols:
            fname = f"{task.method}.{task.table_type}.1-2.{color_col}.html"
            file_path = output_dir / fname
            logger.info(f"Attempting to load: {file_path}")
            try:
                # Check if file exists and is readable
                if not file_path.exists():
                    logger.warning(f"File does not exist: {file_path}")
                    continue
                    
                if file_path.stat().st_size == 0:
                    logger.warning(f"File is empty: {file_path}")
                    continue
                    
                logger.debug(f"Reading HTML file: {file_path}")
                fig = pio.read_html(file_path)[0]
                figures[color_col] = fig
                logger.info(f"Successfully loaded existing figure: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to load existing figure {file_path}: {e}")
                import traceback
                logger.debug(f"Traceback: {traceback.format_exc()}")
                
        return figures

    def _store_figure_paths(self, task: OrdinationTask, output_dir: Path) -> Dict[str, str]:
        """Store file paths to existing figures instead of loading them."""
        logger.info(f"Storing figure paths for {task}")
        figure_paths = {}
        metadata = self.metadata[task.table_type][task.level]
        valid_color_cols = [col for col in self.color_columns if col in metadata.columns]
        
        for color_col in valid_color_cols:
            fname = f"{task.method}.{task.table_type}.1-2.{color_col}.html"
            file_path = output_dir / fname
            logger.info(f"Checking figure path: {file_path}")
            
            if file_path.exists() and file_path.stat().st_size > 0:
                figure_paths[color_col] = str(file_path)
                logger.info(f"Found existing figure: {file_path}")
            else:
                logger.warning(f"Figure file not found or empty: {file_path}")
                
        return figure_paths

    def _calculate_optimal_workers(self) -> int:
        """Calculate optimal number of worker threads."""
        cpu_count = os.cpu_count() or 1
        # For I/O bound tasks with some CPU computation, use more threads
        # but cap at reasonable limit to avoid resource contention
        optimal = min(6, max(2, cpu_count // 2 + 1))
        logger.debug(f"Calculated optimal workers: {optimal}")
        return optimal

    def run(self, output_dir: Optional[Path] = None) -> None:
        """Run ordination analysis with optimized parallel processing."""
        # Early returns
        if not self.tasks:
            logger.info("No tasks to run")
            return
            
        logger.info(f"Starting ordination with {len(self.tasks)} tasks")
        self._initialize_results()
        
        if output_dir is None:
            output_dir = Path(self.config['output_dir'])
            logger.debug(f"Using output directory: {output_dir}")
            
        # Check if we can load existing figures
        ordination_config = self.config.get('ordination', {})
        if ordination_config.get('load_existing_figures', False) and not hasattr(pio, 'read_html'):
            logger.error("Plotly version does not support read_html method. Cannot load existing figures.")
            logger.info("Will store file paths instead for downstream use")
            ordination_config['load_existing_figures'] = False
            # Update the config to reflect this change
            self.config['ordination']['load_existing_figures'] = False
            
        with get_progress_bar() as progress:
            stats_desc = "Running beta diversity"
            stats_task = progress.add_task(  
                _format_task_desc(stats_desc),
                total=len(self.tasks)
            )
            
            max_workers = self._calculate_optimal_workers()
            logger.info(f"Using {max_workers} worker threads")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks at once for better scheduling
                logger.debug("Submitting tasks to executor")
                future_to_task = {
                    executor.submit(
                        self._run_single_ordination,
                        task,
                        output_dir,
                        progress
                    ): task for task in self.tasks
                }
    
                # Process completed futures with timeout
                errors = []
                try:
                    logger.debug("Waiting for tasks to complete")
                    for future in as_completed(future_to_task, timeout=2*3600):
                        task = future_to_task[future]
                        try:
                            result = future.result()
                            if result:  # Only store non-None results
                                self._store_result(*result)
                        except Exception as e:
                            error_msg = f"Ordination failed for {task}: {str(e)}"
                            errors.append(error_msg)
                            logger.error(error_msg)
                            import traceback
                            logger.debug(f"Traceback: {traceback.format_exc()}")
                        finally:
                            progress.update(stats_task, advance=1)
                            
                except TimeoutError:
                    logger.warning("Ordination timeout - proceeding with completed results")
                
                # Log summary of errors if any
                if errors:
                    logger.warning(f"Completed with {len(errors)} errors out of {len(self.tasks)} tasks")
                
            progress.update(stats_task, description=_format_task_desc(stats_desc))
        logger.info("Ordination completed")

    def _store_result(self, table_type: str, level: str, method: str, 
                     ord_result: Any, figures: Dict) -> None:
        """Store ordination results efficiently."""
        logger.debug(f"Storing result for {table_type}/{level}/{method}")
        level_results = self.results[table_type][level]
        level_results[method] = ord_result
        level_results['figures'][method] = figures

    def _run_single_ordination(self, task: OrdinationTask, output_dir: Path, 
                              progress) -> Optional[Tuple]:
        """Run a single ordination task with optimized error handling."""
        method_desc = (
            f"{task.table_type.replace('_', ' ').title()} ({task.level.title()})"
            f" → {self.TEST_CONFIG[task.method].name}"  
        )
        
        logger.info(f"Starting ordination task: {task}")
        
        # Create progress task for this method
        method_task = progress.add_task(
            _format_task_desc(method_desc),
            total=1
        )
        
        try:
            progress.update(method_task, description=_format_task_desc(method_desc))
            
            # Prepare output directory
            table_output_dir = output_dir / 'ordination' / task.table_type / task.level
            logger.debug(f"Creating output directory: {table_output_dir}")
            table_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if we should skip and load existing figures
            if self._should_skip_existing(task, table_output_dir):
                logger.info(f"Checking existing figures for {task}")
                
                # Try to load figures if possible, otherwise store paths
                if hasattr(pio, 'read_html'):
                    figures = self._load_existing_figures(task, table_output_dir)
                    if figures:
                        logger.info(f"Returning loaded figures for {task}")
                        return task.table_type, task.level, task.method, None, figures
                else:
                    # Store file paths instead of loading figures
                    figure_paths = self._store_figure_paths(task, table_output_dir)
                    if figure_paths:
                        logger.info(f"Returning figure paths for {task}")
                        return task.table_type, task.level, task.method, None, figure_paths
                
                logger.info(f"No figures loaded for {task}, proceeding with calculation")
            
            # Get aligned data
            logger.debug(f"Getting aligned data for {task}")
            table = self.tables[task.table_type][task.level]
            metadata = self.metadata[task.table_type][task.level]
            table_aligned, metadata_aligned = update_table_and_meta(table, metadata)
            
            logger.debug(f"Running test for {task}")
            result, figures = self._run_test(  
                table=table_aligned,
                metadata=metadata_aligned,
                symbol_col=self.group_column,  
                task=task,
                output_dir=table_output_dir
            )
            
            logger.info(f"Completed task {task}")
            return task.table_type, task.level, task.method, result, figures
            
        except Exception as e:
            logger.error(f"Ordination {task} failed: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return None
        finally:
            progress.update(method_task, completed=1, visible=False)

    def _get_method_parameters(self, task: OrdinationTask) -> Dict:
        """Get method-specific parameters efficiently."""
        logger.debug(f"Getting method parameters for {task.method}")
        params = {}
        
        if task.method == "pcoa":
            table_config = self.config['ordination']['tables'].get(task.table_type, {})
            params["metric"] = table_config.get("pcoa_metric", "braycurtis")
            logger.info(f"Using PCoA metric: {params['metric']}")
        elif task.method in ("tsne", "umap"):
            params["n_jobs"] = 1  # Thread safety
            logger.debug(f"Setting n_jobs=1 for {task.method} for thread safety")
            
        return params

    def _run_test(self, table: Table, metadata: pd.DataFrame, symbol_col: str,
                 task: OrdinationTask, output_dir: Path) -> Tuple[Any, Dict]:
        """Run ordination test with optimized parameter handling."""
        logger.info(f"Running {task.method} calculation for {task.table_type}/{task.level}")
        method_config = self.TEST_CONFIG[task.method]
        method_params = self._get_method_parameters(task)
        
        try:
            # Thread-safe execution for UMAP/t-SNE
            if task.method in ("tsne", "umap"):
                logger.debug(f"Acquiring lock for {task.method}")
                with umap_lock:
                    os.environ['NUMBA_NUM_THREADS'] = '1'
                    logger.debug(f"Running {task.method} with thread safety")
                    result = method_config.func(table=table, **method_params)
            else:
                logger.debug(f"Running {task.method}")
                result = method_config.func(table=table, **method_params)
                
        except Exception as e:
            logger.error(f"Failed {task}: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return None, {}

        # Generate figures
        try:
            logger.debug(f"Generating figures for {task}")
            figures = self._generate_figures(
                result, metadata, symbol_col, task, output_dir, method_config
            )
            return result, figures
            
        except Exception as e:
            logger.error(f"Plotting failed for {task}: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return result, {}

    def _generate_figures(self, result: Any, metadata: pd.DataFrame, 
                         symbol_col: str, task: OrdinationTask, 
                         output_dir: Path, method_config: OrdinationConfig) -> Dict:
        """Generate figures with optimized parameter preparation."""
        logger.debug(f"Generating figures for {task}")
        figures = {}
        
        # Filter valid color columns once
        valid_color_cols = [col for col in self.color_columns if col in metadata.columns]
        
        if not valid_color_cols:
            logger.warning(f"No valid color columns found for {task}")
            return figures
        
        # Base plot parameters
        base_params = {
            "metadata": metadata,
            "symbol_col": symbol_col,
            "transformation": task.table_type,
            "output_dir": output_dir,
            **method_config.plot_kwargs
        }
        
        # Method-specific parameters
        if task.method == "pca":
            base_params.update({
                "components": result["components"],
                "proportion_explained": result["exp_var_ratio"],
            })
        elif task.method == "pcoa":
            base_params.update({
                "components": result.samples,
                "proportion_explained": result.proportion_explained,
            })
        else:  # t-SNE/UMAP
            base_params["df"] = result

        # Generate figures for each valid color column
        for color_col in valid_color_cols:
            try:
                logger.debug(f"Generating figure for {task} with color column: {color_col}")
                plot_params = {**base_params, "color_col": color_col}
                fig, _ = method_config.plot_func(**plot_params)
                if fig:
                    figures[color_col] = fig
                    logger.debug(f"Successfully generated figure for {color_col}")
            except Exception as e:
                logger.warning(f"Failed to generate figure for {task} with color {color_col}: {e}")
                import traceback
                logger.debug(f"Traceback: {traceback.format_exc()}")
                continue
                
        logger.info(f"Generated {len(figures)} figures for {task}")
        return figures
