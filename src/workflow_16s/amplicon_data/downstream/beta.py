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

    def _get_enabled_tasks(self) -> Tuple[OrdinationTask, ...]:
        """Get enabled tasks as immutable tuple for better performance."""
        ordination_config = self.config.get('ordination', {})
        table_config = ordination_config.get('tables', {})
        tasks = []
        
        for table_type, levels in self.tables.items():
            table_type_config = table_config.get(table_type, {})
            if not table_type_config.get('enabled', False):
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
        
        return tuple(tasks)  # Return immutable tuple

    def _initialize_results(self) -> None:
        """Initialize results storage structure efficiently."""
        for task in self.tasks:
            if task.table_type not in self.results:
                self.results[task.table_type] = {}
            if task.level not in self.results[task.table_type]:
                self.results[task.table_type][task.level] = {'figures': {}}

    @lru_cache(maxsize=32)
    def _should_skip_existing(self, task: OrdinationTask, output_dir: Path) -> bool:
        """Check if we should skip calculation due to existing figures (cached)."""
        if not self.config.get('ordination', {}).get('load_existing_figures', False):
            return False
            
        # Get metadata for this specific task
        metadata = self.metadata[task.table_type][task.level]
        required_color_cols = [col for col in self.color_columns if col in metadata.columns]
        
        if not required_color_cols:
            logger.info(f"Skipping ordination {task}: no valid color columns")
            return True
        
        # Check if all required files exist
        for color_col in required_color_cols:
            fname = f"{task.method}.{task.table_type}.1-2.{color_col}.html"
            if not (output_dir / fname).exists():
                return False
                
        logger.info(f"Skipping ordination {task}: all figures exist")
        return True

    def _load_existing_figures(self, task: OrdinationTask, output_dir: Path) -> Dict[str, Any]:
        """Load existing figures from HTML files."""
        figures = {}
        metadata = self.metadata[task.table_type][task.level]
        valid_color_cols = [col for col in self.color_columns if col in metadata.columns]
        
        for color_col in valid_color_cols:
            fname = f"{task.method}.{task.table_type}.1-2.{color_col}.html"
            file_path = output_dir / fname
            try:
                fig = pio.read_html(file_path)[0]
                figures[color_col] = fig
                logger.info(f"Loaded existing figure: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to load existing figure {file_path}: {e}")
                
        return figures

    def _calculate_optimal_workers(self) -> int:
        """Calculate optimal number of worker threads."""
        cpu_count = os.cpu_count() or 1
        # For I/O bound tasks with some CPU computation, use more threads
        # but cap at reasonable limit to avoid resource contention
        return min(6, max(2, cpu_count // 2 + 1))

    def run(self, output_dir: Optional[Path] = None) -> None:
        """Run ordination analysis with optimized parallel processing."""
        # Early returns
        if not self.tasks:
            return
            
        self._initialize_results()
        
        if output_dir is None:
            output_dir = Path(self.config['output_dir'])
            
        with get_progress_bar() as progress:
            stats_desc = "Running beta diversity"
            stats_task = progress.add_task(  
                _format_task_desc(stats_desc),
                total=len(self.tasks)
            )
            
            max_workers = self._calculate_optimal_workers()
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks at once for better scheduling
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
                        finally:
                            progress.update(stats_task, advance=1)
                            
                except TimeoutError:
                    logger.warning("Ordination timeout - proceeding with completed results")
                
                # Log summary of errors if any
                if errors:
                    logger.warning(f"Completed with {len(errors)} errors out of {len(self.tasks)} tasks")
                
            progress.update(stats_task, description=_format_task_desc(stats_desc))

    def _store_result(self, table_type: str, level: str, method: str, 
                     ord_result: Any, figures: Dict) -> None:
        """Store ordination results efficiently."""
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
        
        # Create progress task for this method
        method_task = progress.add_task(
            _format_task_desc(method_desc),
            total=1
        )
        
        try:
            progress.update(method_task, description=_format_task_desc(method_desc))
            
            # Prepare output directory
            table_output_dir = output_dir / 'ordination' / task.table_type / task.level
            table_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if we should skip and load existing figures
            if self._should_skip_existing(task, table_output_dir):
                figures = self._load_existing_figures(task, table_output_dir)
                if figures:
                    return task.table_type, task.level, task.method, None, figures
            
            # Get aligned data
            table = self.tables[task.table_type][task.level]
            metadata = self.metadata[task.table_type][task.level]
            table_aligned, metadata_aligned = update_table_and_meta(table, metadata)
            
            result, figures = self._run_test(  
                table=table_aligned,
                metadata=metadata_aligned,
                symbol_col=self.group_column,  
                task=task,
                output_dir=table_output_dir
            )
            
            return task.table_type, task.level, task.method, result, figures
            
        except Exception as e:
            logger.error(f"Ordination {task} failed: {e}")
            return None
        finally:
            progress.update(method_task, completed=1, visible=False)

    def _get_method_parameters(self, task: OrdinationTask) -> Dict:
        """Get method-specific parameters efficiently."""
        params = {}
        
        if task.method == "pcoa":
            table_config = self.config['ordination']['tables'].get(task.table_type, {})
            params["metric"] = table_config.get("pcoa_metric", "braycurtis")
            logger.debug(f"Using PCoA metric: {params['metric']}")
        elif task.method in ("tsne", "umap"):
            params["n_jobs"] = 1  # Thread safety
            
        return params

    def _run_test(self, table: Table, metadata: pd.DataFrame, symbol_col: str,
                 task: OrdinationTask, output_dir: Path) -> Tuple[Any, Dict]:
        """Run ordination test with optimized parameter handling."""
        method_config = self.TEST_CONFIG[task.method]
        method_params = self._get_method_parameters(task)
        
        try:
            # Thread-safe execution for UMAP/t-SNE
            if task.method in ("tsne", "umap"):
                with umap_lock:
                    os.environ['NUMBA_NUM_THREADS'] = '1'
                    result = method_config.func(table=table, **method_params)
            else:
                result = method_config.func(table=table, **method_params)
                
        except Exception as e:
            logger.error(f"Failed {task}: {e}")
            return None, {}

        # Generate figures
        try:
            figures = self._generate_figures(
                result, metadata, symbol_col, task, output_dir, method_config
            )
            return result, figures
            
        except Exception as e:
            logger.error(f"Plotting failed for {task}: {e}")
            return result, {}

    def _generate_figures(self, result: Any, metadata: pd.DataFrame, 
                         symbol_col: str, task: OrdinationTask, 
                         output_dir: Path, method_config: OrdinationConfig) -> Dict:
        """Generate figures with optimized parameter preparation."""
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
                plot_params = {**base_params, "color_col": color_col}
                fig, _ = method_config.plot_func(**plot_params)
                if fig:
                    figures[color_col] = fig
            except Exception as e:
                logger.warning(f"Failed to generate figure for {task} with color {color_col}: {e}")
                continue
                
        return figures
