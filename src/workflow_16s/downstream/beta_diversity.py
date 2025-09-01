# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import glob
import logging
import os
import re
import time
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from functools import lru_cache
from collections import defaultdict
# Third‑Party Imports
import json
import pandas as pd
import numpy as np
from biom.table import Table
import plotly.io as pio

# Local Imports
from workflow_16s import constants
from workflow_16s.amplicon_data.helpers import _init_dict_level
from workflow_16s.figures.figures import load_plotly_html
from workflow_16s.figures.merged import (
    pca as plot_pca,
    pcoa as plot_pcoa,
    mds as plot_mds
)
from workflow_16s.constants import GROUP_COLUMNS, MODE
from workflow_16s.diversity import beta_diversity 
from workflow_16s.figures.beta_diversity import beta_diversity_plot
from workflow_16s.downstream.load_data import align_table_and_metadata
from workflow_16s.utils.dataframe import table_to_df
from workflow_16s.utils.dir_utils import SubDirs
from workflow_16s.utils.dir import Dir, ProjectDir
from workflow_16s.utils.progress import get_progress_bar, _format_task_desc


# ========================== INITIALISATION & CONFIGURATION ========================== #

logger = logging.getLogger("workflow_16s")
umap_lock = threading.Lock() # Global lock for UMAP operations to prevent thread conflicts

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
    name: str
    plot_kwargs: Dict = None
    
    def __post_init__(self):
        if self.plot_kwargs is None:
            self.plot_kwargs = {}

# =================================== FUNCTIONS ====================================== #

class Ordination:
    """Performs ordination analyses (PCA, PCoA, t-SNE, UMAP) and stores figures."""
    
    # Class constants for better memory efficiency
    KnownMethods = frozenset(["pca", "pcoa", "tsne", "umap"])
    DefaultMethods = {
        "raw": ("pca",),
        "filtered": ("pca", "pcoa"),
        "normalized": ("pca", "pcoa", "tsne", "umap"),
        "clr_transformed": ("pca", "pcoa", "tsne", "umap"),
        "presence_absence": ("pcoa", "tsne", "umap")
    }
    
    DefaultColorCols = (
        constants.DEFAULT_DATASET_COLUMN,
        constants.DEFAULT_GROUP_COLUMN,
        "env_feature", 
        "env_material", 
        "country"
    )
    
    # Use dataclass for better structure
    TestConfig = {
        "pca": OrdinationConfig("pca", beta_diversity.pca, "PCA"),
        "pcoa": OrdinationConfig("pcoa", beta_diversity.pcoa, "PCoA"),
        "tsne": OrdinationConfig("tsne", beta_diversity.tsne, "t‑SNE", {"mode": "TSNE"}),
        "umap": OrdinationConfig("umap", beta_diversity.umap, "UMAP", {"mode": "UMAP"}),
    }
    
    def __init__(
        self, 
        config: Dict, 
        project_dir: Union[ProjectDir, SubDirs],
        metadata: pd.DataFrame,
        tables: Dict[str, Dict[str, Table]],
        group_column: str = constants.DEFAULT_GROUP_COLUMN,
    ):
        self.config = config
        # Check if ordination is enabled
        ordination_config = self.config.get('ordination', {})
        if not ordination_config.get('enabled', False):
            logger.info("Beta diversity analysis (ordination) disabled")
            self.tasks = ()
            return
      
        self.mode = self.config.get("target_subfragment_mode", MODE)
        self.verbose = self.config("verbose", False)
        self.project_dir = project_dir

        self.group_columns = self.config.get("group_columns", GROUP_COLUMNS)
        self.symbol_col = 'nuclear_contamination_status'
        #self.group_column = group_column
        
        self.metadata = metadata
        self.tables = tables
        
        self.color_columns = tuple(self.config['maps'].get("color_columns", self.DefaultColorCols))

        # Initialize results dict
        self.results = defaultdict(lambda: defaultdict(lambda: {'figures': {}}))
            
        # Check which ordination tasks are enabled    
        self.tasks = self._get_enabled_tasks()          
        if not self.tasks:
            self.log("No methods enabled for beta diversity analysis (ordination)")
        else:
            self.log(f"Found {len(self.tasks)} beta diversity analysis (ordination) tasks to process")

    def log(self, msg):
        return (lambda msg: logger.debug(msg)) if self.verbose else (lambda *_: None)

    def _fetch_data(table_type: str, level: str) -> Tuple:
        metadata = self.metadata.get(table_type, {}).get(level)
        table = self.tables.get(table_type, {}).get(level)
        if table is None or metadata is None:
            raise ValueError(
                f"Missing table or metadata for level '{level}' and table type '{table_type}'"
            )
        return metadata, table
        
    def _get_enabled_tasks(self, ordination_config) -> Tuple[OrdinationTask, ...]:
        """Get enabled tasks."""
        logger.debug("Retrieving enabled ordination tasks from the config file")

        tasks = []
        
        table_config = ordination_config.get('tables', {})        
        for table_type, levels in self.tables.items():
            table_type_config = table_config.get(table_type, {})
            if not table_type_config.get('enabled', False):
                self.log(f"Skipping table type {table_type}: disabled in config")
                continue
                
            # Get valid levels   
            available_levels = set(levels.keys())
            enabled_levels = set(table_type_config.get('levels', available_levels))
            valid_levels = available_levels & enabled_levels

            # Get valid methods
            default_methods = set(self.DefaultMethods.get(table_type, ("pca",)))
            enabled_methods = set(table_type_config.get('methods', default_methods))
            valid_methods = self.KnownMethods & enabled_methods
            
            for level in valid_levels:
                for method in valid_methods:  
                    tasks.append(OrdinationTask(table_type, level, method))
                    self.log(f"Added task: {table_type}/{level}/{method}")
        
        self.log_ok(f"Retrieved {len(tasks)} tasks")
        return tuple(tasks) 

    @lru_cache(maxsize=32)
    def _should_skip_existing(self, task: OrdinationTask, output_dir: Path) -> bool:
        """Check if we should skip calculation due to existing figures (cached)."""
        if not self.config.get('ordination', {}).get('load_existing', False):
            self.log(f"Skipping ordination {task}: disabled in config")
            return False
            
        # Check if color columns exist in metadata
        metadata = self.metadata[task.table_type][task.level]
        required_color_columns = [col for col in self.color_columns if col in metadata.columns]
        if not required_color_columns:
            self.log(f"Skipping ordination {task}: no valid color columns")
            return True
        
        # Check if all required files exist
        for color_col in required_color_columns:
            fname = f"{task.method}.{task.table_type}.1-2.{color_col}.html"
            file_path = output_dir / fname
            if not file_path.exists() or file_path.stat().st_size == 0:
                return False
                
        self.log(f"Skipping ordination {task}: all figures exist")
        return True

    def _load_existing_figures(self, task: OrdinationTask, output_dir: Path) -> Dict[str, Any]:
        print("Not supported")

    def _skip_and_load_existing(self, task: OrdinationTask, task_output_dir: Path):
        print("Not supported")

    def _calculate_optimal_workers(self) -> int:
        """Calculate optimal number of worker threads.
        For I/O bound tasks with some CPU computation, use more threads,
        but cap at reasonable limit to avoid resource contention."""
        cpu_count = os.cpu_count() or 1
        return min(6, max(2, cpu_count // 2 + 1))

    def run(self) -> None:
        """Run ordination analysis with optimized parallel processing."""
        if not self.tasks:
            return
            
        with get_progress_bar() as progress:
            desc = "Running beta diversity module"
            task_id = progress.add_task(_format_task_desc(desc), total=len(self.tasks))
            
            max_workers = self._calculate_optimal_workers()
            self.log(f"Using {max_workers} worker threads")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks at once 
                future_to_task = {executor.submit(self._run_single_ordination, 
                                                  task, self.output_dir, 
                                                  progress): task for task in self.tasks}
    
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
                            errors.append(e)
                            self.log(f"Ordination failed for {task}: {str(e)}; "
                                     f"Traceback: {traceback.format_exc()}")
                        finally:
                            progress.update(task, advance=1)
                except TimeoutError:
                    logger.warning("Ordination timeout - proceeding with completed results")
                
                if errors: # Log summary of errors if any
                    logger.warning(f"Completed with {len(errors)} errors out of {len(self.tasks)} tasks")
                
            progress.update(task_id, description=_format_task_desc(desc))
        self.log("Ordination completed")

    def _store_results(self, task, result: Any, figures: Dict) -> None:
        """Store ordination results efficiently."""
        self.results[task.table_type][task.level][task.method] = result
        self.results[task.table_type][task.level]['figures'][task.method] = figures

    
        
    def _run_single_ordination(self, task: OrdinationTask, progress: Any) -> Optional[Tuple]:
        """Run a single ordination task with optimized error handling."""
        method_desc = (
            f"{task.table_type.replace('_', ' ').title()} ({task.level.title()})"
            f" → {self.TestConfig[task.method].name}"  
        )
        # Prepare output directory
        output_dir = self.project_dir.final / 'ordination' / task.table_type / task.level / task.method
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create progress task for this method
        method_task = progress.add_task(_format_task_desc(method_desc), total=1)
        
        try:
            '''
            # Check if we should skip and load existing figures
            if self._should_skip_existing(task, output_dir):
                result_tuple = self._skip_and_load_existing(task, task_output_dir)
                # If we have a valid result tuple with figures, return it immediately
                if result_tuple and result_tuple[4] is not None:  # Check if figures are not None
                    self.log(f"Returning existing figures for {task}")
                    self.log(result_tuple[4])
                    return result_tuple
                # If we have figure paths but couldn't load them, continue to calculation
                self.log_ok(f"No figures loaded for {task}, proceeding with calculation")
            '''
            result, figures = self._run_test(task, output_dir)
        except Exception as e:
            logger.error(f"Ordination {task} failed: {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            result, figures = None, None
        finally:
            progress.update(method_task, completed=1, visible=False)
        
        return task.table_type, task.level, task.method, result, figures
            
    def _get_method_params(self, task: OrdinationTask) -> Dict:
        """Get method-specific parameters efficiently."""
        params = {}
        if task.method == "pcoa":
            table_config = self.config['ordination']['tables'].get(task.table_type, {})
            params["metric"] = table_config.get("pcoa_metric", "braycurtis")
        elif task.method in ("tsne", "umap"):
            params["n_jobs"] = 1  # Thread safety
        return params

    def _run_test(self, task: OrdinationTask, output_dir: Union[str, Path]) -> Tuple:
        method_config = self.TestConfig[task.method]
        table, metadata = self._fetch_data(task.table_type, task.level)
        table, metadata = align_table_and_meta(table, metadata)
        result = self._calculate(task, method_config, table)
        if not result:
            return None, None
        figures = self._plot(result, task, method_config, metadata, output_dir)
        return result, figures
      
    def _calculate(self, task: OrdinationTask, method_config: OrdinationConfig, table: Table):
        method_params = self._get_method_params(task)
        try:
            if task.method in ("tsne", "umap"): # Thread-safe execution for UMAP/t-SNE
                self.log(f"Acquiring lock for {task.method}")
                with umap_lock:
                    os.environ['NUMBA_NUM_THREADS'] = '1'
                    result = method_config.func(table=table, **method_params)
            else:
                result = method_config.func(table=table, **method_params)
        except Exception as e:
            logger.error(f"Failed {task}: {e}")
            self.log(f"Traceback: {traceback.format_exc()}")
            return None
        return result

    def _plot(
        self, 
        result: Any, 
        task: OrdinationTask, 
        method_config: OrdinationConfig, 
        metadata: pd.DataFrame, 
        output_dir: Union[str, Path]
    ):
        """Generate figures."""
        figures = {}
        
        valid_color_cols = [col for col in self.color_columns if col in metadata.columns]
        if not valid_color_cols:
            logger.warning(f"No valid color columns found for {task}")
            return {}
        
        # Base plot parameters
        base_params = {
            "metadata": metadata,
            "ordination_type": method_config.name,
            "symbol_col": self.symbol_col,
            "dimensions": (1, 2),
            "transformation": task.table_type,
            "output_dir": output_dir
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
            base_params["df"] = result['components']

        # Generate figures for each valid color column
        for color_col in valid_color_cols:
            try:
                self.log(f"Generating figure for {task} with color column: {color_col}")
                plot_params = {**base_params, "color_col": color_col}
                fig = beta_diversity_plot(plot_params)
                figures[color_col] = fig
            except Exception as e:
                logger.warning(f"Failed to generate figure for {task} with color {color_col}: {e}")
                logger.debug(f"Traceback: {traceback.format_exc()}")
                continue
        return figures


def run_beta_diversity(
    config: Dict, 
    project_dir: Union[ProjectDir, SubDirs],
    metadata: pd.DataFrame,
    tables: Dict[str, Dict[str, Table]],
    group_columns: List[str]
):
    results = {}
    for group_column in group_columns:
        beta = Ordination(
            config=config, 
            project_dir=project_dir, 
            metadata=metadata, 
            tables=tables, 
            group_column=group_column['name']
        )
        beta.run()
        results[group_column['name']] = beta.results
    return results
    
        
