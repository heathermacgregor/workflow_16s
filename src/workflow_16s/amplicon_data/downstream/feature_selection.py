# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import json
import logging
import time
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Third-Party Imports
import catboost as cb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from biom.table import Table
from bs4 import BeautifulSoup

# Local Imports
from workflow_16s import constants
from workflow_16s.amplicon_data.helpers import _init_dict_level
from workflow_16s.models.feature_selection import catboost_feature_selection
from workflow_16s.utils.data import table_to_df
from workflow_16s.utils.progress import get_progress_bar, _format_task_desc

# ========================== INITIALISATION & CONFIGURATION ========================== #

logger = logging.getLogger("workflow_16s")

# =================================== FUNCTIONS ====================================== #

def html_to_plotly(html_content: str) -> go.Figure:
    """Convert embedded Plotly JSON inside an HTML string into a Plotly Figure."""
    soup = BeautifulSoup(html_content, "html.parser")
    script_tag = soup.find("script", type="application/json")
    if not script_tag:
        raise ValueError("Plotly JSON data not found in HTML")

    fig_json = json.loads(script_tag.string)
    return go.Figure(fig_json)


def get_required_files(output_subdir: Path, method: str, n_features: int, verbose: bool = False) -> List[Path]:
    """Return the list of required output files for a given method."""
    if verbose:
        logger.info(f"Searching for required files in: {output_subdir}")
        logger.info(f"Method: {method}, Features: {n_features}")
    
    figs = ["bar", "beeswarm", "heatmap", "force"]
    base = output_subdir / method
    
    # Core model files
    core_files = [
        base / "best_model.cbm",
        base / "feature_importances.csv",
        base / "grid_search_results.csv",
        base / "best_confusion_matrix.html",
        base / "best_roc_curve.html",
        base / "best_precision_recall_curve.html",
    ]
    
    # SHAP figure files
    shap_files = [
        base / "figs" / f"shap.summary.{fig}.{n_features}.html"
        for fig in figs
    ]
    
    required_files = core_files + shap_files
    
    if verbose:
        logger.info(f"Total required files: {len(required_files)}")
        logger.info("Required file list:")
        for i, file_path in enumerate(required_files, 1):
            logger.info(f"  {i:2d}. {file_path}")
    
    return required_files


def check_file_existence(required_files: List[Path], verbose: bool = False) -> Tuple[bool, Dict[str, bool]]:
    """Check existence of required files with detailed logging."""
    file_status = {}
    missing_files = []
    existing_files = []
    
    if verbose:
        logger.info(f"Checking existence of {len(required_files)} required files...")
    
    for file_path in required_files:
        exists = file_path.exists()
        file_status[str(file_path)] = exists
        
        if exists:
            existing_files.append(file_path)
            if verbose:
                file_size = file_path.stat().st_size if file_path.is_file() else 0
                logger.info(f"  ✓ Found: {file_path} ({file_size:,} bytes)")
        else:
            missing_files.append(file_path)
            if verbose:
                logger.info(f"  ✗ Missing: {file_path}")
    
    all_exist = len(missing_files) == 0
    
    if verbose:
        logger.info(f"File existence summary: {len(existing_files)}/{len(required_files)} files found")
        if missing_files:
            logger.warning(f"Missing {len(missing_files)} required files:")
            for missing_file in missing_files:
                logger.warning(f"  - {missing_file}")
        else:
            logger.info("All required files are present!")
    
    return all_exist, file_status


def load_html_files(fig_paths: Dict[str, Path], verbose: bool = False) -> Dict[str, Optional[str]]:
    """Load a dictionary of paths into a dictionary of HTML strings."""
    if verbose:
        logger.info(f"Loading {len(fig_paths)} HTML files...")
    
    result = {}
    loaded_count = 0
    failed_count = 0
    
    for key, path in fig_paths.items():
        try:
            if path.exists():
                content = path.read_text()
                result[key] = content
                loaded_count += 1
                if verbose:
                    content_size = len(content)
                    logger.info(f"  ✓ Loaded {key}: {path} ({content_size:,} characters)")
            else:
                result[key] = None
                failed_count += 1
                if verbose:
                    logger.warning(f"  ✗ File not found for {key}: {path}")
        except Exception as e:
            result[key] = None
            failed_count += 1
            if verbose:
                logger.error(f"  ✗ Failed to load {key} from {path}: {e}")
    
    if verbose:
        logger.info(f"HTML loading summary: {loaded_count} loaded, {failed_count} failed/missing")
    
    return result


def search_for_existing_results(output_subdir: Path, method: str, n_features: int, verbose: bool = False) -> Dict[str, Any]:
    """Search for and validate existing model results with detailed logging."""
    if verbose:
        logger.info(f"Searching for existing results in: {output_subdir / method}")
    
    search_results = {
        "model_file": None,
        "feature_importances_file": None,
        "grid_search_file": None,
        "figure_files": {},
        "all_files_found": False,
        "missing_files": [],
        "total_files_checked": 0
    }
    
    # Check if method directory exists
    method_dir = output_subdir / method
    if not method_dir.exists():
        if verbose:
            logger.warning(f"Method directory does not exist: {method_dir}")
        return search_results
    
    if verbose:
        logger.info(f"Method directory found: {method_dir}")
    
    # Get required files
    required_files = get_required_files(output_subdir, method, n_features, verbose=verbose)
    search_results["total_files_checked"] = len(required_files)
    
    # Check file existence
    all_exist, file_status = check_file_existence(required_files, verbose=verbose)
    search_results["all_files_found"] = all_exist
    search_results["missing_files"] = [f for f, exists in file_status.items() if not exists]
    
    # Specifically identify key files
    model_file = method_dir / "best_model.cbm"
    feature_file = method_dir / "feature_importances.csv"
    grid_file = method_dir / "grid_search_results.csv"
    
    if model_file.exists():
        search_results["model_file"] = model_file
        if verbose:
            logger.info(f"Model file found: {model_file}")
    
    if feature_file.exists():
        search_results["feature_importances_file"] = feature_file
        if verbose:
            logger.info(f"Feature importances file found: {feature_file}")
    
    if grid_file.exists():
        search_results["grid_search_file"] = grid_file
        if verbose:
            logger.info(f"Grid search results file found: {grid_file}")
    
    # Check figure files
    fig_paths = {
        "confusion_matrix": method_dir / "best_confusion_matrix.html",
        "roc": method_dir / "best_roc_curve.html",
        "prc": method_dir / "best_precision_recall_curve.html",
        "shap_summary_bar": method_dir / "figs" / f"shap.summary.bar.{n_features}.html",
        "shap_summary_beeswarm": method_dir / "figs" / f"shap.summary.beeswarm.{n_features}.html",
        "shap_summary_heatmap": method_dir / "figs" / f"shap.summary.heatmap.{n_features}.html",
        "shap_summary_force": method_dir / "figs" / f"shap.summary.force.{n_features}.html",
    }
    
    search_results["figure_files"] = fig_paths
    
    if verbose:
        logger.info(f"Search complete. Found {len([f for f in file_status.values() if f])}/{len(file_status)} files")
    
    return search_results


# ================================== CLASS ================================== #

class FeatureSelection:
    """Feature selection class"""

    def __init__(
        self,
        config: Dict,
        metadata: pd.DataFrame,
        tables: Dict[str, Dict[str, Table]],
        verbose: bool = False,
        debug_mode: bool = True,
    ):
        self.config = config
        ml_config = self.config.get("ml", {})

        self.group_column = config.get("group_column", constants.DEFAULT_GROUP_COLUMN)
        self.n_top_features = ml_config.get("num_features", 100)
        self.step_size = ml_config.get("step_size", 100)
        self.permutation_importance = ml_config.get("permutation_importance", {}).get(
            "enabled", True
        )
        self.n_threads = ml_config.get("n_threads", 8)
        self.metadata = metadata
        self.tables = tables
        self.verbose = verbose
        self.debug_mode = debug_mode

        self.models: Dict[str, Any] = {}

        if not ml_config.get("enabled", False):
            logger.info("ML feature selection disabled")
            self.tasks: List[Tuple[str, str, str]] = []
            return

        self.tasks = self.get_enabled_tasks()
        if not self.tasks:
            logger.info("No methods for ML feature selection enabled")

    def get_enabled_tasks(self) -> List[Tuple[str, str, str]]:
        """Determine which feature selection tasks should run."""
        ml_config = self.config.get("ml", {})
        table_config = ml_config.get("tables", {})
        tasks: List[Tuple[str, str, str]] = []

        if self.verbose:
            logger.info("Determining enabled feature selection tasks...")

        for table_type, levels in self.tables.items():
            table_type_config = table_config.get(table_type, {})
            if not table_type_config.get("enabled", False):
                if self.verbose:
                    logger.info(f"Table type '{table_type}' is disabled, skipping...")
                continue

            enabled_levels = table_type_config.get("levels", list(levels.keys()))
            enabled_methods = table_type_config.get("methods", ["rfe"])

            if self.verbose:
                logger.info(f"Table type '{table_type}' enabled:")
                logger.info(f"  - Levels: {enabled_levels}")
                logger.info(f"  - Methods: {enabled_methods}")

            table_tasks = list(product(enabled_levels, enabled_methods))
            for level, method in table_tasks:
                tasks.append((table_type, level, method))
                if self.verbose:
                    logger.info(f"  - Added task: {table_type}/{level}/{method}")

        if self.verbose:
            logger.info(f"Total tasks enabled: {len(tasks)}")

        return tasks

    def run(self, output_dir: Optional[Path] = None) -> None:
        """Run CatBoost feature selection tasks."""
        with get_progress_bar() as progress:
            cb_desc = "Running CatBoost feature selection"
            cb_task = progress.add_task(_format_task_desc(cb_desc), total=len(self.tasks))

            for task_idx, (table_type, level, method) in enumerate(self.tasks, 1):
                method_desc = (
                    f"{table_type.replace('_', ' ').title()} ({level.title()})"
                    f" → {method.title()}"
                )
                progress.update(cb_task, description=_format_task_desc(method_desc))

                if self.verbose:
                    logger.info(f"Processing task {task_idx}/{len(self.tasks)}: {table_type}/{level}/{method}")

                # Init dict storage
                _init_dict_level(self.models, table_type, level)
                data_storage = self.models[table_type][level]

                # Prepare output dir
                output_subdir = output_dir / "ml" / self.group_column / table_type / level
                output_subdir.mkdir(parents=True, exist_ok=True)

                if self.verbose:
                    logger.info(f"Output directory: {output_subdir}")

                try:
                    if self.debug_mode:
                        if self.verbose:
                            logger.info("Debug mode enabled, skipping actual processing")
                        time.sleep(3)
                        return

                    if table_type == "clr_transformed" and method == "chi_squared":
                        if self.verbose:
                            logger.warning("Chi-squared not compatible with CLR data - skipping")
                        logger.warning(
                            "Skipping chi_squared feature selection for CLR data."
                        )
                        data_storage[method] = None
                        continue

                    # Search for existing results
                    search_results = search_for_existing_results(
                        output_subdir, method, self.n_top_features, verbose=self.verbose
                    )

                    try_to_load_old = self.config.get("ml", {}).get("load_old", True)

                    if search_results["all_files_found"] and try_to_load_old:
                        if self.verbose:
                            logger.info(f"All required files found - loading existing results")
                            logger.info(f"Loading existing results for {table_type}/{level}/{method}")

                        # Load model
                        model = cb.CatBoostClassifier()
                        model_path = search_results["model_file"]
                        model.load_model(str(model_path))
                        if self.verbose:
                            logger.info(f"Model loaded from: {model_path}")

                        # Load CSV files
                        feature_importances = pd.read_csv(search_results["feature_importances_file"])
                        grid_search_results = pd.read_csv(search_results["grid_search_file"])
                        
                        if self.verbose:
                            logger.info(f"Feature importances shape: {feature_importances.shape}")
                            logger.info(f"Grid search results shape: {grid_search_results.shape}")

                        # Load HTML figures
                        figures = load_html_files(search_results["figure_files"], verbose=self.verbose)
                        
                        # Convert to Plotly figures
                        plotly_figures = {}
                        conversion_count = 0
                        for key, content in figures.items():
                            if content is not None:
                                try:
                                    plotly_figures[key] = html_to_plotly(content)
                                    conversion_count += 1
                                    if self.verbose:
                                        logger.info(f"  ✓ Converted {key} to Plotly figure")
                                except Exception as e:
                                    plotly_figures[key] = None
                                    if self.verbose:
                                        logger.error(f"  ✗ Failed to convert {key}: {e}")
                            else:
                                plotly_figures[key] = None

                        if self.verbose:
                            logger.info(f"Successfully converted {conversion_count}/{len(figures)} figures")

                        model_result = {
                            "model": model,
                            "feature_importances": feature_importances,
                            "grid_search_results": grid_search_results,
                            "figures": plotly_figures,
                        }

                    else:
                        if self.verbose:
                            if not try_to_load_old:
                                logger.info("Loading old results is disabled - will train new model")
                            else:
                                logger.info(f"Missing files detected - will train new model")
                                logger.info(f"Missing files: {search_results['missing_files']}")

                        logger.info(f"Running model for {table_type}/{level}/{method}")

                        table = self.tables[table_type][level]
                        metadata = self.metadata[table_type][level]
                        X = table_to_df(table)
                        X.index = X.index.str.lower()
                        y = metadata.set_index("#sampleid")[[self.group_column]]
                        y.index = y.index.astype(str).str.lower()
                        idx = X.index.intersection(y.index)
                        X, y = X.loc[idx], y.loc[idx]

                        if self.verbose:
                            logger.info(f"Training data shape: X={X.shape}, y={y.shape}")
                            logger.info(f"Common samples: {len(idx)}")

                        use_perm_importance = (
                            False
                            if method == "select_k_best"
                            else self.permutation_importance
                        )

                        if self.verbose:
                            logger.info(f"Using permutation importance: {use_perm_importance}")

                        model_result = catboost_feature_selection(
                            metadata=y,
                            features=X,
                            output_dir=output_subdir,
                            group_col=self.group_column,
                            method=method,
                            n_top_features=self.n_top_features,
                            step_size=self.step_size,
                            use_permutation_importance=use_perm_importance,
                            thread_count=self.n_threads,
                            progress=progress,
                            task_id=cb_task,
                        )

                        if self.verbose:
                            logger.info(f"Model training completed for {table_type}/{level}/{method}")

                    data_storage[method] = model_result
                    
                    # Check if figures were generated
                    figure_count = sum(1 for fig in model_result["figures"].values() if fig is not None)
                    if figure_count == 0:
                        logger.warning(
                            f"No figures generated for {table_type}/{level}/{method}"
                        )
                    elif self.verbose:
                        logger.info(f"Successfully generated {figure_count} figures")

                except Exception as e:
                    logger.error(
                        f"Model training failed for {table_type}/{level}/{method}: {e}"
                    )
                    if self.verbose:
                        logger.exception(f"Detailed error for {table_type}/{level}/{method}:")
                    self.models[table_type][level][method] = None

                finally:
                    progress.update(cb_task, advance=1)
                    if self.verbose:
                        logger.info(f"Completed task {task_idx}/{len(self.tasks)}")

            progress.update(cb_task, description=_format_task_desc(cb_desc))

        if self.verbose:
            logger.info("Feature selection workflow completed")
