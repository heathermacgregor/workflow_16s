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


def get_required_files(output_subdir: Path, method: str, n_features: int) -> List[Path]:
    """Return the list of required output files for a given method."""
    figs = ["bar", "beeswarm", "heatmap", "force"]
    base = output_subdir / method
    return [
        base / "best_model.cbm",
        base / "feature_importances.csv",
        base / "grid_search_results.csv",
        base / "best_confusion_matrix.html",
        base / "best_roc_curve.html",
        base / "best_precision_recall_curve.html",
        *[
            base / "figs" / f"shap.summary.{fig}.{n_features}.html"
            for fig in figs
        ],
    ]


def load_html_files(fig_paths: Dict[str, Path]) -> Dict[str, Optional[str]]:
    """Load a dictionary of paths into a dictionary of HTML strings."""
    return {k: (p.read_text() if p.exists() else None) for k, p in fig_paths.items()}


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

        for table_type, levels in self.tables.items():
            table_type_config = table_config.get(table_type, {})
            if not table_type_config.get("enabled", False):
                continue

            enabled_levels = table_type_config.get("levels", list(levels.keys()))
            enabled_methods = table_type_config.get("methods", ["rfe"])

            tasks.extend(
                (table_type, level, method)
                for level, method in product(enabled_levels, enabled_methods)
            )
        return tasks

    def run(self, output_dir: Optional[Path] = None) -> None:
        """Run CatBoost feature selection tasks."""
        with get_progress_bar() as progress:
            cb_desc = "Running CatBoost feature selection"
            cb_task = progress.add_task(_format_task_desc(cb_desc), total=len(self.tasks))

            for table_type, level, method in self.tasks:
                method_desc = (
                    f"{table_type.replace('_', ' ').title()} ({level.title()})"
                    f" → {method.title()}"
                )
                progress.update(cb_task, description=_format_task_desc(method_desc))

                # Init dict storage
                _init_dict_level(self.models, table_type, level)
                data_storage = self.models[table_type][level]

                # Prepare output dir
                output_subdir = output_dir / "ml" / self.group_column / table_type / level
                output_subdir.mkdir(parents=True, exist_ok=True)

                try:
                    if self.debug_mode:
                        time.sleep(3)
                        return

                    if table_type == "clr_transformed" and method == "chi_squared":
                        logger.warning(
                            "Skipping chi_squared feature selection for CLR data."
                        )
                        data_storage[method] = None
                        continue

                    # Required files
                    required_files = get_required_files(
                        output_subdir, method, self.n_top_features
                    )
                    all_files_exist = all(f.exists() for f in required_files)
                    try_to_load_old = self.config.get("ml", {}).get("load_old", True)

                    if all_files_exist and try_to_load_old:
                        logger.info(
                            f"Loading existing results for {table_type}/{level}/{method}"
                        )

                        model = cb.CatBoostClassifier()
                        model.load_model(str(output_subdir / method / "best_model.cbm"))

                        feature_importances = pd.read_csv(
                            output_subdir / method / "feature_importances.csv"
                        )
                        grid_search_results = pd.read_csv(
                            output_subdir / method / "grid_search_results.csv"
                        )

                        fig_paths = {
                            "confusion_matrix": output_subdir
                            / method
                            / "best_confusion_matrix.html",
                            "roc": output_subdir / method / "best_roc_curve.html",
                            "prc": output_subdir
                            / method
                            / "best_precision_recall_curve.html",
                            "shap_summary_bar": output_subdir
                            / method
                            / "figs"
                            / f"shap.summary.bar.{self.n_top_features}.html",
                            "shap_summary_beeswarm": output_subdir
                            / method
                            / "figs"
                            / f"shap.summary.beeswarm.{self.n_top_features}.html",
                            "shap_summary_heatmap": output_subdir
                            / method
                            / "figs"
                            / f"shap.summary.heatmap.{self.n_top_features}.html",
                            "shap_summary_force": output_subdir
                            / method
                            / "figs"
                            / f"shap.summary.force.{self.n_top_features}.html",
                        }

                        figures = load_html_files(fig_paths)
                        plotly_figures = {
                            key: (
                                html_to_plotly(content) if content is not None else None
                            )
                            for key, content in figures.items()
                        }

                        model_result = {
                            "model": model,
                            "feature_importances": feature_importances,
                            "grid_search_results": grid_search_results,
                            "figures": plotly_figures,
                        }

                    else:
                        logger.info(f"Running model for {table_type}/{level}/{method}")

                        table = self.tables[table_type][level]
                        metadata = self.metadata[table_type][level]
                        X = table_to_df(table)
                        X.index = X.index.str.lower()
                        y = metadata.set_index("#sampleid")[[self.group_column]]
                        y.index = y.index.astype(str).str.lower()
                        idx = X.index.intersection(y.index)
                        X, y = X.loc[idx], y.loc[idx]

                        use_perm_importance = (
                            False
                            if method == "select_k_best"
                            else self.permutation_importance
                        )

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

                    data_storage[method] = model_result
                    if not any(model_result["figures"].values()):
                        logger.warning(
                            f"No figures generated for {table_type}/{level}/{method}"
                        )

                except Exception as e:
                    logger.error(
                        f"Model training failed for {table_type}/{level}/{method}: {e}"
                    )
                    self.models[table_type][level][method] = None

                finally:
                    progress.update(cb_task, advance=1)

            progress.update(cb_task, description=_format_task_desc(cb_desc))
