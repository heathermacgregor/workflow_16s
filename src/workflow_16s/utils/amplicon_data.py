# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import glob
import logging
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Third‑Party Imports
import pandas as pd
from biom.table import Table

# ================================== LOCAL IMPORTS =================================== #

from workflow_16s.utils.biom import (
    collapse_taxa,
    convert_to_biom,
    export_h5py,
    presence_absence,
)
from workflow_16s.utils.progress import get_progress_bar
from workflow_16s.utils.file_utils import (
    import_merged_table_biom,
    import_merged_meta_tsv,
    filter_and_reorder_biom_and_metadata as update_tables,
)
from workflow_16s.stats.utils import (
    clr_transform_table,
    filter_table,
    merge_table_with_metadata,
    normalize_table,
    table_to_dataframe,
)
from workflow_16s.stats.tests import (
    fisher_exact_bonferroni,
    kruskal_bonferroni,
    mwu_bonferroni,
    ttest,
)
from workflow_16s.stats.beta_diversity import (
    pcoa as calculate_pcoa,
    pca as calculate_pca,
    tsne as calculate_tsne,
    umap as calculate_umap,
)
from workflow_16s.figures.merged import (
    mds as plot_mds,
    pca as plot_pca,
    pcoa as plot_pcoa,
    sample_map_categorical,
)
from workflow_16s.function.faprotax import (
    faprotax_functions_for_taxon,
    get_faprotax_parsed,
)
from workflow_16s.models.feature_selection import (
    catboost_feature_selection,
)

# ========================== INITIALISATION & CONFIGURATION ========================== #

logger = logging.getLogger("workflow_16s")
warnings.filterwarnings("ignore")

# ================================= DEFAULT VALUES =================================== #

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"
DEFAULT_PROGRESS_TEXT_N = 65
DEFAULT_GROUP_COLUMN = "nuclear_contamination_status"
DEFAULT_GROUP_COLUMN_VALUES = [True, False]
DEFAULT_MODE = 'genus'

# ===================================== CLASSES ====================================== #

class _ProcessingMixin:
    """
    Provides reusable methods for processing steps with progress tracking and logging.
    
    Attributes:
        verbose: Flag indicating whether to show detailed progress information.
    """
    
    def _run_processing_step(
        self,
        process_name: str,
        process_func: Callable,
        levels: List[str],
        func_args: tuple,
        get_source: Callable[[str], Table],
        log_template: Optional[str] = None,
        log_action: Optional[str] = None,
    ) -> Dict[str, Table]:
        """
        Executes a processing function across multiple taxonomic levels with progress 
        tracking.
        
        Args:
            process_name:   Name of the process for progress display.
            process_func:   Function to execute for each level.
            levels:         List of taxonomic levels to process.
            func_args:      Additional arguments for process_func.
            get_source:     Function to retrieve input table for a level.
            log_template:   Template string for logging (uses {level} placeholder).
            log_action:     Action description for logging.
        
        Returns:
            Dictionary of processed tables keyed by taxonomic level.
        """
        processed: Dict[str, Table] = {}

        if getattr(self, "verbose", False):
            logger.info(f"{process_name}")
            for level in levels:
                start_time = time.perf_counter()  # More precise timing
                processed[level] = process_func(get_source(level), level, *func_args)
                duration = time.perf_counter() - start_time
                # Only log if we have a template or action
                if log_template or log_action:
                    self._log_level_action(level, log_template, log_action, duration)
        else:
            logger.debug(f"{process_name}")
            with get_progress_bar() as progress:
                parent_task = progress.add_task(
                    process_name,
                    total=len(levels),
                )
                for level in levels:
                    start_time = time.perf_counter()  # More precise timing
                    child_task = progress.add_task(
                        f"Processing {level} level".ljust(DEFAULT_PROGRESS_TEXT_N),
                        parent=parent_task,
                        total=1,
                    )
                    processed[level] = process_func(get_source(level), level, *func_args)
                    duration = time.perf_counter() - start_time

                    # Only log if we have a template or action
                    if log_template or log_action:
                        self._log_level_action(level, log_template, log_action, duration)

                    progress.update(child_task, completed=1)
                    progress.remove_task(child_task)
                    progress.update(parent_task, advance=1)

        return processed

    def _log_level_action(
        self,
        level: str,
        template: Optional[str] = None,
        action: Optional[str] = None,
        duration: Optional[float] = None,
    ) -> None:
        """
        Logs an action for a specific taxonomic level with timing information.
        
        Args:
            level:     Taxonomic level being processed.
            template:  String template with {level} placeholder.
            action:    Action description.
            duration:  Time taken for the action in seconds.
        """
        message = ""
        if template:
            message = template.format(level=level)
        elif action:
            message = f"{level} {action}"

        # Only append duration if we have a message to log
        if message and duration is not None:
            message += f" in {duration:.2f}s"

        if message:
            logger.debug(message)


class StatisticalAnalyzer:
    """
    Performs statistical tests on feature tables to identify significant differences.
    
    Attributes:
        cfg:     Configuration dictionary.
        verbose: Flag for verbose output.
    """
    
    TEST_CONFIG = {
        "fisher": {
            "key": "fisher",
            "func": fisher_exact_bonferroni,
            "name": "Fisher exact (Bonferroni)",
            "effect_col": "proportion_diff",
            "alt_effect_col": "odds_ratio",
        },
        "ttest": {
            "key": "ttest",
            "func": ttest,
            "name": "Student t‑test",
            "effect_col": "mean_difference",
            "alt_effect_col": "cohens_d",
        },
        "mwu_bonferroni": {
            "key": "mwub",
            "func": mwu_bonferroni,
            "name": "Mann–Whitney U (Bonferroni)",
            "effect_col": "effect_size_r",
            "alt_effect_col": "median_difference",
        },
        "kruskal_bonferroni": {
            "key": "kwb",
            "func": kruskal_bonferroni,
            "name": "Kruskal–Wallis (Bonferroni)",
            "effect_col": "epsilon_squared",
            "alt_effect_col": None,
        },
    }

    def __init__(self, cfg: Dict, verbose: bool = False):
        """
        Initializes the StatisticalAnalyzer.
        
        Args:
            cfg:     Configuration dictionary.
            verbose: If True, enables verbose logging.
        """
        self.cfg = cfg
        self.verbose = verbose

    def run_tests(
        self,
        table: Table,
        metadata: pd.DataFrame,
        group_column: str,
        group_values: List[Any],
        enabled_tests: List[str],
    ) -> Dict[str, Any]:
        """
        Runs enabled statistical tests on the feature table.
        
        Args:
            table:         BIOM feature table.
            metadata:      Sample metadata DataFrame.
            group_column:  Column in metadata defining groups.
            group_values:  Values to compare in group_column.
            enabled_tests: List of test names to run.
        
        Returns:
            Dictionary of test results keyed by test identifier.
        """
        results: Dict[str, Any] = {}
        # Pre-align samples once instead of in each test
        table, metadata = update_tables(table, metadata)

        for test_name in enabled_tests:
            if test_name not in self.TEST_CONFIG:
                continue
            cfg = self.TEST_CONFIG[test_name]
            if self.verbose:
                logger.info(f"Running {cfg['name']}...")
            results[cfg["key"]] = cfg["func"](
                table=table,
                metadata=metadata,
                group_column=group_column,
                group_column_values=group_values,
            )
        return results

    def get_effect_size(self, test_name: str, row: pd.Series) -> Optional[float]:
        """
        Extracts effect size from a test result row.
        
        Args:
            test_name: Name of the statistical test.
            row:       Series containing test results.
        
        Returns:
            Effect size value or None if not found.
        """
        if test_name not in self.TEST_CONFIG:
            return None
        cfg = self.TEST_CONFIG[test_name]
        for col in (cfg["effect_col"], cfg["alt_effect_col"]):
            if col and col in row:
                return row[col]
        return None


class Ordination:
    """
    Performs ordination analyses (PCA, PCoA, t-SNE, UMAP) and generates plots.
    
    Attributes:
        cfg:               Configuration dictionary.
        verbose:           Flag for verbose output.
        figure_output_dir: Directory to save generated plots.
        results:           Dictionary of ordination results.
        figures:           Dictionary of generated figures.
        color_columns:     List of metadata columns to use for coloring points.
    """
    
    TEST_CONFIG = {
        "pca": {
            "key": "pca", 
            "func": calculate_pca, 
            "plot_func": plot_pca, 
            "name": "PCA"
        },
        "pcoa": {
            "key": "pcoa", 
            "func": calculate_pcoa, 
            "plot_func": plot_pcoa, 
            "name": "PCoA"
        },
        "tsne": {
            "key": "tsne",
            "func": calculate_tsne,
            "plot_func": plot_mds,
            "name": "t‑SNE",
            "plot_kwargs": {"mode": "TSNE"},
        },
        "umap": {
            "key": "umap",
            "func": calculate_umap,
            "plot_func": plot_mds,
            "name": "UMAP",
            "plot_kwargs": {"mode": "UMAP"},
        },
    }

    def __init__(
        self, 
        cfg: Dict, 
        output_dir: Union[str, Path], 
        verbose: bool = False
    ):
        """
        Initializes the Ordination analyzer.
        
        Args:
            cfg:        Configuration dictionary.
            output_dir: Directory to save generated plots.
            verbose:    If True, enables verbose logging.
        """
        self.cfg = cfg
        self.verbose = verbose
        self.figure_output_dir = Path(output_dir)
        self.results: Dict[str, Any] = {}
        self.figures: Dict[str, Any] = {}
        # Get color columns from config or use default
        self.color_columns = cfg["figures"].get(
            "ordination_color_columns",
            cfg["figures"].get(
                "map_columns",
                ["dataset_name", "nuclear_contamination_status",
                 "env_feature", "env_material", "country"],
            ),
        )

    def run_tests(
        self,
        table: Table,
        metadata: pd.DataFrame,
        symbol_col: str,
        transformation: str,
        enabled_tests: List[str],
        **kwargs,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Runs ordination methods and generates plots for multiple color columns.
        
        Args:
            table:          BIOM feature table.
            metadata:       Sample metadata DataFrame.
            symbol_col:     Column in metadata for point symbols.
            transformation: Data transformation method (e.g., 'normalized').
            enabled_tests:  List of ordination methods to run.
            **kwargs:       Additional keyword arguments for plotting.
        
        Returns:
            Tuple containing:
                - Dictionary of ordination results keyed by method
                - Dictionary of figures keyed by method and color column
        """
        trans_cfg = self.cfg.get("ordination", {}).get(transformation, {})
        tests_to_run = [t for t in enabled_tests if t in self.TEST_CONFIG]
        if not tests_to_run:
            return {}, {}

        table, metadata = update_tables(table, metadata)
        return self._run_without_progress(
            table, metadata, symbol_col, transformation,
            tests_to_run, trans_cfg, kwargs,
        )

    def _run_without_progress(
        self, table, metadata, symbol_col, transformation, tests_to_run, 
        trans_cfg, kwargs,
    ):
        results, figures = {}, {}
        for tname in tests_to_run:
            cfg = self.TEST_CONFIG[tname]
            try:
                res, figs = self._run_ordination_method(
                    cfg, table, metadata, symbol_col, transformation, 
                    trans_cfg, kwargs
                )
                results[cfg["key"]] = res
                figures[cfg["key"]] = figs
            except Exception as e:
                logger.error(f"Failed {tname} for {transformation}: {e}")
                figures[cfg["key"]] = {}
        return results, figures

    def _run_ordination_method(
        self, cfg, table, metadata, symbol_col, transformation, trans_cfg, 
        kwargs
    ):
        """
        Executes a single ordination method and generates plots for all color 
        columns.
        
        Args:
            cfg:            Configuration for the ordination method.
            table:          BIOM feature table.
            metadata:       Sample metadata DataFrame.
            symbol_col:     Column in metadata for point symbols.
            transformation: Data transformation name.
            trans_cfg:      Transformation-specific configuration.
            kwargs:         Additional plotting arguments.
        
        Returns:
            Tuple containing:
                - Ordination result object
                - Dictionary of figures keyed by color column
        """
        method_params = {}
        if cfg["key"] == "pcoa":
            method_params["metric"] = trans_cfg.get("pcoa_metric", "braycurtis")
        
        # Add CPU limiting parameters
        if cfg["key"] in ["tsne", "umap"]:
            cpu_limit = self.cfg.get("ordination", {}).get("cpu_limit", 1)
            method_params["n_jobs"] = cpu_limit

        ord_res = cfg["func"](table=table, **method_params)

        # Generate plots for each color column
        figures = {}
        pkwargs = {**cfg.get("plot_kwargs", {}), **kwargs}
        
        for color_col in self.color_columns:
            if color_col not in metadata.columns:
                logger.warning(f"Color column '{color_col}' not found in metadata")
                continue

            # Set up plot parameters based on method
            if cfg["key"] == "pca":
                pkwargs.update({
                    "components": ord_res["components"],
                    "proportion_explained": ord_res["exp_var_ratio"],
                })
            elif cfg["key"] == "pcoa":
                pkwargs.update({
                    "components": ord_res.samples,
                    "proportion_explained": ord_res.proportion_explained,
                })
            else:  # t-SNE or UMAP
                pkwargs["df"] = ord_res

            fig, _ = cfg["plot_func"](
                metadata=metadata,
                color_col=color_col,
                symbol_col=symbol_col,
                transformation=transformation,
                output_dir=self.figure_output_dir,
                **pkwargs,
            )
            figures[color_col] = fig

        return ord_res, figures


class Plotter:
    """
    Generates various plots for microbiome data analysis.
    
    Attributes:
        cfg:           Configuration dictionary.
        output_dir:    Directory to save generated figures.
        verbose:       Flag for verbose output.
        color_columns: List of metadata columns to use for coloring points.
    """
    
    def __init__(
        self, 
        cfg: Dict, 
        output_dir: Path, 
        verbose: bool = False
    ):
        """
        Initializes the Plotter.
        
        Args:
            cfg:        Configuration dictionary.
            output_dir: Directory to save generated figures.
            verbose:    If True, enables verbose logging.
        """
        self.cfg = cfg
        self.output_dir = output_dir
        self.verbose = verbose
        self.color_columns = cfg["figures"].get(
            "map_columns",
            [
                "dataset_name",
                "nuclear_contamination_status",
                "env_feature",
                "env_material",
                "country",
            ],
        )

    def generate_sample_map(
        self, 
        metadata: pd.DataFrame, 
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generates sample map plots for multiple metadata columns.
        
        Args:
            metadata: Sample metadata DataFrame.
            **kwargs: Additional keyword arguments for plotting.
        
        Returns:
            Dictionary of generated figures keyed by metadata column name.
        """
        valid_columns = [col for col in self.color_columns if col in metadata]
        missing = set(self.color_columns) - set(valid_columns)
        if missing and self.verbose:
            logger.warning(f"Missing columns in metadata: {', '.join(missing)}")

        with get_progress_bar() as progress:
            parent_task = progress.add_task(
                "Generating sample maps...".ljust(DEFAULT_PROGRESS_TEXT_N), 
                total=len(valid_columns)
            )

            figs = {}
            for col in valid_columns:
                child_task = progress.add_task(
                    f"[cyan]Mapping {col}...", 
                    parent=parent_task, total=1
                )
                fig, _ = sample_map_categorical(
                    metadata=metadata,
                    output_dir=self.output_dir,
                    color_col=col,
                    **kwargs,
                )
                figs[col] = fig
                progress.update(child_task, completed=1)
                progress.remove_task(child_task)
                progress.update(parent_task, advance=1)
        return figs


class TopFeaturesAnalyzer:
    """
    Identifies top differentially abundant features based on statistical results.
    
    Attributes:
        cfg:     Configuration dictionary.
        verbose: Flag for verbose output.
    """
    
    def __init__(
        self, 
        cfg: Dict, 
        verbose: bool = False
    ):
        """
        Initializes the TopFeaturesAnalyzer.
        
        Args:
            cfg:     Configuration dictionary.
            verbose: If True, enables verbose logging.
        """
        self.cfg = cfg
        self.verbose = verbose

    def analyze(
        self,
        stats_results: Dict[str, Dict[str, Dict[str, pd.DataFrame]]],
        group_column: str,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Identifies top contaminated and pristine features from statistical results.
        
        Args:
            stats_results: Nested dictionary of statistical test results.
            group_column:  Metadata column used for group comparisons.
        
        Returns:
            Tuple containing:
                - List of top contaminated features
                - List of top pristine features
        """
        san = StatisticalAnalyzer(self.cfg, self.verbose)
        all_features = []

        # Collect all significant features efficiently
        for table_type, tests in stats_results.items():
            for test_name, test_results in tests.items():
                for level, df in test_results.items():
                    sig_df = df[df["p_value"] < 0.05].copy()
                    if sig_df.empty:
                        continue

                    # Vectorized effect size calculation
                    sig_df["effect"] = sig_df.apply(
                        lambda row: san.get_effect_size(test_name, row), axis=1
                    )
                    sig_df = sig_df.dropna(subset=["effect"])

                    # Collect features in a list
                    for _, row in sig_df.iterrows():
                        all_features.append({
                            "feature": row["feature"],
                            "level": level,
                            "table_type": table_type,
                            "test": test_name,
                            "effect": row["effect"],
                            "p_value": row["p_value"],
                            "effect_dir": "positive" if row["effect"] > 0 else "negative",
                        })

        # Efficient sorting and filtering
        cont_feats = [f for f in all_features if f["effect"] > 0]
        pris_feats = [f for f in all_features if f["effect"] < 0]

        cont_feats.sort(key=lambda d: (-d["effect"], d["p_value"]))
        pris_feats.sort(key=lambda d: (d["effect"], d["p_value"]))  # More negative is stronger

        return cont_feats[:100], pris_feats[:100]


class _DataLoader(_ProcessingMixin):
    """
    Loads and processes microbiome data from BIOM files and metadata.
    
    Attributes:
        cfg:         Configuration dictionary.
        project_dir: Project directory structure.
        mode:        Analysis mode ('asv' or 'genus').
        verbose:     Flag for verbose output.
        meta:        Loaded metadata DataFrame.
        table:       Loaded BIOM feature table.
    """
    
    MODE_CONFIG = {
        "asv": ("table", "asv"), 
        "genus": ("table_6", "l6")
    }

    def __init__(
        self, 
        cfg: Dict, 
        project_dir: Any, 
        mode: str, 
        verbose: bool = False
    ):
        """
        Initializes the DataLoader and loads data.
        
        Args:
            cfg:         Configuration dictionary.
            project_dir: Project directory structure.
            mode:        Analysis mode ('asv' or 'genus').
            verbose:     If True, enables verbose logging.
        """
        self.cfg, self.project_dir, self.mode, self.verbose = cfg, project_dir, mode, verbose
        self._validate_mode()
        self._load_metadata()
        self._load_biom_table()
        self._filter_and_align()

    # Public after run
    meta: pd.DataFrame
    table: Table

    def _validate_mode(self) -> None:
        """Validates the analysis mode."""
        if self.mode not in self.MODE_CONFIG:
            raise ValueError(f"Invalid mode: {self.mode}")

    def _get_metadata_paths(self) -> List[Path]:
        """Retrieves paths to metadata files."""
        paths: List[Path] = []
        for bi in self._get_biom_paths():
            ds_dir = bi.parent if bi.is_file() else bi
            tail = ds_dir.parts[-6:-1]
            mp = Path(self.project_dir.metadata_per_dataset).joinpath(*tail, "sample-metadata.tsv")
            if mp.exists():
                paths.append(mp)
        if self.verbose:
            logger.info(f"Found {RED}{len(paths)}{RESET} metadata files")
        return paths

    def _load_metadata(self) -> None:
        """Loads and merges metadata from multiple files."""
        paths = self._get_metadata_paths()
        self.meta = import_merged_meta_tsv(paths, None, self.verbose)

    def _get_biom_paths(self) -> List[Path]:
        """Retrieves paths to BIOM feature tables."""
        table_dir, _ = self.MODE_CONFIG[self.mode]
        pattern = "/".join([
            "*", "*", "*", "*", "FWD_*_REV_*", table_dir, "feature-table.biom",
        ])
        globbed = glob.glob(str(Path(
            self.project_dir.qiime_data_per_dataset
        ) / pattern), recursive=True)
        if self.verbose:
            logger.info(f"Found {RED}{len(globbed)}{RESET} feature tables")
        return [Path(p) for p in globbed]

    def _load_biom_table(self) -> None:
        """Loads and merges BIOM feature tables."""
        biom_paths = self._get_biom_paths()
        if not biom_paths:
            raise FileNotFoundError("No BIOM files found")
        self.table = import_merged_table_biom(biom_paths, "table", self.verbose)

    def _filter_and_align(self) -> None:
        """Filters and aligns feature table with metadata."""
        orig_n = self.table.shape[1]
        self.table, self.meta = update_tables(self.table, self.meta, "#sampleid")
        ftype = "genera" if self.mode == "genus" else "ASVs"
        logger.info(
            f"Loaded metadata: "
            f"{RED}{self.meta.shape[0]}{RESET} samples × "
            f"{RED}{self.meta.shape[1]}{RESET} cols"
        )
        logger.info(
            f"Loaded feature table: "
            f"{RED}{self.table.shape[1]} ({orig_n}){RESET} samples × "
            f"{RED}{self.table.shape[0]}{RESET} {ftype}"
        )


class _TableProcessor(_ProcessingMixin):
    """
    Processes feature tables through various transformations and taxonomical 
    collapses.
    
    Attributes:
        cfg:               Configuration dictionary.
        table:             Input BIOM feature table.
        mode:              Analysis mode ('asv' or 'genus').
        meta:              Sample metadata DataFrame.
        figure_output_dir: Directory for output figures.
        project_dir:       Project directory structure.
        verbose:           Flag for verbose output.
        tables:            Dictionary of processed tables.
    """
    
    def __init__(
        self,
        cfg: Dict,
        table: Table,
        mode: str,
        meta: pd.DataFrame,
        figure_output_dir: Path,
        project_dir: Any,
        verbose: bool,
    ) -> None:
        """
        Initializes the TableProcessor and runs processing pipeline.
        
        Args:
            cfg:               Configuration dictionary.
            table:             Input BIOM feature table.
            mode:              Analysis mode ('asv' or 'genus').
            meta:              Sample metadata DataFrame.
            figure_output_dir: Directory for output figures.
            project_dir:       Project directory structure.
            verbose:           If True, enables verbose logging.
        """
        self.cfg, self.mode, self.verbose = cfg, mode, verbose
        self.meta = meta
        self.figure_output_dir = figure_output_dir
        self.project_dir = project_dir
        self.tables: Dict[str, Dict[str, Table]] = {"raw": {mode: table}}
        self._apply_preprocessing()
        self._collapse_taxa()
        self._create_presence_absence()
        self._save_tables()

    def _apply_preprocessing(self) -> None:
        """Applies filtering, normalization, and CLR transformation."""
        feat_cfg = self.cfg["features"]
        table = self.tables["raw"][self.mode]

        # Pipeline processing to avoid intermediate copies
        if feat_cfg["filter"]:
            table = filter_table(table)
            self.tables.setdefault("filtered", {})[self.mode] = table

        if feat_cfg["normalize"]:
            table = normalize_table(table, axis=1)
            self.tables.setdefault("normalized", {})[self.mode] = table

        if feat_cfg["clr_transform"]:
            table = clr_transform_table(table)
            self.tables.setdefault("clr_transformed", {})[self.mode] = table

    def _collapse_taxa(self) -> None:
        """Collapses feature tables to different taxonomic levels."""
        levels = ["phylum", "class", "order", "family", "genus"]
        for table_type in list(self.tables.keys()):
            base_table = self.tables[table_type][self.mode]
            self.tables[table_type] = self._run_processing_step(
                f"Collapsing taxonomy for {table_type.replace('_', ' ')} tables...".ljust(DEFAULT_PROGRESS_TEXT_N),
                collapse_taxa,
                levels,
                (),
                lambda level, _table=base_table: _table,
                log_template=f"Collapsed {table_type} to {{level}}",
            )

    def _create_presence_absence(self) -> None:
        """Creates presence/absence versions of feature tables."""
        if not self.cfg["features"]["presence_absence"]:
            return
        levels = ["phylum", "class", "order", "family", "genus"]
        raw_table = self.tables["raw"][self.mode]
        self.tables["presence_absence"] = self._run_processing_step(
            "Converting to presence/absence...".ljust(DEFAULT_PROGRESS_TEXT_N),
            presence_absence,
            levels,
            (),
            lambda level: raw_table,
        )

    def _save_tables(self) -> None:
        """Saves processed tables to disk in BIOM format."""
        base = Path(self.project_dir.data) / "merged" / "table"
        base.mkdir(parents=True, exist_ok=True)

        # Prepare all export tasks
        export_tasks = []
        for table_type, levels in self.tables.items():
            tdir = base / table_type
            tdir.mkdir(parents=True, exist_ok=True)
            for level, table in levels.items():
                out = tdir / level / "feature-table.biom"
                out.parent.mkdir(parents=True, exist_ok=True)
                export_tasks.append((table, out, table_type, level))

        # Parallel export
        with ThreadPoolExecutor() as executor:
            futures = []
            for table, out, table_type, level in export_tasks:
                futures.append(executor.submit(export_h5py, table, out))

            # Wait for completion
            for future in futures:
                future.result()


class _AnalysisManager(_ProcessingMixin):
    """
    Manages the analysis pipeline including statistics, ordination, and 
    machine learning.
    
    Attributes:
        cfg:                         Configuration dictionary.
        tables:                      Processed feature tables.
        meta:                        Sample metadata DataFrame.
        figure_output_dir:           Directory for output figures.
        verbose:                     Flag for verbose output.
        stats:                       Statistical test results.
        ordination:                  Ordination analysis results.
        models:                      Machine learning models.
        figures:                     Generated figures.
        top_contaminated_features:   Top features associated with 
                                     contamination.
        top_pristine_features:       Top features associated with 
                                     pristine samples.
        faprotax_enabled:            Flag indicating if FAPROTAX is 
                                     enabled.
        fdb:                         FAPROTAX database.
        _faprotax_cache:             Cache for FAPROTAX annotations.
    """
    
    def __init__(
        self,
        cfg: Dict,
        tables: Dict[str, Dict[str, Table]],
        meta: pd.DataFrame,
        figure_output_dir: Path,
        verbose: bool,
        faprotax_enabled: bool = False,
        fdb: Optional[Dict] = None,
    ) -> None:
        """
        Initializes the AnalysisManager and runs the analysis pipeline.
        
        Args:
            cfg:               Configuration dictionary.
            tables:            Processed feature tables.
            meta:              Sample metadata DataFrame.
            figure_output_dir: Directory for output figures.
            verbose:           If True, enables verbose logging.
            faprotax_enabled:  If True, enables FAPROTAX functional 
                               annotation.
            fdb:               Loaded FAPROTAX database.
        """
        self.cfg, self.tables, self.meta, self.verbose = cfg, tables, meta, verbose
        self.figure_output_dir = figure_output_dir
        self.stats: Dict[str, Any] = {}
        self.ordination: Dict[str, Any] = {}
        self.models: Dict[str, Any] = {}
        self.figures: Dict[str, Any] = {}  # Initialize figures dictionary
        self.top_contaminated_features: List[Dict] = []
        self.top_pristine_features: List[Dict] = []
        self.faprotax_enabled, self.fdb = faprotax_enabled, fdb
        self._faprotax_cache = {}

        # Process in stages and clear intermediates
        self._run_statistical_tests()
        stats_copy = deepcopy(self.stats)

        self._identify_top_features(stats_copy)
        del stats_copy  # Free memory

        self._run_ordination()

        # Keep only necessary tables for ML
        #ml_table_types = {"normalized", "clr_transformed"}
        ml_table_types = {"clr_transformed"}
        ml_tables = {
            t: d for t, d in self.tables.items() if t in ml_table_types
        }
        self._run_ml_feature_selection(ml_tables)
        self._compare_top_features()
        del ml_tables

        # Add FAPROTAX annotations only to top features
        if self.faprotax_enabled and self.top_contaminated_features:
            self._annotate_top_features()

    def _get_cached_faprotax(
        self, 
        taxon: str
    ) -> List[str]:
        """
        Retrieves FAPROTAX functions for a taxon, using cache if available.
        
        Args:
            taxon: Taxonomic string to look up.
        
        Returns:
            List of functional annotations.
        """
        if taxon not in self._faprotax_cache:
            self._faprotax_cache[taxon] = faprotax_functions_for_taxon(
                taxon, self.fdb, include_references=False
            )
        return self._faprotax_cache[taxon]

    def _annotate_top_features(self) -> None:
        """Batch process annotations to minimize DB lookups"""
        all_taxa = {
            f["feature"] for f in self.top_contaminated_features + self.top_pristine_features
        }

        # Batch lookup
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self._get_cached_faprotax, all_taxa))

        taxon_map = dict(zip(all_taxa, results))

        for feat in self.top_contaminated_features:
            feat["faprotax_functions"] = taxon_map.get(feat["feature"], [])

        for feat in self.top_pristine_features:
            feat["faprotax_functions"] = taxon_map.get(feat["feature"], [])

    def _run_statistical_tests(self) -> None:
        """Runs statistical tests on all tables and levels."""
        grp_col = self.cfg.get("group_column", DEFAULT_GROUP_COLUMN)
        grp_vals = self.cfg.get("group_values", [True, False])
        san = StatisticalAnalyzer(self.cfg, self.verbose)

        # Calculate total tests
        total_tests = 0
        for table_type, levels in self.tables.items():
            tests_config = self.cfg["stats"].get(table_type, {})
            enabled_for_table_type = [t for t, flag in tests_config.items() if flag]
            total_tests += len(levels) * len(enabled_for_table_type)

        with get_progress_bar() as progress:
            main_task = progress.add_task(
                "Statistical testing...".ljust(DEFAULT_PROGRESS_TEXT_N), 
                total=total_tests
            )

            for table_type, levels in self.tables.items():
                tests_config = self.cfg["stats"].get(table_type, {})
                enabled_for_table_type = [t for t, flag in tests_config.items() if flag]
                self.stats[table_type] = {}

                for level, table in levels.items():
                    # Align table/metadata once per level
                    table_aligned, meta_aligned = update_tables(table, self.meta)
                    
                    for test_name in enabled_for_table_type:
                        if test_name not in san.TEST_CONFIG:
                            continue
                        cfg = san.TEST_CONFIG[test_name]
                        
                        # Create child task with fixed description
                        child_task = progress.add_task(
                            f"{table_type} | {level} | {cfg['name']}".ljust(DEFAULT_PROGRESS_TEXT_N),
                            parent=main_task,
                            total=1
                        )
                        
                        try:
                            # Run the test directly
                            result = cfg["func"](
                                table=table_aligned,
                                metadata=meta_aligned,
                                group_column=grp_col,
                                group_column_values=grp_vals,
                            )
                            self.stats[table_type].setdefault(cfg['key'], {})[level] = result
                        except Exception as e:
                            logger.error(f"Test failed: {e}")
                        finally:
                            # Complete and remove child task
                            progress.update(child_task, completed=1)
                            progress.remove_task(child_task)
                            progress.advance(main_task)  # Update main task after each test

    def _identify_top_features(self, stats_results: Dict) -> None:
        """Identifies top features from statistical results."""
        tfa = TopFeaturesAnalyzer(self.cfg, self.verbose)
        self.top_contaminated_features, self.top_pristine_features = tfa.analyze(
            stats_results, DEFAULT_GROUP_COLUMN
        )

        if self.verbose:
            logger.info(
                f"Found {len(self.top_contaminated_features)} " 
                f"top contaminated features"
            )
            logger.info(
                f"Found {len(self.top_pristine_features)} " 
                f"top pristine features"
            )

    def _run_ordination(self) -> None:
        """Runs ordination analyses on all tables and levels."""
        KNOWN_METHODS = ["pca", "pcoa", "tsne", "umap"]
        # Calculate total tasks
        total_tasks = 0
        for table_type, levels in self.tables.items():
            ord_config = self.cfg.get("ordination", {}).get(table_type, {})
            enabled_methods = [m for m in KNOWN_METHODS if ord_config.get(m, False)]
            total_tasks += len(levels) * len(enabled_methods)
        
        if not total_tasks:
            return

        # Initialize structures
        self.ordination = {tt: {} for tt in self.tables}
        self.figures = {tt: {} for tt in self.tables}

        with get_progress_bar() as progress:
            main_task = progress.add_task(
                "Ordination analysis...".ljust(DEFAULT_PROGRESS_TEXT_N), 
                total=total_tasks
            )
            
            # Use thread pool with limited workers
            max_workers = self.cfg.get("ordination", {}).get("max_workers", 1)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for table_type, levels in self.tables.items():
                    ord_config = self.cfg.get("ordination", {}).get(table_type, {})
                    enabled_methods = [m for m in KNOWN_METHODS if ord_config.get(m, False)]
                    
                    for level, table in levels.items():
                        # Create output directory
                        ordir = self.figure_output_dir / level / table_type
                        ordir.mkdir(parents=True, exist_ok=True)
                        
                        for method in enabled_methods:
                            # Submit task to thread pool
                            future = executor.submit(
                                self._run_single_ordination,
                                table=table,
                                meta=self.meta,
                                table_type=table_type,
                                level=level,
                                method=method,
                                ordir=ordir
                            )
                            futures.append(future)
                
                # Process results as they complete
                for future in as_completed(futures):
                    table_type, level, method, res, fig = future.result()
                    self.ordination[table_type].setdefault(level, {})[method] = res
                    self.figures[table_type].setdefault(level, {})[method] = fig
                    progress.advance(main_task)

    def _run_single_ordination(self, table, meta, table_type, level, method, ordir):
        """
        Runs a single ordination method in isolation.
        
        Args:
            table:      Feature table to analyze.
            meta:       Sample metadata DataFrame.
            table_type: Type of feature table.
            level:      Taxonomic level.
            method:     Ordination method to run.
            ordir:      Output directory for figures.
        
        Returns:
            Tuple containing:
                - Table type
                - Taxonomic level
                - Method name
                - Ordination results
                - Generated figures
        """
        try:
            ordn = Ordination(self.cfg, ordir, verbose=False)
            # Run just this one method
            res, figs = ordn.run_tests(
                table=table,
                metadata=meta,
                symbol_col="nuclear_contamination_status",
                transformation=table_type,
                enabled_tests=[method],
            )
            method_key = ordn.TEST_CONFIG[method]['key']
            return table_type, level, method, res.get(method_key), figs.get(method_key)
        except Exception as e:
            logger.error(f"Ordination {method} failed for {table_type}/{level}: {e}")
            return table_type, level, method, None, None

    def _run_ml_feature_selection(self, ml_tables: Dict) -> None:
        """Runs machine learning feature selection on normalized tables."""
        group_col = self.cfg.get("group_column", DEFAULT_GROUP_COLUMN)
        tot = sum(
            len(levels) * len(self.cfg.get("ml", {}).get("methods", ["rfe"]))
            for table_type, levels in ml_tables.items()
        )
        if not tot:
            return

        with get_progress_bar() as progress:
            parent_task = progress.add_task(
                "Running ML feature selection...".ljust(DEFAULT_PROGRESS_TEXT_N), 
                total=tot
            )

            for table_type, levels in ml_tables.items():
                self.models[table_type] = {}
                ml_cfg = self.cfg.get("ml", {})
                methods = ml_cfg.get("methods", ["rfe"])

                for level, table in levels.items():
                    self.models[table_type].setdefault(level, {})
                    for method in methods:
                        child_task = progress.add_task(
                            f"{table_type.ljust(15)} + {level.ljust(10)} + {method.ljust(20)}",
                            parent=parent_task,
                            total=1,
                        )

                        X = table_to_dataframe(table)
                        X.index = X.index.str.lower()
                        y = self.meta.set_index("#sampleid")[[group_col]]
                        y.index = y.index.astype(str).str.lower()
                        idx = X.index.intersection(y.index)
                        X, y = X.loc[idx], y.loc[idx]
                        mdir = Path(self.figure_output_dir).parent / "ml" / level / table_type
                        try:
                            # MODIFIED: Capture model results including top features
                            model_result = catboost_feature_selection(
                                metadata=y,
                                features=X,
                                output_dir=mdir,
                                contamination_status_col=group_col,
                                method=method,
                                n_top_features=100  # Collect top 100 features
                            )
                            self.models[table_type][level][method] = model_result
                        except Exception as e:
                            logger.error(f"Model training with {method} failed for {table_type}/{level}: {e}")
                            self.models[table_type][level][method] = None
                        progress.update(child_task, completed=1)
                        progress.remove_task(child_task)
                        progress.update(parent_task, advance=1)
                        
    # NEW METHOD: Compare model-selected features with statistical features
    def _compare_top_features(self) -> None:
        """Compares top features from ML models with statistical results."""
        if not self.models:
            return
            
        # Collect all statistically significant features
        stat_features = {}
        for table_type, tests in self.stats.items():
            for test_name, levels in tests.items():
                for level, df in levels.items():
                    key = (table_type, level)
                    sig_df = df[df["p_value"] < 0.05]
                    if not sig_df.empty:
                        if key not in stat_features:
                            stat_features[key] = set()
                        stat_features[key].update(sig_df["feature"].tolist())
        
        # Compare with model features
        for table_type, levels in self.models.items():
            for level, methods in levels.items():
                key = (table_type, level)
                stat_set = stat_features.get(key, set())
                
                for method, model_result in methods.items():
                    if model_result is None:
                        continue
                        
                    # Get top features from model
                    model_set = set(model_result.get("top_features", []))
                    
                    # Calculate overlap
                    overlap = model_set & stat_set
                    jaccard = len(overlap) / len(model_set | stat_set) if (model_set or stat_set) else 0.0
                    
                    # Log comparison results
                    logger.info(
                        f"Feature comparison ({table_type}/{level}/{method}): "
                        f"Model features: {len(model_set)}, "
                        f"Statistical features: {len(stat_set)}, "
                        f"Overlap: {len(overlap)} ({jaccard:.1%})"
                    )

class AmpliconData:
    """
    Main class for orchestrating 16S amplicon data analysis pipeline.
    
    Attributes:
        cfg:                       Configuration dictionary.
        project_dir:               Project directory structure.
        mode:                      Analysis mode ('asv' or 'genus').
        verbose:                   Flag for verbose output.
        fdb:                       FAPROTAX database if enabled.
        meta:                      Sample metadata DataFrame.
        table:                     Raw BIOM feature table.
        figure_output_dir:         Directory for output figures.
        tables:                    Processed feature tables.
        figures:                   Generated figures.
        stats:                     Statistical test results.
        ordination:                Ordination analysis results.
        models:                    Machine learning models.
        top_contaminated_features: Top features associated with contamination.
        top_pristine_features:     Top features associated with pristine samples.
    """
    
    def __init__(
        self, 
        cfg: Dict, 
        project_dir: Any, 
        mode: str = DEFAULT_MODE, 
        verbose: bool = False
    ):
        """
        Initializes and runs the amplicon data analysis pipeline.
        
        Args:
            cfg:         Configuration dictionary.
            project_dir: Project directory structure.
            mode:        Analysis mode ('asv' or 'genus').
            verbose:     If True, enables verbose logging.
        """
        self.cfg, self.project_dir, self.mode, self.verbose = cfg, project_dir, mode, verbose
        self.fdb = get_faprotax_parsed() if cfg.get("faprotax", False) else None

        # Apply CPU limiting for parallel libraries
        self._apply_cpu_limits()
        
        data = _DataLoader(cfg, project_dir, mode, verbose)
        self.meta, self.table = data.meta, data.table

        # Process
        self.figure_output_dir = Path(self.project_dir.figures)
        tp = _TableProcessor(
            cfg, self.table, mode, self.meta, self.figure_output_dir, 
            project_dir, verbose
        )
        self.tables = tp.tables

        # Figures
        self.figures: Dict[str, Any] = {}

        if cfg["figures"].get("map", False):
            self.plotter = Plotter(cfg, self.figure_output_dir, verbose)
            self.figures["map"] = self.plotter.generate_sample_map(self.meta)

        # Analysis
        am = _AnalysisManager(
            cfg, self.tables, self.meta, self.figure_output_dir,
            verbose, cfg.get("faprotax", False), self.fdb,
        )
        self.stats = am.stats
        self.ordination = am.ordination
        self.models = am.models
        self.top_contaminated_features = am.top_contaminated_features
        self.top_pristine_features = am.top_pristine_features
        self.figures.update(am.figures)

        if verbose:
            logger.info(GREEN + "AmpliconData analysis finished." + RESET)
    
    def _apply_cpu_limits(self):
        """
        Sets environment variables to limit CPU usage in parallel libraries.
        """
        cpu_limit = self.cfg.get("cpu", {}).get("limit", 1)
        
        # Set for common parallel libraries
        os.environ["OMP_NUM_THREADS"] = str(cpu_limit)
        os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_limit)
        os.environ["MKL_NUM_THREADS"] = str(cpu_limit)
        os.environ["BLIS_NUM_THREADS"] = str(cpu_limit)
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_limit)
        os.environ["NUMBA_NUM_THREADS"] = str(cpu_limit)
        os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_limit)
