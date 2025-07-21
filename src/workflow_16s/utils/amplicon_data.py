# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import glob
import logging
import os
import time
import warnings
import threading  # Added for thread locking
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Third‑Party Imports
import pandas as pd
from biom.table import Table

# ================================== LOCAL IMPORTS =================================== #

from workflow_16s.figures.merged import (
    mds as plot_mds,
    pca as plot_pca,
    pcoa as plot_pcoa,
    create_alpha_diversity_boxplot, create_alpha_diversity_stats_plot,
    plot_alpha_correlations, sample_map_categorical, violin_feature
)
from workflow_16s.function.faprotax import (
    faprotax_functions_for_taxon, get_faprotax_parsed
)
from workflow_16s.models.feature_selection import (
    catboost_feature_selection,
)
from workflow_16s.stats.beta_diversity import (
    pcoa as calculate_pcoa,
    pca as calculate_pca,
    tsne as calculate_tsne,
    umap as calculate_umap,
)
from workflow_16s.stats.tests import (
    alpha_diversity, analyze_alpha_diversity, analyze_alpha_correlations,
    fisher_exact_bonferroni, kruskal_bonferroni, mwu_bonferroni, ttest
)
from workflow_16s.utils.data import (
    clr, collapse_taxa, filter, normalize, presence_absence, table_to_df, 
    update_table_and_meta, to_biom
)
from workflow_16s.utils.io import (
    export_h5py, import_merged_metadata_tsv, import_merged_table_biom
)
from workflow_16s.utils.progress import get_progress_bar, _format_task_desc
from workflow_16s.utils.nfc_facilities import find_nearby_nfc_facilities

# ========================== INITIALISATION & CONFIGURATION ========================== #

logger = logging.getLogger("workflow_16s")
warnings.filterwarnings("ignore")

# Global lock for UMAP operations to prevent thread conflicts
umap_lock = threading.Lock()

# ================================= DEFAULT VALUES =================================== #

DEFAULT_PROGRESS_TEXT_N = 65
DEFAULT_N = 65
DEFAULT_DATASET_COLUMN = "dataset_name"
DEFAULT_GROUP_COLUMN = "nuclear_contamination_status"
DEFAULT_SYMBOL_COL = DEFAULT_GROUP_COLUMN
DEFAULT_GROUP_COLUMN_VALUES = [True, False]
DEFAULT_MODE = 'genus'
DEFAULT_ALPHA_METRICS = [
    'shannon', 'observed_features', 'simpson',
    'pielou_evenness', 'chao1', 'ace', 
    'gini_index', 'goods_coverage', 'heip_evenness', 
    'dominance'       
]
PHYLO_METRICS = ['faith_pd', 'pd_whole_tree']
DEFAULT_GROUP_COLUMNS = [
    {
        'name': "nuclear_contamination_status",
        'type': "bool",
        'values': [True, False]
    },
]
debug_mode = False

# =============================== HELPER FUNCTIONS ==================================== #

def _init_dict_level(a, b, c=None, d=None, e=None):
    if b not in a:
        a[b] = {}
    if c and c not in a[b]:
        a[b][c] = {}
    if d and d not in a[b][c]:
        a[b][c][d] = {}
    if e and e not in a[b][c][d]:
        a[b][c][d][e] = {}
        
# ===================================== CLASSES ====================================== #

class _ProcessingMixin:
    """
    Provides reusable methods for processing steps with progress tracking and logging.
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
        processed: Dict[str, Table] = {}

        if getattr(self, "verbose", False):
            for level in levels:
                start_time = time.perf_counter() 
                processed[level] = process_func(get_source(level), level, *func_args)
                duration = time.perf_counter() - start_time
                if log_template or log_action:
                    self._log_level_action(level, log_template, log_action, duration)
        else:
            with get_progress_bar() as progress:
                parent_desc = f"{process_name}"
                parent_task = progress.add_task(_format_task_desc(parent_desc), total=len(levels))
                
                for level in levels:
                    level_desc = f"{parent_desc} ({level})"
                    progress.update(parent_task, description=_format_task_desc(level_desc))
                    
                    start_time = time.perf_counter()  
                    processed[level] = process_func(get_source(level), level, *func_args)
                    duration = time.perf_counter() - start_time
                    if log_template or log_action:
                        self._log_level_action(level, log_template, log_action, duration)

                    progress.update(parent_task, advance=1)
            progress.update(parent_task, description=_format_task_desc(parent_desc))
        return processed

    def _log_level_action(
        self,
        level: str,
        template: Optional[str] = None,
        action: Optional[str] = None,
        duration: Optional[float] = None,
    ) -> None:
        message = ""
        if template:
            message = template.format(level=level)
        elif action:
            message = f"{level} {action}"

        if message and duration is not None:
            message += f" in {duration:.2f}s"

        if message:
            logger.debug(message)


class StatisticalAnalyzer:
    """
    Performs statistical tests on feature tables to identify significant differences.
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
        self.cfg = cfg
        self.verbose = verbose

    def run_tests(
        self,
        table: Table,
        metadata: pd.DataFrame,
        group_column: str,
        group_column_values: List[Any],
        enabled_tests: List[str],
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        table, metadata = update_table_and_meta(table, metadata)

        for test_name in enabled_tests:
            if test_name not in self.TEST_CONFIG:
                continue
            cfg = self.TEST_CONFIG[test_name]
            if self.verbose:
                logger.debug(f"Running {cfg['name']}...")
            results[cfg["key"]] = cfg["func"](table, metadata, group_column, 
                                              group_column_values)
        return results

    def get_effect_size(self, test_name: str, row: pd.Series) -> Optional[float]:
        if test_name not in self.TEST_CONFIG:
            return None
        cfg = self.TEST_CONFIG[test_name]
        for col in (cfg["effect_col"], cfg["alt_effect_col"]):
            if col and col in row:
                return row[col]
        return None


class Ordination:
    """
    Performs ordination analyses (PCA, PCoA, t-SNE, UMAP) and stores figures.
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
        self.cfg = cfg
        self.verbose = verbose
        self.figure_output_dir = Path(output_dir)
        self.color_columns = cfg["maps"].get(
            "color_columns",
            [
                DEFAULT_DATASET_COLUMN, DEFAULT_GROUP_COLUMN,
                "env_feature", "env_material", "country"
            ],
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
        trans_cfg = self.cfg.get("ordination", {}).get(transformation, {})
        tests_to_run = [t for t in enabled_tests if t in self.TEST_CONFIG]
        if not tests_to_run:
            return {}, {}

        table, metadata = update_table_and_meta(table, metadata)
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
        try:
            if debug_mode:
                time.sleep(3)
                return
            method_params = {}
            if cfg["key"] == "pcoa":
                method_params["metric"] = trans_cfg.get("pcoa_metric", "braycurtis")
                logger.info(method_params["metric"])
            
            # Handle UMAP/TSNE thread safety
            if cfg["key"] in ["tsne", "umap"]:
                method_params["n_jobs"] = 1
                # Use global lock to prevent NUMBA thread conflicts
                with umap_lock:
                    # Set NUMBA threads before importing/executing
                    os.environ['NUMBA_NUM_THREADS'] = '1'
                    import numba
                    numba.config.NUMBA_NUM_THREADS = 1
                    ord_res = cfg["func"](table=table, **method_params)
            else:
                ord_res = cfg["func"](table=table, **method_params)
        except Exception as e:
            logger.error(f"Params: {method_params}")
            logger.error(f"Failed {cfg['key']} for {transformation}: {e}")
            return None, {}

        try:
            figures = {}
            pkwargs = {**cfg.get("plot_kwargs", {}), **kwargs}
            
            for color_col in self.color_columns:
                if color_col not in metadata.columns:
                    logger.warning(f"Color column '{color_col}' not found in metadata")
                    continue
    
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
                else:
                    pkwargs["df"] = ord_res
    
                fig, _ = cfg["plot_func"](
                    metadata=metadata,
                    color_col=color_col,
                    symbol_col=symbol_col,
                    transformation=transformation,
                    output_dir=self.figure_output_dir,
                    **pkwargs,
                )
                if fig:  # Only add if figure was created
                    figures[color_col] = fig
            return ord_res, figures

        except Exception as e:
            logger.error(f"Failed {cfg['key']} plot for {transformation}: {e}")
            return ord_res, {}


class MapPlotter:
    """
    Generates sample map plots and stores them internally.
    """
    
    def __init__(
        self, 
        cfg: Dict, 
        output_dir: Path, 
        verbose: bool = False
    ):
        self.cfg = cfg
        self.output_dir = output_dir
        self.verbose = verbose
        self.color_columns = cfg["maps"].get(
            "color_columns",
            [
                DEFAULT_DATASET_COLUMN,
                DEFAULT_GROUP_COLUMN,
                "env_feature",
                "env_material",
                "country",
            ],
        )
        self.figures: Dict[str, Any] = {}

    def generate_sample_map(
        self, 
        metadata: pd.DataFrame, 
        nfc_facility_data: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> Dict[str, Any]:
        valid_columns = [col for col in self.color_columns if col in metadata]
        missing = set(self.color_columns) - set(valid_columns)
        if missing and self.verbose:
            logger.warning(f"Missing columns in metadata: {', '.join(missing)}")

        with get_progress_bar() as progress:
            plot_desc = f"Plotting sample maps"
            plot_task = progress.add_task(_format_task_desc(plot_desc), total=len(valid_columns))

            for col in valid_columns:
                col_desc = f"Plotting sample maps → {col}"
                progress.update(plot_task, description=_format_task_desc(col_desc))
                
                fig, _ = sample_map_categorical(
                    metadata=metadata,
                    nfc_facilities_data=_facility_data,
                    output_dir=self.output_dir,
                    color_col=col,
                    **kwargs,
                )
                self.figures[col] = fig
                
                progress.update(plot_task, advance=1)
            progress.update(plot_task, description=_format_task_desc(plot_desc))
        return self.figures


class TopFeaturesAnalyzer:
    """
    Identifies top differentially abundant features based on statistical results.
    """
    
    def __init__(
        self, 
        cfg: Dict, 
        verbose: bool = False
    ):
        self.cfg = cfg
        self.verbose = verbose

    def analyze(
        self,
        stats_results: Dict[str, Dict[str, Dict[str, pd.DataFrame]]],
        group_column: str,
    ) -> Tuple[List[Dict], List[Dict]]:
        san = StatisticalAnalyzer(self.cfg, self.verbose)
        all_features = []

        for table_type, levels in stats_results.items():  # 1. Table Types
            for level, tests in levels.items():           # 2. Taxonomic Levels
                for test_name, df in tests.items():       # 3. Test Names
                    if df is None or not isinstance(df, pd.DataFrame):
                        continue
                    if "p_value" not in df.columns:
                        continue
                        
                    sig_df = df[df["p_value"] < 0.05].copy()
                    if sig_df.empty:
                        continue

                    sig_df["effect"] = sig_df.apply(
                        lambda row: san.get_effect_size(test_name, row), axis=1
                    )
                    sig_df = sig_df.dropna(subset=["effect"])

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

        cont_feats = [f for f in all_features if f["effect"] > 0]
        pris_feats = [f for f in all_features if f["effect"] < 0]

        cont_feats.sort(key=lambda d: (-d["effect"], d["p_value"]))
        pris_feats.sort(key=lambda d: (d["effect"], d["p_value"]))

        return cont_feats[:100], pris_feats[:100]


class _DataLoader(_ProcessingMixin):
    """
    Loads and processes microbiome data from BIOM files and metadata.
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
        existing_subsets: Dict[str, Dict[str, Path]] = None,
        verbose: bool = False
    ):
        self.cfg, self.project_dir, self.mode, self.existing_subsets, self.verbose = cfg, project_dir, mode, existing_subsets, verbose
        self._validate_mode()
        self._load_metadata()
        self._load_biom_table()
        self._filter_and_align()

    meta: pd.DataFrame
    nfc_facilities: pd.DataFrame
    meta_nfc_facilities: pd.DataFrame
    table: Table

    def _validate_mode(self) -> None:
        if self.mode not in self.MODE_CONFIG:
            raise ValueError(f"Invalid mode: {self.mode}")

    def _get_metadata_paths(self) -> List[Path]:
        metadata_paths = [paths["metadata"] 
                          for subset_id, paths in self.existing_subsets.items()]
        if self.verbose:
            (f"Found {len(metadata_paths)} metadata files")
        return metadata_paths

    def _get_metadata_paths_glob(self) -> List[Path]:
        paths: List[Path] = []
        for bi in self._get_biom_paths_glob():
            ds_dir = bi.parent if bi.is_file() else bi
            tail = ds_dir.parts[-6:-1]
            mp = Path(self.project_dir.metadata_per_dataset).joinpath(*tail, "sample-metadata.tsv")
            if mp.exists():
                paths.append(mp)
        if self.verbose:
            logger.info(f"Found {len(paths)} metadata files")
        return paths

    def _load_metadata(self) -> None:
        if self.existing_subsets == None:
            paths = self._get_metadata_paths_glob()
        else:
            paths = self._get_metadata_paths()
            
        self.meta = import_merged_metadata_tsv(paths, None, self.verbose)

        # Remove duplicated columns
        if self.meta.columns.duplicated().any():
            duplicated_columns = self.meta.columns[self.meta.columns.duplicated()].tolist()
            logger.debug(
                f"Found duplicate columns in metadata: {duplicated_columns}. "
                "Removing duplicates."
            )
            self.meta = self.meta.loc[:, ~self.meta.columns.duplicated()]
            
        # If enabled, find samples within a threshold distance from NFC facilities
        if self.cfg.get("nfc_facilities", {}).get("enabled", False):
            self.nfc_facilities, self.meta_nfc_facilities = find_nearby_nfc_facilities(cfg=self.cfg, meta=self.meta)
            
    def _get_biom_paths(self) -> List[Path]:
        table_dir, _ = self.MODE_CONFIG[self.mode]
        biom_paths = [paths[table_dir] for subset_id, paths in self.existing_subsets.items()]
        if self.verbose:
            logger.info(f"Found {len(biom_paths)} feature tables")
        return biom_paths

    def _get_biom_paths_glob(self) -> List[Path]:
        table_dir, _ = self.MODE_CONFIG[self.mode]
        if self.cfg["target_subfragment_mode"] != 'any' or self.mode != 'genus':
            pattern = "/".join([
                "*", "*", "*", self.cfg["target_subfragment_mode"], 
                "FWD_*_REV_*", table_dir, "feature-table.biom",
            ])
        else:
            pattern = "/".join([
                "*", "*", "*", "*", 
                "FWD_*_REV_*", table_dir, "feature-table.biom",
            ])
        globbed = glob.glob(str(Path(
            self.project_dir.qiime_data_per_dataset
        ) / pattern), recursive=True)
        if self.verbose:
            logger.info(f"Found {len(globbed)} feature tables")
        return [Path(p) for p in globbed]

    def _load_biom_table(self) -> None:
        if self.existing_subsets == None:
            biom_paths = self._get_biom_paths_glob()
        else:
            biom_paths = self._get_biom_paths()
            
        if not biom_paths:
            raise FileNotFoundError("No BIOM files found")
        self.table = import_merged_table_biom(biom_paths, "table", self.verbose)
    
    def _filter_and_align(self) -> None:
        self.table, self.meta = update_table_and_meta(
            self.table, self.meta, 
            self.cfg.get("metadata_id_column", "#sampleid")
        )
        ftype = "genera" if self.mode == "genus" else "ASVs"
        logger.info(
            f"{'Loaded metadata:':<30}{self.meta.shape[0]:>6} samples × {self.meta.shape[1]:>5} cols"
        )
        logger.info(
            f"{'Loaded features:':<30}{self.table.shape[1]:>6} samples × {self.table.shape[0]:>5} {ftype}"
        )


class _TableProcessor(_ProcessingMixin):
    """
    Processes feature tables through various transformations and taxonomical collapses.
    """
    
    def __init__(
        self,
        cfg: Dict,
        table: Table,
        mode: str,
        meta: pd.DataFrame,
        output_dir: Path,
        project_dir: Any,
        verbose: bool,
    ) -> None:
        self.cfg, self.mode, self.verbose = cfg, mode, verbose
        self.meta = meta
        self.output_dir = output_dir
        self.project_dir = project_dir
        self.levels = ["phylum", "class", "order", "family", "genus"]
        self.tables: Dict[str, Dict[str, Table]] = {"raw": {mode: table}}
        self._apply_preprocessing()
        self._collapse_taxa()
        self._create_presence_absence()
        self._save_tables()

    def _apply_preprocessing(self) -> None:
        feat_cfg = self.cfg.get("features", {})
        table = self.tables["raw"][self.mode]

        if feat_cfg.get("filter", True):
            table = filter(table)
            self.tables.setdefault("filtered", {})[self.mode] = table

        if feat_cfg.get("normalize", True):
            table = normalize(table, axis=1)
            self.tables.setdefault("normalized", {})[self.mode] = table

        if feat_cfg.get("clr_transform", True):
            table = clr(table)
            self.tables.setdefault("clr_transformed", {})[self.mode] = table

    def _collapse_taxa(self) -> None:
        with get_progress_bar() as progress:
            ct_desc = "Collapsing taxonomy"
            ct_task = progress.add_task(
                _format_task_desc(ct_desc), 
                total=len(self.tables) * len(self.levels)
            )   
            
            for table_type in list(self.tables.keys()):
                table_desc = f"{table_type.replace('_', ' ').title()}"
                table_task = progress.add_task(
                    _format_task_desc(table_desc),
                    parent=ct_task,
                    total=len(self.levels)
                )
                
                base_table = self.tables[table_type][self.mode]
                processed = {}
                for level in self.levels:
                    level_desc = f"{table_desc} → {level.title()}"
                    progress.update(
                        table_task, 
                        description=_format_task_desc(level_desc)
                    )
                    try:
                        start_time = time.perf_counter()
                        processed[level] = collapse_taxa(
                            base_table, 
                            level, 
                            progress, table_task
                        )
                        duration = time.perf_counter() - start_time
                        logger.debug(f"Collapsed {table_type} to {level} in {duration:.2f}s")
                    except Exception as e:
                        logger.error(f"Taxonomic collapse failed for {table_type}/{level}: {e}")
                        processed[level] = None
                    finally:
                        progress.update(table_task, advance=1)
                        progress.update(ct_task, advance=1)
                    
                self.tables[table_type] = processed
                progress.remove_task(table_task)
    
    def _create_presence_absence(self) -> None:
        if not self.cfg.get("features", {}).get("presence_absence", False):
            return
        with get_progress_bar() as progress:
            pa_desc = "Converting to Presence/Absence"
            pa_task = progress.add_task(
                _format_task_desc(pa_desc),
                total=len(self.levels)  
            )
            processed = {}
            
            for level in self.levels:
                level_desc = f"Converting to Presence/Absence → {level.capitalize()}"
                progress.update(
                    pa_task, 
                    description=_format_task_desc(level_desc)
                )
                try:
                    collapsed_table = self.tables["raw"][level]
                    start_time = time.perf_counter()
                    processed[level] = presence_absence(collapsed_table)
                    duration = time.perf_counter() - start_time
                    if self.verbose:
                        logger.debug(f"Created Presence/Absence table for {level} in {duration:.2f}s")
                except Exception as e:
                    logger.error(f"Presence/Absence failed for {level}: {e}")
                    processed[level] = None
                finally:
                    progress.update(pa_task, advance=1)
                
            self.tables["presence_absence"] = processed
            progress.update(
                pa_task, 
                description=_format_task_desc(pa_desc)
            )
            
    def _save_tables(self) -> None:
        # Create directory if it doesn't exist
        base = Path(self.project_dir.data) / "merged" / "table"
        base.mkdir(parents=True, exist_ok=True)

        export_tasks = []
        for table_type, levels in self.tables.items():
            tdir = base / table_type
            tdir.mkdir(parents=True, exist_ok=True)
            for level, table in levels.items():
                out = tdir / level / "feature-table.biom"
                out.parent.mkdir(parents=True, exist_ok=True)
                export_tasks.append((table, out))

        with ThreadPoolExecutor() as executor:
            futures = []
            for table, out in export_tasks:
                futures.append(executor.submit(export_h5py, table, out))

            for future in futures:
                future.result()


class _AnalysisManager(_ProcessingMixin):
    """
    Manages the analysis pipeline including statistics, ordination, and machine learning.
    """
    
    def __init__(
        self,
        cfg: Dict,
        tables: Dict[str, Dict[str, Table]],
        meta: pd.DataFrame,
        output_dir: Path,
        verbose: bool,
        faprotax_enabled: bool = False,
        fdb: Optional[Dict] = None,
    ) -> None:
        self.cfg, self.tables, self.meta, self.verbose = cfg, tables, meta, verbose
        self.output_dir = output_dir
        self.table_output_dir = Path(output_dir) / 'tables'
        self.stats: Dict[str, Any] = {}
        self.ordination: Dict[str, Any] = {}
        self.models: Dict[str, Any] = {}
        self.alpha_diversity: Dict[str, Any] = {}
        self.top_features_group_1: List[Dict] = [] # contaminated
        self.top_features_group_2: List[Dict] = [] # pristine
        self.faprotax_enabled, self.fdb = faprotax_enabled, fdb
        self._faprotax_cache = {}

        self._run_alpha_diversity_analysis()  
        self._run_statistical_tests()
        stats_copy = deepcopy(self.stats)

        self._identify_top_features(stats_copy)
        del stats_copy
        self._generate_violin_plots(n=cfg.get("violin_plots", {}).get("n", 50))

        self._run_ordination()
        self._run_ml_feature_selection()
        self._compare_top_features()

        if self.faprotax_enabled and self.top_features_group_1:
            self._annotate_top_features()

        self._generate_violin_plots(n=cfg.get("violin_plots", {}).get("n", 50))

    def _get_cached_faprotax(
        self, 
        taxon: str
    ) -> List[str]:
        if taxon not in self._faprotax_cache:
            self._faprotax_cache[taxon] = faprotax_functions_for_taxon(
                taxon, self.fdb, include_references=False
            )
        return self._faprotax_cache[taxon]

    def _annotate_top_features(self) -> None:
        all_taxa = {
            f["feature"] for f in self.top_features_group_1 + self.top_features_group_2
        }

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self._get_cached_faprotax, all_taxa))

        taxon_map = dict(zip(all_taxa, results))

        for feat in self.top_features_group_1:
            feat["faprotax_functions"] = taxon_map.get(feat["feature"], [])

        for feat in self.top_features_group_2:
            feat["faprotax_functions"] = taxon_map.get(feat["feature"], [])

    def _run_alpha_diversity_analysis(self) -> None:      
        alpha_cfg = self.cfg.get("alpha_diversity", {})
        if not alpha_cfg.get("enabled", False):
            ("Alpha diversity analysis disabled.")
            return
        
        group_col = self.cfg.get("group_column", DEFAULT_GROUP_COLUMN)
        
        metrics = alpha_cfg.get("metrics", DEFAULT_ALPHA_METRICS)
        parametric = alpha_cfg.get("parametric", False)
        
        n = 0
        table_cfg = alpha_cfg.get("tables", {})
        for table_type, levels in self.tables.items():
            tt_cfg = table_cfg.get(table_type, {})
            if not tt_cfg.get('enabled', False):
                continue
            enabled_levels = tt_cfg.get("levels", list(levels.keys()))
            enabled_levels = [l for l in enabled_levels if l in list(levels.keys())]
            # Accumulate tasks: levels × metrics for this table
            n += len(enabled_levels) 
        
        if not n:
            return

        with get_progress_bar() as progress:
            alpha_desc = f"Running alpha diversity for '{group_col}'"
            alpha_task = progress.add_task(_format_task_desc(alpha_desc), total=n)
            
            for table_type, levels in self.tables.items():
                tt_cfg = table_cfg.get(table_type, {})
                if not tt_cfg.get('enabled', False):
                    continue
                enabled_levels = tt_cfg.get("levels", list(levels.keys()))
                enabled_levels = [l for l in enabled_levels if l in list(levels.keys())]
                
                tt_desc = f"{table_type.replace('_', ' ').title()}"
                tt_task = progress.add_task(
                    _format_task_desc(tt_desc),
                    parent=alpha_task,
                    total=len(enabled_levels)
                )
                
                for level in enabled_levels:
                    level_desc = f"{tt_desc} ({level.title()})"
                    progress.update(tt_task, description=_format_task_desc(level_desc))

                    _init_dict_level(self.alpha_diversity, table_type, level) 
                    data_storage = self.alpha_diversity[table_type][level]
                    try: 
                        if debug_mode:
                            time.sleep(3)
                            return
                        table = self.tables[table_type][level]
                        table_df = table_to_df(table)
                        
                        alpha_df = alpha_diversity(
                            table_df, metrics=metrics
                        )
                        
                        stats_df = analyze_alpha_diversity(
                            alpha_diversity_df=alpha_df,
                            metadata=self.meta,
                            group_column=group_col,
                            parametric=parametric
                        )
                        
                        # Save results
                        output_dir = self.output_dir / 'alpha_diversity' / table_type / level
                        output_dir.mkdir(parents=True, exist_ok=True)
                        alpha_df.to_csv(output_dir / 'alpha_diversity.tsv', sep='\t', index=True)
                        stats_df.to_csv(output_dir / f'stats_{group_col}.tsv', sep='\t', index=True)
                        
                        data_storage['results'] = alpha_df
                        data_storage['stats'] = stats_df

                        corr_cfg = alpha_cfg.get("correlation_analysis", {})
                        if corr_cfg.get("enabled", False):
                            corr_results = analyze_alpha_correlations(
                                alpha_df,
                                self.meta,
                                max_categories=corr_cfg.get("max_categories", 20),
                                min_samples=corr_cfg.get("min_group_size", 5)
                            )
                            # Save results
                            pd.DataFrame.from_dict([corr_results], orient='index').to_csv(
                                output_dir / f'correlations_{group_col}.tsv', 
                                sep='\t', index=True
                            )
                            data_storage['correlations'] = corr_results

                        plot_cfg = alpha_cfg.get("plots", {})
                        if plot_cfg.get("enabled", True):
                            data_storage['figures'] = {}
                            fig_storage = data_storage['figures']
                            
                            for metric in metrics:
                                if alpha_df[metric].isnull().all():
                                    logger.error(f"All values NaN for metric {metric} in {table_type}/{level}")
                                    
                                fig = create_alpha_diversity_boxplot(
                                    alpha_df=alpha_df,
                                    metadata=self.meta,
                                    group_column=group_col,
                                    metric=metric,
                                    output_dir=output_dir,
                                    show=False,
                                    verbose=self.verbose,
                                    add_points=plot_cfg.get("add_points", True),
                                    add_stat_annot=plot_cfg.get("add_stat_annot", True),
                                    test_type="parametric" if parametric else "nonparametric"
                                )
                                fig_storage[metric] = fig
                            
                            stats_fig = create_alpha_diversity_stats_plot(
                                stats_df=stats_df,
                                output_dir=output_dir,
                                verbose=self.verbose,
                                effect_size_threshold=plot_cfg.get("effect_size_threshold", 0.5)
                            )
                            fig_storage['summary'] = stats_fig
                            
                            if corr_cfg.get("enabled", False):
                                corr_figures = plot_alpha_correlations(
                                    corr_results,
                                    output_dir=output_dir,
                                    top_n=corr_cfg.get("top_n_correlations", 10)
                                )
                                fig_storage['correlations'] = corr_figures
                            
                    except Exception as e:
                        logger.error(f"Alpha diversity analysis failed for {table_type}/{level}: {e}")
                        data_storage = {'results': None, 'stats': None, 'figures': {}}
                        
                    finally:
                        progress.update(tt_task, advance=1)
                        progress.update(alpha_task, advance=1)
                progress.remove_task(tt_task)
            
    def _run_statistical_tests(self) -> None:
        stats_cfg = self.cfg.get("stats", {})
        if not stats_cfg.get("enabled", False):
            ("Statistical analysis disabled.")
            return

        KNOWN_TESTS = ['fisher', 'ttest', 'mwu_bonferroni', 'kruskal_bonferroni']
        default_table_type_tests = {
            "raw": ["ttest"],
            "filtered": ['mwu_bonferroni', 'kruskal_bonferroni'],
            "normalized": ['ttest', 'mwu_bonferroni', 'kruskal_bonferroni'],
            "clr_transformed": ['ttest', 'mwu_bonferroni', 'kruskal_bonferroni'],
            "presence_absence": ["fisher"]
        }
        
        group_col = self.cfg.get("group_column", DEFAULT_GROUP_COLUMN)
        group_vals = self.cfg.get("group_column_values", [True, False])
        san = StatisticalAnalyzer(self.cfg, self.verbose)
        san_cfg = san.TEST_CONFIG
        
        n = 0
        table_cfg = stats_cfg.get("tables", {})
        for table_type, levels in self.tables.items():
            tt_cfg = table_cfg.get(table_type, {})
            if not tt_cfg.get('enabled', False):
                continue
            enabled_levels = tt_cfg.get("levels", list(levels.keys()))
            enabled_levels = [l for l in enabled_levels if l in list(levels.keys())]

            enabled_tests = tt_cfg.get("tests", default_table_type_tests[table_type])
            enabled_tests = [m for m in enabled_tests if m in KNOWN_TESTS]
            # Accumulate tasks: levels × tests for this table
            n += len(enabled_levels) * len(enabled_tests)
        
        if not n:
            return

        with get_progress_bar() as progress:
            stats_desc = f"Running statistics for '{group_col}'"
            stats_task = progress.add_task(_format_task_desc(stats_desc), total=n)
            
            for table_type, levels in self.tables.items():
                # Check that table type is enabled
                tt_cfg = table_cfg.get(table_type, {})
                if not tt_cfg.get('enabled', False):
                    continue

                # Get enabled levels for table type
                enabled_levels = tt_cfg.get("levels", list(levels.keys()))
                enabled_levels = [l for l in enabled_levels if l in list(levels.keys())]

                # Get enabled tests for table type
                enabled_tests = tt_cfg.get("tests", default_table_type_tests[table_type])
                enabled_tests = [m for m in enabled_tests if m in KNOWN_TESTS]
                
                tt_desc = f"{table_type.replace('_', ' ').title()}"
                tt_task = progress.add_task(
                    _format_task_desc(tt_desc),
                    parent=stats_task,
                    total=len(enabled_levels) * len(enabled_tests)
                )

                for level in enabled_levels:
                    level_desc = f"{tt_desc} ({level.title()})"
                    progress.update(tt_task, description=_format_task_desc(level_desc))

                    # Create output directory 
                    output_dir = self.output_dir / 'stats' / table_type / level
                    output_dir.mkdir(parents=True, exist_ok=True)

                    _init_dict_level(self.stats, table_type, level) 
                    data_storage = self.stats[table_type][level]
                    
                    table = self.tables[table_type][level]
                    table_aligned, meta_aligned = update_table_and_meta(table, self.meta)
                    
                    for test in enabled_tests:
                        test_desc = f"{level_desc} → {san_cfg[test]['name']}"
                        progress.update(tt_task, description=_format_task_desc(test_desc))
                        try:
                            if debug_mode:
                                time.sleep(3)
                                return
                            result = san_cfg[test]["func"](
                                table=table_aligned,
                                metadata=meta_aligned,
                                group_column=group_col,
                                group_column_values=group_vals,
                            )
                            # Save results 
                            result.to_csv(output_dir / f'{test}.tsv', sep='\t', index=True)
                            data_storage[test] = result
                            
                            # Log number of significant features
                            if isinstance(result, pd.DataFrame) and "p_value" in result.columns:
                                n_sig = sum(result["p_value"] < 0.05)
                                logger.debug(f"Found {n_sig} significant features for {table_type}/{level}/{test}")
                                
                                # Add debug info for first 5 features if none are significant
                                if n_sig == 0 and self.verbose:
                                    logger.debug(f"Top 5 features by p-value ({test}):")
                                    top_p = result.nsmallest(5, "p_value")[["feature", "p_value", cfg["effect_col"]]]
                                    for _, row in top_p.iterrows():
                                        logger.debug(f"  {row['feature']}: p={row['p_value']:.3e}, effect={row[cfg['effect_col']]:.3f}")
                            
                        except Exception as e:
                            logger.error(f"Test '{test}' failed for {table_type}/{level}: {e}")
                            data_storage[test] = None
                            
                        finally:
                            progress.update(tt_task, advance=1)
                            progress.update(stats_task, advance=1)
                progress.remove_task(tt_task)
                        

    def _identify_top_features(self, stats_results: Dict) -> None:
        if not stats_results:
            logger.warning("No statistical results available for feature selection")
            return [], []
        group_col = self.cfg.get("group_column", DEFAULT_GROUP_COLUMN)
        group_col_values = self.cfg.get("group_column_values", DEFAULT_GROUP_COLUMN_VALUES)
        
        tfa = TopFeaturesAnalyzer(self.cfg, self.verbose)
        self.top_features_group_1, self.top_features_group_2 = tfa.analyze(stats_results, group_col)

        logger.debug(f"Identified {len(self.top_features_group_1)} top features for '{group_col}' = {group_col_values[0]}")
        logger.debug(f"Identified {len(self.top_features_group_2)} top features for '{group_col}' = {group_col_values[1]}")
        
        if not self.top_features_group_1 and not self.top_features_group_2:
            logger.warning("No significant features found in any statistical test. Top features tables and violin plots will be empty.")

    def _run_ordination(self) -> None:
        ordination_cfg = self.cfg.get("ordination", {})
        if not ordination_cfg.get("enabled", False):
            ("Ordination analysis disabled.")
            return

        group_col = self.cfg.get("group_column", DEFAULT_GROUP_COLUMN)
        
        KNOWN_METHODS = ["pca", "pcoa", "tsne", "umap"]
        default_table_type_methods = {
            "raw": ["pca"],
            "filtered": ["pca", "pcoa"],
            "normalized": ["pca", "pcoa", "tsne", "umap"],
            "clr_transformed": ["pca", "pcoa", "tsne", "umap"],
            "presence_absence": ["pcoa", "tsne", "umap"]
        }
        
        n = 0
        table_cfg = ordination_cfg.get("tables", {})
        for table_type, levels in self.tables.items():
            tt_cfg = table_cfg.get(table_type, {})
            if not tt_cfg.get('enabled', False):
                continue
            enabled_levels = tt_cfg.get("levels", list(levels.keys()))
            enabled_levels = [l for l in enabled_levels if l in list(levels.keys())]
            enabled_methods = tt_cfg.get("methods", default_table_type_methods[table_type])
            enabled_methods = [m for m in enabled_methods if m in KNOWN_METHODS]
            # Accumulate tasks: levels × methods for this table
            n += len(enabled_levels) * len(enabled_methods)
        if not n:
            return
    
        self.ordination = {tt: {} for tt in self.tables}
    
        with get_progress_bar() as progress:
            beta_desc = "Running beta diversity analysis"
            beta_task = progress.add_task(_format_task_desc(beta_desc), total=n)
            
            max_workers = min(2, os.cpu_count() // 2)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                future_to_key = {}
                for table_type, levels in self.tables.items():
                    tt_cfg = table_cfg.get(table_type, {})
                    if not tt_cfg.get('enabled', False):
                        continue
                    enabled_levels = tt_cfg.get("levels", list(levels.keys()))
                    enabled_levels = [l for l in enabled_levels if l in list(levels.keys())]
                    
                    enabled_methods = tt_cfg.get("methods", default_table_type_methods[table_type])
                    enabled_methods = [m for m in enabled_methods if m in KNOWN_METHODS]
                    
                    for level in enabled_levels:
                        table = self.tables[table_type][level]
                        table_aligned, meta_aligned = update_table_and_meta(table, self.meta)
                        
                        output_dir = self.output_dir / 'ordination' / table_type / level
                        output_dir.mkdir(parents=True, exist_ok=True)                        
                        
                        for method in enabled_methods:
                            future = executor.submit(
                                self._run_single_ordination,
                                table=table_aligned,
                                meta=meta_aligned,
                                symbol_col=group_col,
                                table_type=table_type,
                                level=level,
                                method=method,
                                output_dir=output_dir
                            )
                            futures.append(future)
                            future_to_key[future] = (table_type, level, method)
            
                completed_results = {}
                errors = {}
                try:
                    for future in as_completed(futures, timeout=2*3600):  # 2 hour timeout
                        key = future_to_key[future]
                        try:
                            if debug_mode:
                                time.sleep(3)
                                return
                            result = future.result()
                            completed_results[key] = result
                        except Exception as e:
                            errors[key] = str(e)
                            logger.error(f"Ordination failed for {key}: {str(e)}")
                        progress.update(beta_task, advance=1)
                except TimeoutError:
                    logger.warning("Ordination timeout - proceeding with completed results")
                
                # Process completed results
                for key, result in completed_results.items():
                    table_type, level, method = key
                    _, _, _, res, figs = result
                    _init_dict_level(self.ordination, table_type, level) 
                    data_storage = self.ordination[table_type][level]
                    data_storage[method] = {'result': res, 'figures': figs}
                
                # Log summary
                completed_count = len(completed_results)
                error_count = len(errors)
                timeout_count = len(futures) - completed_count - error_count
                logger.debug(
                    f"Ordination completed: {completed_count} succeeded, "
                    f"{error_count} failed, {timeout_count} timed out"
                )

    def _run_single_ordination(self, table, meta, symbol_col, table_type, level, method, output_dir):
        try:
            ordn = Ordination(self.cfg, output_dir, verbose=False)
            res, figs = ordn.run_tests(
                table=table,
                metadata=meta,
                symbol_col=symbol_col,
                transformation=table_type,
                enabled_tests=[method]
            )
            method_key = ordn.TEST_CONFIG[method]['key']
            return table_type, level, method, res.get(method_key), figs.get(method_key)
        except Exception as e:
            logger.error(f"Ordination {method} failed for {table_type}/{level}: {e}")
            return table_type, level, method, None, None

    def _run_ml_feature_selection(self) -> None:
        ml_cfg = self.cfg.get("ml", {})
        if not ml_cfg.get("enabled", False):
            logger.info("ML feature selection disabled.")
            return
           
        group_col = self.cfg.get("group_column", DEFAULT_GROUP_COLUMN)
        n_top_features = ml_cfg.get("num_features", 100)
        step_size = ml_cfg.get("step_size", 100)
        permutation_importance = ml_cfg.get("permutation_importance", {}).get("enabled", True)
        n_threads = ml_cfg.get("n_threads", 8)


        n = 0
        table_cfg = ml_cfg.get("tables", {})
        for table_type, levels in self.tables.items():
            tt_cfg = table_cfg.get(table_type, {})
            if not tt_cfg.get('enabled', False):
                continue
            enabled_levels = tt_cfg.get("levels", list(levels.keys()))
            enabled_levels = [l for l in enabled_levels if l in list(levels.keys())]
            enabled_methods = tt_cfg.get("methods", [])#default_table_type_methods[table_type])
            #enabled_methods = [m for m in enabled_methods if m in KNOWN_METHODS]
            # Accumulate tasks: levels × tests for this table
            n += len(enabled_levels) * len(enabled_methods) 
        if not n:
            return

        with get_progress_bar() as progress:
            cb_desc = "Running CatBoost feature selection"
            cb_task = progress.add_task(_format_task_desc(cb_desc), total=n)
            
            for table_type, levels in self.tables.items():
                # Check that table type is enabled
                tt_cfg = table_cfg.get(table_type, {})
                if not tt_cfg.get('enabled', False):
                    continue

                # Get enabled levels for table type
                enabled_levels = tt_cfg.get("levels", list(levels.keys()))
                enabled_levels = [l for l in enabled_levels if l in list(levels.keys())]
                
                tt_desc = f"{table_type.replace('_', ' ').title()}"
                tt_task = progress.add_task(
                    _format_task_desc(tt_desc),
                    parent=cb_task,
                    total=len(enabled_levels)
                )

                # Get enabled tests for table type
                enabled_methods = tt_cfg.get("methods", ["rfe"])#default_table_type_tests[table_type])
                #enabled_methods = [m for m in enabled_methods if m in KNOWN_METHODS]
                
                for level in enabled_levels:
                    level_desc = f"{tt_desc} ({level.title()})"
                    progress.update(tt_task, description=_format_task_desc(level_desc))

                    # Create output directory 
                    output_dir = self.output_dir / 'ml' / table_type / level
                    output_dir.mkdir(parents=True, exist_ok=True)

                    table = self.tables[table_type][level]
                    table_aligned, meta_aligned = update_table_and_meta(table, self.meta)
                    
                    for method in enabled_methods:
                        method_desc = f"{level_desc} → {method.title()}"
                        progress.update(tt_task, description=_format_task_desc(method_desc))

                        _init_dict_level(self.models, table_type, level, method) 
                        data_storage = self.models[table_type][level][method]
                        try:
                            if debug_mode:
                                time.sleep(3)
                                return
                            if table_type == "clr_transformed" and method == "chi_squared":
                                logger.warning(
                                    "Skipping chi_squared feature selection for CLR data."
                                )
                                self.models[table_type][level][method] = None
                            else:
                                X = table_to_df(table)
                                X.index = X.index.str.lower()
                                y = self.meta.set_index("#sampleid")[[group_col]]
                                y.index = y.index.astype(str).str.lower()
                                idx = X.index.intersection(y.index)
                                X, y = X.loc[idx], y.loc[idx]

                                use_permutation_importance = False if method == "select_k_best" else permutation_importance
                                    
                                model_result = catboost_feature_selection(
                                    metadata=y,
                                    features=X,
                                    output_dir=output_dir,
                                    group_col=group_col,
                                    method=method,
                                    n_top_features=n_top_features,
                                    step_size=step_size,
                                    use_permutation_importance=use_permutation_importance,
                                    thread_count=n_threads,
                                    progress=progress, 
                                    task_id=tt_task,
                                )
                                
                                # Log if no figures were generated
                                if not any(model_result['figures'].values()):
                                    logger.warning(f"No figures generated for {table_type}/{level}/{method}")
                                
                                self.models[table_type][level][method] = model_result
                                    
                        except Exception as e:
                            logger.error(f"Model training failed for {table_type}/{level}/{method}: {e}")
                            data_storage = None
                                
                        finally:
                            progress.update(tt_task, advance=1)
                            progress.update(cb_task, advance=1)
                progress.remove_task(tt_task)
                        
    def _compare_top_features(self) -> None:
        if not self.models:
            return
            
        stat_features = {}
        for table_type, tests in self.stats.items():
            for test_name, levels in tests.items():
                for level, df in levels.items():
                    key = (table_type, level)
                    if df is not None and "p_value" in df.columns:
                        sig_df = df[df["p_value"] < 0.05]
                        if not sig_df.empty:
                            if key not in stat_features:
                                stat_features[key] = set()
                            stat_features[key].update(sig_df["feature"].tolist())
        
        for table_type, levels in self.models.items():
            for level, methods in levels.items():
                key = (table_type, level)
                stat_set = stat_features.get(key, set())
                
                for method, model_result in methods.items():
                    if model_result is None:
                        continue
                        
                    model_set = set(model_result.get("top_features", []))
                    overlap = model_set & stat_set
                    jaccard = len(overlap) / len(model_set | stat_set) if (model_set or stat_set) else 0.0
                    
                    logger.debug(
                        f"Feature comparison ({table_type}/{level}/{method}): "
                        f"Overlap: {len(overlap)} ({jaccard:.1%})"
                    )
                    
    def _generate_violin_plots(self, n=50):
        # Create output directory for violin plots
        violin_output_dir = self.output_dir / 'violin_plots'
        violin_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define required metadata columns (add these)
        group_col = self.cfg.get("group_column", DEFAULT_GROUP_COLUMN)
        group_col_values = self.cfg.get("group_column_values", DEFAULT_GROUP_COLUMN_VALUES)
        required_metadata = [
            group_col,
            DEFAULT_DATASET_COLUMN,
            "env_feature",
            "env_material",
            "country"
        ]
        available_metadata = [col for col in required_metadata if col in self.meta.columns]
        
        logger.info(f"Generating violin plots for top {n} features")
        
        # Contaminated features
        if self.top_features_group_1:
            logger.debug(f"Processing {min(n, len(self.top_features_group_1))} '{group_col}'={group_col_values[0]} features")
            for i in range(min(n, len(self.top_features_group_1))):
                feat = self.top_features_group_1[i]
                try:
                    table_type = feat['table_type']
                    level = feat['level']
                    feature_name = feat['feature']
                    
                    # Get the table and convert to DataFrame
                    biom_table = self.tables[table_type][level]
                    table = table_to_df(biom_table)[[feature_name]]
                    meta_ids = self.meta['#sampleid'].astype(str).str.strip().str.lower()
                    table_ids = table.index.astype(str).str.strip().str.lower()
                    shared_ids = set(table_ids) & set(meta_ids)

                    group_map = (
                        self.meta
                        .assign(norm_id=meta_ids)
                        .set_index("norm_id")[group_col]
                    )
                    # Create normalized table index
                    table_normalized_index = table.index.astype(str).str.strip().str.lower()
                    # Map group values using normalized IDs
                    table[group_col] = table_normalized_index.map(group_map)
                    
                    # Verify feature exists
                    if feature_name not in table.columns:
                        logger.warning(f"Feature '{feature_name}' not found in {table_type}/{level} table")
                        continue
                    
                    # Create output directory
                    feature_output_dir = violin_output_dir / f"{group_col}_{group_col_values[0]}" / table_type / level
                    feature_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Generate violin plot
                    fig = violin_feature(
                        df=table,
                        feature=feature_name,
                        output_dir=feature_output_dir,
                        status_col=group_col
                    )
                    feat['violin_figure'] = fig
                except Exception as e:
                    logger.error(f"Failed violin plot for {feature_name} at {level} level: {e}")
                    feat['violin_figure'] = None
        else:
            logger.warning(f"No {group_col}={group_col_values[0]} features for violin plots")
        
        # Pristine features
        if self.top_features_group_2:
            logger.debug(f"Processing {min(n, len(self.top_features_group_2))} '{group_col}'={group_col_values[1]} features")
            for i in range(min(n, len(self.top_features_group_2))):
                feat = self.top_features_group_2[i]
                try:
                    table_type = feat['table_type']
                    level = feat['level']
                    feature_name = feat['feature']
                    
                    # Get the table and convert to DataFrame
                    biom_table = self.tables[table_type][level]
                    table = table_to_df(biom_table)[[feature_name]]
                    meta_ids = self.meta['#sampleid'].astype(str).str.strip().str.lower()
                    table_ids = table.index.astype(str).str.strip().str.lower()
                    shared_ids = set(table_ids) & set(meta_ids)

                    group_map = (
                        self.meta
                        .assign(norm_id=meta_ids)
                        .set_index("norm_id")[group_col]
                    )
                    # Create normalized table index
                    table_normalized_index = table.index.astype(str).str.strip().str.lower()
                    # Map group values using normalized IDs
                    table[group_col] = table_normalized_index.map(group_map)

                    # Verify feature exists
                    if feature_name not in table.columns:
                        logger.warning(f"Feature '{feature_name}' not found in {table_type}/{level} table")
                        continue
                    
                    # Create output directory for this feature
                    feature_output_dir = violin_output_dir / f"{group_col}_{group_col_values[1]}" / table_type / level
                    feature_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Generate violin plot
                    fig = violin_feature(
                        df=table,
                        feature=feature_name,
                        output_dir=feature_output_dir,
                        status_col=group_col
                    )
                    feat['violin_figure'] = fig
                except Exception as e:
                    logger.error(f"Failed violin plot for {feature_name} at {level} level: {e}")
                    feat['violin_figure'] = None
        else:
            logger.warning(f"No {group_col}={group_col_values[1]} features for violin plots")

class AmpliconData:
    """
    Main class for orchestrating 16S amplicon data analysis pipeline.
    """
    
    def __init__(
        self, 
        cfg: Dict, 
        project_dir: Any, 
        mode: str = DEFAULT_MODE, 
        existing_subsets: Optional[Dict[str, Dict[str, Path]]] = None,
        verbose: bool = False
    ):
        self.cfg = cfg
        self.project_dir = project_dir
        self.mode = mode
        self.existing_subsets = existing_subsets
        self.verbose = verbose
        
        # Initialize result containers
        self.maps: Optional[Dict[str, Any]] = None
        self.tables: Dict[str, Any] = {}
        self.stats: Dict[str, Any] = {}
        self.ordination: Dict[str, Any] = {}
        self.models: Dict[str, Any] = {}
        self.alpha_diversity: Dict[str, Any] = {}
        self.top_features_group_1: List[Dict] = []
        self.top_features_group_2: List[Dict] = []
        logger.info("Running amplicon data analysis pipeline...")
        self._execute_pipeline()

    def _execute_pipeline(self):
        """Execute the analysis pipeline in sequence."""
        self._apply_cpu_limits()
        self._load_data()
        self._process_tables()
        self._generate_sample_maps()
        self._run_analysis()
        
        if self.verbose:
            logger.info("AmpliconData analysis finished.")

    def _apply_cpu_limits(self):
        """Set environment variables to control thread usage across libraries"""
        cpu_limit = self.cfg.get("cpu", {}).get("limit", 4)
        # Set consistent environment variables
        os.environ['NUMBA_NUM_THREADS'] = str(cpu_limit)
        os.environ['OMP_NUM_THREADS'] = str(cpu_limit)
        os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_limit)
        os.environ['MKL_NUM_THREADS'] = str(cpu_limit)
        os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_limit)
        os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_limit)
        os.environ['MKL_DYNAMIC'] = "FALSE"
        logger.debug(f"Set CPU thread limits to {cpu_limit} for all libraries")

    def _load_data(self):
        data_loader = _DataLoader(
            self.cfg, 
            self.project_dir, 
            self.mode, 
            self.existing_subsets,
            self.verbose
        )
        self.meta = data_loader.meta
        self.table = data_loader.table
        self.nfc_facilities = data_loader.nfc_facilities

    def _process_tables(self):
        processor = _TableProcessor(
            self.cfg,
            self.table,
            self.mode,
            self.meta,
            Path(self.project_dir.final),
            self.project_dir,
            self.verbose
        )
        self.tables = processor.tables

    def _generate_sample_maps(self):
        if self.cfg["maps"].get("enabled", False):
            # Create output directory for sample maps
            maps_output_dir = Path(self.project_dir.final) / 'sample_maps'
            maps_output_dir.mkdir(parents=True, exist_ok=True)
            
            plotter = MapPlotter(
                self.cfg, 
                maps_output_dir,
                self.verbose
            )
            self.maps = plotter.generate_sample_map(self.meta, nfc_facility_data=self.nfc_facilities)

    def _run_analysis(self):
        analyzer = _AnalysisManager(
            self.cfg,
            self.tables,
            self.meta,
            Path(self.project_dir.final),
            self.verbose,
            self.cfg.get("faprotax", False),
            get_faprotax_parsed() if self.cfg.get("faprotax", False) else None
        )
        
        # Collect results
        self.stats = analyzer.stats
        self.ordination = analyzer.ordination
        self.models = analyzer.models
        self.alpha_diversity = analyzer.alpha_diversity
        self.top_features_group_1 = analyzer.top_features_group_1
        self.top_features_group_2 = analyzer.top_features_group_2
