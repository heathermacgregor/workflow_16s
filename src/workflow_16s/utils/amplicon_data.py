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
from workflow_16s.utils.progress import get_progress_bar

# ========================== INITIALISATION & CONFIGURATION ========================== #

logger = logging.getLogger("workflow_16s")
warnings.filterwarnings("ignore")

# ================================= DEFAULT VALUES =================================== #

DEFAULT_PROGRESS_TEXT_N = 65
DEFAULT_N = DEFAULT_PROGRESS_TEXT_N
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

# ===================================== CLASSES ====================================== #

def _init_dict_level(a, b, c=None, d=None, e=None):
    if b not in a:
        a[b] = {}
    if c and c not in a[b]:
        a[b][c] = {}
    if d and d not in a[b][c]:
        a[b][c][d] = {}
    if e and e not in a[b][c][d]:
        a[b][c][d][e] = {}

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
            logger.info(f"{process_name}")
            for level in levels:
                start_time = time.perf_counter() 
                processed[level] = process_func(get_source(level), level, *func_args)
                duration = time.perf_counter() - start_time
                if log_template or log_action:
                    self._log_level_action(level, log_template, log_action, duration)
        else:
            with get_progress_bar() as prog:
                l0_desc = f"{process_name}"
                logger.debug(l0_desc)
                parent_task = prog.add_task(
                    f"[white]{l0_desc:<{DEFAULT_N}}",
                    total=len(levels),
                )
                
                for level in levels:
                    start_time = time.perf_counter()  
                    l1_desc = f"Processing {level} level"
                    child_task = prog.add_task(
                        f"[white]{l1_desc:<{DEFAULT_N}}",
                        parent=parent_task,
                        total=1
                    )
                    processed[level] = process_func(get_source(level), level, *func_args)
                    duration = time.perf_counter() - start_time
                    if log_template or log_action:
                        self._log_level_action(level, log_template, log_action, duration)

                    prog.update(child_task, completed=1)
                    prog.remove_task(child_task)
                    prog.update(parent_task, advance=1)

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
                logger.info(f"Running {cfg['name']}...")
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
        self.color_columns = cfg["figures"].get(
            "color_columns",
            cfg["figures"].get(
                "map_columns",
                [
                    DEFAULT_DATASET_COLUMN, DEFAULT_GROUP_COLUMN,
                    "env_feature", "env_material", "country"
                ],
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
            method_params = {}
            if cfg["key"] == "pcoa":
                method_params["metric"] = trans_cfg.get("pcoa_metric", "braycurtis")
            
            if cfg["key"] in ["tsne", "umap"]:
                cpu_limit = self.cfg.get("ordination", {}).get("cpu_limit", 1)
                method_params["n_jobs"] = cpu_limit
    
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
        self.color_columns = cfg["figures"].get(
            "map_columns",
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
        **kwargs
    ) -> Dict[str, Any]:
        valid_columns = [col for col in self.color_columns if col in metadata]
        missing = set(self.color_columns) - set(valid_columns)
        if missing and self.verbose:
            logger.warning(f"Missing columns in metadata: {', '.join(missing)}")

        with get_progress_bar() as prog:
            l0_desc = "Plotting sample maps..."
            l0_task = prog.add_task(
                f"[white]{l0_desc:<{DEFAULT_N}}", 
                total=len(valid_columns)
            )

            for col in valid_columns:
                l1_desc = f"Mapping {col}..."
                l1_task = prog.add_task(
                    f"[white]{l1_desc:<{DEFAULT_N}}",
                    parent=l0_task,
                    total=1
                )
                
                fig, _ = sample_map_categorical(
                    metadata=metadata,
                    output_dir=self.output_dir,
                    color_col=col,
                    **kwargs,
                )
                self.figures[col] = fig
                prog.update(l1_task, completed=1)
                prog.remove_task(l1_task)
                prog.update(l0_task, advance=1)
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

        for table_type, tests in stats_results.items():
            for test_name, test_results in tests.items():
                for level, df in test_results.items():
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
    table: Table

    def _validate_mode(self) -> None:
        if self.mode not in self.MODE_CONFIG:
            raise ValueError(f"Invalid mode: {self.mode}")

    def _get_metadata_paths(self) -> List[Path]:
        metadata_paths = [paths["metadata"] 
                          for subset_id, paths in self.existing_subsets.items()]
        if self.verbose:
            logger.info(f"Found {len(metadata_paths)} metadata files")
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
        if self.existing_subsets != None:
            paths = self._get_metadata_paths()
        else:
            paths = self._get_metadata_paths_glob()
        self.meta = import_merged_metadata_tsv(paths, None, self.verbose)
        if self.meta.columns.duplicated().any():
            duplicated_columns = self.meta.columns[self.meta.columns.duplicated()].tolist()
            logger.debug(
                f"Found duplicate columns in metadata: {duplicated_columns}. "
                "Removing duplicates."
            )
            self.meta = self.meta.loc[:, ~self.meta.columns.duplicated()]

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
        if self.existing_subsets != None:
            biom_paths = self._get_biom_paths()
        else:
            biom_paths = self._get_biom_paths_glob()
        if not biom_paths:
            raise FileNotFoundError("No BIOM files found")
        self.table = import_merged_table_biom(biom_paths, "table", self.verbose)
    
    def _filter_and_align(self) -> None:
        orig_n = self.table.shape[1]
        self.table, self.meta = update_table_and_meta(self.table, self.meta, "#sampleid")
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
        self.tables: Dict[str, Dict[str, Table]] = {"raw": {mode: table}}
        self._apply_preprocessing()
        self._collapse_taxa()
        self._create_presence_absence()
        self._save_tables()

    def _apply_preprocessing(self) -> None:
        feat_cfg = self.cfg["features"]
        table = self.tables["raw"][self.mode]

        if feat_cfg["filter"]:
            table = filter(table)
            self.tables.setdefault("filtered", {})[self.mode] = table

        if feat_cfg["normalize"]:
            table = normalize(table, axis=1)
            self.tables.setdefault("normalized", {})[self.mode] = table

        if feat_cfg["clr_transform"]:
            table = clr(table)
            self.tables.setdefault("clr_transformed", {})[self.mode] = table

    def _collapse_taxa(self) -> None:
        levels = ["phylum", "class", "order", "family", "genus"]
        with get_progress_bar() as prog:
            master_desc = "Collapsing taxonomy..."
            master_task = prog.add_task(
                f"[white]{master_desc:<{DEFAULT_N}}",
                total=len(self.tables)
            )   
            
            for table_type in list(self.tables.keys()):
                table_desc = f"Table Type: {table_type.replace('_', ' ').title()}"
                table_task = prog.add_task(
                    f"[white]{table_desc:<{DEFAULT_N}}",
                    parent=master_task,
                    total=len(levels)
                )
                base_table = self.tables[table_type][self.mode]
                processed = {}
                
                for level in levels:
                    level_desc = f"Level: {level.title()}"
                    level_task = prog.add_task(
                        f"[white]{level_desc:<{DEFAULT_N}}",
                        parent=table_task,
                        total=1
                    )
                    try:
                        start_time = time.perf_counter()
                        processed[level] = collapse_taxa(base_table, level, prog, table_task)
                        duration = time.perf_counter() - start_time
                        if self.verbose:
                            logger.debug(f"Collapsed {table_type} to {level} in {duration:.2f}s")
                    except Exception as e:
                        logger.error(f"Taxonomic collapse failed for {table_type}/{level}: {e}")
                        processed[level] = None
                    finally:
                        prog.update(level_task, advance=1)
                        prog.remove_task(level_task)
                        prog.update(table_task, advance=1)
                    
                self.tables[table_type] = processed
                prog.remove_task(table_task)
                prog.update(master_task, advance=1)
    
    def _create_presence_absence(self) -> None:
        if not self.cfg["features"]["presence_absence"]:
            return
               
        levels = ["phylum", "class", "order", "family", "genus"]
        with get_progress_bar() as prog:
            master_desc = "Converting to Presence/Absence..."
            master_task = prog.add_task(
                f"{master_desc:<{DEFAULT_N}}",
                total=len(levels)  
            )
            raw_table = self.tables["raw"][self.mode]
            processed = {}
            
            for level in levels:
                level_desc = f"Level: {level.capitalize()}"
                level_task = prog.add_task(
                    f"[white]{level_desc:<{DEFAULT_N}}",
                    parent=master_task,
                    total=1
                )
                try:
                    start_time = time.perf_counter()
                    processed[level] = presence_absence(raw_table, level)
                    duration = time.perf_counter() - start_time
                    if self.verbose:
                        logger.debug(f"Created Presence/Absence table for {level} in {duration:.2f}s")
                except Exception as e:
                    logger.error(f"Presence/Absence failed for {level}: {e}")
                    processed[level] = None
                finally:
                    prog.update(level_task, advance=1)
                    prog.remove_task(level_task)
                    prog.update(master_task, advance=1)
                
            self.tables["presence_absence"] = processed

    def _save_tables(self) -> None:
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
        self.top_contaminated_features: List[Dict] = []
        self.top_pristine_features: List[Dict] = []
        self.faprotax_enabled, self.fdb = faprotax_enabled, fdb
        self._faprotax_cache = {}

        self._run_alpha_diversity_analysis()  
        self._run_statistical_tests()
        stats_copy = deepcopy(self.stats)

        self._identify_top_features(stats_copy)
        del stats_copy

        self._run_ordination()

        ml_table_types = {"clr_transformed"}
        ml_tables = {
            t: d for t, d in self.tables.items() if t in ml_table_types
        }
        self._run_ml_feature_selection(ml_tables)
        self._compare_top_features()
        del ml_tables

        if self.faprotax_enabled and self.top_contaminated_features:
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
            f["feature"] for f in self.top_contaminated_features + self.top_pristine_features
        }

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self._get_cached_faprotax, all_taxa))

        taxon_map = dict(zip(all_taxa, results))

        for feat in self.top_contaminated_features:
            feat["faprotax_functions"] = taxon_map.get(feat["feature"], [])

        for feat in self.top_pristine_features:
            feat["faprotax_functions"] = taxon_map.get(feat["feature"], [])

    def _run_alpha_diversity_analysis(self) -> None:      
        alpha_cfg = self.cfg.get("alpha_diversity", {})
        if not alpha_cfg.get("enabled", False):
            logger.info("Alpha diversity analysis disabled.")
            return
        
        group_col = self.cfg.get("group_column", DEFAULT_GROUP_COLUMN)
        metrics = alpha_cfg.get("metrics", DEFAULT_ALPHA_METRICS)
        parametric = alpha_cfg.get("parametric", False)
        generate_plots = alpha_cfg.get("generate_plots", True)
        
        n = 0
        enabled_table_types = alpha_cfg.get("tables", {})
        for table_type, levels in self.tables.items():
            if not enabled_table_types.get(table_type, False):
                continue
            enabled_levels = enabled_table_types[table_type].get("levels", list(levels.keys()))
            n += len(enabled_levels)
        
        if not n:
            return

        with get_progress_bar() as prog:
            master_desc = f"Running alpha diversity for '{group_col}'..."
            master_task = prog.add_task(
                f"[white]{master_desc:<{DEFAULT_N}}", 
                total=n,
                start_time=time.time()
            )
            for table_type, levels in self.tables.items():
                if not enabled_table_types.get(table_type, False):
                    continue

                table_cfg = enabled_table_types[table_type]
                enabled_levels = table_cfg.get("levels", list(levels.keys()))
                
                table_desc = f"Table Type: {table_type.replace('_', ' ').title()}"
                table_task = prog.add_task(
                    f"[white]{table_desc:<{DEFAULT_N}}",
                    parent=master_task,
                    total=len(enabled_levels),
                    start_time=time.time()
                )
                for level in enabled_levels:
                    if level not in levels:
                        logger.warning(f"Level '{level}' not found for table type '{table_type}'")
                        continue
                    
                    level_desc = f"Level: {level.title()}"
                    level_task = prog.add_task(
                        f"[white]{level_desc:<{DEFAULT_N}}",
                        parent=table_task,
                        total=1,
                        start_time=time.time()
                    )
                    try: 
                        _init_dict_level(self.alpha_diversity, table_type, level)  
                        df = table_to_df(levels[level])
                        alpha_df = alpha_diversity(df, metrics=metrics)
                        self.alpha_diversity[table_type][level]['results'] = alpha_df
                        
                        # Create output directory for this analysis
                        output_dir = self.output_dir / 'alpha_diversity' / table_type / level
                        output_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Save results
                        alpha_df.to_csv(output_dir / 'alpha_diversity.tsv', sep='\t', index=True)
                        
                        # Run statistical analysis
                        stats_df = analyze_alpha_diversity(
                            alpha_diversity_df=alpha_df,
                            metadata=self.meta,
                            group_column=group_col,
                            parametric=parametric
                        )
                        self.alpha_diversity[table_type][level]['stats'] = stats_df
                        stats_df.to_csv(output_dir / f'stats_{group_col}.tsv', sep='\t', index=True)
                        
                        if self.cfg["alpha_diversity"].get("correlation_analysis", True):
                            corr_results = analyze_alpha_correlations(
                                alpha_df,
                                self.meta,
                                max_categories=self.cfg["alpha_diversity"].get("max_categories", 20),
                                min_samples=self.cfg["alpha_diversity"].get("min_group_size", 5)
                            )
                            self.alpha_diversity[table_type][level]['correlations'] = corr_results
                            pd.DataFrame.from_dict([corr_results], orient='index').to_csv(
                                output_dir / f'correlations_{group_col}.tsv', 
                                sep='\t', index=True
                            )
                        
                        if generate_plots:
                            self.alpha_diversity[table_type][level]['figures'] = {}
                            
                            plot_cfg = alpha_cfg.get("plot", {})
                            for metric in metrics:
                                if alpha_df[metric].isnull().all():
                                    logger.error(f"All values NaN for metric {metric} in {table_type}/{level}")
                                metric_stats = stats_df[stats_df['metric'] == metric].iloc[0]
                                
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
                                self.alpha_diversity[table_type][level]['figures'][metric] = fig
                            
                            stats_fig = create_alpha_diversity_stats_plot(
                                stats_df=stats_df,
                                output_dir=output_dir,
                                verbose=self.verbose,
                                effect_size_threshold=plot_cfg.get("effect_size_threshold", 0.5)
                            )
                            self.alpha_diversity[table_type][level]['figures']["summary"] = stats_fig
                            
                            if self.cfg["alpha_diversity"].get("correlation_analysis", True):
                                corr_figures = plot_alpha_correlations(
                                    corr_results,
                                    output_dir=output_dir,
                                    top_n=self.cfg["alpha_diversity"].get("top_n_correlations", 10)
                                )
                                self.alpha_diversity[table_type][level]['figures']["correlations"] = corr_figures
                            
                    except Exception as e:
                        logger.error(f"Alpha diversity analysis failed for {table_type}/{level}: {e}")
                        self.alpha_diversity[table_type][level] = {'results': None, 'stats': None, 'figures': {}}
                        
                    finally:
                        prog.update(level_task, advance=1)
                        prog.remove_task(level_task)
                        prog.update(table_task, advance=1)
                        prog.update(master_task, advance=1)
                
                prog.remove_task(table_task)
            
    def _run_statistical_tests(self) -> None:
        group_col = self.cfg.get("group_column", DEFAULT_GROUP_COLUMN)
        group_vals = self.cfg.get("group_values", [True, False])
        san = StatisticalAnalyzer(self.cfg, self.verbose)
        
        n = 0
        for table_type, levels in self.tables.items():
            tests_config = self.cfg["stats"].get(table_type, {})
            enabled_for_table_type = [t for t, flag in tests_config.items() if flag]
            n += len(levels) * len(enabled_for_table_type)

        with get_progress_bar() as prog:
            master_desc = f"Running statistical tests for '{group_col}'..."
            master_task = prog.add_task(
                f"{master_desc:<{DEFAULT_N}}", 
                total=n,
                start_time=time.time()
            )
            for table_type, levels in self.tables.items():
                tests_config = self.cfg["stats"].get(table_type, {})
                enabled_for_table_type = [t for t, flag in tests_config.items() if flag]
                
                table_desc = f"{table_type.replace('_', ' ').title()}"
                table_task = prog.add_task(
                    f"{table_desc:<{DEFAULT_N}}",
                    parent=master_task,
                    total=len(levels) * len(enabled_for_table_type),
                    start_time=time.time()
                )
                for level, table in levels.items():
                    level_desc = f"Level: {level.title()}"
                    level_task = prog.add_task(
                        f"{level_desc:<{DEFAULT_N}}",
                        parent=table_task,
                        total=len(enabled_for_table_type),
                        start_time=time.time()
                    )
                    
                    # Create output directory for this analysis
                    output_dir = self.output_dir / 'stats' / table_type / level
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    table_aligned, meta_aligned = update_table_and_meta(table, self.meta)
                
                    for test_name in enabled_for_table_type:
                        if test_name not in san.TEST_CONFIG:
                            continue
                        cfg = san.TEST_CONFIG[test_name]
                        _init_dict_level(self.stats, table_type, level)    
                        test_desc = f"Test: {cfg['name']}"
                        test_task = prog.add_task(
                            test_desc,
                            parent=level_task,
                            total=1,
                            start_time=time.time()
                        )
                        try:
                            result = cfg["func"](
                                table=table_aligned,
                                metadata=meta_aligned,
                                group_column=group_col,
                                group_column_values=group_vals,
                            )
                            self.stats[table_type][level][test_name] = result
                            
                            # Save results to analysis directory
                            result.to_csv(output_dir / f'{test_name}.tsv', sep='\t', index=True)
                            
                        except Exception as e:
                            logger.error(f"Test '{test_name}' failed for {table_type}/{level}: {e}")
                            self.stats[table_type][level][test_name] = None
                            
                        finally:
                            prog.update(test_task, completed=1)
                            prog.remove_task(test_task)
                            prog.update(level_task, advance=1)
                            prog.update(table_task, advance=1)
                            prog.update(master_task, advance=1)
                    prog.remove_task(level_task)
                prog.remove_task(table_task)

    def _identify_top_features(self, stats_results: Dict) -> None:
        tfa = TopFeaturesAnalyzer(self.cfg, self.verbose)
        self.top_contaminated_features, self.top_pristine_features = tfa.analyze(
            stats_results, DEFAULT_GROUP_COLUMN
        )

        if self.verbose:
            logger.info(f"Found {len(self.top_contaminated_features)} top contaminated features")
            logger.info(f"Found {len(self.top_pristine_features)} top pristine features")

    def _run_ordination(self) -> None:
        KNOWN_METHODS = ["pca", "pcoa", "tsne", "umap"]
        default_ord_config = {"pca": False, "pcoa": False, "tsne": False, "umap": False}
        
        total_tasks = 0
        for table_type, levels in self.tables.items():
            ord_config = self.cfg.get("ordination", {}).get(table_type, default_ord_config)
            enabled_methods = [m for m in KNOWN_METHODS if ord_config.get(m, False)]
            total_tasks += len(levels) * len(enabled_methods)
        
        if not total_tasks:
            return
    
        self.ordination = {tt: {} for tt in self.tables}
    
        with get_progress_bar() as prog:
            master_desc = "Running beta diversity analysis..."
            master_task = prog.add_task(f"{master_desc:<{DEFAULT_N}}", total=total_tasks)
            
            max_workers = min(2, os.cpu_count() // 2)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for table_type, levels in self.tables.items():
                    ord_config = self.cfg.get("ordination", {}).get(table_type, default_ord_config)
                    enabled_methods = [m for m in KNOWN_METHODS if ord_config.get(m, False)]
                    
                    for level, table in levels.items():
                        df = table_to_df(table)
                        
                        # Create output directory for this analysis
                        output_dir = self.output_dir / 'ordination' / table_type / level
                        output_dir.mkdir(parents=True, exist_ok=True)
                        
                        for method in enabled_methods:
                            future = executor.submit(
                                self._run_single_ordination,
                                table=table,
                                meta=self.meta,
                                table_type=table_type,
                                level=level,
                                method=method,
                                output_dir=output_dir
                            )
                            futures.append(future)
                
                for future in as_completed(futures, timeout=1800):
                    try:
                        table_type, level, method, res, fig = future.result()
                        _init_dict_level(self.ordination, table_type, level) 
                        self.ordination[table_type][level][method] = {
                            'result': res,
                            'figures': fig
                        }
                    except TimeoutError:
                        logger.error("Ordination task timed out after 30 minutes")
                    finally:
                        prog.advance(master_task)

    def _run_single_ordination(self, table, meta, table_type, level, method, output_dir):
        try:
            ordn = Ordination(self.cfg, output_dir, verbose=False)
            res, figs = ordn.run_tests(
                table=table,
                metadata=meta,
                symbol_col=DEFAULT_GROUP_COLUMN,
                transformation=table_type,
                enabled_tests=[method],
            )
            method_key = ordn.TEST_CONFIG[method]['key']
            return table_type, level, method, res.get(method_key), figs.get(method_key)
        except Exception as e:
            logger.error(f"Ordination {method} failed for {table_type}/{level}: {e}")
            return table_type, level, method, None, None

    def _run_ml_feature_selection(self, ml_tables: Dict) -> None:
        ml_cfg = self.cfg.get("ml", {})
        if not ml_cfg.get("enabled", False):
            logger.info("ML feature selection disabled.")
            return
            
        group_col = self.cfg.get("group_column", DEFAULT_GROUP_COLUMN)
        methods = ml_cfg.get("methods", ["rfe"])
        n_top_features = ml_cfg.get("num_features", 100)
        step_size = ml_cfg.get("step_size", 100)
        permutation_importance = ml_cfg.get("permutation_importance", True)
        n_threads = ml_cfg.get("n_threads", 8)
        
        enabled_table_types = set(ml_cfg.get("table_types", ["clr_transformed"]))
        enabled_levels = set(ml_cfg.get("levels", ["genus"]))
        
        filtered_ml_tables = {}
        for table_type, levels in ml_tables.items():
            if table_type not in enabled_table_types:
                continue
            filtered_levels = {
                level: table 
                for level, table in levels.items() 
                if level in enabled_levels
            }
            if filtered_levels:
                filtered_ml_tables[table_type] = filtered_levels
    
        n = sum(len(levels) * len(methods) for levels in filtered_ml_tables.values())
        if not n:
            return
        
        with get_progress_bar() as prog:
            master_desc = "ML Feature Selection..."
            master_task = prog.add_task(
                f"{master_desc:<{DEFAULT_N}}", 
                total=n,
                start_time=time.time()
            )
            for table_type, levels in filtered_ml_tables.items():
                table_desc = f"{table_type.replace('_', ' ').title()}"
                table_task = prog.add_task(
                    f"{table_desc:<{DEFAULT_N}}",
                    parent=master_task,
                    total=len(levels) * len(methods),
                    start_time=time.time()
                )
                for level, table in levels.items():
                    level_desc = f"Level: {level.title()}"
                    level_task = prog.add_task(
                        f"{level_desc:<{DEFAULT_N}}",
                        parent=table_task,
                        total=len(methods),
                        start_time=time.time()
                    )
                    
                    # Create output directory for this analysis
                    output_dir = self.output_dir / 'ml' / table_type / level
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    for method in methods:
                        method_desc = f"Method: {method.upper()}"
                        method_task = prog.add_task(
                            f"{method_desc:<{DEFAULT_N}}",
                            parent=level_task,
                            total=1,
                            start_time=time.time()
                        )
                        _init_dict_level(self.models, table_type, level, method) 
                        try:
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
                                    progress=prog, 
                                    task_id=level_task,
                                )

                                # Store figures within model result
                                model_result['figures'] = {
                                    'shap_summary_bar': model_result.pop('shap_summary_bar'),
                                    'shap_summary_beeswarm': model_result.pop('shap_summary_beeswarm'),
                                    'shap_dependency': model_result.pop('shap_dependency')
                                }
                                
                                self.models[table_type][level][method] = model_result
                                    
                        except Exception as e:
                            logger.error(f"Model training failed for {table_type}/{level}/{method}: {e}")
                            self.models[table_type][level][method] = None
                                
                        finally:
                            prog.update(method_task, completed=1)
                            prog.remove_task(method_task)
                            prog.update(level_task, advance=1)
                            prog.update(table_task, advance=1)
                            prog.update(master_task, advance=1)
                    prog.remove_task(level_task)
                prog.remove_task(table_task)
                        
    def _compare_top_features(self) -> None:
        if not self.models:
            return
            
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
                    
                    logger.info(
                        f"Feature comparison ({table_type}/{level}/{method}): "
                        f"Overlap: {len(overlap)} ({jaccard:.1%})"
                    )
                    
    def _generate_violin_plots(self, n=50):
        # Create output directory for violin plots
        violin_output_dir = self.output_dir / 'violin_plots'
        violin_output_dir.mkdir(parents=True, exist_ok=True)
        
        for feat in self.top_contaminated_features[:n]:
            try:
                table_type = feat['table_type']
                level = feat['level']
                feature_name = feat['feature']
                
                table = self.tables[table_type][level]
                df = table_to_df(table)
                
                merged_df = df.merge(
                    self.meta[[DEFAULT_GROUP_COLUMN]], 
                    left_index=True, 
                    right_index=True
                )
                
                # Create specific output directory for this feature
                feature_output_dir = violin_output_dir / 'contaminated' / table_type / level
                feature_output_dir.mkdir(parents=True, exist_ok=True)
                
                fig = violin_feature(
                    df=merged_df,
                    feature=feature_name,
                    output_dir=feature_output_dir,
                    status_col=DEFAULT_GROUP_COLUMN
                )
                feat['violin_figure'] = fig
            except Exception as e:
                logger.error(f"Failed violin plot for {feature_name}: {e}")
                feat['violin_figure'] = None
        
        for feat in self.top_pristine_features[:n]:
            try:
                table_type = feat['table_type']
                level = feat['level']
                feature_name = feat['feature']
                
                table = self.tables[table_type][level]
                df = table_to_df(table)
                
                merged_df = df.merge(
                    self.meta[[DEFAULT_GROUP_COLUMN]], 
                    left_index=True, 
                    right_index=True
                )
                
                # Create specific output directory for this feature
                feature_output_dir = violin_output_dir / 'pristine' / table_type / level
                feature_output_dir.mkdir(parents=True, exist_ok=True)
                
                fig = violin_feature(
                    df=merged_df,
                    feature=feature_name,
                    output_dir=feature_output_dir,
                    status_col=DEFAULT_GROUP_COLUMN
                )
                feat['violin_figure'] = fig
            except Exception as e:
                logger.error(f"Failed violin plot for {feature_name}: {e}")
                feat['violin_figure'] = None


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
        self.top_contaminated_features: List[Dict] = []
        self.top_pristine_features: List[Dict] = []
        
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
        cpu_limit = self.cfg.get("cpu", {}).get("limit", 4)
        for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", 
                    "MKL_NUM_THREADS", "BLIS_NUM_THREADS", 
                    "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS"]:
            os.environ[var] = str(cpu_limit)
        os.environ["MKL_DYNAMIC"] = "FALSE"

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
        if self.cfg["figures"].get("map", False):
            # Create output directory for sample maps
            maps_output_dir = Path(self.project_dir.final) / 'sample_maps'
            maps_output_dir.mkdir(parents=True, exist_ok=True)
            
            plotter = MapPlotter(
                self.cfg, 
                maps_output_dir,
                self.verbose
            )
            self.maps = plotter.generate_sample_map(self.meta)

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
        self.top_contaminated_features = analyzer.top_contaminated_features
        self.top_pristine_features = analyzer.top_pristine_features
