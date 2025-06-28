# ===================================== IMPORTS ====================================== #
import glob
import logging
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

# Third‑Party Imports
import numpy as np 
import pandas as pd
from biom.table import Table
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TaskID
)

# ================================== LOCAL IMPORTS =================================== #
from workflow_16s.utils.biom import (
    collapse_taxa,
    convert_to_biom,
    export_h5py,
    presence_absence,
)
from workflow_16s.utils.file_utils import (
    import_merged_table_biom,
    import_merged_meta_tsv,
    filter_and_reorder_biom_and_metadata,
)
from workflow_16s.stats.utils import (
    clr_transform_table,
    filter_table,
    normalize_table,
    merge_table_with_metadata,
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
from workflow_16s.figures.merged.merged import (
    mds as plot_mds,
    pca as plot_pca,
    pcoa as plot_pcoa,
    sample_map_categorical,
)
from workflow_16s.function.faprotax import (
    get_faprotax_parsed,
    faprotax_functions_for_taxon,
)
from workflow_16s.models.feature_selection import (
    filter_data,
    grid_search,
    perform_feature_selection,
    save_feature_importances,
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

# ==================================== FUNCTIONS ===================================== #
def print_structure(obj: Any, indent: int = 0, _key: str = "root") -> None:
    spacer = " " * indent
    tname = type(obj).__name__
    print(f"{spacer}{'|-- ' if indent else ''}{_key} ({tname})")
    if isinstance(obj, dict):
        for k, v in obj.items():
            print_structure(v, indent + 4, k)
    elif isinstance(obj, list) and obj:
        print_structure(obj[0], indent + 4, "[0]")

# ===================================== CLASSES ====================================== #
class _ProcessingMixin:
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
            for lvl in levels:
                processed[lvl] = process_func(get_source(lvl), lvl, *func_args)
                self._log_level_action(lvl, log_template, log_action)
        else:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                transient=True,
                expand=True,
            ) as progress:
                parent_task = progress.add_task(
                    f"[white]{process_name}".ljust(DEFAULT_PROGRESS_TEXT_N),
                    total=len(levels),
                )
                for lvl in levels:
                    child_task = progress.add_task(
                        f"Processing {lvl} level",
                        parent=parent_task,
                        total=1
                    )
                    processed[lvl] = process_func(get_source(lvl), lvl, *func_args)
                    progress.update(child_task, completed=1)
                    progress.update(parent_task, advance=1)
        return processed

    def _log_level_action(
        self,
        level: str,
        template: Optional[str] = None,
        action: Optional[str] = None,
    ) -> None:
        if template:
            logger.info(template.format(level=level))
        elif action:
            logger.info(f"{level} {action}")

class StatisticalAnalyzer:
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
        group_values: List[Any],
        enabled_tests: List[str],
        progress: Optional[Progress] = None,
        task_id: Optional[TaskID] = None,
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        # Pre-align samples once instead of in each test
        table, metadata = filter_and_reorder_biom_and_metadata(table, metadata)
        
        if not (progress and task_id):
            for tname in enabled_tests:
                if tname not in self.TEST_CONFIG:
                    continue
                cfg = self.TEST_CONFIG[tname]
                if self.verbose:
                    logger.info(f"Running {cfg['name']}...")
                results[cfg["key"]] = cfg["func"](
                    table=table,
                    metadata=metadata,
                    group_column=group_column,
                    group_column_values=group_values,
                )
            return results

        parent_task = progress.add_task(
            "[white]Running statistical tests...".ljust(DEFAULT_PROGRESS_TEXT_N), 
            total=len(enabled_tests), 
            parent=task_id
        )
        
        for tname in enabled_tests:
            if tname not in self.TEST_CONFIG:
                continue
            cfg = self.TEST_CONFIG[tname]
            
            child_task = progress.add_task(
                f"[cyan]{cfg['name']}",
                parent=parent_task,
                total=1
            )
            
            try:
                if self.verbose:
                    logger.info(f"Running {cfg['name']}...")
                results[cfg["key"]] = cfg["func"](
                    table=table,
                    metadata=metadata,
                    group_column=group_column,
                    group_column_values=group_values,
                )
            finally:
                progress.update(child_task, completed=1)
                progress.update(parent_task, advance=1)
                
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
    TEST_CONFIG = {
        "pca": {"key": "pca", "func": calculate_pca, "plot_func": plot_pca, "name": "PCA"},
        "pcoa": {"key": "pcoa", "func": calculate_pcoa, "plot_func": plot_pcoa, "name": "PCoA"},
        "tsne": {"key": "tsne", "func": calculate_tsne, "plot_func": plot_mds, "name": "t‑SNE", "plot_kwargs": {"mode": "TSNE"}},
        "umap": {"key": "umap", "func": calculate_umap, "plot_func": plot_mds, "name": "UMAP", "plot_kwargs": {"mode": "UMAP"}},
    }

    def __init__(self, cfg: Dict, output_dir: Union[str, Path], verbose: bool = False):
        self.cfg = cfg
        self.verbose = verbose
        self.figure_output_dir = Path(output_dir)
        self.results: Dict[str, Any] = {}
        self.figures: Dict[str, Any] = {}

    def run_tests(
        self,
        table: Table,
        metadata: pd.DataFrame,
        color_col: str,
        symbol_col: str,
        transformation: str,
        enabled_tests: List[str],
        progress: Optional[Progress] = None,
        task_id: Optional[TaskID] = None,
        **kwargs,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        trans_cfg = self.cfg.get("ordination", {}).get(transformation, {})
        tests_to_run = [t for t in enabled_tests if t in self.TEST_CONFIG]
        if not tests_to_run:
            return {}, {}

        try:
            table, metadata = filter_and_reorder_biom_and_metadata(table, metadata)
            
            if not (progress and task_id):
                return self._run_without_progress(table, metadata, color_col, symbol_col, 
                                                transformation, tests_to_run, trans_cfg, kwargs)

            parent_task = progress.add_task(
                f"[white]Running {transformation} ordination".ljust(DEFAULT_PROGRESS_TEXT_N),
                total=len(tests_to_run),
                parent=task_id
            )
            
            results, figures = {}, {}
            for tname in tests_to_run:
                cfg = self.TEST_CONFIG[tname]
                child_task = progress.add_task(
                    f"[cyan]{cfg['name']}",
                    parent=parent_task,
                    total=1
                )
                
                try:
                    res, fig = self._run_ordination_method(cfg, table, metadata, color_col, 
                                                        symbol_col, transformation, trans_cfg, kwargs)
                    results[cfg["key"]] = res
                    figures[cfg["key"]] = fig
                except Exception as e:
                    logger.error(f"Failed {tname} for {transformation}: {e}")
                    figures[cfg["key"]] = None
                finally:
                    progress.update(child_task, completed=1)
                    progress.update(parent_task, advance=1)
            return results, figures
        finally:
            pass  # Cleanup if needed
            
    def _run_without_progress(self, table, metadata, color_col, symbol_col, 
                            transformation, tests_to_run, trans_cfg, kwargs):
        results, figures = {}, {}
        for tname in tests_to_run:
            cfg = self.TEST_CONFIG[tname]
            try:
                res, fig = self._run_ordination_method(cfg, table, metadata, color_col, 
                                                    symbol_col, transformation, trans_cfg, kwargs)
                results[cfg["key"]] = res
                figures[cfg["key"]] = fig
            except Exception as e:
                logger.error(f"Failed {tname} for {transformation}: {e}")
                figures[cfg["key"]] = None
        return results, figures
        
    def _run_ordination_method(self, cfg, table, metadata, color_col, symbol_col, 
                            transformation, trans_cfg, kwargs):
        method_params = {}
        if cfg["key"] == "pcoa":
            method_params["metric"] = trans_cfg.get("pcoa_metric", "braycurtis")
        
        ord_res = cfg["func"](table=table, **method_params)
        
        pkwargs = {**cfg.get("plot_kwargs", {}), **kwargs}
        if cfg["key"] == 'pca':
            pkwargs.update({
                'components': ord_res['components'],
                'proportion_explained': ord_res['exp_var_ratio']
            })
        elif cfg["key"] == 'pcoa':
            pkwargs.update({
                'components': ord_res.samples,
                'proportion_explained': ord_res.proportion_explained
            })
        else:  # t-SNE or UMAP
            pkwargs['df'] = ord_res
            
        fig, _ = cfg["plot_func"](
            metadata=metadata,
            color_col=color_col,
            symbol_col=symbol_col,
            transformation=transformation,
            output_dir=self.figure_output_dir,
            **pkwargs
        )
        return ord_res, fig

class Plotter:
    def __init__(self, cfg: Dict, output_dir: Path, verbose: bool = False):
        self.cfg = cfg
        self.output_dir = output_dir
        self.verbose = verbose
        self.color_columns = cfg["figures"].get("map_columns", [
            "dataset_name", "nuclear_contamination_status",
            "env_feature", "env_material", "country"
        ])

    def generate_sample_map(self, metadata: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        valid_columns = [col for col in self.color_columns if col in metadata]
        missing = set(self.color_columns) - set(valid_columns)
        if missing and self.verbose:
            logger.warning(f"Missing columns in metadata: {', '.join(missing)}")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            transient=True,
            expand=True,
        ) as progress:
            parent_task = progress.add_task(
                "[white]Generating sample maps".ljust(DEFAULT_PROGRESS_TEXT_N),
                total=len(valid_columns))
            
            figs = {}
            for col in valid_columns:
                child_task = progress.add_task(
                    f"[cyan]Mapping {col}",
                    parent=parent_task,
                    total=1
                )
                fig, _ = sample_map_categorical(
                    metadata=metadata,
                    output_dir=self.output_dir,
                    color_col=col,
                    **kwargs,
                )
                figs[col] = fig
                progress.update(child_task, completed=1)
                progress.update(parent_task, advance=1)
        return figs

class TopFeaturesAnalyzer:
    def __init__(self, cfg: Dict, verbose: bool = False):
        self.cfg = cfg
        self.verbose = verbose

    def analyze(
        self,
        stats_results: Dict[str, Dict[str, Dict[str, pd.DataFrame]]],
        group_column: str,
    ) -> Tuple[List[Dict], List[Dict]]:
        san = StatisticalAnalyzer(self.cfg, self.verbose)
        all_features = []
        
        # Collect all significant features efficiently
        for tbl_type, tests in stats_results.items():
            for tname, t_res in tests.items():
                for lvl, df in t_res.items():
                    sig_df = df[df["p_value"] < 0.05].copy()
                    if sig_df.empty:
                        continue
                    
                    # Vectorized effect size calculation
                    sig_df["effect"] = sig_df.apply(
                        lambda row: san.get_effect_size(tname, row), axis=1
                    )
                    sig_df = sig_df.dropna(subset=["effect"])
                    
                    # Collect features in a list
                    for _, row in sig_df.iterrows():
                        all_features.append({
                            "feature": row["feature"],
                            "level": lvl,
                            "table_type": tbl_type,
                            "test": tname,
                            "effect": row["effect"],
                            "p_value": row["p_value"],
                            "effect_dir": "positive" if row["effect"] > 0 else "negative"
                        })
        
        # Efficient sorting and filtering
        cont_feats = [f for f in all_features if f["effect"] > 0]
        pris_feats = [f for f in all_features if f["effect"] < 0]
        
        cont_feats.sort(key=lambda d: (-d["effect"], d["p_value"]))
        pris_feats.sort(key=lambda d: (d["effect"], d["p_value"]))  # More negative is stronger
        
        return cont_feats[:100], pris_feats[:100]

class _DataLoader(_ProcessingMixin):
    MODE_CONFIG = {"asv": ("table", "asv"), "genus": ("table_6", "l6")}

    def __init__(self, cfg: Dict, project_dir: Any, mode: str, verbose: bool = False):
        self.cfg, self.project_dir, self.mode, self.verbose = cfg, project_dir, mode, verbose
        self._validate_mode()
        self._load_metadata()
        self._load_biom_table()
        self._filter_and_align()

    # Public after run
    meta: pd.DataFrame
    table: Table

    def _validate_mode(self) -> None:
        if self.mode not in self.MODE_CONFIG:
            raise ValueError(f"Invalid mode: {self.mode}")

    def _get_metadata_paths(self) -> List[Path]:
        paths: List[Path] = []
        for bi in self._get_biom_paths():
            ds_dir = bi.parent if bi.is_file() else bi
            tail = ds_dir.parts[-6:-1]
            mp = Path(
               self.project_dir.metadata_per_dataset
            ).joinpath(*tail, "sample-metadata.tsv")
            if mp.exists():
                paths.append(mp)
        if self.verbose:
            logger.info(f"Found {RED}{len(paths)}{RESET} metadata files")
        return paths

    def _load_metadata(self) -> None:
        paths = self._get_metadata_paths()
        self.meta = import_merged_meta_tsv(paths, None, self.verbose)

    def _get_biom_paths(self) -> List[Path]:
        table_dir, _ = self.MODE_CONFIG[self.mode]
        pattern = "/".join([
            "*",
            "*",
            "*",
            "*",
            "FWD_*_REV_*",
            table_dir,
            "feature-table.biom",
        ])
        globbed = glob.glob(
           str(Path(self.project_dir.qiime_data_per_dataset) / pattern), 
           recursive=True
        )
        if self.verbose:
            logger.info(
               f"Found {RED}{len(globbed)}{RESET} feature tables"
            )
        return [Path(p) for p in globbed]

    def _load_biom_table(self) -> None:
        biom_paths = self._get_biom_paths()
        if not biom_paths:
            raise FileNotFoundError("No BIOM files found")
        self.table = import_merged_table_biom(biom_paths, "table", self.verbose)

    def _filter_and_align(self) -> None:
        orig_n = self.table.shape[1]
        self.table, self.meta = filter_and_reorder_biom_and_metadata(
            self.table, self.meta, "#sampleid"
        )
        ftype = "genera" if self.mode == "genus" else "ASVs"
        logger.info(f"Loaded metadata: {RED}{self.meta.shape[0]}{RESET} samples × {RED}{self.meta.shape[1]}{RESET} cols")
        logger.info(f"Loaded feature table: {RED}{self.table.shape[1]} ({orig_n}){RESET} samples × {RED}{self.table.shape[0]}{RESET} {ftype}")

class _TableProcessor(_ProcessingMixin):
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
        feat_cfg = self.cfg["features"]
        tbl = self.tables["raw"][self.mode]
        
        # Pipeline processing to avoid intermediate copies
        if feat_cfg["filter"]:
            tbl = filter_table(tbl)
            self.tables.setdefault("filtered", {})[self.mode] = tbl
        
        if feat_cfg["normalize"]:
            tbl = normalize_table(tbl, axis=1)
            self.tables.setdefault("normalized", {})[self.mode] = tbl
        
        if feat_cfg["clr_transform"]:
            tbl = clr_transform_table(tbl)
            self.tables.setdefault("clr_transformed", {})[self.mode] = tbl

    def _collapse_taxa(self) -> None:
        lvls = ["phylum", "class", "order", "family", "genus"]
        for ttype in list(self.tables.keys()):
            base_tbl = self.tables[ttype][self.mode]
            self.tables[ttype] = self._run_processing_step(
                f"Collapsing taxonomy for {ttype.replace('_', ' ')} tables...",
                collapse_taxa,
                lvls,
                (),
                lambda lvl, _tbl=base_tbl: _tbl,
                log_template=f"Collapsed {ttype} to {{level}}",
            )

    def _create_presence_absence(self) -> None:
        if not self.cfg["features"]["presence_absence"]:
            return
        lvls = ["phylum", "class", "order", "family", "genus"]
        raw_tbl = self.tables["raw"][self.mode]
        self.tables["presence_absence"] = self._run_processing_step(
            "Converting to presence/absence...",
            presence_absence,
            lvls,
            (),
            lambda lvl: raw_tbl,
        )

    def _save_tables(self) -> None:
        base = Path(self.project_dir.data) / "merged" / "table"
        base.mkdir(parents=True, exist_ok=True)
        
        # Prepare all export tasks
        export_tasks = []
        for ttype, lvls in self.tables.items():
            tdir = base / ttype
            tdir.mkdir(parents=True, exist_ok=True)
            for lvl, tbl in lvls.items():
                out = tdir / lvl / "feature-table.biom"
                out.parent.mkdir(parents=True, exist_ok=True)
                export_tasks.append((tbl, out, ttype, lvl))
        
        # Parallel export
        with ThreadPoolExecutor() as executor:
            futures = []
            for tbl, out, ttype, lvl in export_tasks:
                futures.append(executor.submit(export_h5py, tbl, out))
            
            # Wait for completion
            for future in futures:
                future.result()

class _AnalysisManager(_ProcessingMixin):
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
        self.cfg, self.tables, self.meta, self.verbose = cfg, tables, meta, verbose
        self.figure_output_dir = figure_output_dir
        self.stats: Dict[str, Any] = {}
        self.ordination: Dict[str, Any] = {}
        self.models: Dict[str, Any] = {}
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
        ml_tables = {t: d for t, d in self.tables.items() if t in {"normalized", "clr_transformed"}}
        self._run_ml_feature_selection(ml_tables)
        del ml_tables
        
        # Add FAPROTAX annotations only to top features
        if self.faprotax_enabled and self.top_contaminated_features:
            self._annotate_top_features()
        
    def _get_cached_faprotax(self, taxon: str) -> List[str]:
        if taxon not in self._faprotax_cache:
            self._faprotax_cache[taxon] = faprotax_functions_for_taxon(
                taxon, self.fdb, include_references=False
            )
        return self._faprotax_cache[taxon]
    
    def _annotate_top_features(self) -> None:
        """Batch process annotations to minimize DB lookups"""
        all_taxa = {f['feature'] for f in self.top_contaminated_features + self.top_pristine_features}
        
        # Batch lookup
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self._get_cached_faprotax, all_taxa))
        
        taxon_map = dict(zip(all_taxa, results))
        
        for feat in self.top_contaminated_features:
            feat['faprotax_functions'] = taxon_map.get(feat['feature'], [])
        
        for feat in self.top_pristine_features:
            feat['faprotax_functions'] = taxon_map.get(feat['feature'], [])

    def _run_statistical_tests(self) -> None:
        grp_col = self.cfg.get("group_column", DEFAULT_GROUP_COLUMN)
        grp_vals = self.cfg.get("group_values", [True, False])
        san = StatisticalAnalyzer(self.cfg, self.verbose)
        
        # Calculate total tests
        tot = sum(
            len(lvls) * len([t for t, flag in self.cfg["stats"].get(ttype, {}).items() if flag])
            for ttype, lvls in self.tables.items()
        )
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            transient=True,
            expand=True,
        ) as prog:
            parent_task = prog.add_task(
                "[white]Statistical testing".ljust(DEFAULT_PROGRESS_TEXT_N),
                total=tot
            )
            
            for ttype, lvls in self.tables.items():
                self.stats[ttype] = {}
                tests_config = self.cfg["stats"].get(ttype, {})
                enabled_for_ttype = [test for test, flag in tests_config.items() if flag]
                
                for lvl, tbl in lvls.items():
                    if self.verbose:
                        logger.info(f"Processing {ttype} table at {lvl} level")
                    
                    tbl, m = filter_and_reorder_biom_and_metadata(tbl, self.meta)
                    
                    level_task = prog.add_task(
                        f"[cyan]{ttype}/{lvl}",
                        parent=parent_task,
                        total=len(enabled_for_ttype))
                    
                    res = san.run_tests(
                        tbl, m, grp_col, grp_vals, 
                        enabled_for_ttype, prog, level_task
                    )
                    
                    for key, df in res.items():
                        self.stats.setdefault(ttype, {}).setdefault(key, {})[lvl] = df
                    
                    prog.update(level_task, completed=len(enabled_for_ttype))
                    prog.update(parent_task, advance=len(enabled_for_ttype))

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
        tot = sum(
            len(lvls) * len([m for m in KNOWN_METHODS if self.cfg.get("ordination", {}).get(ttype, {}).get(m, False)])
            for ttype, lvls in self.tables.items()
        )
        if not tot:
            return
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            transient=True,
            expand=True,
        ) as prog:
            parent_task = prog.add_task(
                "[white]Ordination analysis".ljust(DEFAULT_PROGRESS_TEXT_N), 
                total=tot
            )
            
            for ttype, lvls in self.tables.items():
                self.ordination[ttype] = {}
                self.figures[ttype] = {}
                ord_config = self.cfg.get("ordination", {}).get(ttype, {})
                enabled_methods = [m for m in KNOWN_METHODS if ord_config.get(m, False)]
                
                for lvl, tbl in lvls.items():
                    ordir = self.figure_output_dir / lvl / ttype
                    ordn = Ordination(self.cfg, ordir, False)
                    
                    level_task = prog.add_task(
                        f"[cyan]{ttype}/{lvl}",
                        parent=parent_task,
                        total=len(enabled_methods))
                    
                    res, figs = ordn.run_tests(
                        table=tbl,
                        metadata=self.meta,
                        color_col="dataset_name",
                        symbol_col="nuclear_contamination_status",
                        transformation=ttype,
                        enabled_tests=enabled_methods,
                        progress=prog,
                        task_id=level_task,
                    )
                    self.ordination[ttype][lvl] = res
                    self.figures[ttype][lvl] = figs
                    
                    prog.update(level_task, completed=len(enabled_methods))
                    prog.update(parent_task, advance=len(enabled_methods))
                
    def _run_ml_feature_selection(self, ml_tables: Dict) -> None:
        tot = sum(
            len(lvls) * len(self.cfg.get("ml", {}).get("methods", ["rfe"]))
            for ttype, lvls in ml_tables.items()
        )
        if not tot:
            return
            
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            transient=True,
            expand=True,
        ) as prog:
            parent_task = prog.add_task(
                "[white]ML feature selection".ljust(DEFAULT_PROGRESS_TEXT_N),
                total=tot
            )
            
            for ttype, lvls in ml_tables.items():
                self.models[ttype] = {}
                ml_cfg = self.cfg.get("ml", {})
                methods = ml_cfg.get("methods", ["rfe"])
                
                for lvl, tbl in lvls.items():
                    for method in methods:
                        child_task = prog.add_task(
                            f"[cyan]{ttype}/{lvl}/{method}",
                            parent=parent_task,
                            total=1
                        )
                        
                        X = table_to_dataframe(tbl)
                        X.index = X.index.str.lower()
                        y = self.meta.set_index("#sampleid")[[DEFAULT_GROUP_COLUMN]]
                        idx = X.index.intersection(y.index)
                        X, y = X.loc[idx], y.loc[idx]
                        mdir = Path(self.figure_output_dir).parent / "ml" / lvl / ttype
                        catboost_feature_selection(
                            metadata=y,
                            features=X,
                            output_dir=mdir,
                            contamination_status_col=DEFAULT_GROUP_COLUMN,
                            method=method,
                        )
                        prog.update(child_task, completed=1)
                        prog.update(parent_task, advance=1)

class AmpliconData:  
    def __init__(
        self, 
        cfg: Dict, 
        project_dir: Any, 
        mode: str = "genus", 
        verbose: bool = False
    ):
        self.cfg, self.project_dir, self.mode, self.verbose = cfg, project_dir, mode, verbose
        self.fdb = get_faprotax_parsed() if cfg.get("faprotax", False) else None
        
        dl = _DataLoader(cfg, project_dir, mode, verbose)
        self.meta, self.table = dl.meta, dl.table
       
        # Process
        self.figure_output_dir = Path(self.project_dir.figures)
        tp = _TableProcessor(
            cfg, 
            self.table, 
            mode, 
            self.meta, 
            self.figure_output_dir, 
            project_dir, 
            verbose
        )
        self.tables = tp.tables
       
        # Figures
        self.figures: Dict[str, Any] = {}
        if cfg["figures"].get("map", False):
            self.plotter = Plotter(cfg, self.figure_output_dir, verbose)
            self.figures["map"] = self.plotter.generate_sample_map(self.meta)
           
        # Analysis
        am = _AnalysisManager(
            cfg, 
            self.tables, 
            self.meta, 
            self.figure_output_dir, 
            verbose, 
            cfg.get("faprotax", False), 
            self.fdb
        )
        self.stats = am.stats
        self.ordination = am.ordination
        self.models = am.models
        self.top_contaminated_features = am.top_contaminated_features
        self.top_pristine_features = am.top_pristine_features
        self.figures.update(am.figures)
        if verbose:
            logger.info(GREEN + "AmpliconData analysis finished." + RESET)
