# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import glob
import logging
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Third‑Party Imports
import numpy as np 
import pandas as pd
from biom.table import Table
from rich.progress import Progress, TaskID

# ================================== LOCAL IMPORTS =================================== #

from workflow_16s.utils.biom import (
    collapse_taxa,
    convert_to_biom,
    export_h5py,
    presence_absence,
)
from workflow_16s.utils.progress import create_progress
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

RED = "\033[91m"  # ANSI colours for console logs
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

DEFAULT_PROGRESS_TEXT_N = 65

DEFAULT_GROUP_COLUMN = "nuclear_contamination_status"
DEFAULT_GROUP_COLUMN_VALUES = [True, False]

# ==================================== FUNCTIONS ===================================== #

def print_structure(obj: Any, indent: int = 0, _key: str = "root") -> None:
    """Recursively pretty‑print the nested structure of *obj*."""
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
    """Tiny mix‑in to DRY out progress‑bar + verbose‑logging loops."""

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
            with create_progress() as progress:
                task = progress.add_task(
                    f"[white]{process_name}".ljust(DEFAULT_PROGRESS_TEXT_N),
                    total=len(levels),
                )
                for lvl in levels:
                    processed[lvl] = process_func(get_source(lvl), lvl, *func_args)
                    progress.update(task, advance=1)
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
    """Handles all statistical analyses for amplicon data."""

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
        if progress and task_id:
            ptask = progress.add_task(
               "[white]Running statistical tests...".ljust(DEFAULT_PROGRESS_TEXT_N), 
               total=len(enabled_tests), 
               parent=task_id
            )
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
            if progress and task_id:
                progress.update(ptask, advance=1)
        return results

    def get_effect_size(
       self, 
       test_name: str, 
       row: pd.Series
    ) -> Optional[float]:
        if test_name not in self.TEST_CONFIG:
            return None
        cfg = self.TEST_CONFIG[test_name]
        for col in (cfg["effect_col"], cfg["alt_effect_col"]):
            if col and col in row:
                return row[col]
        return None


class Ordination:
    """Runs PCA/PCoA/t‑SNE/UMAP and produces figures."""

    TEST_CONFIG = {
        "pca": {
            "key": "pca",
            "func": calculate_pca,
            "plot_func": plot_pca,
            "name": "Principal Components Analysis",
        },
        "pcoa": {
            "key": "pcoa",
            "func": calculate_pcoa,
            "plot_func": plot_pcoa,
            "name": "Principal Coordinates Analysis",
            # Requires metric parameter from config
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
        # Get configuration for current transformation type
        trans_cfg = self.cfg.get("ordination", {}).get(transformation, {})
        
        results = {}
        figures = {}
        tests_to_run = [t for t in enabled_tests if t in self.TEST_CONFIG]
        if not tests_to_run:
            return results, figures

        try:
            # Align samples once per table/level
            table, metadata = filter_and_reorder_biom_and_metadata(
                table, metadata
            )
            
            for tname in tests_to_run:
                cfg = self.TEST_CONFIG[tname]
                key = cfg["key"]
                try:
                    # Prepare method-specific parameters
                    method_params = {}
                    
                    # Handle PCoA metric from config
                    if tname == "pcoa":
                        method_params["metric"] = trans_cfg.get(
                            "pcoa_metric", "braycurtis"  # Default metric
                        )
                    
                    # Run ordination method with parameters
                    ord_res = cfg["func"](table=table, **method_params)
                    results[key] = ord_res
                    
                    # Prepare plot parameters
                    pkwargs = {**cfg.get("plot_kwargs", {}), **kwargs}

                    # Handle different result types for plotting
                    if key == 'pca':
                        pkwargs.update({
                            'components': ord_res['components'],
                            'proportion_explained': ord_res['exp_var_ratio']
                        })
                    elif key == 'pcoa':
                        pkwargs.update({
                            'components': ord_res.samples,
                            'proportion_explained': ord_res.proportion_explained
                        })
                    else:  # t-SNE or UMAP
                        pkwargs['df'] = ord_res
                        
                    # Generate plot
                    fig, _ = cfg["plot_func"](
                        metadata=metadata,
                        color_col=color_col,
                        symbol_col=symbol_col,
                        transformation=transformation,
                        output_dir=self.figure_output_dir,
                        **pkwargs
                    )
                    figures[key] = fig
                except Exception as e:
                    logger.error(
                        f"Failed {tname} for {transformation}: {e}"
                    )
                    figures[key] = None
                finally:
                    # Update parent task after EACH method completion
                    if progress and task_id:
                        progress.update(task_id, advance=1)  
        finally:
            return results, figures
              

class Plotter:
    """Generates sample‑location maps coloured by arbitrary metadata cols."""

    def __init__(
       self, 
       cfg: Dict, 
       output_dir: Path, 
       verbose: bool = False
    ):
        self.cfg = cfg
        self.output_dir = output_dir
        self.verbose = verbose

    def generate_sample_map(
        self,
        metadata: pd.DataFrame,
        color_columns: List[str] = (
           "dataset_name", 
           "nuclear_contamination_status"
        ),
        **kwargs,
    ) -> Dict[str, Any]:
        figs: Dict[str, Any] = {}
        for col in color_columns:
            fig, _ = sample_map_categorical(
                metadata=metadata,
                output_dir=self.output_dir,
                color_col=col,
                **kwargs,
            )
            figs[col] = fig
        return figs


class TopFeaturesAnalyzer:
    """
    Detects strongest positive/negative associations across 
    taxonomic levels.
    """

    def __init__(
       self, 
       cfg: Dict, verbose: bool = False
    ):
        self.cfg = cfg
        self.verbose = verbose

    def analyze(
        self,
        stats_results: Dict[str, Dict[str, Dict[str, pd.DataFrame]]],
        group_column: str,
    ) -> Tuple[List[Dict], List[Dict]]:
        cont_feats: List[Dict] = []
        pris_feats: List[Dict] = []
        san = StatisticalAnalyzer(self.cfg, self.verbose)
        levels = ["phylum", "class", "order", "family", "genus"]
        for level in levels:
            lvl_best: Dict[str, Dict] = {}
            for tbl_type, tests in stats_results.items():
                for tname, t_res in tests.items():
                    if level not in t_res:
                        continue
                    for _, row in t_res[level].iterrows():
                        feat = row["feature"]
                        pval = row["p_value"]
                        if pd.isna(pval) or pval > 0.05:
                            continue
                        eff = san.get_effect_size(tname, row)
                        if eff is None:
                            continue
                        cur = lvl_best.get(feat)
                        if not cur or pval < cur["p_value"]:
                            lvl_best[feat] = {
                                "p_value": pval,
                                "effect": eff,
                                "table_type": tbl_type,
                                "test": tname,
                            }
            for feat, res in lvl_best.items():
                entry = {
                    "feature": feat,
                    "level": level,
                    "table_type": res["table_type"],
                    "test": res["test"],
                    "effect": res["effect"],
                    "p_value": res["p_value"],
                }
                (cont_feats if res["effect"] > 0 else pris_feats).append(entry)
        keyf = lambda d: (-abs(d["effect"]), d["p_value"])  
        cont_feats.sort(key=keyf)
        pris_feats.sort(key=keyf)
        return cont_feats, pris_feats
       

class _DataLoader(_ProcessingMixin):
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
        self.meta = import_merged_meta_tsv(
           self._get_metadata_paths(), None, self.verbose
        )

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
        self.table = import_merged_table_biom(
           biom_paths, "table", self.verbose
        )

    def _filter_and_align(self) -> None:
        orig_n = self.table.shape[1]
        self.table, self.meta = filter_and_reorder_biom_and_metadata(
           self.table, self.meta, "#sampleid"
        )
        logger.info(
            f"Loaded metadata: {RED}{self.meta.shape[0]}{RESET} "
            f"samples × {RED}{self.meta.shape[1]}{RESET} cols"
        )
        ftype = "genera" if self.mode == "genus" else "ASVs"
        logger.info(
            f"Loaded feature table: {RED}{self.table.shape[1]} ({orig_n}){RESET} "
            f"samples × {RED}{self.table.shape[0]}{RESET} {ftype}"
        )


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
        filtering = feat_cfg["filter"]
        norm = filtering and feat_cfg["normalize"]
        clr = norm and feat_cfg["clr_transform"]
        enabled = [filtering, norm, clr]
        n_steps = sum(enabled)
        tbl = self.tables["raw"][self.mode]
        with create_progress() as prog:
            task = prog.add_task(
               f"[white]Preprocessing feature tables at the {self.mode} level...".ljust(DEFAULT_PROGRESS_TEXT_N), 
               total=n_steps
            )
            if filtering:
                tbl = filter_table(tbl)
                self.tables.setdefault("filtered", {})[self.mode] = tbl
                prog.update(task, advance=1)
            if norm:
                tbl = normalize_table(tbl, axis=1)
                self.tables.setdefault("normalized", {})[self.mode] = tbl
                prog.update(task, advance=1)
            if clr:
                tbl = clr_transform_table(tbl)
                self.tables.setdefault("clr_transformed", {})[self.mode] = tbl
                prog.update(task, advance=1)

    def _collapse_taxa(self) -> None:
        lvls = ["phylum", "class", "order", "family", "genus"]
        for ttype in list(self.tables.keys()):
            base_tbl = self.tables[ttype][self.mode]
            self.tables[ttype] = self._run_processing_step(
                f"Collapsing taxonomy for {ttype.replace('_', ' ')} feature tables...",
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
            "Converting raw feature tables to presence/absence...",
            presence_absence,
            lvls,
            (),
            lambda lvl: raw_tbl,
        )

    def _save_tables(self) -> None:
        tot = sum(len(v) for v in self.tables.values())
        base = Path(self.project_dir.data) / "merged" / "table"
        base.mkdir(parents=True, exist_ok=True)
        with create_progress() as prog:
            task = prog.add_task(
                "[white]Exporting feature tables...".ljust(DEFAULT_PROGRESS_TEXT_N), 
                total=tot
            )
            for ttype, lvls in self.tables.items():
                tdir = base / ttype
                tdir.mkdir(parents=True, exist_ok=True)
                for lvl, tbl in lvls.items():
                    out = tdir / lvl / "feature-table.biom"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    export_h5py(tbl, out)
                    if self.verbose:
                        logger.info(
                           f"Wrote {ttype} {lvl} table [{tbl.shape[0]}, "
                           f"{tbl.shape[1]}] to '{out}'"
                        )
                    prog.update(task, advance=1)


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
        self.figures: Dict[str, Any] = {}
        self.top_contaminated_features: List[Dict] = []
        self.top_pristine_features: List[Dict] = []
        self.faprotax_enabled, self.fdb = faprotax_enabled, fdb
        self._run_statistical_tests()
        print_structure(self.stats)
        self._identify_top_features()
        if self.faprotax_enabled and self.top_contaminated_features:
            print(self.top_contaminated_features[0])
            print(self.top_contaminated_features[0]["feature"])
            print(
                faprotax_functions_for_taxon(
                    self.top_contaminated_features[0]["feature"], 
                    self.fdb, 
                    include_references=True
                )
            )
        self._run_ordination()
        print_structure(self.ordination)
        self._run_ml_feature_selection() 
        print_structure(self.models)
        
    def _run_statistical_tests(self) -> None:
        grp_col = self.cfg.get("group_column", DEFAULT_GROUP_COLUMN)
        grp_vals = self.cfg.get("group_values", [True, False])
        san = StatisticalAnalyzer(self.cfg, self.verbose)
        
        # Calculate total tests for progress bar: sum over [table levels × enabled tests per table type]
        tot = 0
        for ttype, lvls in self.tables.items():
            # Get enabled tests for this table type from config; fallback to empty list if undefined
            tests_config = self.cfg["stats"].get(ttype, {})
            enabled_for_ttype = [test for test, flag in tests_config.items() if flag]
            tot += len(lvls) * len(enabled_for_ttype)
        
        with create_progress() as prog:
            task = prog.add_task("[white]Running statistical tests".ljust(DEFAULT_PROGRESS_TEXT_N), total=tot)
            for ttype, lvls in self.tables.items():
                self.stats[ttype] = {}
                # Fetch enabled tests for current table type (skip if config missing)
                tests_config = self.cfg["stats"].get(ttype, {})
                enabled_for_ttype = [test for test, flag in tests_config.items() if flag]
                
                for lvl, tbl in lvls.items():
                    tbl, m = filter_and_reorder_biom_and_metadata(tbl, self.meta)
                    # Run only tests enabled for this table type
                    res = san.run_tests(tbl, m, grp_col, grp_vals, enabled_for_ttype, prog, task)
                    for key, df in res.items():
                        self.stats.setdefault(ttype, {}).setdefault(key, {})[lvl] = df
                    # Advance progress bar by number of tests run for this level
                    prog.update(task, advance=len(enabled_for_ttype))

    def _identify_top_features(self) -> None:
        tfa = TopFeaturesAnalyzer(self.cfg, self.verbose)
        self.top_contaminated_features, self.top_pristine_features = tfa.analyze(
            self.stats, DEFAULT_GROUP_COLUMN
        )
        print_structure(self.top_contaminated_features)

    def _run_ordination(self) -> None:
        # Define known ordination methods
        KNOWN_METHODS = ["pca", "pcoa", "tsne", "umap"]
        
        # Calculate total runs: sum over [table levels × enabled methods per table type]
        tot = 0
        for ttype, lvls in self.tables.items():
            # Get config for this table type (default empty dict)
            ord_config = self.cfg.get("ordination", {}).get(ttype, {})
            # Only include known methods that are enabled in config
            enabled_methods = [m for m in KNOWN_METHODS if ord_config.get(m, False)]
            tot += len(lvls) * len(enabled_methods)
        
        if not tot:
            return
        
        with create_progress() as prog:
            task = prog.add_task(
                "[white]Running ordination...".ljust(DEFAULT_PROGRESS_TEXT_N), 
                total=tot
            )
            for ttype, lvls in self.tables.items():
                self.ordination[ttype] = {}
                self.figures[ttype] = {}
                
                # Get enabled methods for current table type from config
                ord_config = self.cfg.get("ordination", {}).get(ttype, {})
                enabled_methods = [m for m in KNOWN_METHODS if ord_config.get(m, False)]
                
                for lvl, tbl in lvls.items():
                    ordir = self.figure_output_dir / lvl / ttype
                    ordn = Ordination(self.cfg, ordir, False)
                    
                    # Run only enabled methods for this table type
                    res, figs = ordn.run_tests(
                        table=tbl,
                        metadata=self.meta,
                        color_col="dataset_name",
                        symbol_col="nuclear_contamination_status",
                        transformation=ttype,
                        enabled_tests=enabled_methods,
                        progress=prog,
                        task_id=task,
                    )
                    self.ordination[ttype][lvl] = res
                    self.figures[ttype][lvl] = figs
                    
                    # Update progress by number of methods run for this level
                    prog.update(task, advance=len(enabled_methods))
                
    def _run_ml_feature_selection(self) -> None:
        if not self.cfg.get("run_ml", False):
            return
        for ttype, lvls in self.tables.items():
            self.models[ttype] = {}
            for lvl, tbl in lvls.items():
                ml_cfg = self.cfg.get("ml", {})
                for method in ml_cfg.get("methods", ["rfe"]):
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
                   

class AmpliconData:  
    """
    End‑user class that orchestrates loading, processing, and analysis.
    """

    def __init__(
       self, 
       cfg: Dict, 
       project_dir: Any, 
       mode: str = "genus", 
       verbose: bool = False
    ):
        self.cfg, self.project_dir, self.mode, self.verbose = cfg, project_dir, mode, verbose
        self.fdb = get_faprotax_parsed() if cfg.get("faprotax", False) else None
       
        # Load
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
