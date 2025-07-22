# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Third‑Party Imports
import pandas as pd
from biom.table import Table

# ================================== LOCAL IMPORTS =================================== #

from workflow_16s import constants
from workflow_16s.stats.tests import (
    alpha_diversity, analyze_alpha_diversity, analyze_alpha_correlations,
    fisher_exact_bonferroni, kruskal_bonferroni, mwu_bonferroni, ttest
)
from workflow_16s.utils.data import update_table_and_meta
from workflow_16s.utils.progress import get_progress_bar, _format_task_desc

# ========================== INITIALISATION & CONFIGURATION ========================== #

logger = logging.getLogger("workflow_16s")

# ================================= DEFAULT VALUES =================================== #

def get_enabled_tasks(
    config: Dict, 
    tables: Dict[str, Dict[str, Table]]
):
    stats_config = config.get('stats', {})
    table_config = stats_config.get('tables', {})

    tasks = []
    for table_type, levels in tables.items():
        table_type_config = table_config.get(table_type, {})
        if not table_type_config.get('enabled', False):
            continue

        enabled_levels = [
            l for l in table_type_config.get('levels', levels.keys()) 
            if l in levels
        ]
        enabled_tests = [
            t for t in table_type_config.get('tests', DEFAULT_TESTS[table_type]) 
            if t in KNOWN_TESTS
        ]

        for level in enabled_levels:
            for test in enabled_tests:
                tasks.append((table_type, level, test))
    return tasks


def log_test_results(
    result: pd.DataFrame, 
    table_type: str, 
    level: str, 
    test: str
) -> None:
    """Log statistical test results"""
    sig_mask = result["p_value"] < 0.05
    n_sig = sig_mask.sum()
    
    logger.debug(f"Found {n_sig} significant features for {table_type}/{level}/{test}")
    
    if n_sig == 0:
        logger.debug(f"Top 5 features by p-value ({test}):")
        top_features = result.nsmallest(5, "p_value")
        
        for _, row in top_features.iterrows():
            feat = row.get('feature', 'N/A')
            p_val = row.get('p_value', float('nan'))
            effect = row.get('effect_size', float('nan'))
            logger.debug(f"  {feat}: p={p_val:.3e}, effect={effect:.3f}")
            

def run_statistical_tests_for_group(
    config: Dict, 
    tables: Dict[str, Dict[str, Table]], 
    meta: pd.DataFrame(),
    group_column: str = DEFAULT_GROUP_COLUMN,
    group_column_values: List[Any],
    output_dir: Optional[Path] = None,
    verbose: bool = False
):
    stats_config = config.get('stats', {})
    if not stats_config.get('enabled', False):
        return {}

    tasks = get_enabled_tasks(config, tables)
    if not tasks:
        return {}

    analyzer = StatisticalAnalyzer(config, verbose)
    analyzer_config = analyzer.TEST_CONFIG
    
    group_stats = {}
    
    with get_progress_bar() as progress:
        stats_desc = f"Running statistics for '{group_col}'"
        stats_tasks = progress.add_task(
            _format_task_desc(),
            total=len(tasks)
        )

        for table_type, level, test in tasks:
            test_desc = (
                f"{table_type.replace('_', ' ').title()} ({level.title()})"
                f" → {analyzer_config[test]['name']}"
            )
            progress.update(stats_task, description=_format_task_desc(test_desc))

            output_dir = self.output_dir / 'stats' / group_column / table_type / level
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize data storage
            _init_dict_level(group_stats, table_type, level)
            data_storage = group_stats[table_type][level]

            try:
                # Prepare data
                table = self.tables[table_type][level]
                table_aligned, meta_aligned = update_table_and_meta(table, meta)
                
                # Run statistical test
                result = analyzer_config[test]["func"](
                    table=table_aligned,
                    metadata=meta_aligned,
                    group_column=group_column,
                    group_column_values=group_column_values,
                )
                
                # Save and store results
                result.to_csv(output_dir / f'{test}.{group_column}.tsv', sep='\t', index=True)
                data_storage[test] = result
                
                # Log significant features
                if verbose:
                    if isinstance(result, pd.DataFrame) and "p_value" in result.columns:
                        self._log_test_results(result, table_type, level, test)
                
            except Exception as e:
                logger.error(
                    f"Test '{test}' failed for {table_type}/{level}: {str(e)}"
                )
                data_storage[test] = None
                
            finally:
                progress.update(main_task, advance=1)
                
    progress.update(stats_task, description=_format_task_desc(stats_desc))    
    
    return group_stats
    

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

        group_1_features = [f for f in all_features if f["effect"] > 0]
        group_2_features = [f for f in all_features if f["effect"] < 0]

        group_1_features.sort(key=lambda d: (-d["effect"], d["p_value"]))
        group_2_features.sort(key=lambda d: (d["effect"], d["p_value"]))

        return group_1_features[:100], group_2_features[:100]


def _run_statistical_tests_for_group(
        self,
        group_column: str,
        group_values: List[Any]
    ) -> Dict[str, Any]:
    """Run statistical tests for a given group column"""
    stats_cfg = self.cfg.get("stats", {})
    if not stats_cfg.get("enabled", False):
        return {}

    # Configuration setup
    KNOWN_TESTS = {'fisher', 'ttest', 'mwu_bonferroni', 'kruskal_bonferroni'}
    DEFAULT_TESTS = {
        "raw": ["ttest"],
        "filtered": ['mwu_bonferroni', 'kruskal_bonferroni'],
        "normalized": ['ttest', 'mwu_bonferroni', 'kruskal_bonferroni'],
        "clr_transformed": ['ttest', 'mwu_bonferroni', 'kruskal_bonferroni'],
        "presence_absence": ["fisher"]
    }
    
    # Precompute enabled tasks
    tasks = []
    table_cfg = stats_cfg.get("tables", {})
    for table_type, levels in self.tables.items():
        tt_cfg = table_cfg.get(table_type, {})
        if not tt_cfg.get('enabled', False):
            continue
            
        enabled_levels = [l for l in tt_cfg.get("levels", levels.keys()) 
                         if l in levels]
        enabled_tests = [t for t in tt_cfg.get("tests", DEFAULT_TESTS[table_type])
                         if t in KNOWN_TESTS]
        
        for level in enabled_levels:
            for test in enabled_tests:
                tasks.append((table_type, level, test))
    
    if not tasks:
        return {}
    
    # Initialize analyzer and results storage
    san = StatisticalAnalyzer(self.cfg, self.verbose)
    san_cfg = san.TEST_CONFIG
    group_stats = {}
    debug_mode = getattr(self, 'debug_mode', False)

    with get_progress_bar() as progress:
        main_task = progress.add_task(
            _format_task_desc(f"Running statistics for '{group_column}'"), 
            total=len(tasks)
        )
        
        for table_type, level, test in tasks:
            # Setup description and directories
            test_desc = (f"{table_type.replace('_', ' ').title()} "
                         f"({level.title()}) → {san_cfg[test]['name']}")
            progress.update(main_task, description=_format_task_desc(test_desc))
            
            output_dir = self.output_dir / 'stats' / group_column / table_type / level
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize data storage
            _init_dict_level(group_stats, table_type, level)
            data_storage = group_stats[table_type][level]
            
            # Handle debug mode
            if debug_mode:
                time.sleep(3)
                return {}
            
            try:
                # Prepare data
                table = self.tables[table_type][level]
                table_aligned, meta_aligned = update_table_and_meta(table, self.meta)
                
                # Run statistical test
                result = san_cfg[test]["func"](
                    table=table_aligned,
                    metadata=meta_aligned,
                    group_column=group_column,
                    group_column_values=group_values,
                )
                
                # Save and store results
                result.to_csv(output_dir / f'{test}.tsv', sep='\t', index=True)
                data_storage[test] = result
                
                # Log significant features
                if isinstance(result, pd.DataFrame) and "p_value" in result.columns:
                    self._log_test_results(result, table_type, level, test)
                
            except Exception as e:
                logger.error(
                    f"Test '{test}' failed for {table_type}/{level}: {str(e)}"
                )
                data_storage[test] = None
                
            finally:
                progress.update(main_task, advance=1)
                    
    return group_stats

def _log_test_results(self, result: pd.DataFrame, table_type: str, 
                     level: str, test: str) -> None:
    """Helper to log statistical test results"""
    sig_mask = result["p_value"] < 0.05
    n_sig = sig_mask.sum()
    
    logger.debug(f"Found {n_sig} significant features for {table_type}/{level}/{test}")
    
    if n_sig == 0 and self.verbose:
        logger.debug(f"Top 5 features by p-value ({test}):")
        top_features = result.nsmallest(5, "p_value")
        
        for _, row in top_features.iterrows():
            feat = row.get('feature', 'N/A')
            p_val = row.get('p_value', float('nan'))
            effect = row.get('effect_size', float('nan'))
            logger.debug(f"  {feat}: p={p_val:.3e}, effect={effect:.3f}")
