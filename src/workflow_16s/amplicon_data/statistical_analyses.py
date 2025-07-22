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
