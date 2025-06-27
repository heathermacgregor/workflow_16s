class TopFeaturesAnalyzer:
    """
    Detects strongest positive/negative associations across 
    taxonomic levels with performance optimizations.
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
        
        # Pre-filter significant results only (p < 0.05)
        sig_results = {}
        for tbl_type, tests in stats_results.items():
            sig_results[tbl_type] = {}
            for tname, t_res in tests.items():
                sig_results[tbl_type][tname] = {}
                for lvl, df in t_res.items():
                    sig_df = df[df["p_value"] < 0.05].copy()
                    if not sig_df.empty:
                        sig_results[tbl_type][tname][lvl] = sig_df
        
        for level in levels:
            lvl_best: Dict[str, Dict] = {}
            for tbl_type, tests in sig_results.items():
                for tname, t_res in tests.items():
                    if level not in t_res:
                        continue
                    # Vectorized effect size calculation
                    t_res[level]["effect"] = t_res[level].apply(
                        lambda row: san.get_effect_size(tname, row), axis=1
                    )
                    # Drop rows with no effect size
                    t_res[level] = t_res[level].dropna(subset=["effect"])
                    
                    for _, row in t_res[level].iterrows():
                        feat = row["feature"]
                        pval = row["p_value"]
                        eff = row["effect"]
                        
                        # Capture effect direction
                        effect_dir = "positive" if eff > 0 else "negative"
                        
                        cur = lvl_best.get(feat)
                        if not cur or pval < cur["p_value"]:
                            lvl_best[feat] = {
                                "p_value": pval,
                                "effect": eff,
                                "effect_dir": effect_dir,
                                "table_type": tbl_type,
                                "test": tname,
                                "level": level,
                            }
            
            # Convert to list of features
            for feat, res in lvl_best.items():
                entry = {
                    "feature": feat,
                    "level": res["level"],
                    "table_type": res["table_type"],
                    "test": res["test"],
                    "effect": res["effect"],
                    "p_value": res["p_value"],
                    "effect_dir": res["effect_dir"],
                }
                (cont_feats if res["effect"] > 0 else pris_feats).append(entry)
                
        # Sort with efficient key function
        keyf = lambda d: (-abs(d["effect"]), d["p_value"])  
        cont_feats.sort(key=keyf)
        pris_feats.sort(key=keyf)
        
        # Limit to top 100 features per category
        return cont_feats[:100], pris_feats[:100]


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
        self._faprotax_cache = {} if faprotax_enabled else None
        
        self._run_statistical_tests()
        
        if self.verbose:
            logger.info("Identifying top features...")
        self._identify_top_features()
        
        # Add FAPROTAX annotations only to top features
        if self.faprotax_enabled and self.top_contaminated_features:
            if self.verbose:
                logger.info("Annotating top features with FAPROTAX...")
            self._annotate_top_features()

    def _get_cached_faprotax(self, taxon: str) -> List[str]:
        """Cached FAPROTAX lookup with memoization"""
        if taxon not in self._faprotax_cache:
            self._faprotax_cache[taxon] = faprotax_functions_for_taxon(
                taxon, self.fdb, include_references=False
            )
        return self._faprotax_cache[taxon]
    
    def _annotate_top_features(self) -> None:
        """Annotate top features using cached lookups"""
        # Process contaminated features
        for feat in self.top_contaminated_features:
            feat['faprotax_functions'] = self._get_cached_faprotax(feat['feature'])
        
        # Process pristine features
        for feat in self.top_pristine_features:
            feat['faprotax_functions'] = self._get_cached_faprotax(feat['feature'])

    def _run_statistical_tests(self) -> None:
        grp_col = self.cfg.get("group_column", DEFAULT_GROUP_COLUMN)
        grp_vals = self.cfg.get("group_values", [True, False])
        san = StatisticalAnalyzer(self.cfg, self.verbose)
        
        # Calculate total tests
        tot = 0
        for ttype, lvls in self.tables.items():
            tests_config = self.cfg["stats"].get(ttype, {})
            enabled_for_ttype = [test for test, flag in tests_config.items() if flag]
            tot += len(lvls) * len(enabled_for_ttype)
        
        with create_progress() as prog:
            task = prog.add_task("[white]Running statistical tests".ljust(DEFAULT_PROGRESS_TEXT_N), total=tot)
            for ttype, lvls in self.tables.items():
                self.stats[ttype] = {}
                tests_config = self.cfg["stats"].get(ttype, {})
                enabled_for_ttype = [test for test, flag in tests_config.items() if flag]
                
                for lvl, tbl in lvls.items():
                    if self.verbose:
                        logger.info(f"Processing {ttype} table at {lvl} level")
                    
                    tbl, m = filter_and_reorder_biom_and_metadata(tbl, self.meta)
                    res = san.run_tests(tbl, m, grp_col, grp_vals, enabled_for_ttype, prog, task)
                    
                    # Store results
                    for key, df in res.items():
                        self.stats.setdefault(ttype, {}).setdefault(key, {})[lvl] = df
                    
                    prog.update(task, advance=len(enabled_for_ttype))

    def _identify_top_features(self) -> None:
        tfa = TopFeaturesAnalyzer(self.cfg, self.verbose)
        self.top_contaminated_features, self.top_pristine_features = tfa.analyze(
            self.stats, DEFAULT_GROUP_COLUMN
        )
        
        if self.verbose:
            logger.info(f"Found {len(self.top_contaminated_features)} contaminated features")
            logger.info(f"Found {len(self.top_pristine_features)} pristine features")
