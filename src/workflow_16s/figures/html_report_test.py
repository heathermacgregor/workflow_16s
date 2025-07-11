# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import base64
import itertools
import json
import logging
import uuid
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

# Third Party Imports
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from plotly.offline import get_plotlyjs_version

# ================================== LOCAL IMPORTS =================================== #

from workflow_16s.utils.io import import_js_as_str
from workflow_16s.utils.amplicon_data import AmpliconData

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger('workflow_16s')

script_dir = Path(__file__).parent  
tables_js_path = script_dir / "tables.js"  
css_path = script_dir / "style.css"  
html_template_path = script_dir / "template.html"  

# ===================================== CLASSES ====================================== #

level_titles = {
    "color_col": "Color",
    "level": "Taxonomic Level",
    "method": "Method",
    "metric": "Metric",
    "table_type": "Table",
    "test_name": "Test",
}

section_info = {
    "stats": { 
        "title": "Statistical Testing",
        "level_1": "table_type",
        "level_2": "level",
        "level_3": "test_name",
    },
    "alpha_diversity": { 
        "title": "Alpha Diversity",
        "level_1": "table_type",
        "level_2": "level",
        "special_level_3": {
            "results": {"title": "Results"},
            "stats": {"title": "Statistics Summary"}
        }
    },
    "ordination": {
        "title": "Beta Diversity",
        "level_1": "table_type",
        "level_2": "level",
        "level_3": "method"
    },
    "ml": {
        "title": "Machine Learning",
        "level_1": "table_type",
        "level_2": "level",
        "level_3": "method"
    }
}

section_figure_info = {
    "alpha_diversity": { 
        "title": "Alpha Diversity",
        "level_1": "table_type",
        "level_2": "level",
        "level_3": "metric",
        "special_level_3": {"summary": {"title": "Summary",}}
    },
    "ordination": {},
    "map": {
        "title": "Sample Map",
        "level_1": "color_col"
    },
    "ml": { 
        "title": "Machine Learning",
        "level_1": "table_type",
        "level_2": "level",
        "level_3": "method",
        "special_level_4": {
            'shap_summary_bar': {"title": "SHAP Summary (Bar)"},
            'shap_summary_beeswarm': {"title": "SHAP Summary (Beeswarm)"},
            'shap_dependency': {"title": "SHAP Dependency"}
        }
    }
}
cols_to_rename = {
    'feature': 'Feature',
    't_statistic': 'T-statistic',
    'p_value': 'P-value', 
    'mean_difference': 'Mean Difference',
    'cohens_d': 'Cohen\'s D',
    'u_statistic': 'U-statistic',
    'median_difference': 'Median Difference',
    'effect_size_r': 'Effect Size (r)',
    'h_statistic': 'H-statistic',
    'epsilon_squared': 'ε²',
    'groups_tested': 'Groups Tested'
}

def rename_columns(df: pd.DataFrame, rename_map: dict = cols_to_rename) -> pd.DataFrame:
    return df.rename(columns=rename_map)
    
debug_mode = True    



class Section:
    def __init__(self, amplicon_data: AmpliconData, target_section: str):
        self.amplicon_data = amplicon_data
        self.section = self._get_section(target_section)
        self.figures = self._get_figures(target_section)
        self.params = self._get_info(target_section)

        self.raw_results = self._extract_data(self.section, self.params)
        self.results = self._process_data(target_section, self.raw_results)

        self.figure_results = self._extract_data(self.figures, section_figure_info.get(target_section, {}), is_figure=True)

        if debug_mode:
            logger.info(f"Results: {self.results}")
            logger.info(f"Figures: {self.figure_results}")

    # ======================== Data Extraction ======================== #

    def _extract_data(self, source, meta_info, is_figure=False):
        if not source or not meta_info:
            return {}

        # Identify ordered level keys and data keys
        level_keys = sorted(
            [k for k in meta_info if k.startswith("level_")],
            key=lambda x: int(x.split("_")[1])
        )
        data_keys = [meta_info[k] for k in level_keys]
        results = defaultdict(list)

        def recurse(d, depth=0, path={}):
            if depth == len(data_keys):
                if is_figure and isinstance(d, dict):
                    for k, v in d.items():
                        results["main"].append({
                            **path,
                            data_keys[-1]: k,
                            f"{level_keys[-1]}_title": level_titles.get(data_keys[-1], data_keys[-1]),
                            "figure": v
                        })
                elif not is_figure:
                    results["main"].append({
                        **path,
                        "result": d,
                        "result_type": type(d).__name__
                    })
                return

            key = data_keys[depth]
            level = level_keys[depth]
            title = level_titles.get(key, key)

            if debug_mode:
                logger.info(f"[extract_data] Extracting {'figures' if is_figure else 'data'}...")
                logger.info(f"  Level keys: {level_keys}")
                logger.info(f"  Data keys: {data_keys}")


            if not isinstance(d, dict):
                return

            for k, v in d.items():
                recurse(v, depth + 1, {**path, level: k, f"{level}_title": title})

        recurse(source)

        # Handle any special levels (figures or data)
        for spec_key, spec_map in meta_info.items():
            if not spec_key.startswith("special_level_"):
                continue

            depth = int(spec_key.split("_")[-1])
            for name, meta in spec_map.items():
                for path in self._enumerate_paths(source, data_keys[:depth - 1]):
                    value = self._get_nested_value(source, path + [name])
                    if value:
                        record = {f"level_{i+1}": path[i] for i in range(len(path))}
                        for i in range(len(path)):
                            record[f"level_{i+1}_title"] = level_titles.get(data_keys[i], data_keys[i])
                        record["figure" if is_figure else "result"] = value
                        record["result_type" if not is_figure else "figure_type"] = type(value).__name__
                        record["special_level_key"] = name
                        record["special_level_title"] = meta.get("title", name)
                        results[name].append(record)
        
        return results

    # ========================= Data Processing ======================== #

    def _process_data(self, section_name, grouped_results):
        combined = {}
        special_names = self._get_special_names(self.params)
        if debug_mode:
            logger.info(f"[process_data] Processing section '{section_name}'")
            logger.info(f"  Result groups: {list(grouped_results.keys())}")

        for group_name, items in grouped_results.items():
            print(group_name)
            dfs = []
            for item in items:
                print(item)
                df = item.get("result")
                if not isinstance(df, pd.DataFrame):
                    continue

                df = df.copy()
                if section_name == "stats" and group_name == "main":
                    df["Significant Features"] = df["p_value"].lt(0.05).sum() if "p_value" in df.columns else 0
                    df["Total Features"] = len(df)

                for col, val in item.items():
                    if col not in {"result", "result_type"}:
                        df[col] = val

                dfs.append(df)

            if dfs:
                combined[group_name] = rename_columns(pd.concat(dfs, ignore_index=True))

        return combined

    # ========================== Helpers =========================== #

    def _get_section(self, section_name: str):
        return getattr(self.amplicon_data, section_name, None)

    def _get_figures(self, section_name: str):
        return getattr(getattr(self.amplicon_data, "figures", None), section_name, None)

    def _get_info(self, section_name: str) -> dict:
        return section_info.get(section_name, {})

    def _get_special_names(self, params: dict) -> set:
        names = set()
        for key, val in params.items():
            if key.startswith("special_level_") and isinstance(val, dict):
                names.update(val.keys())
        return names

    def _enumerate_paths(self, nested_dict, keys) -> list:
        """Get all valid paths down to depth = len(keys)."""
        def walk(d, depth=0, path=[]):
            if depth == len(keys) or not isinstance(d, dict):
                return [path]
            paths = []
            for k in d:
                paths.extend(walk(d[k], depth + 1, path + [k]))
            return paths
        return walk(nested_dict)

    def _get_nested_value(self, d, keys):
        for key in keys:
            if not isinstance(d, dict):
                return None
            d = d.get(key)
        return d

    @classmethod
    def create(cls, amplicon_data, target_section: str):
        if not hasattr(amplicon_data, target_section):
            return None
        return cls(amplicon_data, target_section)
        


def _prepare_stats_summary(stats: Dict) -> pd.DataFrame:
    """
    Create statistical test summary table.
    
    Args:
        stats: Nested dictionary of statistical results.
    
    Returns:
        DataFrame summarizing number of significant features per test/level.
    """
    summary = []
    for table_type, tests in stats.items():
        for test_name, levels in tests.items():
            for level, df in levels.items():
                n_sig = sum(df["p_value"] < 0.05) if "p_value" in df.columns else 0
                summary.append({
                    "Table Type": table_type,
                    "Test": test_name,
                    "Level": level,
                    "Significant Features": n_sig,
                    "Total Features": len(df)
                })
    
    return pd.DataFrame(summary)


def generate_html_report(
    amplicon_data: AmpliconData,
    output_path: Union[str, Path],
    include_sections: Optional[List[str]] = None,
    max_features: int = 20  
) -> None:
    """
    Generate interactive HTML report for 16S analysis results.
    
    Compiles visualizations, statistical summaries, and machine learning
    results into a self-contained HTML file with interactive elements.
    
    Args:
        amplicon_data:    Analysis results container.
        output_path:      Destination path for HTML report.
        include_sections: Sections to include (default: all non-empty sections).
        max_features:     Maximum features to display in differential analysis tables.
    
    Raises:
        IOError: If report file cannot be written.
    """
    include_sections = include_sections or [
        k for k, v in amplicon_data.figures.items() if v
    ]
    if 'violin' in amplicon_data.figures and 'violin' not in include_sections:
        include_sections.append('violin')
    ts = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare tables section 
    tables_html = ""
    
    # Top features tables
    contam_df = _prepare_features_table(
        amplicon_data.top_contaminated_features,
        max_features,
        "Contaminated"
    )
    pristine_df = _prepare_features_table(
        amplicon_data.top_pristine_features,
        max_features,
        "Pristine"
    )
    
    # Stats summary
    stats_df = _prepare_stats_summary(
        amplicon_data.stats
    )
    
    # ML summary
    ml_metrics, ml_features = _prepare_ml_summary(
        amplicon_data.models,
        amplicon_data.top_contaminated_features,
        amplicon_data.top_pristine_features
    )
    ml_html = _format_ml_section(ml_metrics, ml_features)
    
    # Build tables section HTML
    tables_html = f"""
    <div class="subsection">
        <h3>Top Features</h3>
        <h4>Contaminated-Associated Features</h4>
        {_add_table_functionality(contam_df, 'contam-table')}
        
        <h4>Pristine-Associated Features</h4>
        {_add_table_functionality(pristine_df, 'pristine-table')}
    </div>
    
    <div class="subsection">
        <h3>Statistical Summary</h3>
        {_add_table_functionality(stats_df, 'stats-table')}
    </div>
    
    <div class="subsection">
        <h3>Machine Learning Results</h3>
        {ml_html}
    </div>
    """

    # Prepare figures section
    id_counter = itertools.count()
    sections, plot_data = _prepare_sections(
        amplicon_data.figures, include_sections, id_counter
    )
    sections_html = "\n".join(_section_html(s) for s in sections)

    # Prepare navigation section
    nav_items = [
        ("Analysis Summary", "analysis-summary"),
        *[(sec['title'], sec['id']) for sec in sections]
    ]
    
    # Generate navigation HTML
    nav_html = """
    <div class="toc">
        <h2>Table of Contents</h2>
        <ul>
    """
    for title, section_id in nav_items:
        nav_html += f'<li><a href="#{section_id}">{title}</a></li>\n'
    nav_html += "        </ul>\n    </div>"

    # Prepare CDN tag for Plotly 
    try:
        plotly_ver = get_plotlyjs_version()
    except Exception:
        plotly_ver = "3.0.1"
    plotly_js_tag = (
        f'<script src="https://cdn.plot.ly/plotly-{plotly_ver}.min.js"></script>'
    )

    # JSON payload (escape "</" so it can never close the <script>)
    payload = json.dumps(plot_data, cls=NumpySafeJSONEncoder, ensure_ascii=False)
    payload = payload.replace("</", "<\\/")  # safety

    # Table functionality JavaScript
    try:
        table_js = import_js_as_str(tables_js_path)
    except Exception as e:
        logger.error(f"Error reading JavaScript file: {e}")
        table_js = ""
    # CSS
    # Append tooltip CSS to existing styles
    tooltip_css = """
    /* Tooltip styles */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
        border-bottom: 1px dashed #3498db;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 280px;
        background-color: #222;
        color: #fff;
        text-align: left;
        border-radius: 6px;
        padding: 12px;
        position: absolute;
        z-index: 1000;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 14px;
        line-height: 1.5;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    
    .tooltip .tooltiptext::after {
        content: "";
        position: absolute;
        top: 100%;
        left: 50%;
        margin-left: -5px;
        border-width: 5px;
        border-style: solid;
        border-color: #222 transparent transparent transparent;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    """
    
    # CSS
    try:
        css_content = css_path.read_text(encoding='utf-8')
        css_content += tooltip_css  # Append tooltip styles
    except Exception as e:
        logger.error(f"Error reading CSS file: {e}")
        css_content = tooltip_css  # Fallback to just tooltip styles
    # HTML template
    try:
        html_template = html_template_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.error(f"Error loading HTML template: {e}")
        # Fallback minimal template
        html_template = """<!DOCTYPE html>
        <html>
        <head><title>Error</title></head>
        <body>Report generation failed: Missing template</body>
        </html>"""

    # Build the full HTML
    html = html_template.format(
        plotly_js_tag=plotly_js_tag,
        generated_ts=ts,
        section_list=", ".join(include_sections),
        nav_html=nav_html,
        tables_html=tables_html,
        sections_html=sections_html,
        plot_data_json=payload,
        table_js=table_js,
        css_content=css_content
    )
    output_path.write_text(html, encoding="utf-8")
