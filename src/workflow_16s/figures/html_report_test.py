# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import base64
import itertools
import json
import logging
import uuid
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
        self.results = self._get_section_data()
        self._handle_section(target_section)
        self.figure_results = self._get_section_figures(target_section)
        if debug_mode:
            logger.info(f"Results: {self.results}")
            logger.info(f"Figures: {self.figure_results}")

    def _get_section_figures(self, target_section: str):
        fig_info = section_figure_info.get(target_section, {})
        figures = self.figures
        if not figures or not fig_info:
            return {}
    
        level_keys = sorted(
            [key for key in fig_info if key.startswith("level_")],
            key=lambda x: int(x.split("_")[1])
        )
        logger.info(data_keys)
        data_keys = [fig_info[level_key] for level_key in level_keys]
        logger.info(data_keys)
        results = {"main": []}
    
        def recursive_collect(d, depth=0, path={}):
            if depth == len(data_keys):
                # Final level: collect regular figures
                if isinstance(d, dict):
                    for k, v in d.items():
                        results["main"].append({
                            **path,
                            data_keys[-1]: k,
                            f"{level_keys[-1]}_title": level_titles.get(data_keys[-1], data_keys[-1]),
                            "figure": v
                        })
                return
    
            current_key = data_keys[depth]
            current_level_key = level_keys[depth]
            title_name = level_titles.get(current_key, current_key)
    
            if not isinstance(d, dict):
                return
    
            for k, v in d.items():
                path_update = {
                    current_level_key: k,
                    f"{current_level_key}_title": title_name
                }
                recursive_collect(v, depth + 1, {**path, **path_update})
    
        recursive_collect(figures)
    
        # Collect special figures
        special_levels = {
            key: val for key, val in fig_info.items() if key.startswith("special_level_")
        }
        logger.info(special_levels)
    
        for special in special_levels.values():
            for fig_key, meta in special.items():
                for path in self._enumerate_paths(figures, data_keys[:-1]):
                    subfig = self._get_nested_value(figures, path + [fig_key])
                    if subfig:
                        result = {f"level_{i+1}": path[i] for i in range(len(path))}
                        for i in range(len(path)):
                            data_key = data_keys[i]
                            result[f"level_{i+1}_title"] = level_titles.get(data_key, data_key)
                        result[f"special_level_key"] = fig_key
                        result[f"special_level_title"] = meta.get("title", fig_key)
                        result["figure"] = subfig
                        results.setdefault(fig_key, []).append(result)
    
        return results

    
    def _get_section_data(self):
        if not self.section or not self.params:
            return {"main": []}
    
        level_keys = sorted(
            [key for key in self.params if key.startswith('level_')],
            key=lambda x: int(x.split('_')[1])
        )
        data_keys = [self.params[level_key] for level_key in level_keys]
    
        results = {"main": []}
    
        def recursive_collect(d, depth=0, path={}):
            if depth == len(data_keys):
                # Collect regular terminal node
                results["main"].append({
                    **path,
                    'result': d,
                    'result_type': type(d).__name__
                })
                return
    
            current_data_key = data_keys[depth]
            current_level_key = level_keys[depth]
            title_name = level_titles.get(current_data_key, current_data_key)
    
            if not isinstance(d, dict):
                return
    
            for k, v in d.items():
                path_update = {
                    current_level_key: k,
                    f"{current_level_key}_title": title_name
                }
                recursive_collect(v, depth + 1, {**path, **path_update})
    
        recursive_collect(self.section)
    
        # === Handle special levels ===
        special_levels = {
            key: val for key, val in self.params.items() if key.startswith("special_level_")
        }
    
        for special_key, mapping in special_levels.items():
            special_depth = int(special_key.split("_")[-1])
            for special_name, meta in mapping.items():
                # Walk all paths down to special_depth - 1
                for path in self._enumerate_paths(self.section, data_keys[:special_depth - 1]):
                    subdata = self._get_nested_value(self.section, path + [special_name])
                    if isinstance(subdata, (pd.DataFrame, dict, list)):
                        result = {f"level_{i+1}": path[i] for i in range(len(path))}
                        for i in range(len(path)):
                            data_key = data_keys[i]
                            result[f"level_{i+1}_title"] = level_titles.get(data_key, data_key)
                        result[f"special_level_key"] = special_name
                        result[f"special_level_title"] = meta.get("title", special_name)
                        result["result"] = subdata
                        result["result_type"] = type(subdata).__name__
                        results.setdefault(special_name, []).append(result)
    
        return results


    def _handle_section(self, target_section: str):
        combined = {}
        
        # Gather all special levels (any depth) into a flat dict
        special_keys = {}
        for k, v in self.params.items():
            if k.startswith("special_level_") and isinstance(v, dict):
                special_keys.update(v)
    
        for item in self.results:
            df = item.get("result")
    
            if isinstance(df, pd.DataFrame):
                # Determine the key: use special level name if matched, otherwise 'main'
                level_3 = item.get("level_3")
                key = level_3 if level_3 in special_keys else "main"
    
                if target_section == "stats" and key == "main":
                    # Add stats summary
                    n_sig = df["p_value"].lt(0.05).sum() if "p_value" in df.columns else 0
                    df = df.copy()
                    df["Significant Features"] = n_sig
                    df["Total Features"] = len(df)
    
                # Add metadata columns
                meta = {k: v for k, v in item.items() if k not in {"result", "result_type"}}
                for col, val in meta.items():
                    df[col] = val
    
                combined.setdefault(key, []).append(df)
    
        # Final results dict: key -> DataFrame
        self.results = {
            k: rename_columns(pd.concat(dfs, ignore_index=True))
            for k, dfs in combined.items()
        }


    
    def _get_section(self, target_section: str):
        """Get the section attribute from amplicon_data."""
        return getattr(self.amplicon_data, target_section, None)

    def _get_figures(self, target_section: str):
        """Get the section attribute from amplicon_data.figures if it exists."""
        if hasattr(self.amplicon_data, "figures"):
            return getattr(self.amplicon_data.figures, target_section, None)
        return None
        
    def _get_info(self, target_section: str) -> Dict:
        return section_info.get(target_section, {})  # Return empty dict if missing

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
