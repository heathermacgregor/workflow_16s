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

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger('workflow_16s')

script_dir = Path(__file__).parent  
tables_js_path = script_dir / "tables.js"  
css_path = script_dir / "style.css"  
html_template_path = script_dir / "template.html"  

# ===================================== CLASSES ====================================== #

class NumpySafeJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles NumPy data types.
    Extends `json.JSONEncoder` to support encoding of NumPy integers, floats,
    arrays, and booleans into native Python types for JSON serialization.
    
    Methods:
        default: Overridden method to handle NumPy types.
    """
    def default(self, obj) -> Any:  
        """
        Convert NumPy types to Python types for JSON serialization.
        
        Args:
            obj: Object to encode. Supported NumPy types: integer, float,
                 ndarray, bool_.
        
        Returns:
            Python-native representation of the NumPy object.
        
        Note:
            Falls back to default JSON encoder for unsupported types.
        """
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

# ================================== CORE HELPERS =================================== #

def _prepare_sections(
    figures: Dict,
    include_sections: List[str],
    id_counter: Iterator[int],
) -> Tuple[List[Dict], Dict]:
    """
    Organize figures into report sections with hierarchical structure.
    
    Processes visualization figures into HTML-ready section structures. Handles
    special section types (ordination, alpha diversity, etc.) with nested tab
    layouts and prepares plot data for client-side rendering.
    
    Args:
        figures:          Dictionary of section names to figure objects.
        include_sections: List of section names to include in report.
        id_counter:       Iterator generating unique IDs for DOM elements.
    
    Returns:
        Tuple containing:
            - sections: List of section dictionaries with HTML metadata
            - plot_data: Dictionary of plot data for client-side rendering
    """
    sections = []
    plot_data: Dict[str, Any] = {}

    for sec in include_sections:
        if sec not in figures:
            continue

        sec_data = {
            "id": f"sec-{uuid.uuid4().hex}", 
            "title": sec.title(), 
            "subsections": []
        }

        if sec == "ordination":
            # Use nested tab structure for ordination
            btns, tabs, pd = _ordination_to_nested_html(
                figures[sec], id_counter, sec_data["id"]
            )
            plot_data.update(pd)
            sec_data["subsections"].append({
                "title": "Ordination",
                "tabs_html": tabs,
                "buttons_html": btns
            })
        
        elif sec == "alpha_correlations":
            # Use nested structure for alpha correlations
            btns, tabs, pd = _alpha_correlations_to_nested_html(
                figures[sec], id_counter, sec_data["id"]
            )
            plot_data.update(pd)
            sec_data["subsections"].append({
                "title": "Alpha Diversity Correlations",
                "tabs_html": tabs,
                "buttons_html": btns
            })
        elif sec == "map":
            # Sample map section
            flat: Dict[str, Any] = {}
            _flatten(figures[sec], [], flat)
            if flat:
                tabs, btns, pd = _figs_to_html(
                    flat, id_counter, sec_data["id"]
                )
                plot_data.update(pd)
                sec_data["subsections"].append({
                    "title": "Sample Maps",
                    "tabs_html": tabs,
                    "buttons_html": btns
                })
        elif sec == "shap":
            # SHAP plots section
            btns, tabs, pd = _shap_to_nested_html(
                figures[sec], id_counter, sec_data["id"]
            )
            plot_data.update(pd)
            sec_data["subsections"].append({
                "title": "SHAP Interpretability",
                "tabs_html": tabs,
                "buttons_html": btns
            })
        else:
            # Existing non-ordination sections
            flat: Dict[str, Any] = {}
            _flatten(figures[sec], [], flat)
            if flat:
                tabs, btns, pd = _figs_to_html(
                    flat, id_counter, sec_data["id"], row_label="color_col"
                )
                plot_data.update(pd)
                sec_data["subsections"].append({
                    "title": "All",
                    "tabs_html": tabs,
                    "buttons_html": btns
                })
        
        if sec_data["subsections"]:
            sections.append(sec_data)

    return sections, plot_data


def _flatten(tree: Dict, keys: List[str], out: Dict) -> None:
    """
    Recursively flatten a nested dictionary structure.
    
    Converts nested dictionaries into flat key-value pairs where keys are
    concatenated using ' - ' separator.
    
    Args:
        tree: Nested dictionary to flatten.
        keys: Current key path (used recursively).
        out:  Target dictionary for flattened output.
    """
    for k, v in tree.items():
        new_keys = keys + [k]
        if isinstance(v, dict):
            _flatten(v, new_keys, out)
        else:
            out[" - ".join(new_keys)] = v


def _figs_to_html(
    figs: Dict[str, Any], 
    counter: Iterator[int], 
    prefix: str, 
    *, 
    square: bool = False,
    row_label: Optional[str] = None
) -> Tuple[str, str, Dict]:
    """
    Convert figure objects to HTML tab structures and serialized plot data.
    
    Generates HTML tabs/buttons for figure navigation and prepares plot data
    in a format suitable for client-side rendering (Plotly/Matplotlib).
    
    Args:
        figs:      Dictionary of figure titles to figure objects.
        counter:   Iterator for generating unique DOM IDs.
        prefix:    HTML ID prefix for generated elements.
        square:    Whether plots should maintain square aspect ratio.
        row_label: Optional label for row grouping in UI.
    
    Returns:
        Tuple containing:
            - HTML string for tab panes
            - HTML string for tab buttons
            - Dictionary of plot data for client-side rendering
    
    Note:
        Supports Plotly figures and Matplotlib Figure objects.
    """
    tabs, btns, plot_data = [], [], {}

    for title, fig in figs.items():
        idx     = next(counter)
        tab_id  = f"{prefix}-tab-{idx}"
        plot_id = f"{prefix}-plot-{idx}"

        btns.append(
            f'<button class="tab-button {"active" if idx==0 else ""}" '
            f'data-tab="{tab_id}" '
            f'onclick="showTab(\'{tab_id}\', \'{plot_id}\')">{title}</button>'
        )

        tabs.append(
            f'<div id="{tab_id}" class="tab-pane" '
            f'style="display:{"block" if idx==0 else "none"}" '
            f'data-plot-id="{plot_id}">'
            f'<div id="container-{plot_id}" class="plot-container"></div></div>'
        )

        try:
            if hasattr(fig, "to_plotly_json"):
                pj = fig.to_plotly_json()
                pj.setdefault("layout", {})["showlegend"] = False
                plot_data[plot_id] = {
                    "type": "plotly",
                    "data": pj["data"],
                    "layout": pj["layout"],
                    "square": square
                }
            elif isinstance(fig, Figure):
                buf = BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight")
                buf.seek(0)
                plot_data[plot_id] = {
                    "type": "image",
                    "data": base64.b64encode(buf.read()).decode()
                }
            else:
                plot_data[plot_id] = {
                    "type": "error",
                    "error": f"Unsupported figure type {type(fig)}"
                }
        except Exception as exc:  # pragma: no cover
            logger.exception("Serialising figure failed")
            plot_data[plot_id] = {
                "type": "error", 
                "error": str(exc)
            }
    buttons_html = "\n".join(btns)
    if row_label:
        buttons_html = (
            f'<div class="tabs" data-label="{row_label}">'
            f'{buttons_html}</div>'
        )
    else:
        buttons_html = f'<div class="tabs">{buttons_html}</div>'
        
    return "\n".join(tabs), buttons_html, plot_data


def _section_html(sec: Dict) -> str:
    """
    Generate HTML for a report section with subsections.
    
    Args:
        sec: Section dictionary containing:
            - id: HTML ID for section
            - title: Section title
            - subsections: List of subsection dictionaries
    
    Returns:
        HTML string representing the full section structure.
    """
    sub_html = "\n".join(
        f'<div class="subsection">\n'
        f'  <h3>{sub["title"]}</h3>\n'
        f'  <div class="tab-content">\n'          
        f'    <div class="tabs">{sub["buttons_html"]}</div>\n'
        f'    {sub["tabs_html"]}\n'
        f'  </div>\n'                             
        f'</div>'
        for sub in sec["subsections"]
    )
    return f'<div class="section" id="{sec["id"]}">\n' \
           f'  <h2>{sec["title"]}</h2>\n{sub_html}\n</div>'


def _prepare_features_table(
    features: List[Dict], 
    max_features: int,
    category: str
) -> pd.DataFrame:
    """
    Format top differential features into display-ready DataFrame.
    
    Args:
        features:     List of feature dictionaries with statistical results.
        max_features: Maximum number of features to include.
        category:     Feature category label ('Contaminated'/'Pristine').
    
    Returns:
        Formatted DataFrame with selected columns and formatted values.
    
    Note:
        Returns placeholder DataFrame if no features found.
    """
    if not features:
        return pd.DataFrame({"Message": [f"No significant {category} features found"]})
    
    df = pd.DataFrame(features[:max_features])
    # Simplify column names and select important columns
    df = df.rename(columns={
        "feature": "Feature",
        "level": "Taxonomic Level",
        "test": "Test",
        "effect": "Effect Size",
        "p_value": "P-value",
        "effect_dir": "Direction"
    })
    
    # Add FAPROTAX annotations if available
    if "faprotax_functions" in df.columns:
        df["Functions"] = df["faprotax_functions"].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else ""
        )
    
    # Format numeric columns
    df["Effect Size"] = df["Effect Size"].apply(lambda x: f"{x:.4f}")
    df["P-value"] = df["P-value"].apply(lambda x: f"{x:.2e}")
    
    return df[["Feature", "Taxonomic Level", "Test", "Effect Size", 
               "P-value", "Direction", "Functions"]]


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


def _prepare_ml_summary(
    models: Dict, 
    top_contaminated: List[Dict], 
    top_pristine: List[Dict]
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Compile machine learning results into summary DataFrames.
    
    Args:
        models:           Nested dictionary of ML model results.
        top_contaminated: Top contaminated-associated features.
        top_pristine:     Top pristine-associated features.
    
    Returns:
        Tuple containing:
            - Model metrics DataFrame (accuracy, AUC, etc.)
            - Feature importance DataFrame
    """
    if not models:
        return None, None

    metrics_summary = []
    features_summary = []
    
    for table_type, levels in models.items():
        for level, methods in levels.items():
            for method, result in methods.items():
                if not result:
                    continue
                
                # Extract metrics
                test_scores = result.get("test_scores", {})
                metrics = {
                    "Table Type": table_type,
                    "Level": level,
                    "Method": method,
                    "Top Features": len(result.get("top_features", [])),
                    "Accuracy": f"{test_scores.get('accuracy', 0):.4f}",
                    "F1 Score": f"{test_scores.get('f1', 0):.4f}",
                    "MCC": f"{test_scores.get('mcc', 0):.4f}",
                    "ROC AUC": f"{test_scores.get('roc_auc', 0):.4f}",
                    "PR AUC": f"{test_scores.get('pr_auc', 0):.4f}"
                }
                metrics_summary.append(metrics)
                
                # Extract top features
                feat_imp = result.get("feature_importances", {})
                top_features = result.get("top_features", [])[:10]
                for i, feat in enumerate(top_features, 1):
                    importance = feat_imp.get(feat, 0)
                    features_summary.append({
                        "Table Type": table_type,
                        "Level": level,
                        "Method": method,
                        "Rank": i,
                        "Feature": feat,
                        "Importance": f"{importance:.4f}"
                    })
    
    metrics_df = pd.DataFrame(metrics_summary) if metrics_summary else None
    features_df = pd.DataFrame(features_summary) if features_summary else None
    
    return metrics_df, features_df


def _format_ml_section(
    ml_metrics: pd.DataFrame, 
    ml_features: pd.DataFrame
) -> str:
    """
    Generate HTML for machine learning results section.
    
    Args:
        ml_metrics:  Model performance metrics DataFrame.
        ml_features: Feature importance DataFrame.
    
    Returns:
        HTML string for ML section with interactive tables.
    """
    if ml_metrics is None or ml_metrics.empty:
        return "<p>No ML results available</p>"
    
    return f"""
    <div class="ml-section">
        <h3>Model Performance</h3>
        {_add_table_functionality(ml_metrics, 'ml-metrics-table')}
        
        <h3>Top Features</h3>
        {_add_table_functionality(ml_features, 'ml-features-table')}
    </div>
    """


def _shap_to_nested_html(
    figures: Dict[str, Any],
    id_counter: Iterator[int],
    prefix: str,
) -> Tuple[str, str, Dict]:
    """
    Generate nested HTML structure for SHAP visualization section.
    
    Creates 4-level tab hierarchy (table_type → level → method → plot type).
    
    Args:
        figures:    Nested dictionary of SHAP plots.
        id_counter: Iterator for unique DOM IDs.
        prefix:     HTML ID prefix for generated elements.
    
    Returns:
        Tuple containing:
            - HTML for section buttons
            - HTML for tab panes
            - Serialized plot data dictionary
    """
    buttons_html, panes_html, plot_data = [], [], {}
    
    for table_type, levels in figures.items():
        # Table type button
        table_id = f"{prefix}-table-{next(id_counter)}"
        is_first_table = not buttons_html
        buttons_html.append(
            f'<button class="table-button {"active" if is_first_table else ""}" '
            f'data-table="{table_id}" '
            f'onclick="showTable(\'{table_id}\')">{table_type}</button>'
        )
        
        # Build table pane
        level_btns, level_panes = [], []
        for l_idx, (level, methods) in enumerate(levels.items()):
            level_id = f"{table_id}-level-{next(id_counter)}"
            
            # Level button
            level_btns.append(
                f'<button class="level-button {"active" if l_idx == 0 else ""}" '
                f'data-level="{level_id}" '
                f'onclick="showLevel(\'{level_id}\')">{level}</button>'
            )
            
            # Build level pane
            method_btns, method_panes = [], []
            for m_idx, (method, plots) in enumerate(methods.items()):
                method_id = f"{level_id}-method-{next(id_counter)}"
                
                # Method button
                method_btns.append(
                    f'<button class="method-button {"active" if m_idx == 0 else ""}" '
                    f'data-method="{method_id}" '
                    f'onclick="showMethod(\'{method_id}\')">{method}</button>'
                )
                
                # Build method pane with plots
                plot_btns, plot_tabs, pd = _figs_to_html(
                    plots, id_counter, method_id
                )
                plot_data.update(pd)
                
                method_panes.append(
                    f'<div id="{method_id}" class="method-pane" '
                    f'style="display:{"block" if m_idx == 0 else "none"};">'
                    f'{plot_btns}'
                    f'{plot_tabs}'
                    f'</div>'
                )
            
            level_panes.append(
                f'<div id="{level_id}" class="level-pane" '
                f'style="display:{"block" if l_idx == 0 else "none"};">'
                f'<div class="tabs" data-label="method">{"".join(method_btns)}</div>'
                f'{"".join(method_panes)}'
                f'</div>'
            )
        
        panes_html.append(
            f'<div id="{table_id}" class="table-pane" '
            f'style="display:{"block" if is_first_table else "none"};">'
            f'<div class="tabs" data-label="level">{"".join(level_btns)}</div>'
            f'{"".join(level_panes)}'
            f'</div>'
        )
    
    buttons_row = (
        f'<div class="tabs" data-label="table_type">{"".join(buttons_html)}</div>'
    )
    return buttons_row, "".join(panes_html), plot_data


def _add_table_functionality(df: pd.DataFrame, table_id: str) -> str:
    """
    Enhance DataFrame HTML with interactive features.
    
    Adds sorting, pagination, and row controls to generated tables.
    
    Args:
        df:       DataFrame to display.
        table_id: Unique HTML ID for the table.
    
    Returns:
        HTML string with table and interactive controls.
    """
    # Generate base table HTML
    table_html = df.to_html(index=False, classes=f'dynamic-table', table_id=table_id)
    
    # Add sorting and pagination controls
    enhanced_html = f"""
    <div class="table-container" id="container-{table_id}">
        {table_html}
        <div class="table-controls">
            <div class="pagination-controls">
                <span>Rows per page:</span>
                <select class="rows-per-page" onchange="changePageSize('{table_id}', this.value)">
                    <option value="5">5</option>
                    <option value="10" selected>10</option>
                    <option value="20">20</option>
                    <option value="50">50</option>
                    <option value="100">100</option>
                    <option value="-1">All</option>
                </select>
                <div class="pagination-buttons" id="pagination-{table_id}"></div>
                <span class="pagination-indicator" id="indicator-{table_id}"></span>
            </div>
        </div>
    </div>
    """
    return enhanced_html

# ======================== ALPHA DIVERSITY PROCESSING FUNCTION ======================== #

def _alpha_diversity_to_nested_html(
    alpha_diversity_results: Dict[str, Dict[str, Dict[str, pd.DataFrame]]],
    alpha_diversity_stats: Dict[str, Dict[str, Dict[str, pd.DataFrame]]],
    project_dir: Path,
    output_dir: Path,
) -> Tuple[str, str, Dict]:
    """
    Converts alpha diversity results into nested HTML content (buttons and tabs).
    
    Args:
        alpha_diversity_results: Dictionary of alpha diversity results
        alpha_diversity_stats: Dictionary of statistical results
        project_dir: Project root directory
        output_dir: Output directory for the report
        
    Returns:
        Tuple of (buttons HTML, tabs HTML, plot data dictionary)
    """
    buttons = []
    tabs = []
    plot_data = {}
    
    # Create nested structure: table_type -> level -> metric
    for table_type, levels in alpha_diversity_results.items():
        table_buttons = []
        table_tabs = []
        
        for level, metrics in levels.items():
            level_buttons = []
            level_tabs = []
            
            # Get stats for this level
            stats = alpha_diversity_stats.get(table_type, {}).get(level, {})
            
            for metric, result_df in metrics.items():
                # Skip summary entries
                if metric == "summary":
                    continue
                    
                # Create unique ID
                tab_id = f"alpha-{table_type}-{level}-{metric}".replace("_", "-")
                
                # Create button
                level_buttons.append(
                    f'<button class="tablinks" onclick="openTab(event, \'{tab_id}\')">{metric}</button>'
                )
                
                # Get plot path
                plot_path = project_dir / "figures" / "alpha_diversity" / table_type / level / f"{metric}.png"
                rel_plot_path = plot_path.relative_to(output_dir)
                
                # Get statistical results for this metric
                metric_stats = stats.get(metric, pd.DataFrame())
                
                # Create tab content
                tab_content = f'<div class="tabcontent" id="{tab_id}">'
                tab_content += f'<img src="{rel_plot_path}" alt="{metric} alpha diversity">'
                
                if not metric_stats.empty:
                    tab_content += '<div class="stats-summary">'
                    tab_content += metric_stats.to_html()
                    tab_content += '</div>'
                
                tab_content += '</div>'
                level_tabs.append(tab_content)
                
                # Store plot data
                plot_data[f"{table_type}-{level}-{metric}"] = {
                    "type": "alpha_diversity",
                    "path": str(rel_plot_path),
                    "stats": metric_stats.to_dict()
                }
            
            # Add summary if exists
            if "summary" in metrics:
                tab_id = f"alpha-{table_type}-{level}-summary".replace("_", "-")
                plot_path = project_dir / "figures" / "alpha_diversity" / table_type / level / "summary.png"
                rel_plot_path = plot_path.relative_to(output_dir)
                
                level_buttons.append(
                    f'<button class="tablinks" onclick="openTab(event, \'{tab_id}\')">Summary</button>'
                )
                
                tab_content = f'<div class="tabcontent" id="{tab_id}">'
                tab_content += f'<img src="{rel_plot_path}" alt="Alpha diversity summary">'
                tab_content += '</div>'
                level_tabs.append(tab_content)
                
                plot_data[f"{table_type}-{level}-summary"] = {
                    "type": "alpha_summary",
                    "path": str(rel_plot_path)
                }
            
            # Wrap level content
            level_id = f"alpha-{table_type}-{level}".replace("_", "-")
            level_button = f'<button class="accordion" onclick="toggleAccordion(event)">{level.capitalize()}</button>'
            level_div = f'<div class="panel"><div class="tab">{"".join(level_buttons)}</div>{"".join(level_tabs)}</div>'
            
            table_buttons.append(level_button)
            table_tabs.append(level_div)
        
        # Wrap table type content
        table_id = f"alpha-{table_type}".replace("_", "-")
        table_button = f'<button class="accordion" onclick="toggleAccordion(event)">{table_type.replace("_", " ").title()}</button>'
        table_div = f'<div class="panel">{"".join(table_buttons)}{"".join(table_tabs)}</div>'
        
        buttons.append(table_button)
        tabs.append(table_div)
    
    return "".join(buttons), "".join(tabs), plot_data
    
def generate_html_report(
    amplicon_data: "AmpliconData",
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
    try:
        css_content = css_path.read_text(encoding='utf-8')
    except Exception as e:
        logger.error(f"Error reading CSS file: {e}")
        css_content = ""
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
        table_js=table_js
    )
    output_path.write_text(html, encoding="utf-8")
