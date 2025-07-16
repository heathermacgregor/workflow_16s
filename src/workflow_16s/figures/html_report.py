# ===================================== IMPORTS ====================================== #
import base64
import itertools
import json
import logging
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from plotly.offline import get_plotlyjs_version

from workflow_16s.utils.io import import_js_as_str

# ========================== INITIALIZATION & CONFIGURATION ========================== #
logger = logging.getLogger('workflow_16s')
script_dir = Path(__file__).parent  
tables_js_path = script_dir / "tables.js"  
css_path = script_dir / "style.css"  
html_template_path = script_dir / "template.html"  

# ===================================== CLASSES ====================================== #
class NumpySafeJSONEncoder(json.JSONEncoder):
    def default(self, obj) -> Any:  
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
def _extract_figures(amplicon_data: "AmpliconData") -> Dict[str, Any]:
    figures = {}
    
    # Ordination figures
    ordination_figures = {}
    for table_type, levels in amplicon_data.ordination.items():
        for level, methods in levels.items():
            for method, data in methods.items():
                if data and 'figures' in data and data['figures']:
                    if table_type not in ordination_figures:
                        ordination_figures[table_type] = {}
                    if level not in ordination_figures[table_type]:
                        ordination_figures[table_type][level] = {}
                    ordination_figures[table_type][level][method] = data['figures']
    figures['ordination'] = ordination_figures

    # Alpha diversity figures
    alpha_figures = {}
    for table_type, levels in amplicon_data.alpha_diversity.items():
        for level, data in levels.items():
            if 'figures' in data and data['figures']:
                if table_type not in alpha_figures:
                    alpha_figures[table_type] = {}
                alpha_figures[table_type][level] = data['figures']
    figures['alpha_diversity'] = alpha_figures

    # Sample maps
    if amplicon_data.maps:
        figures['map'] = amplicon_data.maps

    # SHAP figures
    shap_figures = {}
    for table_type, levels in amplicon_data.models.items():
        for level, methods in levels.items():
            for method, model_result in methods.items():
                if model_result and 'figures' in model_result:
                    if table_type not in shap_figures:
                        shap_figures[table_type] = {}
                    if level not in shap_figures[table_type]:
                        shap_figures[table_type][level] = {}
                    shap_figures[table_type][level][method] = model_result['figures']
    figures['shap'] = shap_figures

    # Violin plots
    violin_figures = {'contaminated': {}, 'pristine': {}}
    for feat in amplicon_data.top_contaminated_features:
        if 'violin_figure' in feat and feat['violin_figure']:
            violin_figures['contaminated'][feat['feature']] = feat['violin_figure']
    for feat in amplicon_data.top_pristine_features:
        if 'violin_figure' in feat and feat['violin_figure']:
            violin_figures['pristine'][feat['feature']] = feat['violin_figure']
    figures['violin'] = violin_figures

    return figures

def _prepare_sections(
    figures: Dict,
    include_sections: List[str],
    id_counter: Iterator[int],
) -> Tuple[List[Dict], Dict]:
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
            btns, tabs, pd = _ordination_to_nested_html(
                figures[sec], id_counter, sec_data["id"]
            )
            plot_data.update(pd)
            sec_data["subsections"].append({
                "title": "Ordination",
                "tabs_html": tabs,
                "buttons_html": btns
            })
        
        elif sec == "alpha_diversity":
            btns, tabs, pd = _alpha_diversity_to_nested_html(
                figures[sec], id_counter, sec_data["id"]
            )
            plot_data.update(pd)
            sec_data["subsections"].append({
                "title": "Alpha Diversity",
                "tabs_html": tabs,
                "buttons_html": btns
            })
        
        elif sec == "map":
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
            btns, tabs, pd = _shap_to_nested_html(
                figures[sec], id_counter, sec_data["id"]
            )
            plot_data.update(pd)
            sec_data["subsections"].append({
                "title": "SHAP Interpretability",
                "tabs_html": tabs,
                "buttons_html": btns
            })
        elif sec == 'violin':
            btns, tabs, pd = _violin_to_nested_html(
                figures[sec], id_counter, sec_data["id"]
            )
            plot_data.update(pd)
            sec_data["subsections"].append({
                "title": "Violin Plots",
                "tabs_html": tabs,
                "buttons_html": btns
            })
        else:
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
            if fig is None:
                raise ValueError("Figure object is None")
                
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
                fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
                buf.seek(0)
                plot_data[plot_id] = {
                    "type": "image",
                    "data": base64.b64encode(buf.read()).decode()
                }
            else:
                plot_data[plot_id] = {
                    "type": "error",
                    "error": f"Unsupported figure type {type(fig)}"
                }
        except Exception as exc:
            logger.exception("Serializing figure failed")
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

def _alpha_correlations_to_nested_html(
    figures: Dict[str, Any],
    id_counter: Iterator[int],
    prefix: str,
) -> Tuple[str, str, Dict]:
    buttons_html, panes_html, plot_data = [], [], {}
    
    for t_idx, (table_type, levels) in enumerate(figures.items()):
        table_id = f"{prefix}-table-{next(id_counter)}"
        is_active_table = t_idx == 0
        
        buttons_html.append(
            f'<button class="table-button {"active" if is_active_table else ""}" '
            f'data-table="{table_id}" '
            f'onclick="showTable(\'{table_id}\')">{table_type}</button>'
        )
        
        level_btns, level_panes = [], []
        for l_idx, (level, variables) in enumerate(levels.items()):
            level_id = f"{table_id}-level-{next(id_counter)}"
            is_active_level = l_idx == 0 and is_active_table
            
            level_btns.append(
                f'<button class="level-button {"active" if is_active_level else ""}" '
                f'data-level="{level_id}" '
                f'onclick="showLevel(\'{level_id}\')">{level}</button>'
            )
            
            var_btns, var_tabs, var_plot_data = _figs_to_html(
                variables, id_counter, level_id
            )
            plot_data.update(var_plot_data)
            
            level_panes.append(
                f'<div id="{level_id}" class="level-pane" '
                f'style="display:{"block" if is_active_level else "none"};">'
                f'<div class="tabs" data-label="variable">{var_btns}</div>'
                f'{var_tabs}'
                f'</div>'
            )
        
        panes_html.append(
            f'<div id="{table_id}" class="table-pane" '
            f'style="display:{"block" if is_active_table else "none"};">'
            f'<div class="tabs" data-label="level">{"".join(level_btns)}</div>'
            f'{"".join(level_panes)}'
            f'</div>'
        )
    
    buttons_row = f'<div class="tabs" data-label="table_type">{"".join(buttons_html)}</div>'
    return buttons_row, "".join(panes_html), plot_data
    
def _prepare_features_table(
    features: List[Dict], 
    max_features: int,
    category: str
) -> pd.DataFrame:
    if not features:
        return pd.DataFrame({"Message": [f"No significant {category} features found"]})
    
    df = pd.DataFrame(features[:max_features])
    df = df.rename(columns={
        "feature": "Feature",
        "level": "Taxonomic Level",
        "test": "Test",
        "effect": "Effect Size",
        "p_value": "P-value",
        "effect_dir": "Direction"
    })
    
    if "faprotax_functions" in df.columns:
        df["Functions"] = df["faprotax_functions"].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else ""
        )
    
    df["Effect Size"] = df["Effect Size"].apply(lambda x: f"{x:.4f}")
    df["P-value"] = df["P-value"].apply(lambda x: f"{x:.2e}")
    
    return df[["Feature", "Taxonomic Level", "Test", "Effect Size", 
               "P-value", "Direction", "Functions"]]

def _prepare_stats_summary(stats: Dict) -> pd.DataFrame:
    summary = []
    for table_type, tests in stats.items():
        for test_name, levels in tests.items():
            for level, df in levels.items():
                if isinstance(df, pd.DataFrame) and "p_value" in df.columns:
                    n_sig = sum(df["p_value"] < 0.05)
                else:
                    n_sig = 0
                summary.append({
                    "Table Type": table_type,
                    "Test": test_name,
                    "Level": level,
                    "Significant Features": n_sig,
                    "Total Features": len(df) if isinstance(df, pd.DataFrame) else 0
                })
    
    return pd.DataFrame(summary)

def _prepare_ml_summary(
    models: Dict, 
    top_contaminated: List[Dict], 
    top_pristine: List[Dict]
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Dict]:
    if not models:
        return None, None, {}

    metrics_summary = []
    features_summary = []
    shap_reports = {}
    
    for table_type, levels in models.items():
        for level, methods in levels.items():
            for method, result in methods.items():
                if not result:
                    continue
                
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
                
                # Capture SHAP report if available
                if "shap_report" in result:
                    key = (table_type, level, method)
                    shap_reports[key] = result["shap_report"]
    
    metrics_df = pd.DataFrame(metrics_summary) if metrics_summary else None
    features_df = pd.DataFrame(features_summary) if features_summary else None
    
    return metrics_df, features_df, shap_reports

def _format_shap_report(report: str) -> str:
    """Convert SHAP report to styled HTML tables matching the report's theme"""
    sections = report.split("\n\n")
    html_output = []
    
    for section in sections:
        section = section.strip()
        if not section:
            continue
            
        # Extract section header
        if section.startswith("**") and "**" in section[2:]:
            header, content = section.split("**", 2)[1], section.split("**", 2)[2]
            header = header.strip().rstrip(':')
        else:
            header = "Report"
            content = section
            
        # Process different section types
        if "Top" in header and "features by average impact" in header:
            # Top features table
            table_data = []
            for line in content.split('\n'):
                if '•' in line:
                    parts = line.split('`')
                    if len(parts) >= 2:
                        feature = parts[1]
                        value = parts[2].split('=')[-1].strip().rstrip(')')
                        table_data.append({"Feature": feature, "Mean |SHAP|": value})
            
            if table_data:
                df = pd.DataFrame(table_data)
                html_output.append(f'<h4>{header}</h4>')
                html_output.append(df.to_html(index=False, classes='dynamic-table'))
                
        elif "Beeswarm interpretation" in header:
            # Beeswarm table
            table_data = []
            for line in content.split('\n'):
                if '•' in line:
                    parts = line.split('`')
                    if len(parts) >= 2:
                        feature = parts[1]
                        interpretation = parts[2].split(':', 1)[1].split('(')[0].strip()
                        rho = line.split('ρ = ')[-1].rstrip(')')
                        table_data.append({
                            "Feature": feature, 
                            "Interpretation": interpretation,
                            "Spearman's ρ": rho
                        })
            
            if table_data:
                df = pd.DataFrame(table_data)
                html_output.append(f'<h4>{header}</h4>')
                html_output.append(df.to_html(index=False, classes='dynamic-table'))
                
        elif "Dependency" in header and "interpretations" in header:
            # Dependency plot table
            table_data = []
            for line in content.split('\n'):
                if '•' in line:
                    parts = line.split('`')
                    if len(parts) >= 2:
                        feature = parts[1]
                        relationship = line.split('shows a ')[1].split('(')[0].strip()
                        rho = line.split('ρ = ')[-1].rstrip(')')
                        table_data.append({
                            "Feature": feature, 
                            "Relationship": relationship,
                            "Spearman's ρ": rho
                        })
            
            if table_data:
                df = pd.DataFrame(table_data)
                html_output.append(f'<h4>{header}</h4>')
                html_output.append(df.to_html(index=False, classes='dynamic-table'))
                
        elif "Interaction summaries" in header:
            # Interaction table
            table_data = []
            for line in content.split('\n'):
                if '•' in line:
                    parts = line.split('`')
                    if len(parts) >= 4:
                        feature = parts[1]
                        partner = parts[3]
                        score = line.split('mean |interaction SHAP| = ')[1].split(')')[0]
                        relationship = line.split('relationship: ')[1].split(' (ρ')[0]
                        rho_feat = line.split('ρ_feat→SHAP = ')[1].split(',')[0]
                        rho_partner = line.split('ρ_partner→SHAP = ')[1].split(')')[0]
                        table_data.append({
                            "Feature": feature,
                            "Partner Feature": partner,
                            "Interaction Strength": score,
                            "Relationship": relationship,
                            "ρ (Feature)": rho_feat,
                            "ρ (Partner)": rho_partner
                        })
            
            if table_data:
                df = pd.DataFrame(table_data)
                html_output.append(f'<h4>{header}</h4>')
                html_output.append(df.to_html(index=False, classes='dynamic-table'))
                
        else:
            # Default formatting for other sections
            html_output.append(f'<h4>{header}</h4>')
            html_output.append(f'<div class="shap-report-content">{content}</div>')
    
    return "\n".join(html_output)

def _format_ml_section(
    ml_metrics: pd.DataFrame, 
    ml_features: pd.DataFrame,
    shap_reports: Dict
) -> str:
    if ml_metrics is None or ml_metrics.empty:
        return "<p>No ML results available</p>"
    
    ml_metrics_html = ml_metrics.to_html(index=False, classes='dynamic-table', table_id='ml-metrics-table')
    
    tooltip_map = {
        "MCC": "Balanced classifier metric (-1 to 1) that considers all confusion matrix values...",
        "ROC AUC": "Probability that random positive ranks higher than random negative...",
        "F1 Score": "Balance between precision and recall...",
        "PR AUC": "Positive-class focused metric for imbalanced data..."
    }
    ml_metrics_html = _add_header_tooltips(ml_metrics_html, tooltip_map)
    
    enhanced_metrics = f"""
    <div class="table-container" id="container-ml-metrics-table">
        {ml_metrics_html}
        <div class="table-controls">
            <div class="pagination-controls">
                <span>Rows per page:</span>
                <select class="rows-per-page" onchange="changePageSize('ml-metrics-table', this.value)">
                    <option value="5">5</option>
                    <option value="10" selected>10</option>
                    <option value="20">20</option>
                    <option value="50">50</option>
                    <option value="100">100</option>
                    <option value="-1">All</option>
                </select>
                <div class="pagination-buttons" id="pagination-ml-metrics-table"></div>
                <span class="pagination-indicator" id="indicator-ml-metrics-table"></span>
            </div>
        </div>
    </div>
    """
    
    features_html = _add_table_functionality(ml_features, 'ml-features-table') if ml_features is not None else "<p>No feature importance data available</p>"
    
    # SHAP Reports section
    shap_html = ""
    if shap_reports:
        shap_html = "<h3>SHAP Interpretability Report</h3>"
        for (table_type, level, method), report in shap_reports.items():
            shap_html += f"""
            <div class="shap-report-section">
                <h4>{table_type} - {level} - {method}</h4>
                <div class="shap-report-content">
                    {_format_shap_report(report)}
                </div>
            </div>
            """
    
    return f"""
    <div class="ml-section">
        <h3>Model Performance</h3>
        {enhanced_metrics}
        
        <h3>Top Features</h3>
        {features_html}
        
        {shap_html}
    </div>
    """

def _shap_to_nested_html(
    figures: Dict[str, Any],
    id_counter: Iterator[int],
    prefix: str,
) -> Tuple[str, str, Dict]:
    buttons_html, panes_html, plot_data = [], [], {}
    
    for table_type, levels in figures.items():
        table_id = f"{prefix}-table-{next(id_counter)}"
        is_first_table = not buttons_html
        buttons_html.append(
            f'<button class="table-button {"active" if is_first_table else ""}" '
            f'data-table="{table_id}" '
            f'onclick="showTable(\'{table_id}\')">{table_type}</button>'
        )
        
        level_btns, level_panes = [], []
        for l_idx, (level, methods) in enumerate(levels.items()):
            level_id = f"{table_id}-level-{next(id_counter)}"
            
            level_btns.append(
                f'<button class="level-button {"active" if l_idx == 0 else ""}" '
                f'data-level="{level_id}" '
                f'onclick="showLevel(\'{level_id}\')">{level}</button>'
            )
            
            method_btns, method_panes = [], []
            for m_idx, (method, plots) in enumerate(methods.items()):
                method_id = f"{level_id}-method-{next(id_counter)}"
                
                # Flatten any list values in the plots dictionary
                flattened_plots = {}
                for plot_type, fig in plots.items():
                    if isinstance(fig, list):
                        # Convert list to dictionary with generated keys
                        for idx, sub_fig in enumerate(fig):
                            flattened_plots[f"{plot_type} - Feature {idx}"] = sub_fig
                    else:
                        flattened_plots[plot_type] = fig
                
                plot_btns, plot_tabs, pd = _figs_to_html(
                    flattened_plots, id_counter, method_id
                )
                plot_data.update(pd)
                
                method_btns.append(
                    f'<button class="method-button {"active" if m_idx == 0 else ""}" '
                    f'data-method="{method_id}" '
                    f'onclick="showMethod(\'{method_id}\')">{method}</button>'
                )
                
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

def _violin_to_nested_html(
    figures_dict: Dict[str, Any],
    id_counter: Iterator[int],
    prefix: str
) -> Tuple[str, str, Dict]:
    buttons_html = []
    tabs_html = []
    plot_data = {}
    cat_counter = itertools.count()
    
    for category, features in figures_dict.items():
        if not features:
            continue
            
        cat_idx = next(cat_counter)
        cat_id = f"{prefix}-cat-{cat_idx}"
        
        buttons_html.append(
            f'<button class="tab-button {"active" if cat_idx==0 else ""}" '
            f'data-tab="{cat_id}" '
            f'onclick="showTab(\'{cat_id}\')">{category.title()}</button>'
        )
        
        feature_tabs, feature_btns, feature_plot_data = _figs_to_html(
            features, id_counter, cat_id
        )
        plot_data.update(feature_plot_data)
        
        tabs_html.append(
            f'<div id="{cat_id}" class="tab-pane" '
            f'style="display:{"block" if cat_idx==0 else "none"}">'
            f'{feature_btns}'
            f'{feature_tabs}'
            f'</div>'
        )
    
    return "\n".join(buttons_html), "\n".join(tabs_html), plot_data

def _alpha_diversity_to_nested_html(
    figures: Dict[str, Any],
    id_counter: Iterator[int],
    prefix: str,
) -> Tuple[str, str, Dict]:
    buttons_html, panes_html, plot_data = [], [], {}
    
    for t_idx, (table_type, levels) in enumerate(figures.items()):
        table_id = f"{prefix}-table-{next(id_counter)}"
        is_active_table = t_idx == 0
        
        buttons_html.append(
            f'<button class="table-button {"active" if is_active_table else ""}" '
            f'data-table="{table_id}" '
            f'onclick="showTable(\'{table_id}\')">{table_type}</button>'
        )
        
        level_btns, level_panes = [], []
        for l_idx, (level, metrics) in enumerate(levels.items()):
            level_id = f"{table_id}-level-{next(id_counter)}"
            is_active_level = l_idx == 0 and is_active_table
            
            level_btns.append(
                f'<button class="level-button {"active" if is_active_level else ""}" '
                f'data-level="{level_id}" '
                f'onclick="showLevel(\'{level_id}\')">{level}</button>'
            )
            
            metric_btns, metric_tabs, metric_plot_data = _figs_to_html(
                metrics, id_counter, level_id
            )
            plot_data.update(metric_plot_data)
            
            level_panes.append(
                f'<div id="{level_id}" class="level-pane" '
                f'style="display:{"block" if is_active_level else "none"};">'
                f'<div class="tabs" data-label="metric">{metric_btns}</div>'
                f'{metric_tabs}'
                f'</div>'
            )
        
        panes_html.append(
            f'<div id="{table_id}" class="table-pane" '
            f'style="display:{"block" if is_active_table else "none"};">'
            f'<div class="tabs" data-label="level">{"".join(level_btns)}</div>'
            f'{"".join(level_panes)}'
            f'</div>'
        )
    
    buttons_row = f'<div class="tabs" data-label="table_type">{"".join(buttons_html)}</div>'
    return buttons_row, "".join(panes_html), plot_data
    
def _add_header_tooltips(
    table_html: str, 
    tooltip_map: Dict[str, str]
) -> str:
    for header, tooltip_text in tooltip_map.items():
        tooltip_html = (
            f'<span class="tooltip">{header}'
            f'<span class="tooltiptext">{tooltip_text}</span>'
            f'</span>'
        )
        table_html = table_html.replace(
            f'<th>{header}</th>', 
            f'<th>{tooltip_html}</th>'
        )
    return table_html
    
def _add_table_functionality(df: pd.DataFrame, table_id: str) -> str:
    if df is None or df.empty:
        return "<p>No data available</p>"
    
    table_html = df.to_html(index=False, classes=f'dynamic-table', table_id=table_id)
    
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
    
def _ordination_to_nested_html(
    figures: Dict[str, Any],
    id_counter: Iterator[int],
    prefix: str,
) -> Tuple[str, str, Dict]:
    buttons_html, panes_html, plot_data = [], [], {}
    
    for t_idx, (table_type, levels) in enumerate(figures.items()):
        table_id = f"{prefix}-table-{next(id_counter)}"
        is_active_table = t_idx == 0
        
        buttons_html.append(
            f'<button class="table-button {"active" if is_active_table else ""}" '
            f'data-table="{table_id}" '
            f'onclick="showTable(\'{table_id}\')">{table_type}</button>'
        )
        
        level_btns, level_panes = [], []
        for l_idx, (level, methods) in enumerate(levels.items()):
            level_id = f"{table_id}-level-{next(id_counter)}"
            is_active_level = l_idx == 0 and is_active_table
            
            level_btns.append(
                f'<button class="level-button {"active" if is_active_level else ""}" '
                f'data-level="{level_id}" '
                f'onclick="showLevel(\'{level_id}\')">{level}</button>'
            )
            
            method_btns, method_tabs, method_plot_data = _figs_to_html(
                methods, id_counter, level_id
            )
            plot_data.update(method_plot_data)
            
            level_panes.append(
                f'<div id="{level_id}" class="level-pane" '
                f'style="display:{"block" if is_active_level else "none"};">'
                f'<div class="tabs" data-label="method">{method_btns}</div>'
                f'{method_tabs}'
                f'</div>'
            )
        
        panes_html.append(
            f'<div id="{table_id}" class="table-pane" '
            f'style="display:{"block" if is_active_table else "none"};">'
            f'<div class="tabs" data-label="level">{"".join(level_btns)}</div>'
            f'{"".join(level_panes)}'
            f'</div>'
        )
    
    buttons_row = f'<div class="tabs" data-label="table_type">{"".join(buttons_html)}</div>'
    return buttons_row, "".join(panes_html), plot_data    
    
def generate_html_report(
    amplicon_data: "AmpliconData",
    output_path: Union[str, Path],
    include_sections: Optional[List[str]] = None,
    max_features: int = 20  
) -> None:
    figures_dict = _extract_figures(amplicon_data)
    
    include_sections = include_sections or [
        k for k, v in figures_dict.items() if v
    ]
    if 'violin' in figures_dict and 'violin' not in include_sections:
        include_sections.append('violin')
    
    ts = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
    ml_metrics, ml_features, shap_reports = _prepare_ml_summary(
        amplicon_data.models,
        amplicon_data.top_contaminated_features,
        amplicon_data.top_pristine_features
    )
    ml_html = _format_ml_section(ml_metrics, ml_features, shap_reports) if ml_metrics is not None else "<p>No ML results available</p>"
    
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

    id_counter = itertools.count()
    sections, plot_data = _prepare_sections(
        figures_dict, include_sections, id_counter
    )
    sections_html = "\n".join(_section_html(s) for s in sections)

    nav_items = [
        ("Analysis Summary", "analysis-summary"),
        *[(sec['title'], sec['id']) for sec in sections]
    ]
    
    nav_html = """
    <div class="toc">
        <h2>Table of Contents</h2>
        <ul>
    """
    for title, section_id in nav_items:
        nav_html += f'<li><a href="#{section_id}">{title}</a></li>\n'
    nav_html += "        </ul>\n    </div>"

    try:
        plotly_ver = get_plotlyjs_version()
    except Exception:
        plotly_ver = "3.0.1"
    plotly_js_tag = (
        f'<script src="https://cdn.plot.ly/plotly-{plotly_ver}.min.js"></script>'
    )

    payload = json.dumps(plot_data, cls=NumpySafeJSONEncoder, ensure_ascii=False)
    payload = payload.replace("</", "<\\/")

    try:
        table_js = import_js_as_str(tables_js_path)
    except Exception as e:
        logger.error(f"Error reading JavaScript file: {e}")
        table_js = ""
        
    tooltip_css = """
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
    
    # Add styling for SHAP report
    shap_css = """
        .shap-report-section {
            margin-top: 30px;
            padding: 20px;
            background: #2a2a2a;  /* Lighter background */
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }
        .shap-report-content {
            margin-top: 15px;
            padding: 15px;
            background: #2a2a2a;  /* Same as section background */
            border-radius: 5px;
            font-family: inherit;  /* Use default font */
            white-space: pre-wrap;
            line-height: 1.6;
            color: #e0e0e0;       /* Lighter text color */
        }
        """
    
    try:
        css_content = css_path.read_text(encoding='utf-8') + tooltip_css + shap_css
    except Exception as e:
        logger.error(f"Error reading CSS file: {e}")
        css_content = tooltip_css + shap_css
        
    try:
        html_template = html_template_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.error(f"Error loading HTML template: {e}")
        html_template = """<!DOCTYPE html>
        <html>
        <head><title>Error</title></head>
        <body>Report generation failed: Missing template</body>
        </html>"""

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
