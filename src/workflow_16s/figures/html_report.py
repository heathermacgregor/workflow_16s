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
    group_1_name = 'contaminated'
    group_2_name = 'pristine'
    violin_figures = {group_1_name: {}, group_2_name: {}}
    for feat in amplicon_data.top_features_group_1:
        if 'violin_figure' in feat and feat['violin_figure']:
            violin_figures[group_1_name][feat['feature']] = feat['violin_figure']
    for feat in amplicon_data.top_features_group_2:
        if 'violin_figure' in feat and feat['violin_figure']:
            violin_figures[group_2_name][feat['feature']] = feat['violin_figure']
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
            out[" - ".join(new_keys)] = v

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
            f'onclick="showTab(event, \'{tab_id}\', \'{plot_id}\')">{title}</button>'
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
        f'    {sub["buttons_html"]}\n'
        f'    {sub["tabs_html"]}\n'
        f'  </div>\n'
        f'</div>'
        for sub in sec["subsections"]
    )

    return f'''
    <div class="section" id="{sec["id"]}">
        <div class="section-header" onclick="toggleSection('{sec["id"]}-content', this)">
            <h2>{sec["title"]}</h2>
            <span class="toggle-icon">▼</span>
        </div>
        <div class="section-content" id="{sec["id"]}-content">
            {sub_html}
        </div>
    </div>
    '''

def _ordination_to_nested_html( # NEW FUNCTION
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
            f'onclick="showTable(\'{table_id}\', this)">{table_type}</button>'
        )

        level_btns, level_panes = [], []
        for l_idx, (level, methods) in enumerate(levels.items()):
            level_id = f"{table_id}-level-{next(id_counter)}"
            is_active_level = l_idx == 0 and is_active_table

            level_btns.append(
                f'<button class="level-button {"active" if is_active_level else ""}" '
                f'data-level="{level_id}" '
                f'onclick="showLevel(\'{level_id}\', this)">{level}</button>'
            )

            method_btns, method_tabs, method_plot_data = [], [], {}
            for m_idx, (method, plots) in enumerate(methods.items()):
                method_id = f"{level_id}-method-{next(id_counter)}"
                is_active_method = m_idx == 0 and is_active_level

                # Flatten any list values in the plots dictionary
                flattened_plots = {}
                for plot_type, fig in plots.items():
                    if isinstance(fig, list):
                        for idx, sub_fig in enumerate(fig):
                            flattened_plots[f"{plot_type} - {idx+1}"] = sub_fig
                    else:
                        flattened_plots[plot_type] = fig

                plot_tabs_html, plot_btns_html, pd = _figs_to_html(
                    flattened_plots, id_counter, method_id
                )
                method_plot_data.update(pd)


                method_btns.append(
                    f'<button class="method-button {"active" if is_active_method else ""}" '
                    f'data-method="{method_id}" '
                    f'onclick="showMethod(\'{method_id}\', this)">{method}</button>'
                )

                method_tabs.append(
                    f'<div id="{method_id}" class="method-pane" '
                    f'style="display:{"block" if is_active_method else "none"};">'
                    f'<div class="tabs" data-label="plot_type">{plot_btns_html}</div>'
                    f'{plot_tabs_html}'
                    f'</div>'
                )
            plot_data.update(method_plot_data)

            level_panes.append(
                f'<div id="{level_id}" class="level-pane" '
                f'style="display:{"block" if is_active_level else "none"};">'
                f'<div class="tabs" data-label="method">{"".join(method_btns)}</div>'
                f'{"".join(method_tabs)}'
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

def _parse_shap_report(report: str) -> Dict[str, Dict[str, str]]:
    """Parse SHAP report string into structured feature data"""
    shap_data = {}
    sections = report.split("\n\n")

    for section in sections:
        section = section.strip()
        if not section:
            continue

        header, _, content = section.partition('\n')
        header = header.strip().strip('*').strip()

        # Top features section
        if "Top features by average impact" in header:
            for line in content.split('\n'):
                if '•' in line and '`' in line:
                    parts = line.split('`')
                    if len(parts) >= 2:
                        feature = parts[1]
                        value = parts[2].split('=')[-1].strip().rstrip(')')
                        if feature not in shap_data:
                            shap_data[feature] = {}
                        shap_data[feature]['mean_shap'] = value

        # Beeswarm interpretation
        elif "Beeswarm interpretation" in header:
            for line in content.split('\n'):
                if '•' in line and '`' in line:
                    parts = line.split('`')
                    if len(parts) >= 2:
                        feature = parts[1]
                        interpretation = parts[2].split(':', 1)[1].split('(')[0].strip()
                        rho = line.split('ρ = ')[-1].rstrip(')') if 'ρ = ' in line else ''
                        if feature not in shap_data:
                            shap_data[feature] = {}
                        shap_data[feature]['beeswarm_interpretation'] = interpretation
                        shap_data[feature]['spearman_rho'] = rho

        # Dependency plot interpretations
        elif "Dependency plot interpretations" in header:
            for line in content.split('\n'):
                if '•' in line and '`' in line:
                    parts = line.split('`')
                    if len(parts) >= 2:
                        feature = parts[1]
                        relationship = line.split('shows a ')[1].split('(')[0].strip() if 'shows a ' in line else ''
                        rho = line.split('ρ = ')[-1].rstrip(')') if 'ρ = ' in line else ''
                        if feature not in shap_data:
                            shap_data[feature] = {}
                        shap_data[feature]['dependency_relationship'] = relationship
                        shap_data[feature]['dependency_rho'] = rho

        # Interaction summaries
        elif "Interaction summaries" in header:
            for line in content.split('\n'):
                if '•' in line and '`' in line:
                    parts = line.split('`')
                    if len(parts) >= 4:
                        feature = parts[1]
                        partner = parts[3]
                        score = line.split('mean |interaction SHAP| = ')[1].split(')')[0] if 'mean |interaction SHAP| = ' in line else ''
                        relationship = line.split('relationship: ')[1].split(' (ρ')[0] if 'relationship: ' in line else ''
                        rho_feat = line.split('ρ_feat→SHAP = ')[1].split(',')[0] if 'ρ_feat→SHAP = ' in line else ''
                        rho_partner = line.split('ρ_partner→SHAP = ')[1].split(')')[0] if 'ρ_partner→SHAP = ' in line else ''

                        if feature not in shap_data:
                            shap_data[feature] = {}
                        shap_data[feature]['partner_feature'] = partner
                        shap_data[feature]['interaction_strength'] = score
                        shap_data[feature]['interaction_relationship'] = relationship
                        shap_data[feature]['rho_feature'] = rho_feat
                        shap_data[feature]['rho_partner'] = rho_partner
    return shap_data

def _aggregate_shap_data(shap_reports: Dict) -> Dict[str, Dict[str, str]]:
    """Combine SHAP reports from different models into single feature dictionary"""
    aggregated = {}
    for report in shap_reports.values():
        data = _parse_shap_report(report)
        for feature, values in data.items():
            if feature not in aggregated:
                aggregated[feature] = values
    return aggregated

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
    top_group_1: List[Dict],
    top_group_2: List[Dict]
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

def _prepare_shap_table(shap_reports: Dict) -> pd.DataFrame:
    """Prepare comprehensive SHAP data table for ML section"""
    rows = []
    for (table_type, level, method), report in shap_reports.items():
        shap_data = _parse_shap_report(report)
        for feature, values in shap_data.items():
            row = {
                "Table Type": table_type,
                "Level": level,
                "Method": method,
                "Feature": feature,
                "Mean |SHAP|": values.get("mean_shap", ""),
                "Beeswarm Interpretation": values.get("beeswarm_interpretation", ""),
                "Spearman's ρ": values.get("spearman_rho", ""),
                "Dependency plot interpretation Relationship": values.get("dependency_relationship", ""),
                "Partner Feature": values.get("partner_feature", ""),
                "Interaction Strength": values.get("interaction_strength", ""),
                "Relationship": values.get("interaction_relationship", ""),
                "ρ (Feature)": values.get("rho_feature", ""),
                "ρ (Partner)": values.get("rho_partner", "")
            }
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=[
            "Table Type", "Level", "Method", "Feature", "Mean |SHAP|",
            "Beeswarm Interpretation", "Spearman's ρ",
            "Dependency plot interpretation Relationship", "Partner Feature",
            "Interaction Strength", "Relationship", "ρ (Feature)", "ρ (Partner)"
        ])

    return pd.DataFrame(rows)

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

    # SHAP Analysis table
    shap_html = ""
    if shap_reports:
        shap_df = _prepare_shap_table(shap_reports)
        if not shap_df.empty:
            shap_html = """
            <h3>SHAP Analysis</h3>
            <p>Comprehensive SHAP analysis for top features across all models:</p>
            """ + _add_table_functionality(shap_df, 'shap-table')

    return f"""
    <div class="ml-section">
        <h3>Model Performance</h3>
        {enhanced_metrics}

        <h3>Top Features by Importance</h3>
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
            f'onclick="showTable(\'{table_id}\', this)">{table_type}</button>'
        )

        level_btns, level_panes = [], []
        for l_idx, (level, methods) in enumerate(levels.items()):
            level_id = f"{table_id}-level-{next(id_counter)}"

            level_btns.append(
                f'<button class="level-button {"active" if l_idx == 0 and is_first_table else ""}" '
                f'data-level="{level_id}" '
                f'onclick="showLevel(\'{level_id}\', this)">{level}</button>'
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
                            flattened_plots[f"{plot_type} - Feature {idx+1}"] = sub_fig
                    else:
                        flattened_plots[plot_type] = fig

                plot_btns, plot_tabs, pd = _figs_to_html(
                    flattened_plots, id_counter, method_id
                )
                plot_data.update(pd)

                method_btns.append(
                    f'<button class="method-button {"active" if m_idx == 0 and l_idx == 0 and is_first_table else ""}" '
                    f'data-method="{method_id}" '
                    f'onclick="showMethod(\'{method_id}\', this)">{method}</button>'
                )

                method_panes.append(
                    f'<div id="{method_id}" class="method-pane" '
                    f'style="display:{"block" if m_idx == 0 and l_idx == 0 and is_first_table else "none"};">'
                    f'{plot_btns}'
                    f'{plot_tabs}'
                    f'</div>'
                )

            level_panes.append(
                f'<div id="{level_id}" class="level-pane" '
                f'style="display:{"block" if l_idx == 0 and is_first_table else "none"};">'
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
            f'onclick="showTab(event, \'{cat_id}\')">{category.title()}</button>'
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

    return f'<div class="tabs" data-label="category">{"".join(buttons_html)}</div>', "\n".join(tabs_html), plot_data

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
            f'onclick="showTable(\'{table_id}\', this)">{table_type}</button>'
        )

        level_btns, level_panes = [], []
        for l_idx, (level, metrics) in enumerate(levels.items()):
            level_id = f"{table_id}-level-{next(id_counter)}"
            is_active_level = l_idx == 0 and is_active_table

            level_btns.append(
                f'<button class="level-button {"active" if is_active_level else ""}" '
                f'data-level="{level_id}" '
                f'onclick="showLevel(\'{level_id}\', this)">{level}</button>'
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

    # Ensure the table_id is unique and valid for HTML
    clean_table_id = "".join(c for c in table_id if c.isalnum() or c == '-')

    table_html = df.to_html(index=False, classes='dynamic-table', table_id=clean_table_id)

    # Add sorting indicators
    # This is a basic addition; the actual sorting logic will be in JavaScript
    table_html = table_html.replace('<th>', '<th class="sortable">')

    # Add data attributes for sorting and resizing
    # Thead has to be added if not present (to_html does this)
    # The javascript will attach resize handles and sort events

    return f"""
    <div class="table-container" id="container-{clean_table_id}">
        {table_html}
        <div class="table-controls">
            <div class="pagination-controls">
                <span>Rows per page:</span>
                <select class="rows-per-page" onchange="changePageSize('{clean_table_id}', this.value)">
                    <option value="5">5</option>
                    <option value="10" selected>10</option>
                    <option value="20">20</option>
                    <option value="50">50</option>
                    <option value="100">100</option>
                    <option value="-1">All</option>
                </select>
                <div class="pagination-buttons" id="pagination-{clean_table_id}"></div>
                <span class="pagination-indicator" id="indicator-{clean_table_id}"></span>
            </div>
            <div class="search-controls">
                <input type="text" class="search-input" onkeyup="filterTable('{clean_table_id}', this.value)" placeholder="Search table...">
            </div>
            <div class="column-visibility-controls">
                <button onclick="toggleColumnVisibilityDropdown(this)">Show/Hide Columns</button>
                <div class="column-visibility-dropdown">
                    </div>
            </div>
        </div>
    </div>
    """
