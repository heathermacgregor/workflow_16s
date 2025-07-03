# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import base64
import itertools
import json
import logging
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Third Party Imports
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from plotly.offline import get_plotlyjs_version

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger('workflow_16s')

# ===================================== CLASSES ====================================== #

class NumpySafeJSONEncoder(json.JSONEncoder):
    def default(self, obj):  # noqa: D401
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
    id_counter,
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
        elif sec == "alpha_diversity":
            # Use nested structure for alpha diversity
            btns, tabs, pd = _alpha_diversity_to_nested_html(
                figures[sec], id_counter, sec_data["id"]
            )
            plot_data.update(pd)
            sec_data["subsections"].append({
                "title": "Alpha Diversity",
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


def _flatten(tree, keys, out):
    for k, v in tree.items():
        new_keys = keys + [k]
        if isinstance(v, dict):
            _flatten(v, new_keys, out)
        else:
            out[" - ".join(new_keys)] = v


def _figs_to_html(
    figs: Dict[str, Any], 
    counter, 
    prefix, 
    *, 
    square=False,
    row_label: str | None = None
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


def _section_html(sec):
    sub_html = "\n".join(
        f'<div class="subsection">\n'
        f'  <h3>{sub["title"]}</h3>\n'
        f'  <div class="tab-content">\n'           # ⟵  wrapper starts
        f'    <div class="tabs">{sub["buttons_html"]}</div>\n'
        f'    {sub["tabs_html"]}\n'
        f'  </div>\n'                             # ⟵  wrapper ends
        f'</div>'
        for sub in sec["subsections"]
    )
    return f'<div class="section" id="{sec["id"]}">\n' \
           f'  <h2>{sec["title"]}</h2>\n{sub_html}\n</div>'


# ==================================== FUNCTIONS ===================================== #

def _prepare_features_table(
    features: List[Dict], 
    max_features: int,
    category: str
) -> pd.DataFrame:
    """Prepare top features table for HTML display"""
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
    """Prepare statistical summary table."""
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

def _prepare_ml_summary(models: Dict) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[str]]:
    """Prepare detailed ML results for HTML display."""
    if not models:
        return None, None, None

    metrics_summary = []
    features_summary = []
    shap_plot_base64 = None
    best_mcc = -1  # Track best MCC for SHAP plot selection
    
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
                
                # Track best model for SHAP plot
                current_mcc = test_scores.get("mcc", -1)
                shap_path = result.get("shap_summary_bar_path")
                if current_mcc > best_mcc and shap_path and Path(shap_path).exists():
                    try:
                        with open(shap_path, "rb") as img_file:
                            shap_plot_base64 = base64.b64encode(img_file.read()).decode("utf-8")
                        best_mcc = current_mcc
                    except Exception:
                        logger.warning(f"Could not load SHAP plot from {shap_path}")
    
    metrics_df = pd.DataFrame(metrics_summary) if metrics_summary else None
    features_df = pd.DataFrame(features_summary) if features_summary else None
    
    return metrics_df, features_df, shap_plot_base64

def _format_ml_section(ml_metrics, ml_features, shap_plot_base64):
    """Format the machine learning results section."""
    if ml_metrics is None or ml_metrics.empty:
        return "<p>No ML results available</p>"
    
    ml_html = f"""
    <div class="ml-section">
        <h3>Model Performance</h3>
        {ml_metrics.to_html(index=False, classes='ml-metrics-table dynamic-table', table_id='ml-metrics')}
        
        <h3>Top Features</h3>
        {ml_features.to_html(index=False, classes='ml-features-table dynamic-table', table_id='ml-features')}
    """
    
    if shap_plot_base64:
        ml_html += f"""
        <h3>Feature Importance (Best Model)</h3>
        <div class="figure-container">
            <img src="data:image/png;base64,{shap_plot_base64}" alt="SHAP Summary">
        </div>
        """
    
    ml_html += "</div>"
    return ml_html


def _ordination_to_nested_html(
    figures: Dict[str, Any],
    id_counter,
    prefix: str,
) -> Tuple[str, str, Dict]:
    """
    Build the three‑level nested tab structure used for the ordination section.

    ┌ method (PCA/PCoA/…) ┐
        ┌ table_type (ASV/Genus/…) ┐
            ┌ level (L2/L3/…) ┐
                colour buttons (SampleType/Site/…)
    """
    buttons_html, panes_html, plot_data = [], [], {}
    ordination_methods = ["pca", "pcoa", "tsne", "umap"]

    for meth in ordination_methods:
        # ---------- collect figures for this ordination method ----------
        meth_figs: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for table_type, levels in figures.items():
            for level, methods in levels.items():
                if meth in methods:
                    meth_figs.setdefault(table_type, {})[level] = methods[meth]

        if not meth_figs:      # skip empty methods
            continue

        # ---------- IDs ----------
        meth_id = f"{prefix}-meth-{next(id_counter)}"

        # ---------- method‑level button ----------
        is_first_meth = not buttons_html  # first method becomes active
        buttons_html.append(
            f'<button class="method-button {"active" if is_first_meth else ""}" '
            f'data-method="{meth_id}" '
            f'onclick="showMethod(\'{meth_id}\')">{meth.upper()}</button>'
        )

        # ---------- build method pane ----------
        table_btns, table_panes = [], []
        for t_idx, (table_type, levels) in enumerate(meth_figs.items()):
            table_id = f"{meth_id}-table-{next(id_counter)}"

            # table_type button
            table_btns.append(
                f'<button class="table-button {"active" if t_idx == 0 else ""}" '
                f'data-table="{table_id}" '
                f'onclick="showTable(\'{table_id}\')">{table_type}</button>'
            )

            # ---------- build table‑type pane ----------
            level_btns, level_panes = [], []
            for l_idx, (level, colours) in enumerate(levels.items()):
                level_id = f"{table_id}-level-{next(id_counter)}"

                # level button
                level_btns.append(
                    f'<button class="level-button {"active" if l_idx == 0 else ""}" '
                    f'data-level="{level_id}" '
                    f'onclick="showLevel(\'{level_id}\')">{level}</button>'
                )

                # colour row & panes (comes fully wrapped/labelled)
                colour_tabs, colour_btns, pd = _figs_to_html(
                    colours,
                    id_counter,
                    level_id,
                    square=True,
                    row_label="color_col",
                )
                plot_data.update(pd)

                # level pane: colour buttons + colour panes
                level_panes.append(
                    f'<div id="{level_id}" class="level-pane" '
                    f'style="display:{"block" if l_idx == 0 else "none"};">'
                    f'{colour_btns}'
                    f'{colour_tabs}'
                    f'</div>'
                )

            # table‑type pane: level buttons + level panes
            table_panes.append(
                f'<div id="{table_id}" class="table-pane" '
                f'style="display:{"block" if t_idx == 0 else "none"};">'
                f'<div class="tabs" data-label="level">{"".join(level_btns)}</div>'
                f'{"".join(level_panes)}'
                f'</div>'
            )

        # method pane: table_type buttons + table panes
        panes_html.append(
            f'<div id="{meth_id}" class="method-pane" '
            f'style="display:{"block" if is_first_meth else "none"};">'
            f'<div class="tabs" data-label="table_type">{"".join(table_btns)}</div>'
            f'{"".join(table_panes)}'
            f'</div>'
        )

    # wrap top‑level (method) buttons row
    buttons_row = (
        f'<div class="tabs" data-label="method">{"".join(buttons_html)}</div>'
    )
    return buttons_row, "".join(panes_html), plot_data

def _alpha_diversity_to_nested_html(
    figures: Dict[str, Any],
    id_counter,
    prefix: str,
) -> Tuple[str, str, Dict]:
    """
    Build the nested tab structure for alpha diversity section.
    
    ┌ table_type (raw/normalized/...) ┐
        ┌ level (phylum/class/...) ┐
            ┌ metric (shannon/observed_features/...) ┐
    """
    buttons_html, panes_html, plot_data = [], [], {}
    
    for table_type, levels in figures.items():
        # ---------- table type button ----------
        table_id = f"{prefix}-table-{next(id_counter)}"
        is_first_table = not buttons_html
        buttons_html.append(
            f'<button class="table-button {"active" if is_first_table else ""}" '
            f'data-table="{table_id}" '
            f'onclick="showTable(\'{table_id}\')">{table_type}</button>'
        )
        
        # ---------- build table pane ----------
        level_btns, level_panes = [], []
        for l_idx, (level, metrics) in enumerate(levels.items()):
            level_id = f"{table_id}-level-{next(id_counter)}"
            
            # level button
            level_btns.append(
                f'<button class="level-button {"active" if l_idx == 0 else ""}" '
                f'data-level="{level_id}" '
                f'onclick="showLevel(\'{level_id}\')">{level}</button>'
            )
            
            # ---------- build level pane ----------
            metric_btns, metric_panes = [], []
            for m_idx, (metric, fig) in enumerate(metrics.items()):
                metric_id = f"{level_id}-metric-{next(id_counter)}"
                plot_id = f"{prefix}-plot-{next(id_counter)}"
                
                # metric button
                metric_btns.append(
                    f'<button class="metric-button {"active" if m_idx == 0 else ""}" '
                    f'data-metric="{metric_id}" '
                    f'onclick="showMetric(\'{metric_id}\', \'{plot_id}\')">{metric}</button>'
                )
                
                # metric pane
                metric_panes.append(
                    f'<div id="{metric_id}" class="metric-pane" '
                    f'style="display:{"block" if m_idx == 0 else "none"};">'
                    f'<div id="container-{plot_id}" class="plot-container"></div>'
                    f'</div>'
                )
                
                # Add plot data
                try:
                    if hasattr(fig, "to_plotly_json"):
                        pj = fig.to_plotly_json()
                        plot_data[plot_id] = {
                            "type": "plotly",
                            "data": pj["data"],
                            "layout": pj["layout"],
                        }
                    elif isinstance(fig, Figure):
                        buf = BytesIO()
                        fig.savefig(buf, format="png", bbox_inches="tight")
                        buf.seek(0)
                        plot_data[plot_id] = {
                            "type": "image",
                            "data": base64.b64encode(buf.read()).decode()
                        }
                except Exception as exc:
                    logger.exception("Serializing figure failed")
                    plot_data[plot_id] = {
                        "type": "error", 
                        "error": str(exc)
                    }
            
            # level pane: metric buttons + metric panes
            level_panes.append(
                f'<div id="{level_id}" class="level-pane" '
                f'style="display:{"block" if l_idx == 0 else "none"};">'
                f'<div class="tabs" data-label="metric">{"".join(metric_btns)}</div>'
                f'{"".join(metric_panes)}'
                f'</div>'
            )
        
        # table pane: level buttons + level panes
        panes_html.append(
            f'<div id="{table_id}" class="table-pane" '
            f'style="display:{"block" if is_first_table else "none"};">'
            f'<div class="tabs" data-label="level">{"".join(level_btns)}</div>'
            f'{"".join(level_panes)}'
            f'</div>'
        )
    
    # Wrap top-level buttons
    buttons_row = (
        f'<div class="tabs" data-label="table_type">{"".join(buttons_html)}</div>'
    )
    return buttons_row, "".join(panes_html), plot_data

def _alpha_correlations_to_nested_html(
    figures: Dict[str, Any],
    id_counter,
    prefix: str,
) -> Tuple[str, str, Dict]:
    """
    Build the nested tab structure for alpha diversity correlations.
    
    ┌ table_type (raw/normalized/...) ┐
        ┌ level (phylum/class/...) ┐
            ┌ metric (shannon/observed_features/...) ┐
    """
    buttons_html, panes_html, plot_data = [], [], {}
    
    for table_type, levels in figures.items():
        # ---------- table type button ----------
        table_id = f"{prefix}-table-{next(id_counter)}"
        is_first_table = not buttons_html
        buttons_html.append(
            f'<button class="table-button {"active" if is_first_table else ""}" '
            f'data-table="{table_id}" '
            f'onclick="showTable(\'{table_id}\')">{table_type}</button>'
        )
        
        # ---------- build table pane ----------
        level_btns, level_panes = [], []
        for l_idx, (level, metrics) in enumerate(levels.items()):
            level_id = f"{table_id}-level-{next(id_counter)}"
            
            # level button
            level_btns.append(
                f'<button class="level-button {"active" if l_idx == 0 else ""}" '
                f'data-level="{level_id}" '
                f'onclick="showLevel(\'{level_id}\')">{level}</button>'
            )
            
            # ---------- build level pane ----------
            metric_btns, metric_panes = [], []
            for m_idx, (metric, fig) in enumerate(metrics.items()):
                metric_id = f"{level_id}-metric-{next(id_counter)}"
                plot_id = f"{prefix}-plot-{next(id_counter)}"
                
                # metric button
                metric_btns.append(
                    f'<button class="metric-button {"active" if m_idx == 0 else ""}" '
                    f'data-metric="{metric_id}" '
                    f'onclick="showMetric(\'{metric_id}\', \'{plot_id}\')">{metric}</button>'
                )
                
                # metric pane
                metric_panes.append(
                    f'<div id="{metric_id}" class="metric-pane" '
                    f'style="display:{"block" if m_idx == 0 else "none"};">'
                    f'<div id="container-{plot_id}" class="plot-container"></div>'
                    f'</div>'
                )
                
                # Add plot data
                try:
                    if hasattr(fig, "to_plotly_json"):
                        pj = fig.to_plotly_json()
                        plot_data[plot_id] = {
                            "type": "plotly",
                            "data": pj["data"],
                            "layout": pj["layout"],
                        }
                except Exception as exc:
                    logger.exception("Serializing figure failed")
                    plot_data[plot_id] = {
                        "type": "error", 
                        "error": str(exc)
                    }
            
            # level pane: metric buttons + metric panes
            level_panes.append(
                f'<div id="{level_id}" class="level-pane" '
                f'style="display:{"block" if l_idx == 0 else "none"};">'
                f'<div class="tabs" data-label="metric">{"".join(metric_btns)}</div>'
                f'{"".join(metric_panes)}'
                f'</div>'
            )
        
        # table pane: level buttons + level panes
        panes_html.append(
            f'<div id="{table_id}" class="table-pane" '
            f'style="display:{"block" if is_first_table else "none"};">'
            f'<div class="tabs" data-label="level">{"".join(level_btns)}</div>'
            f'{"".join(level_panes)}'
            f'</div>'
        )
    
    # Wrap top-level buttons
    buttons_row = (
        f'<div class="tabs" data-label="table_type">{"".join(buttons_html)}</div>'
    )
    return buttons_row, "".join(panes_html), plot_data

# ================================== TABLE HELPERS ================================== #
def _add_table_functionality(df: pd.DataFrame, table_id: str) -> str:
    """Enhance table HTML with sorting, pagination, and row selection functionality."""
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
            </div>
        </div>
    </div>
    """
    return enhanced_html

# ================================ REPORT GENERATION ================================= #
def generate_html_report(
    amplicon_data: "AmpliconData",
    output_path: Union[str, Path],
    include_sections: List[str] | None = None,
    max_features: int = 20  
) -> None:
    """Write an HTML file with interactive Plotly/Matplotlib figures and analysis tables."""
    include_sections = include_sections or [
        k for k, v in amplicon_data.figures.items() if v
    ]
    ts = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Prepare tables section ────────────────────────────────────────────────
    tables_html = ""
    
    # Top features tables
    contam_df = _prepare_features_table(
        getattr(amplicon_data, "top_contaminated_features", []),
        max_features,
        "Contaminated"
    )
    pristine_df = _prepare_features_table(
        getattr(amplicon_data, "top_pristine_features", []),
        max_features,
        "Pristine"
    )
    
    # Stats summary
    stats_df = _prepare_stats_summary(
        getattr(amplicon_data, "stats", {})
    )
    
    # ML summary
    ml_metrics, ml_features, shap_plot_base64 = _prepare_ml_summary(
        getattr(amplicon_data, "models", {})
    )
    ml_html = _format_ml_section(ml_metrics, ml_features, shap_plot_base64)
    
    # Build tables section HTML
    tables_html = f"""
    <div class="section">
        <h2>Analysis Summary</h2>
        
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
    </div>
    """

    # ── Prepare figures sections ───────────────────────────────────────────────
    id_counter = itertools.count()
    sections, plot_data = _prepare_sections(
        amplicon_data.figures, include_sections, id_counter
    )
    sections_html = "\n".join(_section_html(s) for s in sections)

    # ── CDN tag for Plotly ────────────────────────────────────────────────────
    try:
        plotly_ver = get_plotlyjs_version()
    except Exception:
        plotly_ver = "3.0.1"
    plotly_js_tag = (
        f'<script src="https://cdn.plot.ly/plotly-{plotly_ver}.min.js"></script>'
    )

    # ── JSON payload (escape "</" so it can never close the <script>) ────────
    payload = json.dumps(plot_data, cls=NumpySafeJSONEncoder, ensure_ascii=False)
    payload = payload.replace("</", "<\\/")  # safety

    # ── Table functionality JavaScript ────────────────────────────────────────
    table_js = """
    /* ======================= TABLE FUNCTIONALITY ======================= */
    function sortTable(tableId, columnIndex, isNumeric) {
        const table = document.getElementById(tableId);
        const tbody = table.querySelector('tbody');
        const rows = Array.from(tbody.querySelectorAll('tr'));
        const header = table.querySelectorAll('thead th')[columnIndex];
        const isAscending = !header.classList.contains('asc');
        
        // Clear previous sort indicators
        table.querySelectorAll('thead th').forEach(th => {
            th.classList.remove('asc', 'desc');
        });
        
        // Set new sort indicator
        header.classList.add(isAscending ? 'asc' : 'desc');
        
        rows.sort((a, b) => {
            const aVal = a.cells[columnIndex].textContent.trim();
            const bVal = b.cells[columnIndex].textContent.trim();
            
            if (isNumeric) {
                const numA = parseFloat(aVal) || 0;
                const numB = parseFloat(bVal) || 0;
                return isAscending ? numA - numB : numB - numA;
            }
            return isAscending 
                ? aVal.localeCompare(bVal) 
                : bVal.localeCompare(aVal);
        });
        
        // Clear and re-add sorted rows
        tbody.innerHTML = '';
        rows.forEach(row => tbody.appendChild(row));
        
        // Reapply pagination
        const select = table.closest('.table-container')
                          .querySelector('.rows-per-page');
        changePageSize(tableId, select.value);
    }
    
    function setupTableSorting(tableId) {
        const table = document.getElementById(tableId);
        if (!table) return;
        
        const headers = table.querySelectorAll('thead th');
        
        headers.forEach((header, index) => {
            // Check if column is numeric
            const firstRow = table.querySelector('tbody tr');
            const isNumeric = firstRow && !isNaN(parseFloat(firstRow.cells[index].textContent));
            
            header.style.cursor = 'pointer';
            header.addEventListener('click', () => {
                sortTable(tableId, index, isNumeric);
            });
        });
    }
    
    function paginateTable(tableId, pageSize) {
        const table = document.getElementById(tableId);
        if (!table) return;
        
        const rows = table.querySelectorAll('tbody tr');
        const paginationDiv = document.getElementById(`pagination-${tableId}`);
        const totalPages = Math.ceil(rows.length / pageSize);
        
        // Hide all rows
        rows.forEach(row => row.style.display = 'none');
        
        // Show rows for first page
        const start = 0;
        const end = Math.min(start + pageSize, rows.length);
        for (let i = start; i < end; i++) {
            rows[i].style.display = '';
        }
        
        // Generate pagination buttons
        paginationDiv.innerHTML = '';
        if (totalPages > 1) {
            const prevButton = document.createElement('button');
            prevButton.textContent = '◄';
            prevButton.classList.add('pagination-btn');
            prevButton.disabled = true;
            prevButton.addEventListener('click', () => {
                changePage(tableId, 0, pageSize); // Go to first page
            });
            paginationDiv.appendChild(prevButton);
            
            const pageButton = document.createElement('button');
            pageButton.textContent = '1';
            pageButton.classList.add('pagination-btn', 'active');
            pageButton.addEventListener('click', () => {
                changePage(tableId, 0, pageSize);
            });
            paginationDiv.appendChild(pageButton);
            
            if (totalPages > 2) {
                const ellipsis = document.createElement('span');
                ellipsis.textContent = totalPages > 5 ? '...' : '';
                paginationDiv.appendChild(ellipsis);
            }
            
            const nextButton = document.createElement('button');
            nextButton.textContent = '►';
            nextButton.classList.add('pagination-btn');
            nextButton.disabled = totalPages <= 1;
            nextButton.addEventListener('click', () => {
                changePage(tableId, totalPages - 1, pageSize); // Go to last page
            });
            paginationDiv.appendChild(nextButton);
        }
    }
    
    function changePage(tableId, pageNumber, pageSize) {
        const table = document.getElementById(tableId);
        if (!table) return;
        
        const rows = table.querySelectorAll('tbody tr');
        const paginationDiv = document.getElementById(`pagination-${tableId}`);
        const totalPages = Math.ceil(rows.length / pageSize);
        
        // Validate page number
        pageNumber = Math.max(0, Math.min(pageNumber, totalPages - 1));
        
        // Hide all rows
        rows.forEach(row => row.style.display = 'none');
        
        // Show rows for current page
        const start = pageNumber * pageSize;
        const end = Math.min(start + pageSize, rows.length);
        for (let i = start; i < end; i++) {
            rows[i].style.display = '';
        }
        
        // Update pagination UI
        const buttons = paginationDiv.querySelectorAll('.pagination-btn');
        buttons.forEach(button => button.classList.remove('active'));
        
        // Only activate current page button if it exists
        if (buttons[pageNumber + 1]) {  // +1 to skip the prev button
            buttons[pageNumber + 1].classList.add('active');
        }
        
        // Update button states
        buttons[0].disabled = pageNumber === 0;  // Prev button
        buttons[buttons.length - 1].disabled = pageNumber === totalPages - 1;  // Next button
    }
    
    function changePageSize(tableId, newSize) {
        const pageSize = newSize === '-1' ? 10000 : parseInt(newSize);
        paginateTable(tableId, pageSize);
    }
    
    function initTables() {
        document.querySelectorAll('.dynamic-table').forEach(table => {
            const tableId = table.id;
            setupTableSorting(tableId);
            changePageSize(tableId, 10);  // Initialize with 10 rows per page
        });
    }
    """

    # ── build the full HTML ───────────────────────────────────────────────────
    html = _HTML_TEMPLATE.format(
        plotly_js_tag=plotly_js_tag,
        generated_ts=ts,
        section_list=", ".join(include_sections),
        tables_html=tables_html,
        sections_html=sections_html,
        plot_data_json=payload,
        table_js=table_js
    )
    output_path.write_text(html, encoding="utf‑8")

# ================================= HTML TEMPLATE ================================== #
_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>16S Analysis Report</title>
  {plotly_js_tag}
  <style>
    /* ======================= BASE STYLES ======================= */
    body                                 {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
    .section                             {{ margin-bottom: 30px; padding-bottom: 20px; border-bottom: 1px solid #eee; }}
    .subsection                          {{ margin-left: 20px; margin-bottom: 20px; }}
    
    .tabs                                {{ display: flex; margin: 0 0 10px 0; flex-wrap: wrap; }}
    .tabs[data-label]::before            {{ content: attr(data-label) ": "; font-weight: bold; margin-right: 6px; white-space: nowrap; }}
    .tabs[data-label]                    {{ display: flex; flex-wrap: wrap; align-items: center; margin-top: 6px; }}
    
    .tab-content                         {{ border: 1px solid #ccc; padding: 10px; border-radius: 4px; }}
    
    .method-pane                         {{ display: none; }}
    .method-pane:first-child             {{ display: block; }}
    
    .table-tabs, .level-tabs             {{ display: flex; flex-wrap: wrap; margin-bottom: 5px; }}
    .table-pane                          {{ display: none; margin-left: 15px; }}
    .table-pane:first-child              {{ display: block; }}
    
    .level-pane                          {{ display: none; margin-left: 15px; }}
    .level-pane:first-child              {{ display: block; }}

    :root                                {{ --indent-step: clamp(8px, 2.2vw, 24px); }}

    .subsection  > .tab-content > .tabs  {{ margin-left: 0; }}
    
    .method-pane > .tabs                 {{ margin-left: calc(var(--indent-step) * 1); }}
    .table-pane  > .tabs                 {{ margin-left: calc(var(--indent-step) * 2); }}
    .level-pane  > .tabs                 {{ margin-left: calc(var(--indent-step) * 3); }}
    .plot-container                      {{ }}
    
    .error                               {{ color: #d32f2f; padding: 10px; border: 1px solid #ffcdd2; background: #ffebee; }}
    
    .section-controls                    {{ margin: 10px 0; }}
    .section-button                      {{ background: #f0f0f0; border: 1px solid #ddd; padding: 5px 10px; cursor: pointer; border-radius: 4px; 
                                            margin-right: 5px; }}
                         
    /* ======================= TABLE STYLES ======================= */
    table                                {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
    th, td                               {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th                                   {{ background-color: #f2f2f2; position: relative; }}
    .feature-table tr:nth-child(even)    {{ background-color: #f9f9f9; }}
    .ml-metrics-table tr:nth-child(even) {{ background-color: #f0f8ff; }}
    .ml-features-table tr:nth-child(even) {{ background-color: #fff0f5; }}
    
    .table-container                     {{ margin: 20px 0; overflow-x: auto; }}
    
    .table-controls                      {{ margin-top: 10px; display: flex; justify-content: space-between; align-items: center; }}
    
    .pagination-controls                 {{ display: flex; align-items: center; gap: 10px; }}
    
    .pagination-controls select          {{ padding: 5px; border: 1px solid #ddd; border-radius: 4px; }}
    
    .pagination-controls button          {{ padding: 5px 10px; background: #f0f0f0; border: 1px solid #ddd; cursor: pointer; border-radius: 4px; 
                                            min-width: 32px; }}
    
    .pagination-controls button:disabled {{ background: #ddd; cursor: not-allowed; }}
    
    .pagination-controls button.active   {{ background: #000; color: #fff; }}
    
    .dynamic-table th                    {{ cursor: pointer; }}
    
    .dynamic-table th:hover              {{ background-color: #e6e6e6; }}
    
    .dynamic-table th.asc::after         {{ content: " ▲"; font-size: 0.8em; position: absolute; right: 8px; }}
    .dynamic-table th.desc::after        {{ content: " ▼"; font-size: 0.8em; position: absolute; right: 8px; }}
    
    /* ML Section Styling */
    .ml-section                          {{ margin-top: 30px; padding: 20px; background-color: #f8f9fa; border-radius: 8px; border: 1px solid #dee2e6; }}
    .ml-section h3                       {{ margin-top: 0; }}
    .figure-container                    {{ margin-top: 20px; text-align: center; }}
    .figure-container img                {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; padding: 5px; }}
  </style>
</head>
<body>
  <h1>16S Amplicon Analysis Report</h1>
  <p>Generated: {generated_ts}</p>
  <p>Sections: {section_list}</p>

  <div class="section-controls">
    <button class="section-button" onclick="toggleAllSections(true)">Expand All</button>
    <button class="section-button" onclick="toggleAllSections(false)">Collapse All</button>
  </div>

  <!-- Tables Section -->
  {tables_html}

  <!-- Figures Sections -->
  {sections_html}

  <!-- serialised figure data -->
  <script id="plot-data" type="application/json">{plot_data_json}</script>

  <script>
    /* ======================= FIGURE FUNCTIONALITY ======================= */
    /* ---- data ---- */
    const plotData = JSON.parse(document.getElementById('plot-data').textContent);
    
    /* ---- state ---- */
    const rendered = new Set();
    const MAX_WEBGL_CONTEXTS = 6;  // Conservative limit for most browsers
    const activeWebGLPlots = new Set();

    /* ---- helpers ---- */
    function purgePlot(plotId) {{
        const plotDiv = document.getElementById(plotId);
        if (plotDiv && Plotly) {{
            Plotly.purge(plotDiv);
        }}
        const container = document.getElementById(`container-${{plotId}}`);
        if (container) container.innerHTML = '';
        rendered.delete(plotId);
        activeWebGLPlots.delete(plotId);
    }}

    function enforceWebGLLimit() {{
        while (activeWebGLPlots.size > MAX_WEBGL_CONTEXTS) {{
            const oldest = activeWebGLPlots.values().next().value;
            purgePlot(oldest);
        }}
    }}

    function renderPlot(containerId, plotId) {{
        const container = document.getElementById(containerId);
        if (!container) return console.error('Missing container', containerId);

        container.innerHTML = '';
        const div = document.createElement('div');
        div.id = plotId;
        div.className = 'plot-container';
        container.appendChild(div);

        const payload = plotData[plotId];
        if (!payload) {{
            div.innerHTML = '<div class="error">Plot data unavailable</div>';
            return;
        }}

        // Compute responsive width (min 500px, max 1000px)
        const fullWidth = container.clientWidth || window.innerWidth;
        const minWidth  = fullWidth * 0.15;                    // 25 % floor
        const width     = Math.max(minWidth, Math.min(1000, fullWidth * 0.95));
        // Square only when payload.square === true
        const height = payload.square ? width : Math.round(width * 0.6);
        
        // Check if this is a 3D plot
        const is3D = payload.data?.some(d => d.type.includes('3d'));

        try {{
            if (payload.type === 'plotly') {{
                if (payload.layout) {{
                    payload.layout.showlegend = false;
                    payload.layout.width = width;
                    payload.layout.height = height;
                    
                    // Optimize 3D plots
                    if (is3D) {{
                        payload.layout.scene = payload.layout.scene || {{}};
                        payload.layout.scene.aspectmode = 'data';
                        payload.layout.uirevision = 'constant';
                    }}
                }}
                
                const config = {{
                    responsive: true,
                    webglOptions: {{ preserveDrawingBuffer: false }}
                }};
                
                Plotly.newPlot(plotId, payload.data, payload.layout, config)
                    .then(() => {{
                        if (is3D) {{
                            activeWebGLPlots.add(plotId);
                            enforceWebGLLimit();
                        }}
                    }})
                    .catch(err => {{
                        div.innerHTML = `<div class="error">Plotly error: ${{err}}</div>`;
                        console.error(err);
                    }});
            }}
            else if (payload.type === 'image') {{
                const img = document.createElement('img');
                img.src = 'data:image/png;base64,' + payload.data;
                img.style.maxWidth = '100%';
                img.style.height = 'auto';
                div.appendChild(img);
            }} 
            else if (payload.type === 'error') {{
                div.innerHTML = `<div class="error">${{payload.error}}</div>`;
            }} 
            else {{
                div.innerHTML = '<div class="error">Unknown plot type</div>';
            }}
        }} catch (err) {{
            div.innerHTML = `<div class="error">Rendering error: ${{err}}</div>`;
            console.error(err);
        }}
    }}

    /* ---- tab logic ---- */
    function showTab(tabId, plotId) {{
        const pane = document.getElementById(tabId);
        if (!pane) return;

        const subsection = pane.closest('.subsection');
        if (!subsection) return;
        
        // Purge previous plot in this subsection
        const prevPane = subsection.querySelector('.tab-pane[style*="display: block"]');
        if (prevPane) {{
            const prevPlotId = prevPane.dataset.plotId;
            if (rendered.has(prevPlotId)) {{
                purgePlot(prevPlotId);
            }}
        }}

        // Update UI
        subsection.querySelectorAll('.tab-pane').forEach(p => p.style.display = 'none');
        subsection.querySelectorAll('.tab-button').forEach(b => b.classList.remove('active'));
        
        pane.style.display = 'block';
        const button = subsection.querySelector(`[data-tab="${{tabId}}"]`);
        if (button) button.classList.add('active');

        // Render new plot
        if (!rendered.has(plotId)) {{
            renderPlot(`container-${{plotId}}`, plotId);
            rendered.add(plotId);
        }}
    }}

    /* ---- nested tab management ---- */
    function showMethod(methodId) {{
        // Purge all plots in current method
        const currentMethod = document.querySelector('.method-pane[style*="display: block"]');
        if (currentMethod) {{
            currentMethod.querySelectorAll('.tab-pane[data-plot-id]').forEach(pane => {{
                const plotId = pane.dataset.plotId;
                if (rendered.has(plotId)) purgePlot(plotId);
            }});
        }}

        // Update UI
        document.querySelectorAll('.method-pane').forEach(pane => {{
            pane.style.display = 'none';
        }});
        document.querySelectorAll('.method-button').forEach(btn => {{
            btn.classList.remove('active');
        }});
        
        const newMethod = document.getElementById(methodId);
        if (newMethod) newMethod.style.display = 'block';
        document.querySelector(`[data-method="${{methodId}}"]`).classList.add('active');
        
        // Show first table
        const firstTable = newMethod.querySelector('.table-pane');
        if (firstTable) showTable(firstTable.id);
    }}
    
    function showTable(tableId) {{
        // Purge all plots in current table
        const methodPane = document.getElementById(tableId).closest('.method-pane');
        const currentTable = methodPane.querySelector('.table-pane[style*="display: block"]');
        if (currentTable) {{
            currentTable.querySelectorAll('.tab-pane[data-plot-id]').forEach(pane => {{
                const plotId = pane.dataset.plotId;
                if (rendered.has(plotId)) purgePlot(plotId);
            }});
        }}

        // Update UI
        methodPane.querySelectorAll('.table-pane').forEach(pane => {{
            pane.style.display = 'none';
        }});
        methodPane.querySelectorAll('.table-button').forEach(btn => {{
            btn.classList.remove('active');
        }});
        
        const newTable = document.getElementById(tableId);
        if (newTable) newTable.style.display = 'block';
        document.querySelector(`[data-table="${{tableId}}"]`).classList.add('active');
        
        // Show first level
        const firstLevel = newTable.querySelector('.level-pane');
        if (firstLevel) showLevel(firstLevel.id);
    }}
    
    function showLevel(levelId) {{
        // Purge all plots in current level
        const tablePane = document.getElementById(levelId).closest('.table-pane');
        const currentLevel = tablePane.querySelector('.level-pane[style*="display: block"]');
        if (currentLevel) {{
            currentLevel.querySelectorAll('.tab-pane[data-plot-id]').forEach(pane => {{
                const plotId = pane.dataset.plotId;
                if (rendered.has(plotId)) purgePlot(plotId);
            }});
        }}

        // Update UI
        tablePane.querySelectorAll('.level-pane').forEach(pane => {{
            pane.style.display = 'none';
        }});
        tablePane.querySelectorAll('.level-button').forEach(btn => {{
            btn.classList.remove('active');
        }});
        
        const newLevel = document.getElementById(levelId);
        if (newLevel) newLevel.style.display = 'block';
        document.querySelector(`[data-level="${{levelId}}"]`).classList.add('active');
        
        // Show first colour plot
        const firstColour = newLevel.querySelector('.tab-pane');
        if (firstColour) showTab(firstColour.id, firstColour.dataset.plotId);
    }}
    
    function showMetric(metricId, plotId) {{
        // Purge any existing plot in this metric pane
        const container = document.getElementById(`container-${{plotId}}`);
        if (container) {{
            container.innerHTML = '';
        }}
        
        // Update UI
        const metricPane = document.getElementById(metricId);
        if (!metricPane) return;
        
        const levelPane = metricPane.closest('.level-pane');
        if (!levelPane) return;
        
        levelPane.querySelectorAll('.metric-pane').forEach(pane => {{
            pane.style.display = 'none';
        }});
        levelPane.querySelectorAll('.metric-button').forEach(btn => {{
            btn.classList.remove('active');
        }});
        
        metricPane.style.display = 'block';
        document.querySelector(`[data-metric="${{metricId}}"]`).classList.add('active');
        
        // Render new plot
        if (!rendered.has(plotId)) {{
            renderPlot(`container-${{plotId}}`, plotId);
            rendered.add(plotId);
        }}
    }}

    /* ---- section toggles ---- */
    function toggleAllSections(show) {{
        document.querySelectorAll('.section').forEach(s => {{
            s.style.display = show ? 'block' : 'none';
        }});
    }}

    /* ======================= TABLE FUNCTIONALITY ======================= */
    {table_js}

    /* ---- initialization ---- */
    document.addEventListener('DOMContentLoaded', () => {{
        // Initialize all first-level plots
        document.querySelectorAll('.subsection').forEach(sub => {{
            const first = sub.querySelector('.tab-pane');
            if (first) showTab(first.id, first.dataset.plotId);
        }});
        
        // Initialize nested tabs
        document.querySelectorAll('.method-pane').forEach(pane => {{
            const firstTable = pane.querySelector('.table-pane');
            if (firstTable) showTable(firstTable.id);
        }});
        
        // Initialize alpha diversity tabs
        document.querySelectorAll('.table-pane').forEach(pane => {{
            const firstLevel = pane.querySelector('.level-pane');
            if (firstLevel) {{
                const firstMetric = firstLevel.querySelector('.metric-pane');
                const firstButton = firstLevel.querySelector('.metric-button');
                if (firstMetric && firstButton) {{
                    const plotId = firstMetric.querySelector('.plot-container').id.replace('container-', '');
                    showMetric(firstMetric.id, plotId);
                }}
            }}
        }});
        
        // Initialize tables
        initTables();
    }});
  </script>
</body>
</html>
"""
