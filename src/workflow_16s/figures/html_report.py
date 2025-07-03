# ===================================== IMPORTS ====================================== #

# -------- Standard Library --------
import base64
import json
import logging
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Optional

# -------- Third‑party -------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from plotly.offline import get_plotlyjs_version

# ==================================== LOGGING ====================================== #

logger = logging.getLogger(__name__)

# =============================================================================
# CUSTOM JSON ENCODER FOR NUMPY TYPES
# =============================================================================

class NumpySafeJSONEncoder(json.JSONEncoder):
    """JSON encoder that safely converts numpy scalars/arrays to python types."""

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

# ==================================== PUBLIC API =================================== #

def generate_html_report(
    amplicon_data: "AmpliconData",
    output_path: Union[str, Path],
    max_features: int = 20,
) -> None:
    """Top‑level HTML report writer that now bundles the new *Sample‑Map* tabs.

    Besides integrating the new sample‑map visualisation (tabs per colour column),
    all existing sections (Top‑Features, Stats, ML, other visualisations) remain
    untouched.  Plots that are **not** sample maps are handled by the original
    helper but using a copy of `amplicon_data.figures` that excludes the "map"
    entry so we do not render them twice.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # 1. Data tables & summaries (unchanged)
    # ---------------------------------------------------------------------
    contam_df = _prepare_features_table(
        amplicon_data.top_contaminated_features, max_features, "Contaminated"
    )
    pristine_df = _prepare_features_table(
        amplicon_data.top_pristine_features, max_features, "Pristine"
    )
    stats_summary = _prepare_stats_summary(amplicon_data.stats)
    ml_metrics, ml_features, shap_plot = _prepare_ml_summary(amplicon_data.models)

    # ---------------------------------------------------------------------
    # 2. Visualisations – split into (a) new Sample‑Map tabs, (b) everything else
    # ---------------------------------------------------------------------
    map_tabs_html, map_buttons_html, map_plot_data = _prepare_map_figures(
        amplicon_data.figures
    )

    # remove sample maps so they are not duplicated by the legacy helper
    other_figures_dict: Dict[str, Any] = {
        k: v for k, v in amplicon_data.figures.items() if k != "map"
    }
    other_figures_html = _prepare_other_figures(other_figures_dict)

    # ---------------------------------------------------------------------
    # 3. Assets that depend on plotly version / serialised map data
    # ---------------------------------------------------------------------
    try:
        plotly_js_version = get_plotlyjs_version()
    except Exception:  # pragma: no cover  – fallback for very old Plotly
        plotly_js_version = "3.0.1"
    plotly_js_tag = (
        f"<script src=\"https://cdn.plot.ly/plotly-{plotly_js_version}.min.js\"></script>"
    )

    map_plot_data_json = json.dumps(map_plot_data, cls=NumpySafeJSONEncoder)

    # ---------------------------------------------------------------------
    # 4. Compose final HTML
    # ---------------------------------------------------------------------
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset='utf-8'>
    <title>16S Analysis Report</title>
    {plotly_js_tag}
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1, h2, h3, h4 {{ color: #2c3e50; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .section {{ margin-bottom: 40px; }}

        /* ===== Grid for legacy (non‑map) figures ===== */
        .figure-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(500px, 1fr));
            gap: 20px;
        }}
        .figure-container {{
            border: 1px solid #ddd;
            padding: 10px;
            height: 500px;
            overflow: hidden;
        }}
        .plot-wrapper {{
            width: 100%;
            height: 400px;
            display: flex;
            justify-content: center;
            align-items: center;
            overflow: hidden;
        }}
        .plot-container {{
            width: 100%;
            height: 100%;
            display: none;
        }}
        .plot-container.active {{
            display: flex;
            justify-content: center;
            align-items: center;
        }}
        .plot-container img {{
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }}
        .ml-feature-table tr:nth-child(even) {{ background-color: #f9f9f9; }}

        /* ===== New tabbed Sample‑Map styles ===== */
        .tabs {{ display: flex; margin-bottom: -1px; }}
        .tab-button {{
            padding: 10px 15px;
            background: #eee;
            border: 1px solid #ccc;
            cursor: pointer;
            border-radius: 5px 5px 0 0;
            margin-right: 5px;
        }}
        .tab-button.active {{
            background: #fff;
            border-bottom: 1px solid #fff;
            position: relative;
            z-index: 2;
        }}
        .tab-content {{
            border: 1px solid #ccc;
            padding: 20px;
            border-radius: 0 5px 5px 5px;
            position: relative;
            z-index: 1;
        }}
        .map‑plot-container {{ width: 900px; height: 600px; }}
        .error {{ color: #d32f2f; padding: 10px; border: 1px solid #ffcdd2; background: #ffebee; }}
    </style>
</head>
<body>
    <h1>16S Amplicon Analysis Report</h1>
    <p>Generated on {timestamp}</p>

    <!-- ========================== TOP FEATURES ========================== -->
    <div class="section">
        <h2>Top Features</h2>
        <h3>Contaminated‑Associated Features</h3>
        {contam_df.to_html(index=False, classes='feature-table')}
        <h3>Pristine‑Associated Features</h3>
        {pristine_df.to_html(index=False, classes='feature-table')}
    </div>

    <!-- ========================== STAT SUMMARY ========================== -->
    <div class="section">
        <h2>Statistical Summary</h2>
        {stats_summary.to_html(index=False)}
    </div>

    <!-- ========================== ML RESULTS ============================ -->
    <div class="section">
        <h2>Machine Learning Results</h2>
        {_format_ml_section(ml_metrics, ml_features, shap_plot)}
    </div>

    <!-- ========================== SAMPLE MAPS (new) ===================== -->
    <div class="section">
        <h2>Sample Maps</h2>
        <div class="tabs">{map_buttons_html}</div>
        <div class="tab-content">{map_tabs_html}</div>
    </div>

    <!-- ========================== OTHER FIGURES (legacy) ================ -->
    <div class="section">
        <h2>Other Visualisations</h2>
        <div class="figure-grid">{other_figures_html}</div>
    </div>

    <!-- ========================== JS FOR SAMPLE MAP TABS ================ -->
    <script>
        const plotData = {map_plot_data_json};
        const initialised = new Set();

        function renderPlot(containerId, plotId) {{
            const container = document.getElementById(containerId);
            if (!container) return;
            container.innerHTML = "";  // clear
            const div = document.createElement("div");
            div.id = plotId;
            div.className = "map‑plot-container";  // note: zero‑width space prevented clash
            container.appendChild(div);

            const payload = plotData[plotId];
            if (!payload) {{
                div.innerHTML = '<div class="error">Plot data unavailable</div>';
                return;
            }}
            try {{
                if (payload.layout) payload.layout.showlegend = false;
                Plotly.newPlot(plotId, payload.data, payload.layout).catch(err => {{
                    div.innerHTML = `<div class='error'>Plotly error: ${err}</div>`;
                    console.error(err);
                }});
            }} catch (err) {{
                div.innerHTML = `<div class='error'>Render error: ${err}</div>`;
                console.error(err);
            }}
        }}

        function showTab(tabId, plotId) {{
            document.querySelectorAll('.tab-pane').forEach(tab => tab.style.display = 'none');
            document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
            const pane = document.getElementById(tabId);
            if (pane) {{
                pane.style.display = 'block';
                if (!initialised.has(plotId)) {{
                    renderPlot(`container-${plotId}`, plotId);
                    initialised.add(plotId);
                }}
            }}
            const btn = document.querySelector(`[data-tab="${tabId}"]`);
            if (btn) btn.classList.add('active');
        }}

        document.addEventListener('DOMContentLoaded', () => {{
            const first = document.querySelector('.tab-pane');
            if (first) showTab(first.id, first.dataset.plotId);
        }});
    </script>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")

# ============================= INTERNAL HELPERS =============================== #

def _prepare_map_figures(figures: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
    """Extract *sample‑map* figures and transform into tabbed HTML.

    This is essentially the helper from the user's *new method*, shortened a bit
    and renamed so it does not clash with the original `_prepare_figures`.
    """
    if not figures or "map" not in figures:
        return (
            '<div class="error">No sample maps available.</div>',
            '<div class="error">No data</div>',
            {},
        )

    maps = [(c, f) for c, f in figures["map"].items() if f][:2]
    if not maps:
        return (
            '<div class="error">No sample maps available.</div>',
            '<div class="error">No data</div>',
            {},
        )

    tab_elems: List[str] = []
    btn_elems: List[str] = []
    plot_data: Dict[str, Any] = {}

    for idx, (colour_col, fig) in enumerate(maps):
        tab_id = f"tab-{idx}"
        plot_id = f"plot-{idx}"
        active_cls = "active" if idx == 0 else ""

        # buttons
        btn_elems.append(
            f"<button class='tab-button {active_cls}' data-tab='{tab_id}' "
            f"onclick=\"showTab('{tab_id}', '{plot_id}')\">{colour_col}</button>"
        )

        # tab panes – initially only the first one displayed
        display = "block" if idx == 0 else "none"
        tab_elems.append(
            f"<div id='{tab_id}' class='tab-pane' style='display:{display}' "
            f"data-plot-id='{plot_id}'>"
            f"<div id='container-{plot_id}' class='map‑plot-container'></div>"
            "</div>"
        )

        # serialise plotly fig -> dict, hide legend server‑side
        if hasattr(fig, "to_plotly_json"):
            try:
                pj = fig.to_plotly_json()
                layout = pj.get("layout", {}) or {}
                layout["showlegend"] = False
                plot_data[plot_id] = {"data": pj["data"], "layout": layout}
            except Exception as e:  # noqa: BLE001
                logger.error("Error processing %s figure: %s", colour_col, e)
                plot_data[plot_id] = {"error": str(e)}
        else:
            plot_data[plot_id] = {"error": f"Unsupported figure type: {type(fig)}"}

    return "\n".join(tab_elems), "\n".join(btn_elems), plot_data

# ---------------------------------------------------------------------------
# Remaining helpers from the **original** script (unaltered except for renaming
# _prepare_figures -> _prepare_other_figures and removing map handling)
# ---------------------------------------------------------------------------

def _prepare_other_figures(figures: Dict) -> str:  # noqa: C901 – keep legacy complexity
    """Original helper minus the *map* handling (now done elsewhere)."""
    if not figures:
        return "<p>No visualisations available</p>"

    html_parts: List[str] = []

    # ================= Alpha diversity =================
    alpha_html: List[str] = []
    for table_type, levels in figures.items():
        for level, plots in (levels or {}).items():
            if not isinstance(plots, dict):
                continue
            for plot_type, fig in plots.items():
                if "alpha" in plot_type and fig:
                    alpha_html.append(
                        _figure_to_html(fig, f"Alpha Diversity – {table_type} – {level} – {plot_type}")
                    )
    if alpha_html:
        html_parts.append("<h3>Alpha Diversity</h3>" + "\n".join(alpha_html))

    # ================= Beta diversity (with colour selectors) =============
    beta_groups: Dict[str, Dict[str, Any]] = {}
    for table_type, levels in figures.items():
        for level, methods in (levels or {}).items():
            if not isinstance(methods, dict):
                continue
            for method, colour_figs in methods.items():
                if method in {"pca", "pcoa", "tsne", "umap"} and isinstance(colour_figs, dict):
                    key = f"{table_type}_{level}_{method}"
                    beta_groups[key] = {
                        "title": f"{method.upper()} – {table_type} – {level}",
                        "figures": colour_figs,
                    }

    if beta_groups:
        html_parts.append("<div class='beta-diversity-section'><h3>Beta Diversity</h3>")
        for key, group in beta_groups.items():
            container_id = f"{key}_container"
            html_parts.append(f"<h4>{group['title']}</h4>")

            valid_figs = [(c, f) for c, f in group["figures"].items() if f is not None]
            if not valid_figs:
                continue

            # dropdown
            select_el = [f"<select class='color-selector' onchange=\"showPlot(this, '{container_id}')\">"]
            plot_divs: List[str] = []
            for i, (col, fig) in enumerate(valid_figs):
                plot_id = f"{key}_{col.replace(' ', '_')}"
                selected = "selected" if i == 0 else ""
                select_el.append(f"<option value='{plot_id}' {selected}>{col}</option>")
                plot_divs.append(
                    f"<div id='{plot_id}' class='plot-container' style='display: {'flex' if i==0 else 'none'}'>"
                    f"{_figure_to_html(fig, f'Colored by: {col}', include_caption=False)}"
                    "</div>"
                )
            select_el.append("</select>")

            html_parts.append("".join(select_el))
            html_parts.append(
                f"<div id='{container_id}' class='plot-wrapper'>" + "\n".join(plot_divs) + "</div>"
            )
        html_parts.append("</div>")  # close beta-diversity-section

    # ================= Everything else =================
    other_html: List[str] = []
    for plot_type, levels in figures.items():
        if plot_type in {"pca", "pcoa", "tsne", "umap"}:
            continue
        for level, methods in (levels or {}).items():
            if not isinstance(methods, dict):
                continue
            for method, fig in methods.items():
                if fig and not isinstance(fig, dict):
                    other_html.append(_figure_to_html(fig, f"{plot_type} – {level} – {method}"))
    if other_html:
        html_parts.append("<h3>Other Visualisations</h3>" + "\n".join(other_html))

    return "\n".join(html_parts) if html_parts else "<p>No visualisations available</p>"

# ---------------------------------------------------------------------------
# *** Everything below this line is copied verbatim from the original script ***
#   (helper functions for tables, ML section, _figure_to_html, etc.)
# ---------------------------------------------------------------------------

def _format_ml_section(ml_metrics, ml_features, shap_plot):
    """(Unchanged)"""
    if ml_metrics is None or ml_metrics.empty:
        return "<p>No ML results available</p>"

    ml_html = f"""
    <div class="ml-section">
        <h3>Model Performance</h3>
        {ml_metrics.to_html(index=False)}

        <h3>Top Features</h3>
        {ml_features.to_html(index=False, classes='ml-feature-table')}
    """

    if shap_plot:
        ml_html += f"""
        <h3>Feature Importance (Best Model)</h3>
        <div class="figure-container">
            <img src="data:image/png;base64,{shap_plot}" alt="SHAP Summary">
        </div>
        """

    ml_html += "</div>"
    return ml_html


def _prepare_features_table(
    features: List[Dict], max_features: int, category: str
) -> pd.DataFrame:
    """(Unchanged)"""
    if not features:
        return pd.DataFrame({"Message": [f"No significant {category} features found"]})

    df = pd.DataFrame(features[:max_features]).rename(
        columns={
            "feature": "Feature",
            "level": "Taxonomic Level",
            "test": "Test",
            "effect": "Effect Size",
            "p_value": "P-value",
            "effect_dir": "Direction",
        }
    )

    if "faprotax_functions" in df.columns:
        df["Functions"] = df["faprotax_functions"].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else ""
        )

    df["Effect Size"] = df["Effect Size"].apply(lambda x: f"{x:.4f}")
    df["P-value"] = df["P-value"].apply(lambda x: f"{x:.2e}")

    return df[
        [
            "Feature",
            "Taxonomic Level",
            "Test",
            "Effect Size",
            "P-value",
            "Direction",
        ]
    ]


def _prepare_stats_summary(stats: Dict) -> pd.DataFrame:
    """(Unchanged)"""
    summary = []
    for table_type, tests in stats.items():
        for test_name, levels in tests.items():
            for level, df in levels.items():
                n_sig = sum(df["p_value"] < 0.05) if "p_value" in df.columns else 0
                summary.append(
                    {
                        "Table Type": table_type,
                        "Test": test_name,
                        "Level": level,
                        "Significant Features": n_sig,
                        "Total Features": len(df),
                    }
                )
    return pd.DataFrame(summary)


def _prepare_ml_summary(models: Dict) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[str]]:
    """(Unchanged – lifted from original)"""
    if not models:
        return None, None, None

    metrics_summary: List[Dict[str, Any]] = []
    features_summary: List[Dict[str, Any]] = []
    shap_plot_base64: Optional[str] = None
    best_mcc = -1

    for table_type, levels in models.items():
        for level, methods in levels.items():
            for method, result in methods.items():
                if not result:
                    continue

                scores = result.get("test_scores", {})
                metrics_summary.append(
                    {
                        "Table Type": table_type,
                        "Level": level,
                        "Method": method,
                        "Top Features": len(result.get("top_features", [])),
                        "Accuracy": f"{scores.get('accuracy', 0):.4f}",
                        "F1 Score": f"{scores.get('f1', 0):.4f}",
                        "MCC": f"{scores.get('mcc', 0):.4f}",
                        "ROC AUC": f"{scores.get('roc_auc', 0):.4f}",
                        "PR AUC": f"{scores.get('pr_auc', 0):.4f}",
                    }
                )

                imp = result.get("feature_importances", {})
                for i, feat in enumerate(result.get("top_features", [])[:10], 1):
                    features_summary.append(
                        {
                            "Table Type": table_type,
                            "Level": level,
                            "Method": method,
                            "Rank": i,
                            "Feature": feat,
                            "Importance": f"{imp.get(feat, 0):.4f}",
                        }
                    )

                current_mcc = scores.get("mcc", -1)
                shap_path = result.get("shap_summary_bar_path")
                if current_mcc > best_mcc and shap_path:
                    try:
                        with open(shap_path, "rb") as fh:
                            shap_plot_base64 = base64.b64encode(fh.read()).decode()
                        best_mcc = current_mcc
                    except Exception as err:  # noqa: BLE001
                        logger.warning("Couldn't load SHAP plot: %s", err)

    metrics_df = pd.DataFrame(metrics_summary) if metrics_summary else None
    features_df = pd.DataFrame(features_summary) if features_summary else None
    return metrics_df, features_df, shap_plot_base64


def _figure_to_html(fig: Any, caption: str, include_caption: bool = True) -> str:
    """(Unchanged – legacy helper for plt/plotly conversion)"""
    if fig is None:
        return f"<div class='figure-container'><p>Missing figure: {caption}</p></div>"

    try:
        # -------- Plotly figures --------
        if hasattr(fig, "to_html") and callable(fig.to_html):
            html = fig.to_html(full_html=False, include_plotlyjs=False)
            cap = f"<p>{caption}</p>" if include_caption else ""
            return (
                "<div class='figure-container'><div class='plot-wrapper'>" + html + "</div>" + cap + "</div>"
            )

        # -------- Matplotlib / Seaborn --------
        buf = BytesIO()
        if hasattr(fig, "savefig"):
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
            plt.close(fig)
        elif hasattr(fig, "figure"):
            fig.figure.savefig(buf, format="png", bbox_inches="tight", dpi=100)
            plt.close(fig.figure)
        else:
            return (
                f"<div class='figure-container'><p>Unsupported figure type: {type(fig)}</p><p>{caption}</p></div>"
            )
        buf.seek(0)
        img64 = base64.b64encode(buf.read()).decode()
        cap = f"<p>{caption}</p>" if include_caption else ""
        return (
            "<div class='figure-container'><div class='plot-wrapper'>"
            f"<img src='data:image/png;base64,{img64}' alt='{caption}'>"
            "</div>" + cap + "</div>"
        )
    except Exception as err:  # noqa: BLE001
        logger.error("Error rendering figure: %s", err)
        return (
            f"<div class='figure-container'><p>Error rendering figure: {err}</p><p>{caption}</p></div>"
        )
