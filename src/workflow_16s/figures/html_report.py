# ===================================== IMPORTS ======================================

# -------- Standard Library --------
import base64
import logging
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Optional

# -------- Third‑Party -------------
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import plotly.graph_objects as go

# ========================== INITIALIZATION & CONFIGURATION ==========================

logger = logging.getLogger("workflow_16s")
timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

# ==================================== MAIN API =====================================

def generate_html_report(
    amplicon_data: "AmpliconData",
    output_path: Union[str, Path],
    max_features: int = 20,
) -> None:
    """
    Generate the full 16S analysis HTML report.

    Args
    ----
    amplicon_data : AmpliconData
        Object holding results, figures, models, etc.
    output_path   : str | Path
        Where to write the HTML file.
    max_features  : int, default 20
        Maximum rows to include in the top‑features tables.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Top‑feature tables
    contam_df  = _prepare_features_table(
        amplicon_data.top_contaminated_features, max_features, "Contaminated"
    )
    pristine_df = _prepare_features_table(
        amplicon_data.top_pristine_features, max_features, "Pristine"
    )

    # 2. Stats summary
    stats_summary = _prepare_stats_summary(amplicon_data.stats)

    # 3. ML summary
    ml_metrics, ml_features, shap_plot = _prepare_ml_summary(amplicon_data.models)

    # 4. Figures (with Plotly‑native embedding)
    figures_html = _prepare_figures(amplicon_data.figures)

    # 5. Compose document
    html_content = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>16S Analysis Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 40px; }}
    h1, h2, h3, h4 {{ color: #2c3e50; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background-color: #f2f2f2; }}
    .section {{ margin-bottom: 40px; }}
    .figure-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(500px, 1fr)); gap: 20px; }}
    .figure-container {{ border: 1px solid #ddd; padding: 10px; }}
    .plot-wrapper {{ width: 100%; height: 400px; display: flex; justify-content: center; align-items: center; overflow: hidden; }}
    .plot-container {{ display: none; width: 100%; height: 100%; }}
    .plot-container.active {{ display: flex; justify-content: center; align-items: center; }}
    .color-selector {{ margin: 10px 0; padding: 5px; width: 100%; }}
    .ml-feature-table tr:nth-child(even) {{ background-color: #f9f9f9; }}
  </style>
</head>
<body>
  <h1>16S Amplicon Analysis Report</h1>
  <p>Generated on {timestamp}</p>

  <div class='section'>
    <h2>Top Features</h2>
    <h3>Contaminated‑Associated Features</h3>
    {contam_df.to_html(index=False, classes='feature-table')}
    <h3>Pristine‑Associated Features</h3>
    {pristine_df.to_html(index=False, classes='feature-table')}
  </div>

  <div class='section'>
    <h2>Statistical Summary</h2>
    {stats_summary.to_html(index=False)}
  </div>

  <div class='section'>
    <h2>Machine‑Learning Results</h2>
    {_format_ml_section(ml_metrics, ml_features, shap_plot)}
  </div>

  <div class='section'>
    <h2>Visualisations</h2>
    <div class='figure-grid'>
      {figures_html}
    </div>
  </div>
</body>
</html>
"""
    output_path.write_text(html_content, encoding="utf-8")

# =============================== HELPER FUNCTIONS ===================================

def _format_ml_section(
    ml_metrics: Optional[pd.DataFrame],
    ml_features: Optional[pd.DataFrame],
    shap_plot: Optional[str],
) -> str:
    """Render ML section HTML."""
    if ml_metrics is None or ml_metrics.empty:
        return "<p>No ML results available.</p>"

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
    features: List[Dict],
    max_features: int,
    category: str,
) -> pd.DataFrame:
    """Top‑features table."""
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

    cols = ["Feature", "Taxonomic Level", "Test", "Effect Size", "P-value", "Direction"]
    return df[cols]


def _prepare_stats_summary(stats: Dict) -> pd.DataFrame:
    """Statistical summary table."""
    rows = []
    for table_type, tests in stats.items():
        for test_name, levels in tests.items():
            for level, df in levels.items():
                n_sig = sum(df["p_value"] < 0.05) if "p_value" in df.columns else 0
                rows.append(
                    {
                        "Table Type": table_type,
                        "Test": test_name,
                        "Level": level,
                        "Significant Features": n_sig,
                        "Total Features": len(df),
                    }
                )
    return pd.DataFrame(rows)


def _prepare_ml_summary(
    models: Dict,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[str]]:
    """Collate ML metrics, feature importances, best SHAP plot."""
    if not models:
        return None, None, None

    metrics_rows, feature_rows = [], []
    shap_plot_b64, best_mcc = None, -1

    for table_type, levels in models.items():
        for level, methods in levels.items():
            for method, result in methods.items():
                if not result:
                    continue

                sc = result.get("test_scores", {})
                metrics_rows.append(
                    {
                        "Table Type": table_type,
                        "Level": level,
                        "Method": method,
                        "Top Features": len(result.get("top_features", [])),
                        "Accuracy": f"{sc.get('accuracy', 0):.4f}",
                        "F1 Score": f"{sc.get('f1', 0):.4f}",
                        "MCC": f"{sc.get('mcc', 0):.4f}",
                        "ROC AUC": f"{sc.get('roc_auc', 0):.4f}",
                        "PR AUC": f"{sc.get('pr_auc', 0):.4f}",
                    }
                )

                feat_imp = result.get("feature_importances", {})
                for rank, feat in enumerate(result.get("top_features", [])[:10], 1):
                    feature_rows.append(
                        {
                            "Table Type": table_type,
                            "Level": level,
                            "Method": method,
                            "Rank": rank,
                            "Feature": feat,
                            "Importance": f"{feat_imp.get(feat, 0):.4f}",
                        }
                    )

                cur_mcc = sc.get("mcc", -1)
                shap_path = result.get("shap_summary_bar_path")
                if cur_mcc > best_mcc and shap_path:
                    try:
                        with open(shap_path, "rb") as fh:
                            shap_plot_b64 = base64.b64encode(fh.read()).decode()
                        best_mcc = cur_mcc
                    except Exception as e:
                        logger.warning("Couldn't load SHAP plot: %s", e)

    metrics_df  = pd.DataFrame(metrics_rows)  if metrics_rows  else None
    features_df = pd.DataFrame(feature_rows) if feature_rows else None
    return metrics_df, features_df, shap_plot_b64


# --------------------------- FIGURE HANDLING (UPDATED) ------------------------------

def _figure_to_html(
    fig: Any,
    caption: str,
    *,
    width: int | None = None,
    height: int | None = None,
    hide_legend: bool = False,
) -> str:
    """
    Convert a Plotly or Matplotlib figure to HTML.

    Plotly   → embed via `to_html` (safer than manual JSON).
    Matplotlib → fallback to base64 PNG.
    """
    # ---------- Plotly -------------------------------------------------------------
    if hasattr(fig, "to_html"):
        if width or height or hide_legend:
            fig.update_layout(
                width=width   or fig.layout.width,
                height=height or fig.layout.height,
                showlegend=False if hide_legend else fig.layout.showlegend,
            )
        plot_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
        return f"""
<div class='figure-container'>
  {plot_html}
  <p>{caption}</p>
</div>
"""

    # ---------- Matplotlib / Seaborn ----------------------------------------------
    buf = BytesIO()
    if isinstance(fig, Figure):
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        plt.close(fig)
    elif hasattr(fig, "figure"):
        fig.figure.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        plt.close(fig.figure)
    else:
        return f"<div class='figure-container'><p>Unsupported figure type: {type(fig)}</p><p>{caption}</p></div>"

    img_b64 = base64.b64encode(buf.getvalue()).decode()
    return f"""
<div class='figure-container'>
  <div class='plot-wrapper'>
    <img src='data:image/png;base64,{img_b64}' alt='{caption}' style='max-width:100%;'>
  </div>
  <p>{caption}</p>
</div>
"""


def _prepare_figures(figures: Dict) -> str:
    """Build HTML for every figure category."""
    if not figures:
        return "<p>No visualisations available</p>"

    html_parts: List[str] = []

    # ───────────────────────────── SAMPLE MAPS ────────────────────────────────────
    if "map" in figures:
        html_parts.append("<h3>Sample Maps</h3>")
        for col, fig in figures["map"].items():
            if fig:
                html_parts.append(
                    _figure_to_html(
                        fig,
                        f"Sample Map: {col}",
                        width=900,
                        height=600,
                        hide_legend=True,
                    )
                )

    # ───────────────────────────── ALPHA DIV ─────────────────────────────────────
    alpha_html: List[str] = []
    for table_type, levels in figures.items():
        if table_type == "map":
            continue
        for level, plots in levels.items():
            if not isinstance(plots, dict):
                continue
            for plot_type, fig in plots.items():
                if "alpha" in plot_type and fig:
                    alpha_html.append(
                        _figure_to_html(
                            fig,
                            f"Alpha Diversity – {table_type} – {level} – {plot_type}",
                        )
                    )
    if alpha_html:
        html_parts.append("<h3>Alpha Diversity</h3>" + "\n".join(alpha_html))

    # ───────────────────────────── BETA DIV (+ dropdown) ─────────────────────────
    beta_groups: Dict[str, Dict[str, Any]] = {}
    for table_type, levels in figures.items():
        if table_type == "map":
            continue
        for level, methods in levels.items():
            if not isinstance(methods, dict):
                continue
            for method, color_figs in methods.items():
                if method in {"pca", "pcoa", "tsne", "umap"} and isinstance(color_figs, dict):
                    key = f"{table_type}_{level}_{method}"
                    beta_groups[key] = {
                        "title": f"{method.upper()} – {table_type} – {level}",
                        "figures": color_figs,
                    }

    if beta_groups:
        html_parts.append("<div class='beta-diversity-section'><h3>Beta Diversity</h3>")
        for key, group in beta_groups.items():
            container_id = f"{key}_container"
            html_parts.append(f"<h4>{group['title']}</h4>")

            select_html = f"<select class='color-selector' onchange='showPlot(this, \"{container_id}\")'>"
            options, plot_divs = [], []

            valid_figs = [(col, fig) for col, fig in group["figures"].items() if fig]
            if not valid_figs:
                continue

            for i, (col, fig) in enumerate(valid_figs):
                plot_id = f"{key}_{col.replace(' ', '_')}"
                opt_sel = "selected" if i == 0 else ""
                options.append(f"<option value='{plot_id}' {opt_sel}>{col}</option>")
                plot_divs.append(
                    f"<div id='{plot_id}' class='plot-container{' active' if i==0 else ''}'>"
                    f"{_figure_to_html(fig, f'Colored by: {col}')}"
                    "</div>"
                )

            select_html += "\n".join(options) + "</select>"
            plot_wrapper = f"<div id='{container_id}' class='plot-wrapper'>" + "\n".join(plot_divs) + "</div>"

            html_parts.append(select_html)
            html_parts.append(plot_wrapper)

        html_parts.append("</div>")  # end beta section

    # ───────────────────────────── OTHER PLOTS ───────────────────────────────────
    other_html: List[str] = []
    for plot_type, levels in figures.items():
        if plot_type in {"map", "pca", "pcoa", "tsne", "umap"}:
            continue
        if not isinstance(levels, dict):
            continue
        for level, methods in levels.items():
            if not isinstance(methods, dict):
                continue
            for method, fig in methods.items():
                if fig and not isinstance(fig, dict):
                    other_html.append(
                        _figure_to_html(fig, f"{plot_type} – {level} – {method}")
                    )
    if other_html:
        html_parts.append("<h3>Other Visualisations</h3>" + "\n".join(other_html))

    return "\n".join(html_parts) if html_parts else "<p>No visualisations available</p>"
