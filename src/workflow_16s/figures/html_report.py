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
    Generate a debug 16S analysis report that shows only the first TWO
    sample‑map figures with a color‑column dropdown.
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

    # 4. Figures – only the first two sample maps + dropdown
    figures_html = _prepare_figures(amplicon_data.figures)

    # 5. Compose document
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>16S Analysis Report – DEBUG</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 40px; }}
  h1, h2, h3, h4 {{ color: #2c3e50; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
  th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
  th {{ background-color: #f2f2f2; }}
  .section {{ margin-bottom: 40px; }}
  .figure-container {{ border: 1px solid #ddd; padding: 10px; }}
  .plot-wrapper {{ width: 100%; height: 600px; position: relative; }}
  .plot-container {{ display: none; width: 100%; height: 100%; }}
  .plot-container.active {{ display: block; }}
  .color-selector {{ padding: 5px; margin-bottom: 10px; }}
</style>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<script>
function showSampleMap(sel) {{
  const plots = document.querySelectorAll('#sample_map_wrapper .plot-container');
  plots.forEach(p => p.classList.remove('active'));
  const tgt = document.getElementById(sel.value);
  if (tgt) tgt.classList.add('active');
}}
</script>
</head>
<body>
<h1>16S Amplicon Analysis Report – DEBUG</h1>
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
  <h2>Sample Map (first two color columns)</h2>
  {figures_html}
</div>

</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")

# =============================== HELPER FUNCTIONS ===================================

def _format_ml_section(
    ml_metrics: Optional[pd.DataFrame],
    ml_features: Optional[pd.DataFrame],
    shap_plot: Optional[str],
) -> str:
    if ml_metrics is None or ml_metrics.empty:
        return "<p>No ML results available.</p>"

    out = [ml_metrics.to_html(index=False)]
    out.append("<h3>Top Features</h3>")
    out.append(ml_features.to_html(index=False, classes='ml-feature-table')
               if ml_features is not None else "<p>No feature table.</p>")

    if shap_plot:
        out.append(
            f'<div class="figure-container"><img src="data:image/png;base64,{shap_plot}"'
            ' alt="SHAP Summary"></div>'
        )
    return "\n".join(out)


def _prepare_features_table(
    features: List[Dict],
    max_features: int,
    category: str,
) -> pd.DataFrame:
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
    df["Effect Size"] = df["Effect Size"].apply(lambda x: f"{x:.4f}")
    df["P-value"] = df["P-value"].apply(lambda x: f"{x:.2e}")
    cols = ["Feature", "Taxonomic Level", "Test", "Effect Size", "P-value", "Direction"]
    return df[cols]


def _prepare_stats_summary(stats: Dict) -> pd.DataFrame:
    rows = []
    for ttype, tests in stats.items():
        for test, lvls in tests.items():
            for lvl, df in lvls.items():
                n_sig = sum(df["p_value"] < 0.05) if "p_value" in df.columns else 0
                rows.append(
                    {"Table Type": ttype, "Test": test, "Level": lvl,
                     "Significant Features": n_sig, "Total Features": len(df)}
                )
    return pd.DataFrame(rows)


def _prepare_ml_summary(
    models: Dict,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[str]]:
    if not models:
        return None, None, None

    metrics, feats = [], []
    shap_b64, best_mcc = None, -1
    for ttype, lvls in models.items():
        for lvl, methods in lvls.items():
            for mtd, res in methods.items():
                if not res:
                    continue
                sc = res.get("test_scores", {})
                metrics.append(
                    {"Table": ttype, "Level": lvl, "Method": mtd,
                     "Accuracy": f"{sc.get('accuracy',0):.4f}",
                     "F1": f"{sc.get('f1',0):.4f}",
                     "MCC": f"{sc.get('mcc',0):.4f}"}
                )

                for rk, feat in enumerate(res.get("top_features", [])[:10], 1):
                    feats.append(
                        {"Table": ttype, "Level": lvl, "Method": mtd,
                         "Rank": rk, "Feature": feat}
                    )

                mcc = sc.get("mcc", -1)
                shap_path = res.get("shap_summary_bar_path")
                if mcc > best_mcc and shap_path:
                    try:
                        with open(shap_path, "rb") as fh:
                            shap_b64 = base64.b64encode(fh.read()).decode()
                        best_mcc = mcc
                    except Exception:
                        pass

    return (pd.DataFrame(metrics) if metrics else None,
            pd.DataFrame(feats) if feats else None,
            shap_b64)

# --------------------------- FIGURE HANDLING (DEBUG) --------------------------------

def _figure_to_html(
    fig: Any,
    *,
    width: int = 900,
    height: int = 600,
) -> str:
    """Embed a Plotly figure as raw HTML (no legend)."""
    if hasattr(fig, "to_html"):
        fig.update_layout(width=width, height=height, showlegend=False)
        return fig.to_html(full_html=False, include_plotlyjs="cdn")
    # Matplotlib fallback
    buf = BytesIO()
    if isinstance(fig, Figure):
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        plt.close(fig)
    elif hasattr(fig, "figure"):
        fig.figure.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        plt.close(fig.figure)
    else:
        return f"<p>Unsupported figure type: {type(fig)}</p>"
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    return f'<img src="data:image/png;base64,{img_b64}" style="max-width:100%;">'

def _prepare_figures(figures: Dict) -> str:
    """
    Build a dropdown + single container to switch between
    the first two sample‑map Plotly figures.
    Assumes your <head> already has:
      <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    """
    # 1. grab up to two maps
    maps = [(col, fig) for col, fig in figures.get("map", {}).items() if fig][:2]
    if not maps:
        return "<p>No sample maps available.</p>"

    # 2. resize + hide legend + serialize each to JSON
    blobs = []
    for col, fig in maps:
        fig.update_layout(width=900, height=600, showlegend=False)
        blobs.append((col, fig.to_json()))

    # 3. build <option>s and <script type="application/json"> blocks
    opts = []
    scripts = []
    for i, (col, blob) in enumerate(blobs):
        tag = f"fig{i}"
        selected = " selected" if i == 0 else ""
        opts.append(f"<option value='{tag}'{selected}>{col}</option>")
        scripts.append(
            f"<script id='{tag}' type='application/json'>\n{blob}\n</script>"
        )

    dropdown = (
        "<select class='color-selector' onchange='renderMap()'>\n"
        + "\n".join(opts)
        + "\n</select>"
    )

    # 4. single div where we’ll draw
    container = "<div id='map_container' style='width:900px;height:600px;'></div>"

    # 5. inline script to actually render
    runner = """
<script>
function renderMap() {
  const sel = document.querySelector('.color-selector');
  const raw = document.getElementById(sel.value);
  if (!raw) return;
  const fig = JSON.parse(raw.textContent);
  Plotly.newPlot('map_container', fig.data, fig.layout, {responsive:true});
}
document.addEventListener('DOMContentLoaded', renderMap);
</script>
"""

    # 6. glue it all together
    return "\n".join([dropdown, container] + scripts + [runner])
