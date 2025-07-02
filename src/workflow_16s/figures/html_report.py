# ---------------------------------- DEBUG SAMPLE MAP REPORT ----------------------------------
"""
Generate an HTML report that embeds **only the first two sample‑map Plotly figures** found in
``amplicon_data.figures["map"]``. Each figure is exported as its own Plotly HTML fragment and
shown/hidden with a dropdown. No raw JSON blobs (which require the Plotly binary plugin) are used.
This keeps things simple and reliable.

Usage
-----
>>> generate_html_report(amplicon_data, "report_debug.html")
"""
from __future__ import annotations

# -------- Standard library --------
import base64
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

# -------- Third‑party ------------
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import plotly.graph_objects as go  # noqa: F401 – imported for type checking only

logger = logging.getLogger(__name__)

# =================================================================================================
# Top‑level API
# =================================================================================================

def generate_html_report(
    amplicon_data: "AmpliconData",
    output_path: Union[str, Path],
    max_features: int = 20,
) -> None:
    """Write an HTML page for debugging the first two sample maps."""
    ts = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ layout sections (minimal)
    figures_html = _prepare_figures(amplicon_data.figures)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset=\"utf-8\">
<title>16S Sample‑Map Debug</title>
<script src=\"https://cdn.plot.ly/plotly-3.0.1.min.js\"></script>
<style>
  body {{ font-family: Arial, sans-serif; margin: 40px; }}
  h1 {{ color: #2c3e50; }}
  .section {{ margin-bottom: 40px; }}
  .plot-container {{ display: none; }}
  .plot-container.active {{ display: block; }}
  .color-selector {{ margin-bottom: 10px; padding: 5px; }}
</style>
<script>
function showMap(id) {{
  document.querySelectorAll('.plot-container').forEach(div => div.classList.remove('active'));
  const tgt = document.getElementById(id);
  if (tgt) tgt.classList.add('active');
}}
</script>
</head>
<body>
  <h1>16S Amplicon Analysis – Sample‑Map Debug</h1>
  <p>Generated: {ts}</p>

  <div class='section'>
    <h2>Sample Map (first two colour columns)</h2>
    {figures_html}
  </div>
</body>
</html>"""
    output_path.write_text(html, encoding="utf-8")

# =================================================================================================
# Figure handling (debug only)
# =================================================================================================

def _figure_to_html(fig: Any, *, width: int = 900, height: int = 600) -> str:
    """Return a Plotly or Matplotlib figure embedded as HTML/PNG."""
    if hasattr(fig, "to_html"):
        fig.update_layout(width=width, height=height, showlegend=False)
        return fig.to_html(full_html=False, include_plotlyjs=False)

    # Fallback to PNG if not Plotly
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
    return f'<img src="data:image/png;base64,{img_b64}" style="max-width:100%">'


def _prepare_figures(figures: Dict) -> str:
    """Return dropdown + two Plotly map fragments wrapped for toggling."""
    if not figures or "map" not in figures:
        return "<p>No sample maps available.</p>"

    # Pick first two maps that are not None
    maps = [(col, fig) for col, fig in figures["map"].items() if fig][:2]
    if not maps:
        return "<p>No sample maps available.</p>"

    # Build dropdown
    options_html: List[str] = []
    divs_html: List[str] = []

    for i, (col, fig) in enumerate(maps):
        div_id = f"map{i}"
        sel = " selected" if i == 0 else ""
        options_html.append(f"<option value='{div_id}'{sel}>{col}</option>")

        div_class = "plot-container active" if i == 0 else "plot-container"
        divs_html.append(
            f"<div id='{div_id}' class='{div_class}'>\n{_figure_to_html(fig)}\n</div>"
        )

    dropdown = (
        "<select class='color-selector' onchange='showMap(this.value)'>\n"
        + "\n".join(options_html)
        + "\n</select>"
    )
    wrapper = "<div id='map_wrapper'>" + "\n".join(divs_html) + "</div>"

    return dropdown + "\n" + wrapper + "\n"
