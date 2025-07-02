# ---------------------------------- DEBUG SAMPLE MAP REPORT ----------------------------------
"""
Generate an HTML report that embeds **only the first two sample‑map Plotly figures** found in
``amplicon_data.figures["map"]``. Each figure is exported as its own Plotly HTML fragment and
shown/hidden with a dropdown. No raw JSON blobs are used.

Usage
-----
>>> generate_html_report(amplicon_data, "report_debug.html")
"""
from __future__ import annotations

# -------- Standard library --------
import base64
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Union
import logging

# -------- Third‑party -------------
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import plotly

logger = logging.getLogger(__name__)

# =============================================================================
# PUBLIC API
# =============================================================================

def generate_html_report(
    amplicon_data: "AmpliconData",
    output_path: Union[str, Path],
) -> None:
    """Write an HTML debug page with exactly two sample‑map plots."""
    ts = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figures_html = _prepare_figures(amplicon_data.figures)

    # Match JS to the Python package version
    plotly_js_tag = (
        f"<script src='https://cdn.plot.ly/plotly-3.0.1.min.js'></script>"
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset='utf-8'>
  <title>16S Sample‑Map Debug</title>
  {plotly_js_tag}
  <style>
    body {{ font-family: Arial, sans-serif; margin: 40px; }}
    .plot-container {{ display: none; }}
    .plot-container.active {{ display: block; }}
    .color-selector {{ margin-bottom: 10px; padding: 5px; }}
  </style>
  <script>
    function showMap(id) {{
      document.querySelectorAll('.plot-container').forEach(d => d.classList.remove('active'));
      const tgt = document.getElementById(id);
      if (tgt) tgt.classList.add('active');
    }}
  </script>
</head>
<body>
  <h1>16S Amplicon Analysis – Sample‑Map Debug</h1>
  <p>Generated: {ts}</p>

  <h2>Sample Map (first two colour columns)</h2>
  {figures_html}
</body>
</html>"""
    output_path.write_text(html, encoding="utf-8")

# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _figure_to_html(fig: Any, *, w: int = 900, h: int = 600) -> str:
    """Convert a Plotly or Matplotlib figure to an embeddable HTML fragment."""
    if hasattr(fig, "to_html"):
        fig.update_layout(width=w, height=h, showlegend=False)
        return fig.to_html(full_html=False, include_plotlyjs=False)

    buf = BytesIO()
    if isinstance(fig, Figure):
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        plt.close(fig)
    elif hasattr(fig, "figure"):
        fig.figure.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        plt.close(fig.figure)
    else:
        return f"<p>Unsupported figure type: {type(fig)}</p>"
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'<img src="data:image/png;base64,{b64}" style="max-width:100%">'


def _prepare_figures(figures: Dict) -> str:
    """Create dropdown + two map divs."""
    if not figures or "map" not in figures:
        return "<p>No sample maps available.</p>"

    maps = [(c, f) for c, f in figures["map"].items() if f][:2]
    if not maps:
        return "<p>No sample maps available.</p>"

    opts, divs = [], []
    for i, (col, fig) in enumerate(maps):
        div_id = f"map{i}"
        sel = " selected" if i == 0 else ""
        cls = "plot-container active" if i == 0 else "plot-container"
        opts.append(f"<option value='{div_id}'{sel}>{col}</option>")
        divs.append(f"<div id='{div_id}' class='{cls}'>\n{_figure_to_html(fig)}\n</div>")

    dropdown = (
        "<select class='color-selector' onchange='showMap(this.value)'>\n"
        + "\n".join(opts) + "\n</select>"
    )
    return dropdown + "\n" + "\n".join(divs)
