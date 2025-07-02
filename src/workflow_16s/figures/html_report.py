# ===================================== IMPORTS ======================================

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Union

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import plotly.graph_objects as go

# ===================================== CONSTANTS ====================================

_TIMESTAMP = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

# ===================================== HTML HELPERS =================================


def _figure_to_html(fig: Any, caption: str = "") -> str:
    """Return an <img> (matplotlib) **or** a Plotly container (<div>) for `fig`."""
    # --- Plotly (preferred) ---------------------------------------------------------
    if hasattr(fig, "to_json"):
        json_str = fig.to_json()
        return (
            f"<div id='plotly-target' style='width:100%;height:500px'"
            f"      data-figure='{json_str}'></div>"
            f"<p>{caption}</p>"
        )

    # --- Matplotlib fallback --------------------------------------------------------
    buf = BytesIO()

    if isinstance(fig, Figure):
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        plt.close(fig)
    elif hasattr(fig, "figure"):  # seaborn / axes‑level
        fig.figure.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        plt.close(fig.figure)
    else:
        return f"<p>Unsupported figure type: {type(fig)!r}</p>"

    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"<img src='data:image/png;base64,{img_b64}' alt='{caption}' style='max-width:100%'><p>{caption}</p>"


# ===================================== MAIN ENTRY ===================================


def generate_html_report(
    amplicon_data: "AmpliconData",
    output_path: Union[str, Path],
) -> None:
    """
    Write an HTML page that *only* shows the first sample map contained in
    `amplicon_data.figures["map"]`.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ find the map
    first_map_fig = None
    if isinstance(amplicon_data.figures, Dict) and "map" in amplicon_data.figures:
        for _col, fig in amplicon_data.figures["map"].items():
            if fig is not None:  # keep the very first non‑None entry
                first_map_fig = fig
                break

    if first_map_fig is None:
        body_html = "<p><strong>No sample map figure found.</strong></p>"
    else:
        body_html = _figure_to_html(first_map_fig, caption="Sample map #1")

    # -------------------------------------------------------------- assemble document
    html_doc = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>16S Sample‑Map Debug</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
      body {{ font-family: Arial, sans-serif; margin: 40px; }}
  </style>
  <script>
    document.addEventListener("DOMContentLoaded", () => {{
        const tgt = document.getElementById("plotly-target");
        if (tgt && tgt.dataset.figure) {{
            const fig = JSON.parse(tgt.dataset.figure);
            Plotly.newPlot(tgt, fig.data, fig.layout, {{responsive: true}});
        }}
    }});
  </script>
</head>
<body>
  <h1>16S Amplicon Analysis – Sample Map Debug</h1>
  <p>Generated: {_TIMESTAMP}</p>
  {body_html}
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html_doc)
