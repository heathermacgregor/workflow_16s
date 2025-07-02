# ===================================== IMPORTS ======================================
from pathlib import Path
from typing import Union
import pandas as pd
# ------------------------------------------------------------------- only Plotly used
import plotly.graph_objects as go

# ===================================== DEBUG REPORT ================================

def generate_html_report(amplicon_data: "AmpliconData",
                         output_path: Union[str, Path]) -> None:
    """
    Write an HTML page that shows *only* the first Plotly sample‑map in
    `amplicon_data.figures["map"]`. Other sections are omitted so you can
    debug figure embedding.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ts = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── grab the first available map figure ────────────────────────────────────────
    fig = None
    if isinstance(amplicon_data.figures, dict) and "map" in amplicon_data.figures:
        for _col, f in amplicon_data.figures["map"].items():
            if f is not None and hasattr(f, "to_json"):
                fig = f
                break

    if fig is None:
        body_html = "<p><strong>No sample‑map figure found.</strong></p>"
    else:
        # Let Plotly serialise its own structure (handles ndarrays):
        fig_json = fig.to_json()
        body_html = f"""
<div id="plotly-target" style="width:100%; height:500px;"></div>

<!-- Raw figure JSON lives here; no need to escape quotes -->
<script id="fig-json" type="application/json">
{fig_json}
</script>
"""

    # ── assemble the minimal HTML document ─────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>16S Sample‑Map Debug</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>body{{font-family:Arial,sans-serif;margin:40px;}}</style>
<script>
document.addEventListener("DOMContentLoaded", () => {{
  const tgt  = document.getElementById("plotly-target");
  const json = document.getElementById("fig-json");
  if (tgt && json) {{
    try {{
      const fig = JSON.parse(json.textContent);
      Plotly.newPlot(tgt, fig.data, fig.layout, {{responsive:true}});
    }} catch(e) {{
      tgt.innerHTML = "<p>Error rendering Plotly figure.</p>";
      console.error(e);
    }}
  }}
}});
</script>
</head>
<body>
<h1>16S Amplicon Analysis – Sample‑Map Debug</h1>
<p>Generated: {ts}</p>
{body_html}
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
