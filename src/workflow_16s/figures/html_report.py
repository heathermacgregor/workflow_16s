# ===================================== IMPORTS ======================================
from pathlib import Path
from typing import Union
import pandas as pd
import plotly.io as pio

# ===================================== DEBUG REPORT ================================

def generate_html_report(amplicon_data: "AmpliconData",
                         output_path: Union[str, Path]) -> None:
    """
    Minimal HTML writer that embeds ONLY the first sample‑map figure,
    passing data, layout *and* config back to Plotly.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ts = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Grab the first Plotly figure in `figures["map"]`
    fig = None
    if isinstance(amplicon_data.figures, dict) and "map" in amplicon_data.figures:
        for _col, f in amplicon_data.figures["map"].items():
            if f is not None and hasattr(f, "to_json"):
                fig = f
                break

    if fig is None:
        body_html = "<p><strong>No sample‑map figure found.</strong></p>"
    else:
        # Let Plotly serialise itself (handles NumPy arrays)
        fig_json = fig.to_json()
        body_html = f"""
<div id="plotly-target" style="width:100%; height:500px;"></div>

<!-- Raw figure JSON -->
<script id="fig-json" type="application/json">
{fig_json}
</script>
"""

    # ── Build the HTML document
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
  const raw  = document.getElementById("fig-json");
  if (!tgt || !raw) return;

  const fig = JSON.parse(raw.textContent);

  // ── OPTIONAL DEBUG: ensure markers are visible ──
  // fig.data.forEach(tr => {{
  //   if (tr.type.startsWith("scatter")) {{
  //     tr.marker = tr.marker || {{}};
  //     tr.marker.size = tr.marker.size || 8;
  //     tr.marker.color = tr.marker.color || "red";
  //     tr.marker.opacity = 1;
  //   }}
  // }});

  Plotly.newPlot(
    tgt,
    fig.data,
    fig.layout,
    {{...fig.config, responsive:true}}   // <= include CONFIG!
  ).catch(e => {{
    tgt.innerHTML = "<p>Error rendering Plotly figure.</p>";
    console.error(e);
  }});
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
