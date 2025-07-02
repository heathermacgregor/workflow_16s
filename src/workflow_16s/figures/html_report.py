# ===================================== IMPORTS ======================================
from pathlib import Path
from typing import Union
import pandas as pd
import plotly.graph_objects as go   # only to satisfy type checkers

# ===================================== DEBUG‑ONLY REPORT ============================

def generate_html_report(amplicon_data: "AmpliconData",
                         output_path: Union[str, Path]) -> None:
    """
    Write an HTML page that shows ONLY the first Plotly map figure, embedded
    exactly as Plotly generates it (no JSON processing).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ts = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    # -- find the first sample‑map figure ------------------------------------------
    fig_html = "<p><strong>No sample‑map figure found.</strong></p>"
    if isinstance(amplicon_data.figures, dict) and "map" in amplicon_data.figures:
        for _col, f in amplicon_data.figures["map"].items():
            if f is not None and hasattr(f, "to_html"):
                # Plotly generates a <div> with the data + PlotlyJS loader.
                fig_html = f.to_html(full_html=False, include_plotlyjs="cdn")
                break

    # -- write the minimal wrapper --------------------------------------------------
    html_doc = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>16S Sample‑Map Debug</title>
  <style>body{{font-family:Arial,sans-serif;margin:40px;}}</style>
</head>
<body>
  <h1>16S Amplicon Analysis – Sample‑Map Debug</h1>
  <p>Generated: {ts}</p>
  {fig_html}
</body>
</html>"""

    output_path.write_text(html_doc, encoding="utf-8")
