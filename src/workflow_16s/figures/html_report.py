import json
from pathlib import Path
from typing import Union
import pandas as pd

def generate_html_report(
    amplicon_data: "AmpliconData",
    output_path: Union[str, Path],
) -> None:
    """Render the first sample map from amplicon_data.figures['map'] as HTML with Plotly."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    # Get the first available sample map Plotly figure
    fig = None
    if isinstance(amplicon_data.figures, dict) and "map" in amplicon_data.figures:
        for _col, f in amplicon_data.figures["map"].items():
            if f is not None and hasattr(f, "to_dict"):
                fig = f
                break

    if fig is None:
        figure_html = "<p><strong>No sample map figure found.</strong></p>"
    else:
        # Convert Plotly figure to dict, then JSON and escape quotes
        fig_json = json.dumps(fig.to_dict()).replace('"', "&quot;")
        figure_html = f"""
<div id="plotly-target" style="width:100%; height:500px;" data-figure="{fig_json}"></div>
"""

    html_doc = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>16S Sample Map Debug</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 40px; }}
  </style>
  <script>
    document.addEventListener("DOMContentLoaded", () => {{
      const tgt = document.getElementById("plotly-target");
      if (tgt && tgt.dataset.figure) {{
        try {{
          const fig = JSON.parse(tgt.dataset.figure.replaceAll('&quot;', '"'));
          Plotly.newPlot(tgt, fig.data, fig.layout, {{ responsive: true }});
        }} catch (err) {{
          tgt.innerHTML = '<p>Error rendering Plotly figure.</p>';
          console.error("Plotly rendering error:", err);
        }}
      }}
    }});
  </script>
</head>
<body>
  <h1>16S Amplicon Analysis – Sample Map Debug</h1>
  <p>Generated: {timestamp}</p>
  {figure_html}
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_doc)
