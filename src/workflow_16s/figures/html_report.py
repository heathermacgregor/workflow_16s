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
from plotly.offline import get_plotlyjs_version  # Add this import


logger = logging.getLogger(__name__)

# =============================================================================
# PUBLIC API
# =============================================================================

import json
import numpy as np
from plotly.offline import get_plotlyjs_version

# =============================================================================
# NEW HELPER FUNCTION FOR NUMPY SERIALIZATION
# =============================================================================
def numpy_to_json(obj):
    """Recursively convert NumPy objects to JSON-serializable types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: numpy_to_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_json(item) for item in obj]
    return obj

# =============================================================================
# UPDATED REPORT FUNCTION
# =============================================================================
def generate_html_report(
    amplicon_data: "AmpliconData",
    output_path: Union[str, Path],
) -> None:
    ts = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figures_html, plot_data = _prepare_figures(amplicon_data.figures)

    # Get Plotly.js version dynamically
    plotly_js_version = get_plotlyjs_version()
    plotly_js_tag = f'<script src="https://cdn.plot.ly/plotly-{plotly_js_version}.min.js"></script>'

    # Convert plot_data to JSON-safe format
    safe_plot_data = numpy_to_json(plot_data)
    plot_data_json = json.dumps(safe_plot_data)

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset='utf-8'>
  <title>16S Sample‑Map Debug</title>
  {plotly_js_tag}
  <style>
    body {{ font-family: Arial, sans-serif; margin: 40px; }}
    .tab {{ display: none; }}
    .tab.active {{ display: block; }}
    .tab-button {{ 
      padding: 10px 15px;
      background: #eee;
      border: 1px solid #ccc;
      cursor: pointer;
    }}
    .tab-button.active {{ 
      background: #fff; 
      border-bottom: none;
    }}
    .tab-container {{ 
      border: 1px solid #ccc;
      padding: 20px;
      margin-top: -1px;
    }}
    .tabs {{
      display: flex;
      margin-bottom: -1px;
    }}
  </style>
</head>
<body>
  <h1>16S Amplicon Analysis – Sample‑Map Debug</h1>
  <p>Generated: {ts}</p>

  <h2>Sample Map (first two colour columns)</h2>
  <div class="tabs">
    {''.join([f'<div id="btn{i}" class="tab-button{" active" if i==0 else ""}" onclick="showTab({i})">{col}</div>' 
              for i, col in enumerate(plot_data.keys())])}
  </div>
  
  <div class="tab-container">
    {figures_html}
  </div>

  <script>
    const plotData = {plot_data_json};
    
    function renderPlot(containerId, data) {{
      const container = document.getElementById(containerId);
      if (!container) return;
      
      // Clean existing plot
      while(container.firstChild) container.removeChild(container.firstChild);
      
      Plotly.newPlot(container, data.data, data.layout)
        .catch(err => {{
          container.innerHTML = `<div style="color:red">Plot error: ${{err}}</div>`;
        }});
    }}
    
    function showTab(index) {{
      // Update buttons
      document.querySelectorAll('.tab-button').forEach((btn, i) => {{
        btn.classList.toggle('active', i === index);
      }});
      
      // Update tabs
      document.querySelectorAll('.tab').forEach((tab, i) => {{
        tab.classList.toggle('active', i === index);
        if (i === index) {{
          const containerId = `plot${{i}}`;
          if (!window.PLOTLY_INITIALIZED${{i}}) {{
            renderPlot(containerId, plotData[Object.keys(plotData)[i]]);
            window.PLOTLY_INITIALIZED${{i}} = true;
          }}
        }}
      }});
    }}
    
    // Initialize first tab
    document.addEventListener('DOMContentLoaded', () => showTab(0));
  </script>
</body>
</html>"""
    output_path.write_text(html, encoding="utf-8")

# =============================================================================
# UPDATED FIGURE PREPARATION
# =============================================================================
def _prepare_figures(figures: Dict) -> tuple:
    """Prepare HTML tabs and store sanitized plot data."""
    if not figures or "map" not in figures:
        return "<div class='tab active'><p>No sample maps available.</p></div>", {}
    
    maps = [(c, f) for c, f in figures["map"].items() if f][:2]
    if not maps:
        return "<div class='tab active'><p>No sample maps available.</p></div>", {}
    
    tabs = []
    plot_data = {}
    
    for i, (col, fig) in enumerate(maps):
        active = "active" if i == 0 else ""
        
        if hasattr(fig, "to_plotly_json"):
            plot_json = fig.to_plotly_json()
            # Store sanitized data
            plot_data[col] = {
                "data": plot_json["data"],
                "layout": plot_json["layout"]
            }
            tabs.append(f"""
            <div id="tab{i}" class="tab {active}">
                <div id="plot{i}" style="width:900px;height:600px;"></div>
            </div>""")
        else:
            # Fallback for matplotlib figures
            tabs.append(f"""
            <div id="tab{i}" class="tab {active}">
                {_figure_to_html(fig)}
            </div>""")
    
    return "\n".join(tabs), plot_data


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _figure_to_html(fig: Any, *, w: int = 900, h: int = 600) -> str:
    """Convert figure to HTML with safe layout updates."""
    if hasattr(fig, "to_html"):
        try:
            # Safely update layout
            fig.update_layout(width=w, height=h, showlegend=False)
        except Exception as e:
            logger.error(f"Layout update failed: {e}")
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

