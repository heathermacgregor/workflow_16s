# -------- Standard library --------
import base64
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Union
import logging
import json
import numpy as np
from plotly.offline import get_plotlyjs_version

# -------- Third‑party -------------
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

# =============================================================================
# CUSTOM JSON ENCODER FOR NUMPY TYPES
# =============================================================================
class NumpySafeJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NumPy data types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

# =============================================================================
# PUBLIC API
# =============================================================================

def generate_html_report(
    amplicon_data: "AmpliconData",
    output_path: Union[str, Path],
) -> None:
    """Write an HTML debug page with interactive sample-map plots with hidden legends."""
    ts = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare figures and plot data
    tabs_html, buttons_html, plot_data = _prepare_figures(amplicon_data.figures)
    
    # Get Plotly.js version dynamically
    try:
        plotly_js_version = get_plotlyjs_version()
    except Exception:
        plotly_js_version = "3.0.1"  # Fallback version
    plotly_js_tag = f'<script src="https://cdn.plot.ly/plotly-{plotly_js_version}.min.js"></script>'

    # Convert plot data to JSON with numpy support
    plot_data_json = json.dumps(plot_data, cls=NumpySafeJSONEncoder)

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset='utf-8'>
  <title>16S Sample‑Map Debug</title>
  {plotly_js_tag}
  <style>
    body {{
        font-family: Arial, sans-serif;
        margin: 40px;
        line-height: 1.6;
    }}
    .tabs {{
        display: flex;
        margin-bottom: -1px;
    }}
    .tab-button {{
        padding: 10px 15px;
        background: #eee;
        border: 1px solid #ccc;
        cursor: pointer;
        border-radius: 5px 5px 0 0;
        margin-right: 5px;
    }}
    .tab-button.active {{
        background: #fff;
        border-bottom: 1px solid #fff;
        position: relative;
        z-index: 2;
    }}
    .tab-content {{
        border: 1px solid #ccc;
        padding: 20px;
        border-radius: 0 5px 5px 5px;
        position: relative;
        z-index: 1;
    }}
    .plot-container {{
        width: 900px;
        height: 600px;
    }}
    .error {{
        color: #d32f2f;
        padding: 10px;
        border: 1px solid #ffcdd2;
        background: #ffebee;
    }}
  </style>
</head>
<body>
  <h1>16S Amplicon Analysis – Sample‑Map Debug</h1>
  <p>Generated: {ts}</p>

  <h2>Sample Maps</h2>
  
  <div class="tabs">
    {buttons_html}
  </div>
  
  <div class="tab-content">
    {tabs_html}
  </div>

  <script>
    // Store plot data
    const plotData = {plot_data_json};
    
    // Track initialized plots
    const initializedPlots = new Set();
    
    function renderPlot(containerId, plotId) {{
        const container = document.getElementById(containerId);
        if (!container) {{
            console.error('Container not found:', containerId);
            return;
        }}
        
        // Clear previous content
        container.innerHTML = '';
        
        // Create plot div
        const plotDiv = document.createElement('div');
        plotDiv.id = plotId;
        plotDiv.className = 'plot-container';
        container.appendChild(plotDiv);
        
        // Get plot data
        const data = plotData[plotId];
        if (!data) {{
            plotDiv.innerHTML = '<div class="error">Plot data not available</div>';
            return;
        }}
        
        try {{
            // Ensure legend is hidden on client side
            if (data.layout) {{
                data.layout.showlegend = false;
            }}
            
            // Render the plot
            Plotly.newPlot(plotId, data.data, data.layout)
                .catch(error => {{
                    plotDiv.innerHTML = `<div class="error">Plot error: ${{error}}</div>`;
                    console.error('Plotly error:', error);
                }});
        }} catch (error) {{
            plotDiv.innerHTML = `<div class="error">JS error: ${{error}}</div>`;
            console.error('Rendering error:', error);
        }}
    }}
    
    function showTab(tabId, plotId) {{
        // Hide all tabs
        document.querySelectorAll('.tab-pane').forEach(tab => {{
            tab.style.display = 'none';
        }});
        
        // Remove active class from buttons
        document.querySelectorAll('.tab-button').forEach(btn => {{
            btn.classList.remove('active');
        }});
        
        // Show selected tab
        const tab = document.getElementById(tabId);
        if (tab) {{
            tab.style.display = 'block';
            
            // Render plot if not initialized
            if (!initializedPlots.has(plotId)) {{
                renderPlot(`container-${{plotId}}`, plotId);
                initializedPlots.add(plotId);
            }}
        }}
        
        // Activate button
        const btn = document.querySelector(`[data-tab="${{tabId}}"]`);
        if (btn) btn.classList.add('active');
    }}
    
    // Initialize first tab
    document.addEventListener('DOMContentLoaded', () => {{
        const firstTab = document.querySelector('.tab-pane');
        if (firstTab) {{
            const plotId = firstTab.dataset.plotId;
            showTab(firstTab.id, plotId);
        }}
    }});
  </script>
</body>
</html>"""
    output_path.write_text(html, encoding="utf-8")

# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _prepare_figures(figures: Dict) -> tuple:
    """Prepare HTML tabs and plot data with legends hidden."""
    if not figures or "map" not in figures:
        return (
            '<div class="error">No sample maps available.</div>',
            '<div class="error">No data</div>',
            {}
        )
    
    maps = [(c, f) for c, f in figures["map"].items() if f]
    if not maps:
        return (
            '<div class="error">No sample maps available.</div>',
            '<div class="error">No data</div>',
            {}
        )
    
    tabs = []
    buttons = []
    plot_data = {}
    
    for i, (col, fig) in enumerate(maps):
        tab_id = f"tab-{i}"
        plot_id = f"plot-{i}"
        
        # Add tab button
        active = "active" if i == 0 else ""
        buttons.append(
            f'<button class="tab-button {active}" data-tab="{tab_id}" '
            f'onclick="showTab(\'{tab_id}\', \'{plot_id}\')">{col}</button>'
        )
        
        # Add tab content
        tabs.append(
            f'<div id="{tab_id}" class="tab-pane" style="display:{"block" if i == 0 else "none"}" '
            f'data-plot-id="{plot_id}">'
            f'<div id="container-{plot_id}" class="plot-container"></div>'
            f'</div>'
        )
        
        # Prepare plot data with legend hidden
        if hasattr(fig, "to_plotly_json"):
            try:
                plot_json = fig.to_plotly_json()
                
                # Hide legend in layout configuration
                layout = plot_json.get("layout", {})
                layout["showlegend"] = False  # Hide legend on server side
                
                plot_data[plot_id] = {
                    "data": plot_json["data"],
                    "layout": layout
                }
            except Exception as e:
                logger.error(f"Error processing {col} figure: {e}")
                plot_data[plot_id] = {"error": str(e)}
        else:
            plot_data[plot_id] = {"error": f"Unsupported figure type: {type(fig)}"}
    
    return "\n".join(tabs), "\n".join(buttons), plot_data
