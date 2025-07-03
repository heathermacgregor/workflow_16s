# -------- Standard library --------
import base64
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Union, Optional, Tuple
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
    include_sections: List[str] = ["map", "ordination"]
) -> None:
    """Write an HTML debug page with interactive sample-map plots with hidden legends."""
    ts = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare sections with subsections
    sections_data = _prepare_sections(amplicon_data.figures, include_sections)
    
    # Get Plotly.js version dynamically
    try:
        plotly_js_version = get_plotlyjs_version()
    except Exception:
        plotly_js_version = "3.0.1"  # Fallback version
    plotly_js_tag = f'<script src="https://cdn.plot.ly/plotly-{plotly_js_version}.min.js"></script>'

    # Convert plot data to JSON with numpy support
    plot_data_json = json.dumps(sections_data["plot_data"], cls=NumpySafeJSONEncoder)

    # Generate HTML for all sections and subsections
    sections_html = ""
    for section in sections_data["sections"]:
        section_id = section["id"]
        section_title = section["title"]
        subsections_html = ""
        
        for i, subsection in enumerate(section["subsections"]):
            subsection_id = f"{section_id}-{i}"
            tabs_html = subsection["tabs_html"]
            buttons_html = subsection["buttons_html"]
            
            subsections_html += f"""
            <div class="subsection">
                <h3>{subsection['title']}</h3>
                <div class="tabs">
                    {buttons_html}
                </div>
                <div class="tab-content">
                    {tabs_html}
                </div>
            </div>
            """
        
        sections_html += f"""
        <div class='section' id='{section_id}'>
            <h2>{section_title}</h2>
            {subsections_html}
        </div>
        """

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset='utf-8'>
  <title>16S Analysis Debug Report</title>
  {plotly_js_tag}
  <style>
    body {{
        font-family: Arial, sans-serif;
        margin: 40px;
        line-height: 1.6;
    }}
    .section {{
        margin-bottom: 30px;
        padding-bottom: 20px;
        border-bottom: 1px solid #eee;
    }}
    .subsection {{
        margin-left: 20px;
        margin-bottom: 20px;
    }}
    .tabs {{
        display: flex;
        margin-bottom: -1px;
        flex-wrap: wrap;
    }}
    .tab-button {{
        padding: 8px 12px;
        background: #eee;
        border: 1px solid #ccc;
        cursor: pointer;
        border-radius: 4px 4px 0 0;
        margin-right: 5px;
        margin-bottom: 5px;
        font-size: 0.9em;
    }}
    .tab-button.active {{
        background: #fff;
        border-bottom: 1px solid #fff;
    }}
    .tab-content {{
        border: 1px solid #ccc;
        padding: 15px;
        border-radius: 0 4px 4px 4px;
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
    .section-controls {{
        margin: 10px 0;
    }}
    .section-button {{
        background: #f0f0f0;
        border: 1px solid #ddd;
        padding: 5px 10px;
        cursor: pointer;
        border-radius: 4px;
        margin-right: 5px;
    }}
  </style>
</head>
<body>
  <h1>16S Amplicon Analysis – Debug Report</h1>
  <p>Generated: {ts}</p>
  <p>Showing sections: {', '.join(include_sections)}</p>

  <div class="section-controls">
    <button class="section-button" onclick="toggleAllSections(true)">Expand All</button>
    <button class="section-button" onclick="toggleAllSections(false)">Collapse All</button>
  </div>
  
  {sections_html}

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
        
        // Handle different plot types
        if (data.type === "plotly") {{
            // Hide legend on client side
            if (data.layout) {{
                data.layout.showlegend = false;
            }}
            // Render the plot
            Plotly.newPlot(plotId, data.data, data.layout)
                .catch(error => {{
                    plotDiv.innerHTML = `<div class="error">Plot error: ${{error}}</div>`;
                    console.error('Plotly error:', error);
                }});
        }} else if (data.type === "image") {{
            const img = document.createElement('img');
            img.src = 'data:image/png;base64,' + data.data;
            img.style.maxWidth = '100%';
            plotDiv.appendChild(img);
        }} else if (data.type === "error") {{
            plotDiv.innerHTML = `<div class="error">${{data.error}}</div>`;
        }} else {{
            plotDiv.innerHTML = '<div class="error">Unknown plot type</div>';
        }}
    }}
    
    function showTab(tabId, plotId) {{
        // Hide all tabs in this subsection
        const tabPanes = document.querySelectorAll(`#${{tabId}}`).closest('.subsection').querySelectorAll('.tab-pane');
        tabPanes.forEach(tab => tab.style.display = 'none');
        
        // Remove active class from buttons in this subsection
        const tabButtons = document.querySelectorAll(`#${{tabId}}`).closest('.subsection').querySelectorAll('.tab-button');
        tabButtons.forEach(btn => btn.classList.remove('active'));
        
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
    
    // Toggle section visibility
    function toggleSection(sectionId) {{
        const section = document.getElementById(sectionId);
        if (section) {{
            section.style.display = section.style.display === 'none' ? 'block' : 'none';
        }}
    }}
    
    // Toggle all sections
    function toggleAllSections(show) {{
        document.querySelectorAll('.section').forEach(section => {{
            section.style.display = show ? 'block' : 'none';
        }});
    }}
    
    // Initialize first tab in each subsection
    document.addEventListener('DOMContentLoaded', () => {{
        document.querySelectorAll('.subsection').forEach(subsection => {{
            const firstTab = subsection.querySelector('.tab-pane');
            if (firstTab) {{
                const plotId = firstTab.dataset.plotId;
                showTab(firstTab.id, plotId);
            }}
        }});
    }});
  </script>
</body>
</html>"""
    output_path.write_text(html, encoding="utf-8")

# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _prepare_sections(
    figures_dict: Dict,
    include_sections: List[str]
) -> Dict:
    """Prepare HTML sections and subsections with filtering."""
    sections = []
    plot_data = {}
    
    # Define ordination subsections
    ordination_methods = ["pca", "pcoa", "tsne", "umap"]
    
    for section_title in include_sections:
        if section_title not in figures_dict:
            continue
            
        section_data = {
            "id": section_title.replace(" ", "_"),
            "title": section_title.title(),
            "subsections": []
        }
        
        # Special handling for ordination section
        if section_title == "ordination":
            for method in ordination_methods:
                subsection_title = method.upper()
                subsection_figures = {}
                
                # Find all figures for this method
                for table_type, levels in figures_dict[section_title].items():
                    for level, methods in levels.items():
                        if method in methods:
                            for color_col, fig in methods[method].items():
                                key = f"{table_type} - {level} - {color_col}"
                                subsection_figures[key] = fig
                
                if not subsection_figures:
                    continue
                    
                # Prepare subsection
                tabs, buttons, subsection_plot_data = _prepare_figures_from_dict(subsection_figures)
                plot_data.update(subsection_plot_data)
                section_data["subsections"].append({
                    "title": subsection_title,
                    "tabs_html": tabs,
                    "buttons_html": buttons
                })
        
        # Handle other sections (maps, alpha_diversity, etc.)
        else:
            subsection_figures = {}
            _flatten_figures(figures_dict[section_title], [], subsection_figures)
            
            if not subsection_figures:
                continue
                
            # Prepare subsection
            tabs, buttons, subsection_plot_data = _prepare_figures_from_dict(subsection_figures)
            plot_data.update(subsection_plot_data)
            section_data["subsections"].append({
                "title": "All",
                "tabs_html": tabs,
                "buttons_html": buttons
            })
        
        if section_data["subsections"]:
            sections.append(section_data)
    
    return {
        "sections": sections,
        "plot_data": plot_data
    }

def _flatten_figures(
    figures_dict: Dict,
    parent_keys: List[str],
    output_dict: Dict
) -> None:
    """Recursively flatten nested figures dictionary."""
    for key, value in figures_dict.items():
        current_keys = parent_keys + [key]
        if isinstance(value, dict):
            _flatten_figures(value, current_keys, output_dict)
        else:
            title = " - ".join(current_keys)
            output_dict[title] = value

def _prepare_figures_from_dict(
    figures_dict: Dict[str, Any]
) -> Tuple[str, str, Dict]:
    """Prepare HTML tabs and plot data from a flat figures dictionary."""
    tabs = []
    buttons = []
    plot_data = {}
    
    for i, (title, fig) in enumerate(figures_dict.items()):
        tab_id = f"tab-{i}"
        plot_id = f"plot-{i}"
        
        # Add tab button
        active = "active" if i == 0 else ""
        buttons.append(
            f'<button class="tab-button {active}" data-tab="{tab_id}" '
            f'onclick="showTab(\'{tab_id}\', \'{plot_id}\')">{title}</button>'
        )
        
        # Add tab content
        tabs.append(
            f'<div id="{tab_id}" class="tab-pane" style="display:{"block" if i == 0 else "none"}" '
            f'data-plot-id="{plot_id}">'
            f'<div id="container-{plot_id}" class="plot-container"></div>'
            f'</div>'
        )
        
        # Process different figure types
        if hasattr(fig, "to_plotly_json"):  # Plotly figure
            try:
                plot_json = fig.to_plotly_json()
                layout = plot_json.get("layout", {})
                layout["showlegend"] = False
                plot_data[plot_id] = {
                    "type": "plotly",
                    "data": plot_json["data"],
                    "layout": layout
                }
            except Exception as e:
                logger.error(f"Error processing Plotly figure: {e}")
                plot_data[plot_id] = {"type": "error", "error": str(e)}
                
        elif isinstance(fig, Figure):  # Matplotlib figure
            try:
                # Convert to base64-encoded image
                buf = BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight")
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode("utf-8")
                plot_data[plot_id] = {
                    "type": "image",
                    "data": img_str
                }
            except Exception as e:
                logger.error(f"Error processing Matplotlib figure: {e}")
                plot_data[plot_id] = {"type": "error", "error": str(e)}
                
        else:
            plot_data[plot_id] = {"type": "error", "error": f"Unsupported figure type: {type(fig)}"}
    
    return "\n".join(tabs), "\n".join(buttons), plot_data
