# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import base64
import logging
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Union, Tuple, Optional

# Third‑Party Imports
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import plotly.graph_objects as go  

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger("workflow_16s")

# ==================================== FUNCTIONS ===================================== #

def generate_html_report(
    amplicon_data: 'AmpliconData',
    output_path: Union[str, Path],
    max_features: int = 20
) -> None:
    """
    Generates an HTML report of key analysis results.
    
    Args:
        amplicon_data: AmpliconData object containing analysis results
        output_path: Path to save the HTML report
        max_features: Maximum number of top features to display per category
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. Prepare top features tables
    contam_df = _prepare_features_table(
        amplicon_data.top_contaminated_features, 
        max_features,
        "Contaminated"
    )
    pristine_df = _prepare_features_table(
        amplicon_data.top_pristine_features, 
        max_features,
        "Pristine"
    )
    
    # 2. Prepare statistical summary
    stats_summary = _prepare_stats_summary(amplicon_data.stats)
    
    # 3. Prepare ML summary - now more detailed
    ml_metrics, ml_features, shap_plot = _prepare_ml_summary(amplicon_data.models)
    
    # 4. Prepare figures with new categorization
    figures_html = _prepare_figures(amplicon_data.figures)
    
    # 5. Generate HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>16S Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1, h2, h3, h4 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .section {{ margin-bottom: 40px; }}
            .figure-grid {{ 
                display: grid; 
                grid-template-columns: repeat(auto-fill, minmax(500px, 1fr));
                gap: 20px;
            }}
            .figure-container {{ 
                border: 1px solid #ddd; 
                padding: 10px; 
                height: 500px;
                overflow: hidden;
            }}
            .plot-wrapper {{
                width: 100%;
                height: 400px;
                display: flex;
                justify-content: center;
                align-items: center;
                overflow: hidden;
            }}
            .plot-container {{
                width: 100%;
                height: 100%;
                display: none;
            }}
            .plot-container.active {{
                display: flex;
                justify-content: center;
                align-items: center;
            }}
            .plot-container img {{
                max-width: 100%;
                max-height: 100%;
                object-fit: contain;
            }}
            .color-selector {{
                margin: 10px 0;
                padding: 5px;
                width: 100%;
            }}
            .beta-diversity-section h3 {{
                margin-top: 30px;
                border-bottom: 1px solid #ddd;
                padding-bottom: 5px;
            }}
            .ml-feature-table tr:nth-child(even) {{ background-color: #f9f9f9; }}
            /* MODIFIED: Plotly container styling */
            .plotly-container {{
                width: 100%;
                height: 100%;
            }}
            .plotly-graph-div {{
                width: 100% !important;
                height: 100% !important;
            }}
        </style>
        <!-- ADDED: Plotly.js CDN -->
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <script>
            // MODIFIED: Render only active Plotly figures
            document.addEventListener('DOMContentLoaded', function() {{
                renderAllPlotlyFigures();
            }});
            
            function renderAllPlotlyFigures() {{
                // Select only active plot containers
                var activeContainers = document.querySelectorAll('.plot-container.active .plotly-container');
                activeContainers.forEach(function(container) {{
                    if (!container.hasChildNodes()) {{
                        try {{
                            var figure = JSON.parse(container.dataset.figure);
                            Plotly.newPlot(container, figure.data, figure.layout, {{responsive: true}});
                        }} catch (e) {{
                            console.error('Error rendering Plotly figure:', e);
                            container.innerHTML = '<p>Error rendering Plotly figure</p>';
                        }}
                    }}
                }});
            }}
            
            function showPlot(selectElement, containerId) {{
                var container = document.getElementById(containerId);
                var plots = container.getElementsByClassName('plot-container');
                for (var i = 0; i < plots.length; i++) {{
                    plots[i].classList.remove('active');
                }}
                var selectedPlot = document.getElementById(selectElement.value);
                if (selectedPlot) {{
                    selectedPlot.classList.add('active');
                    // Now render any Plotly figures in the newly active container
                    renderAllPlotlyFigures();
                }}
            }}
        </script>
    </head>
    <body>
        <h1>16S Amplicon Analysis Report</h1>
        <p>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="section">
            <h2>Top Features</h2>
            <h3>Contaminated-Associated Features</h3>
            {contam_df.to_html(index=False, classes='feature-table')}
            
            <h3>Pristine-Associated Features</h3>
            {pristine_df.to_html(index=False, classes='feature-table')}
        </div>
        
        <div class="section">
            <h2>Statistical Summary</h2>
            {stats_summary.to_html(index=False)}
        </div>
        
        <div class="section">
            <h2>Machine Learning Results</h2>
            {_format_ml_section(ml_metrics, ml_features, shap_plot)}
        </div>
        
        <div class="section">
            <h2>Visualizations</h2>
            <div class="figure-grid">
                {figures_html}
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html_content)

# ... [The rest of your functions remain unchanged until _figure_to_html] ...

def _figure_to_html(fig: Any, caption: str, include_caption: bool = True) -> str:
    """Convert figures to HTML embedding with consistent size"""
    if fig is None:
        return f"<div class='figure-container'><p>Missing figure: {caption}</p></div>"
    
    try:
        # MODIFIED: Improved Plotly handling
        if hasattr(fig, 'to_json') and callable(fig.to_json):
            plot_json = fig.to_json()
            caption_html = f"<p>{caption}</p>" if include_caption else ""
            
            # MODIFIED: Use direct HTML embedding instead of JSON parsing
            # Generate Plotly HTML string
            plotly_html = fig.to_html(
                full_html=False, 
                include_plotlyjs=False, 
                config={'responsive': True}
            )
            
            return f"""
            <div class="figure-container">
                <div class="plot-wrapper">
                    <!-- MODIFIED: Directly embed Plotly HTML -->
                    <div class="plotly-container">{plotly_html}</div>
                </div>
                {caption_html}
            </div>
            """
        
        # Handle Matplotlib/Seaborn figures (unchanged)
        buf = BytesIO()
        dpi = 100
        if hasattr(fig, 'savefig'):
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
            plt.close(fig)  # Close figure to free memory
        elif hasattr(fig, 'figure'):  # Seaborn grid
            fig.figure.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
            plt.close(fig.figure)
        else:
            return f"<div class='figure-container'><p>Unsupported figure type: {type(fig)}</p><p>{caption}</p></div>"
        
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        caption_html = f"<p>{caption}</p>" if include_caption else ""
        
        return f"""
        <div class="figure-container">
            <div class="plot-wrapper">
                <img src="data:image/png;base64,{img_base64}" alt="{caption}">
            </div>
            {caption_html}
        </div>
        """
    
    except Exception as e:
        logger.error(f"Error rendering figure: {str(e)}")
        return f"""
        <div class="figure-container">
            <p>Error rendering figure: {str(e)}</p>
            <p>{caption}</p>
        </div>
        """
