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

def _prepare_features_table(
    features: List[Dict], 
    max_features: int,
    category: str
) -> pd.DataFrame:
    """Prepare top features table for HTML display"""
    if not features:
        return pd.DataFrame({"Message": [f"No significant {category} features found"]})
    
    df = pd.DataFrame(features[:max_features])
    # Simplify column names and select important columns
    df = df.rename(columns={
        "feature": "Feature",
        "level": "Taxonomic Level",
        "test": "Test",
        "effect": "Effect Size",
        "p_value": "P-value",
        "effect_dir": "Direction"
    })
    
    # Add FAPROTAX annotations if available
    if "faprotax_functions" in df.columns:
        df["Functions"] = df["faprotax_functions"].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else ""
        )
    
    # Format numeric columns
    df["Effect Size"] = df["Effect Size"].apply(lambda x: f"{x:.4f}")
    df["P-value"] = df["P-value"].apply(lambda x: f"{x:.2e}")
    
    return df[["Feature", "Taxonomic Level", "Test", "Effect Size", "P-value", "Direction"]]

def _prepare_stats_summary(stats: Dict) -> pd.DataFrame:
    """Prepare statistical summary table"""
    summary = []
    for table_type, tests in stats.items():
        for test_name, levels in tests.items():
            for level, df in levels.items():
                n_sig = sum(df["p_value"] < 0.05) if "p_value" in df.columns else 0
                summary.append({
                    "Table Type": table_type,
                    "Test": test_name,
                    "Level": level,
                    "Significant Features": n_sig,
                    "Total Features": len(df)
                })
    
    return pd.DataFrame(summary)

def _prepare_ml_summary(models: Dict) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[str]]:
    """Prepare detailed ML results for HTML display"""
    if not models:
        return None, None, None

    metrics_summary = []
    features_summary = []
    shap_plot_base64 = None
    best_mcc = -1  # Track best MCC for SHAP plot selection
    
    for table_type, levels in models.items():
        for level, methods in levels.items():
            for method, result in methods.items():
                if not result:
                    continue
                
                # Extract metrics
                test_scores = result.get("test_scores", {})
                metrics = {
                    "Table Type": table_type,
                    "Level": level,
                    "Method": method,
                    "Top Features": len(result.get("top_features", [])),
                    "Accuracy": f"{test_scores.get('accuracy', 0):.4f}",
                    "F1 Score": f"{test_scores.get('f1', 0):.4f}",
                    "MCC": f"{test_scores.get('mcc', 0):.4f}",
                    "ROC AUC": f"{test_scores.get('roc_auc', 0):.4f}",
                    "PR AUC": f"{test_scores.get('pr_auc', 0):.4f}"
                }
                metrics_summary.append(metrics)
                
                # Extract top features
                feat_imp = result.get("feature_importances", {})
                top_features = result.get("top_features", [])[:10]
                for i, feat in enumerate(top_features, 1):
                    importance = feat_imp.get(feat, 0)
                    features_summary.append({
                        "Table Type": table_type,
                        "Level": level,
                        "Method": method,
                        "Rank": i,
                        "Feature": feat,
                        "Importance": f"{importance:.4f}"
                    })
                
                # Track best model for SHAP plot
                current_mcc = test_scores.get("mcc", -1)
                shap_path = result.get("shap_summary_bar_path")
                if current_mcc > best_mcc and shap_path:
                    try:
                        with open(shap_path, "rb") as img_file:
                            shap_plot_base64 = base64.b64encode(img_file.read()).decode("utf-8")
                        best_mcc = current_mcc
                    except Exception as e:
                        logger.warning(f"Couldn't load SHAP plot: {str(e)}")
    
    metrics_df = pd.DataFrame(metrics_summary) if metrics_summary else None
    features_df = pd.DataFrame(features_summary) if features_summary else None
    
    return metrics_df, features_df, shap_plot_base64

def _prepare_figures(figures: Dict) -> str:
    """Organize figures into categories with dropdowns for color selection"""
    if not figures:
        return "<p>No visualizations available</p>"
    
    html_parts = []
    
    # Sample Maps
    if "map" in figures:
        html_parts.append("<h3>Sample Maps</h3>")
        for col, fig in figures["map"].items():
            if fig:
                html_parts.append(
                    _figure_to_html(fig, f"Sample Map: {col}")
                )
    
    # Alpha Diversity
    alpha_html = []
    for table_type, levels in figures.items():
        if table_type == "map":
            continue
        for level, plots in levels.items():
            if not isinstance(plots, dict):
                continue
            for plot_type, fig in plots.items():
                if "alpha" in plot_type and fig:
                    alpha_html.append(
                        _figure_to_html(fig, f"Alpha Diversity - {table_type} - {level} - {plot_type}")
                    )
    if alpha_html:
        html_parts.append("<h3>Alpha Diversity</h3>" + "\n".join(alpha_html))
    
    # Beta Diversity
    beta_groups = {}
    for table_type, levels in figures.items():
        if table_type == "map":
            continue
        for level, methods in levels.items():
            if not isinstance(methods, dict):
                continue
            for method, color_figs in methods.items():
                if method in ["pca", "pcoa", "tsne", "umap"] and isinstance(color_figs, dict):
                    key = f"{table_type}_{level}_{method}"
                    beta_groups[key] = {
                        "title": f"{method.upper()} - {table_type} - {level}",
                        "figures": color_figs
                    }
    
    if beta_groups:
        html_parts.append("<div class='beta-diversity-section'><h3>Beta Diversity</h3>")
        for key, group in beta_groups.items():
            container_id = f"{key}_container"
            html_parts.append(f"<h4>{group['title']}</h4>")
            
            # Create dropdown
            select_html = f"<select class='color-selector' onchange='showPlot(this, \"{container_id}\")'>"
            options = []
            plot_divs = []
            
            valid_figs = [(col, fig) for col, fig in group["figures"].items() if fig is not None]
            if not valid_figs:
                continue
                
            for i, (col, fig) in enumerate(valid_figs):
                plot_id = f"{key}_{col.replace(' ', '_')}"
                selected = "selected" if i == 0 else ""
                options.append(f"<option value='{plot_id}' {selected}>{col}</option>")
                plot_divs.append(
                    f"<div id='{plot_id}' class='plot-container' style='display: {'flex' if i==0 else 'none'}'>"
                    f"{_figure_to_html(fig, f'Colored by: {col}', include_caption=False)}"
                    "</div>"
                )
            
            select_html += "\n".join(options) + "</select>"
            plot_container = f"<div id='{container_id}' class='plot-wrapper'>" + "\n".join(plot_divs) + "</div>"
            
            html_parts.append(select_html)
            html_parts.append(plot_container)
        
        html_parts.append("</div>")  # Close beta-diversity-section
    
    # Other plots
    other_html = []
    for plot_type, levels in figures.items():
        if plot_type in ["map", "pca", "pcoa", "tsne", "umap"]:
            continue
        if not isinstance(levels, dict):
            continue
        for level, methods in levels.items():
            if not isinstance(methods, dict):
                continue
            for method, fig in methods.items():
                if fig and not isinstance(fig, dict):  # Single figure
                    other_html.append(
                        _figure_to_html(fig, f"{plot_type} - {level} - {method}")
                    )
    if other_html:
        html_parts.append("<h3>Other Visualizations</h3>" + "\n".join(other_html))
    
    return "\n".join(html_parts) if html_parts else "<p>No visualizations available</p>"


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
