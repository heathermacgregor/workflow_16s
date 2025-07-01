# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import base64
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Union

# Third‑Party Imports
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

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
                display: block;
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
        </style>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <script>
            function showPlot(selectElement, containerId) {{
                var container = document.getElementById(containerId);
                var plots = container.getElementsByClassName('plot-container');
                for (var i = 0; i < plots.length; i++) {{
                    plots[i].classList.remove('active');
                }}
                var selectedPlot = document.getElementById(selectElement.value);
                if (selectedPlot) {{
                    selectedPlot.classList.add('active');
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

def _format_ml_section(ml_metrics, ml_features, shap_plot):
    """Format the machine learning results section"""
    if ml_metrics is None:
        return "<p>No ML results available</p>"
    
    ml_html = f"""
    <div class="ml-section">
        <h3>Model Performance</h3>
        {ml_metrics.to_html(index=False)}
        
        <h3>Top Features</h3>
        {ml_features.to_html(index=False, classes='ml-feature-table')}
    """
    
    if shap_plot:
        ml_html += f"""
        <h3>Feature Importance (Best Model)</h3>
        <div class="figure-container">
            <img src="data:image/png;base64,{shap_plot}" alt="SHAP Summary">
        </div>
        """
    
    ml_html += "</div>"
    return ml_html

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
                n_sig = sum(df["p_value"] < 0.05)
                summary.append({
                    "Table Type": table_type,
                    "Test": test_name,
                    "Level": level,
                    "Significant Features": n_sig,
                    "Total Features": len(df)
                })
    
    return pd.DataFrame(summary)

def _prepare_ml_summary(models: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
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
                if result is None:
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
                for i, feat in enumerate(result.get("top_features", [])[:10], 1):
                    features_summary.append({
                        "Table Type": table_type,
                        "Level": level,
                        "Method": method,
                        "Rank": i,
                        "Feature": feat,
                        "Importance": f"{feat_imp.get(feat, 0):.4f}"
                    })
                
                # Track best model for SHAP plot
                current_mcc = test_scores.get("mcc", -1)
                if current_mcc > best_mcc and "shap_summary_bar_path" in result:
                    try:
                        with open(result["shap_summary_bar_path"], "rb") as img_file:
                            shap_plot_base64 = base64.b64encode(img_file.read()).decode("utf-8")
                        best_mcc = current_mcc
                    except Exception as e:
                        logger.warning(f"Couldn't load SHAP plot: {str(e)}")
    
    metrics_df = pd.DataFrame(metrics_summary) if metrics_summary else None
    features_df = pd.DataFrame(features_summary) if features_summary else None
    
    return metrics_df, features_df, shap_plot_base64

def _prepare_figures(figures: Dict) -> str:
    """Organize figures into categories with dropdowns for color selection"""
    html_parts = []
    
    # Sample Maps
    if "map" in figures:
        html_parts.append("<h3>Sample Maps</h3>")
        for col, fig in figures["map"].items():
            html_parts.append(
                _figure_to_html(fig, f"Sample Map: {col}")
            )
    
    # Alpha Diversity
    alpha_html = []
    for table_type, levels in figures.items():
        if table_type == "map":
            continue
        for level, plots in levels.items():
            for plot_type, fig in plots.items():
                if "alpha" in plot_type:
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
            
            for i, (col, fig) in enumerate(group["figures"].items()):
                plot_id = f"{key}_{col.replace(' ', '_')}"
                options.append(f"<option value='{plot_id}' {'selected' if i==0 else ''}>{col}</option>")
                plot_divs.append(
                    f"<div id='{plot_id}' class='plot-container' style='display: {'block' if i==0 else 'none'}'>"
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
        for level, methods in levels.items():
            for method, fig in methods.items():
                if not isinstance(fig, dict):  # Single figure
                    other_html.append(
                        _figure_to_html(fig, f"{plot_type} - {level} - {method}")
                    )
    if other_html:
        html_parts.append("<h3>Other Visualizations</h3>" + "\n".join(other_html))
    
    return "\n".join(html_parts)

def _figure_to_html(fig: Any, caption: str, include_caption: bool = True) -> str:
    """Convert figures to HTML embedding with consistent size"""
    if fig is None:
        return ""
    
    try:
        # Handle Plotly figures
        if hasattr(fig, 'to_html') and callable(fig.to_html):
            plot_html = fig.to_html(full_html=False, include_plotlyjs=False)
            caption_html = f"<p>{caption}</p>" if include_caption else ""
            return f"""
            <div class="figure-container">
                <div class="plot-wrapper">
                    {plot_html}
                </div>
                {caption_html}
            </div>
            """
        
        # Handle Matplotlib/Seaborn figures
        buf = BytesIO()
        if hasattr(fig, 'savefig'):
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        elif hasattr(fig, 'figure'):  # Seaborn grid
            fig.figure.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        caption_html = f"<p>{caption}</p>" if include_caption else ""
        
        return f"""
        <div class="figure-container">
            <div class="plot-wrapper">
                <img src="data:image/png;base64,{img_base64}" alt="{caption}" style="max-height: 100%; max-width: 100%;">
            </div>
            {caption_html}
        </div>
        """
    
    except Exception as e:
        return f"""
        <div class="figure-container">
            <p>Error rendering figure: {str(e)}</p>
            <p>{caption}</p>
        </div>
        """
