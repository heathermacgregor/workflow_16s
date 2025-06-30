# ===================================== IMPORTS ====================================== #

import base64
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Union
import pandas as pd
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
    
    # 3. Prepare ML summary
    ml_summary = _prepare_ml_summary(amplicon_data.models)
    
    # 4. Prepare figures
    figure_html = _prepare_figures(amplicon_data.figures)
    
    # 5. Generate HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>16S Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1, h2 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            img {{ max-width: 100%; height: auto; margin: 10px 0; }}
            .section {{ margin-bottom: 40px; }}
            .figure-grid {{ 
                display: grid; 
                grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
                gap: 20px;
            }}
            .figure-container {{ border: 1px solid #ddd; padding: 10px; }}
        </style>
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
            <h2>Machine Learning Feature Selection</h2>
            {ml_summary.to_html(index=False) if ml_summary is not None else "<p>No ML results available</p>"}
        </div>
        
        <div class="section">
            <h2>Visualizations</h2>
            <div class="figure-grid">
                {figure_html}
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
                n_sig = sum(df["p_value"] < 0.05)
                summary.append({
                    "Table Type": table_type,
                    "Test": test_name,
                    "Level": level,
                    "Significant Features": n_sig,
                    "Total Features": len(df)
                })
    
    return pd.DataFrame(summary)

def _prepare_ml_summary(models: Dict) -> Union[pd.DataFrame, None]:
    """Prepare machine learning feature selection summary"""
    if not models:
        return None
        
    summary = []
    for table_type, levels in models.items():
        for level, methods in levels.items():
            for method, result in methods.items():
                if result is None:
                    continue
                summary.append({
                    "Table Type": table_type,
                    "Level": level,
                    "Method": method,
                    "Top Features": len(result.get("top_features", [])),
                    "Accuracy": f"{result.get('accuracy', 0):.2f}"
                })
    
    return pd.DataFrame(summary) if summary else None

def _prepare_figures(figures: Dict) -> str:
    """Convert figures to HTML image tags"""
    html_parts = []
    
    # Sample maps
    if "map" in figures:
        for col, fig in figures["map"].items():
            html_parts.append(
                _figure_to_html(fig, f"Sample Map: {col}")
            )
    
    # Ordination plots
    for plot_type, levels in figures.items():
        if plot_type == "map":
            continue
            
        for level, methods in levels.items():
            for method, color_figs in methods.items():
                if not isinstance(color_figs, dict):
                    continue
                for col, fig in color_figs.items():
                    html_parts.append(
                        _figure_to_html(fig, f"{plot_type.upper()} - {level} - {method}<br>Colored by: {col}")
                    )
    
    return "\n".join(html_parts)

def _figure_to_html(fig: Figure, caption: str) -> str:
    """Convert matplotlib figure to HTML image tag with caption"""
    if fig is None:
        return ""
        
    # Save figure to in-memory bytes buffer
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    
    # Convert to base64
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    
    return f"""
    <div class="figure-container">
        <img src="data:image/png;base64,{img_base64}" alt="{caption}">
        <p>{caption}</p>
    </div>
    """
