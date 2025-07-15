import os
import base64
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from jinja2 import Environment, FileSystemLoader

def generate_html_report(amplicon_data, output_path):
    """
    Generate an interactive HTML report for AmpliconData analysis
    
    Args:
        amplicon_data: AmpliconData object containing analysis results
        output_path: Path to save the HTML report
    """
    # Create report data structure
    report_data = {
        "sections": [],
        "has_maps": amplicon_data.maps is not None,
        "has_alpha": bool(amplicon_data.alpha_diversity),
        "has_stats": bool(amplicon_data.stats),
        "has_ordination": bool(amplicon_data.ordination),
        "has_top_features": bool(amplicon_data.top_contaminated_features or 
                                amplicon_data.top_pristine_features),
        "has_ml": bool(amplicon_data.models),
    }
    
    # 1. Sample Maps Section
    if report_data["has_maps"]:
        map_options = []
        for color_col, fig in amplicon_data.maps.items():
            map_options.append({
                "name": color_col,
                "figure": fig.to_html(full_html=False, include_plotlyjs=False)
            })
        
        report_data["sections"].append({
            "id": "sample-maps",
            "title": "Sample Maps",
            "description": "Geographical distribution of samples colored by metadata variables",
            "dropdown": {
                "id": "map-selector",
                "options": map_options,
                "default": map_options[0]["name"] if map_options else ""
            }
        })
    
    # 2. Alpha Diversity Section
    if report_data["has_alpha"]:
        alpha_options = []
        
        for table_type, levels in amplicon_data.alpha_diversity.items():
            for level, data in levels.items():
                for metric, fig in data.get('figures', {}).items():
                    # Skip summary and correlation figures for now
                    if metric in ['summary', 'correlations']:
                        continue
                    
                    alpha_options.append({
                        "name": f"{table_type} - {level} - {metric}",
                        "figure": fig.to_html(full_html=False, include_plotlyjs=False)
                    })
        
        report_data["sections"].append({
            "id": "alpha-diversity",
            "title": "Alpha Diversity",
            "description": "Measures of within-sample diversity across different groups",
            "dropdown": {
                "id": "alpha-selector",
                "options": alpha_options,
                "default": alpha_options[0]["name"] if alpha_options else ""
            }
        })
    
    # 3. Statistical Tests Summary
    if report_data["has_stats"]:
        # Create summary table of significant features
        stats_summary = []
        
        for table_type, levels in amplicon_data.stats.items():
            for level, tests in levels.items():
                for test_name, df in tests.items():
                    if isinstance(df, pd.DataFrame) and "p_value" in df.columns:
                        sig_count = sum(df["p_value"] < 0.05)
                        stats_summary.append({
                            "table_type": table_type,
                            "level": level,
                            "test": test_name,
                            "significant_features": sig_count,
                            "total_features": len(df)
                        })
        
        # Convert to HTML table
        stats_df = pd.DataFrame(stats_summary)
        stats_table = stats_df.to_html(
            classes="table table-striped table-hover", 
            index=False
        ) if not stats_df.empty else "<p>No significant features found</p>"
        
        report_data["sections"].append({
            "id": "statistical-tests",
            "title": "Statistical Tests",
            "description": "Summary of significant differences between groups",
            "content": stats_table
        })
    
    # 4. Top Features Section
    if report_data["has_top_features"]:
        # Prepare top features tables
        top_features_options = []
        
        # Contaminated features
        if amplicon_data.top_contaminated_features:
            contam_df = pd.DataFrame(amplicon_data.top_contaminated_features)
            contam_table = contam_df[['feature', 'level', 'table_type', 'test', 
                                      'effect', 'p_value', 'effect_dir']].to_html(
                classes="table table-striped table-hover", index=False)
            
            # Create violin plots section
            violin_options = []
            for feature in amplicon_data.top_contaminated_features[:10]:  # Limit to top 10
                if 'violin_figure' in feature and feature['violin_figure']:
                    violin_options.append({
                        "name": feature['feature'],
                        "figure": feature['violin_figure'].to_html(
                            full_html=False, include_plotlyjs=False)
                    })
            
            top_features_options.append({
                "type": "contaminated",
                "name": "Contaminated Features",
                "table": contam_table,
                "violin_plots": violin_options
            })
        
        # Pristine features
        if amplicon_data.top_pristine_features:
            pristine_df = pd.DataFrame(amplicon_data.top_pristine_features)
            pristine_table = pristine_df[['feature', 'level', 'table_type', 'test', 
                                         'effect', 'p_value', 'effect_dir']].to_html(
                classes="table table-striped table-hover", index=False)
            
            # Create violin plots section
            violin_options = []
            for feature in amplicon_data.top_pristine_features[:10]:  # Limit to top 10
                if 'violin_figure' in feature and feature['violin_figure']:
                    violin_options.append({
                        "name": feature['feature'],
                        "figure": feature['violin_figure'].to_html(
                            full_html=False, include_plotlyjs=False)
                    })
            
            top_features_options.append({
                "type": "pristine",
                "name": "Pristine Features",
                "table": pristine_table,
                "violin_plots": violin_options
            })
        
        report_data["sections"].append({
            "id": "top-features",
            "title": "Top Features",
            "description": "Most differentially abundant features between groups",
            "tabs": top_features_options
        })
    
    # 5. Ordination Section
    if report_data["has_ordination"]:
        ordination_options = []
        
        for table_type, levels in amplicon_data.ordination.items():
            for level, methods in levels.items():
                for method, data in methods.items():
                    for color_col, fig in data.get('figures', {}).items():
                        ordination_options.append({
                            "name": f"{table_type} - {level} - {method} - {color_col}",
                            "figure": fig.to_html(full_html=False, include_plotlyjs=False)
                        })
        
        report_data["sections"].append({
            "id": "ordination",
            "title": "Ordination Plots",
            "description": "Multidimensional scaling of samples based on beta diversity",
            "dropdown": {
                "id": "ordination-selector",
                "options": ordination_options,
                "default": ordination_options[0]["name"] if ordination_options else ""
            }
        })
    
    # 6. Machine Learning Section
    if report_data["has_ml"]:
        ml_options = []
        
        for table_type, levels in amplicon_data.models.items():
            for level, methods in levels.items():
                for method, model_data in methods.items():
                    if model_data and 'figures' in model_data:
                        # Add evaluation plots
                        for plot_type, fig in model_data['figures'].items():
                            if fig and plot_type in ['eval_plots', 'shap_summary_bar', 
                                                   'shap_summary_beeswarm']:
                                ml_options.append({
                                    "name": f"{table_type} - {level} - {method} - {plot_type}",
                                    "figure": fig.to_html(full_html=False, include_plotlyjs=False)
                                })
        
        report_data["sections"].append({
            "id": "machine-learning",
            "title": "Machine Learning",
            "description": "Feature importance from machine learning models",
            "dropdown": {
                "id": "ml-selector",
                "options": ml_options,
                "default": ml_options[0]["name"] if ml_options else ""
            }
        })
    
    # Load Jinja2 template
    env = Environment(loader=FileSystemLoader(os.path.dirname(os.path.abspath(__file__))))
    template = env.get_template("report_template.html")
    
    # Render and save report
    html_output = template.render(report_data=report_data)
    
    with open(output_path, 'w') as f:
        f.write(html_output)
    
    return output_path


# Example usage:
# amplicon_data = AmpliconData(cfg, project_dir, mode="genus")
# generate_html_report(amplicon_data, "analysis_report.html")
