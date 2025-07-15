import os
import pandas as pd
import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader
import json
import numpy as np
import pandas as pd
from plotly.utils import PlotlyJSONEncoder

class NumpySafeJSONEncoder(PlotlyJSONEncoder):
    """Custom JSON encoder that handles NumPy types and pandas Timestamps"""
    def default(self, obj) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if pd.api.types.is_integer(obj):
            return int(obj)
        if pd.api.types.is_float(obj):
            return float(obj)
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return super().default(obj)
        
def generate_html_report(amplicon_data, output_path):
    """
    Generate an interactive HTML report for AmpliconData analysis
    
    Args:
        amplicon_data: AmpliconData object containing analysis results
        output_path: Path to save the HTML report
    """
    # Create JSON encoder instance
    json_encoder = NumpySafeJSONEncoder()
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
            # Update figure size to be responsive
            fig.update_layout(
                autosize=True,
                width=None,
                height=600,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            map_options.append({
                "name": color_col,
                "figure": fig.to_dict() 
            })
        
        report_data["sections"].append({
            "id": "sample-maps",
            "title": "Sample Maps",
            "description": "Geographical distribution of samples colored by metadata variables",
            "dropdowns": [{
                "id": "map-selector",
                "label": "Color by:",
                "options": [opt["name"] for opt in map_options],
                "values": map_options
            }]
        })
    
    # 2. Alpha Diversity Section
    if report_data["has_alpha"]:
        # Collect all available options
        table_types = set()
        levels = set()
        metrics = set()
        alpha_options = []
        
        for table_type, level_data in amplicon_data.alpha_diversity.items():
            table_types.add(table_type)
            for level, data in level_data.items():
                levels.add(level)
                for metric, fig in data.get('figures', {}).items():
                    # Skip summary and correlation figures
                    if metric in ['summary', 'correlations']:
                        continue
                    
                    metrics.add(metric)
                    
                    # Update figure size to be responsive
                    fig.update_layout(
                        autosize=True,
                        width=None,
                        height=500,
                        margin=dict(l=50, r=50, t=60, b=50)
                    )
                    
                    alpha_options.append({
                        "table_type": table_type,
                        "level": level,
                        "metric": metric,
                        "figure": fig.to_dict()
                    })
    
        
        # Create dropdown options
        dropdowns = [
            {
                "id": "alpha-table-type",
                "label": "Table Type:",
                "options": sorted(table_types),
                "default": sorted(table_types)[0] if table_types else ""
            },
            {
                "id": "alpha-level",
                "label": "Taxonomic Level:",
                "options": sorted(levels),
                "default": sorted(levels)[0] if levels else ""
            },
            {
                "id": "alpha-metric",
                "label": "Metric:",
                "options": sorted(metrics),
                "default": sorted(metrics)[0] if metrics else ""
            }
        ]
        
        report_data["sections"].append({
            "id": "alpha-diversity",
            "title": "Alpha Diversity",
            "description": "Measures of within-sample diversity across different groups",
            "dropdowns": dropdowns,
            "figures": alpha_options
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
                    # Update figure size to be responsive
                    fig = feature['violin_figure']
                    fig.update_layout(
                        autosize=True,
                        width=None,
                        height=450,
                        margin=dict(l=50, r=50, t=60, b=50)
                    )
                    
                    violin_options.append({
                        "feature": feature['feature'],
                        "level": feature['level'],
                        "table_type": feature['table_type'],
                        "figure": fig.to_dict()
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
                    # Update figure size to be responsive
                    fig = feature['violin_figure']
                    fig.update_layout(
                        autosize=True,
                        width=None,
                        height=450,
                        margin=dict(l=50, r=50, t=60, b=50)
                    )
                    
                    violin_options.append({
                        "feature": feature['feature'],
                        "level": feature['level'],
                        "table_type": feature['table_type'],
                        "figure": fig.to_dict()
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
        # Collect all available options
        table_types = set()
        levels = set()
        methods = set()
        color_cols = set()
        ordination_options = []
        
        for table_type, level_data in amplicon_data.ordination.items():
            table_types.add(table_type)
            for level, method_data in level_data.items():
                levels.add(level)
                for method, data in method_data.items():
                    methods.add(method)
                    for color_col, fig in data.get('figures', {}).items():
                        color_cols.add(color_col)
                        
                        # Update figure size to be responsive
                        fig.update_layout(
                            autosize=True,
                            width=None,
                            height=600,
                            margin=dict(l=50, r=50, t=60, b=50)
                        )
                        
                        # Convert figure to JSON
                        ordination_options.append({
                            "table_type": table_type,
                            "level": level,
                            "method": method,
                            "color_col": color_col,
                            "figure": fig.to_dict()
                        })
        
        # Create dropdown options
        dropdowns = [
            {
                "id": "ord-table-type",
                "label": "Table Type:",
                "options": sorted(table_types),
                "default": sorted(table_types)[0] if table_types else ""
            },
            {
                "id": "ord-level",
                "label": "Taxonomic Level:",
                "options": sorted(levels),
                "default": sorted(levels)[0] if levels else ""
            },
            {
                "id": "ord-method",
                "label": "Method:",
                "options": sorted(methods),
                "default": sorted(methods)[0] if methods else ""
            },
            {
                "id": "ord-color",
                "label": "Color By:",
                "options": sorted(color_cols),
                "default": sorted(color_cols)[0] if color_cols else ""
            }
        ]
        
        report_data["sections"].append({
            "id": "ordination",
            "title": "Ordination Plots",
            "description": "Multidimensional scaling of samples based on beta diversity",
            "dropdowns": dropdowns,
            "figures": ordination_options
        })
    
    # 6. Machine Learning Section
    if report_data["has_ml"]:
        # Collect all available options
        table_types = set()
        levels = set()
        methods = set()
        plot_types = set()
        ml_options = []
        
        for table_type, level_data in amplicon_data.models.items():
            table_types.add(table_type)
            for level, method_data in level_data.items():
                levels.add(level)
                for method, model_data in method_data.items():
                    if model_data and 'figures' in model_data:
                        methods.add(method)
                        for plot_type, fig in model_data['figures'].items():
                            if fig and plot_type in ['eval_plots', 'shap_summary_bar', 
                                                   'shap_summary_beeswarm']:
                                plot_types.add(plot_type)
                                
                                # Update figure size to be responsive
                                fig.update_layout(
                                    autosize=True,
                                    width=None,
                                    height=500,
                                    margin=dict(l=50, r=50, t=60, b=50)
                                )
                                
                                # Convert figure to JSON
                                ml_options.append({
                                    "table_type": table_type,
                                    "level": level,
                                    "method": method,
                                    "plot_type": plot_type,
                                    "figure": fig.to_dict()  
                                })
        
        # Create dropdown options
        dropdowns = [
            {
                "id": "ml-table-type",
                "label": "Table Type:",
                "options": sorted(table_types),
                "default": sorted(table_types)[0] if table_types else ""
            },
            {
                "id": "ml-level",
                "label": "Taxonomic Level:",
                "options": sorted(levels),
                "default": sorted(levels)[0] if levels else ""
            },
            {
                "id": "ml-method",
                "label": "Method:",
                "options": sorted(methods),
                "default": sorted(methods)[0] if methods else ""
            },
            {
                "id": "ml-plot-type",
                "label": "Plot Type:",
                "options": sorted(plot_types),
                "default": sorted(plot_types)[0] if plot_types else ""
            }
        ]
        
        report_data["sections"].append({
            "id": "machine-learning",
            "title": "Machine Learning",
            "description": "Feature importance from machine learning models",
            "dropdowns": dropdowns,
            "figures": ml_options
        })
    
    # Load Jinja2 template
    env = Environment(loader=FileSystemLoader(os.path.dirname(os.path.abspath(__file__))))
    template = env.get_template("report_template.html")
    
    # Render and save report
    html_output = template.render(report_data=report_data)
    
    with open(output_path, 'w') as f:
        f.write(html_output)
    
    return output_path
