# ===================================== IMPORTS ====================================== #
import base64
import itertools
import json
import logging
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from plotly.offline import get_plotlyjs_version
import plotly.io as pio

from workflow_16s.utils.io import import_js_as_str

# ========================== INITIALIZATION & CONFIGURATION ========================== #
logger = logging.getLogger('workflow_16s')
script_dir = Path(__file__).parent  
tables_js_path = script_dir / "tables.js"  
css_path = script_dir / "style.css"  
html_template_path = script_dir / "template.html"  

DEFAULT_GROUP_COLUMN = 'nuclear_contamination_status'
DEFAULT_GROUP_COLUMN_VALUES = [True, False]

# ===================================== CLASSES ====================================== #
class NumpySafeJSONEncoder(json.JSONEncoder):
    def default(self, obj) -> Any:  
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

# ========================== PLOTLY SELECTOR INTEGRATION ========================== #

def _generate_plotly_selector_html(figures_dict: Dict[str, Any], 
                                 container_id: str = "plotly-container",
                                 section_title: str = "Plots") -> str:
    """
    Generate HTML with interactive selection UI for nested dictionary of Plotly figures.
    Integrated version for the amplicon analysis report.
    """
    
    def flatten_dict(d: Dict, parent_key: str = '', sep: str = ' > ') -> Dict[str, Any]:
        """Flatten nested dictionary and create display labels"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict) and not hasattr(v, 'to_plotly_json'):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def create_selector_options(d: Dict, parent_key: str = '', level: int = 0) -> str:
        """Create hierarchical option elements for the selector"""
        options_html = ""
        indent = "  " * level
        
        for key, value in d.items():
            full_key = f"{parent_key} > {key}" if parent_key else key
            
            if isinstance(value, dict) and not hasattr(value, 'to_plotly_json'):
                # Create optgroup for nested dictionaries
                options_html += f'{indent}<optgroup label="{key}">\n'
                options_html += create_selector_options(value, full_key, level + 1)
                options_html += f'{indent}</optgroup>\n'
            else:
                # Create option for figure
                options_html += f'{indent}<option value="{full_key}">{key}</option>\n'
        
        return options_html
    
    # Flatten the dictionary to get all figures with their paths
    flat_figures = flatten_dict(figures_dict)
    
    if not flat_figures:
        return f'<div id="{container_id}">No valid figures found in {section_title.lower()}.</div>'
    
    # Get the first figure to display initially
    first_key = list(flat_figures.keys())[0]
    
    # Convert all figures to JSON using existing conversion function
    figures_json = {}
    for key, fig in flat_figures.items():
        serialized = _convert_figure_to_serializable(fig)
        # Use custom encoder for all figures
        figures_json[key] = json.dumps(serialized, cls=NumpySafeJSONEncoder)  # Add encoder here
    
    # Create the selector options HTML
    selector_options = create_selector_options(figures_dict)
    
    # Generate the HTML with integrated styling
    html_template = f"""
<div id="{container_id}" class="plotly-selector-container">
    <div class="selector-controls">
        <label for="{container_id}-selector" class="selector-label">Select {section_title}:</label>
        <select id="{container_id}-selector" class="figure-dropdown">
{selector_options}        </select>
    </div>
    <div id="{container_id}-plot" class="plotly-selector-plot"></div>
</div>

<script>
(function() {{
    // Store figures data for {container_id}
    const figuresData_{container_id.replace('-', '_')} = {json.dumps(figures_json)};
    
    // Get DOM elements
    const selector = document.getElementById('{container_id}-selector');
    const plotDiv = document.getElementById('{container_id}-plot');
    
    // Function to display a plot
    function displayPlot_{container_id.replace('-', '_')}(figureKey) {{
        if (figuresData_{container_id.replace('-', '_')}[figureKey]) {{
            const figureData = JSON.parse(figuresData_{container_id.replace('-', '_')}[figureKey]);
            
            if (figureData.type === 'plotly') {{
                // Handle Plotly figures
                Plotly.newPlot(plotDiv, figureData.data, figureData.layout, {{responsive: true}});
            }} else if (figureData.type === 'image') {{
                // Handle matplotlib/image figures
                plotDiv.innerHTML = `<img src="data:image/png;base64,${{figureData.data}}" style="max-width: 100%; height: auto;" alt="Plot">`;
            }} else if (figureData.type === 'error') {{
                // Handle errors
                plotDiv.innerHTML = `<div class="error-message">Error loading figure: ${{figureData.error}}</div>`;
            }}
        }}
    }}
    
    // Event listener for selector change
    selector.addEventListener('change', function() {{
        displayPlot_{container_id.replace('-', '_')}(this.value);
    }});
    
    // Display initial plot
    if (selector.value) {{
        displayPlot_{container_id.replace('-', '_')}(selector.value);
    }}
}})();
</script>
"""
    
    return html_template

# ================================== CORE HELPERS =================================== #
def _extract_figures(amplicon_data: "AmpliconData") -> Dict[str, Any]:
    figures = {}
    
    # Ordination figures
    ordination_figures = {}
    for table_type, levels in amplicon_data.ordination.items():
        for level, methods in levels.items():
            for method, data in methods.items():
                if data and 'figures' in data and data['figures']:
                    if table_type not in ordination_figures:
                        ordination_figures[table_type] = {}
                    if level not in ordination_figures[table_type]:
                        ordination_figures[table_type][level] = {}
                    ordination_figures[table_type][level][method] = data['figures']
    figures['ordination'] = ordination_figures

    # Alpha diversity figures
    alpha_figures = {}
    for group_column, table_types in amplicon_data.alpha_diversity.items():
        alpha_figures.setdefault(group_column, {})
        for table_type, levels in table_types.items():
            alpha_figures[group_column].setdefault(table_type, {})
            for level, data in levels.items():
                if 'figures' in data and data['figures']:
                    # Store figures directly under group_key -> level
                    alpha_figures[group_column][table_type][level] = data['figures']
                    
    figures['alpha_diversity'] = alpha_figures
    
    # Sample maps
    if amplicon_data.maps:
        figures['map'] = amplicon_data.maps

    # SHAP figures
    shap_figures = {}
    for table_type, levels in amplicon_data.models.items():
        for level, methods in levels.items():
            for method, model_result in methods.items():
                if model_result and 'figures' in model_result:
                    if table_type not in shap_figures:
                        shap_figures[table_type] = {}
                    if level not in shap_figures[table_type]:
                        shap_figures[table_type][level] = {}
                    shap_figures[table_type][level][method] = model_result['figures']
    figures['shap'] = shap_figures

    # Violin plots
    violin_figures = {}
    for col, vals in amplicon_data.top_features.items():
        for val, features in vals.items():
            group_key = f"{col}={val}"
            violin_figures.setdefault(col, {})
            for feature in features:
                if isinstance(feature, dict):
                    if feature.get('violin_figure'):
                        violin_figures[col][feature['feature']] = feature['violin_figure']
    figures['violin'] = violin_figures
    logger.info(figures)

    return figures

def _convert_figure_to_serializable(fig):
    """Convert figure object to serializable dict"""
    try:
        if fig is None:
            return {"type": "error", "error": "Figure object is None"}
            
        if hasattr(fig, "to_plotly_json"):
            pj = fig.to_plotly_json()
            pj.setdefault("layout", {})["showlegend"] = False
            return {
                "type": "plotly",
                "data": pj["data"],
                "layout": pj["layout"],
                "square": False
            }
        elif isinstance(fig, Figure):
            buf = BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
            buf.seek(0)
            return {
                "type": "image",
                "data": base64.b64encode(buf.read()).decode()
            }
        else:
            return {
                "type": "error",
                "error": f"Unsupported figure type {type(fig)}"
            }
    except Exception as exc:
        logger.exception("Serializing figure failed")
        return {"type": "error", "error": str(exc)}

def _flatten_figures_tree(tree, prefix="", delimiter=" - "):
    """Flatten a nested figures tree into a list of (path, figure) tuples"""
    flat = []
    if not isinstance(tree, dict):
        return [("", tree)]
    
    for key, value in tree.items():
        new_prefix = f"{prefix}{delimiter}{key}" if prefix else key
        if isinstance(value, dict):
            flat.extend(_flatten_figures_tree(value, new_prefix, delimiter))
        else:
            flat.append((new_prefix, value))
    return flat

def _prepare_sections(
    figures: Dict,
    include_sections: List[str],
) -> List[Dict]:
    sections = []

    for sec in include_sections:
        if sec not in figures or not figures[sec]:
            continue

        # Use the new Plotly selector for this section
        container_id = f"plotly-selector-{sec}"
        section_html = _generate_plotly_selector_html(
            figures[sec], 
            container_id, 
            sec.title()
        )

        sec_data = {
            "id": f"sec-{uuid.uuid4().hex}", 
            "title": sec.title(), 
            "html_content": section_html  # Store the complete HTML
        }

        sections.append(sec_data)

    return sections

def _section_html(sec: Dict) -> str:
    """Generate section HTML using the integrated Plotly selector"""
    return f'''
    <div class="section" id="{sec["id"]}">
        <div class="section-header" onclick="toggleSection(event)">
            <h2>{sec["title"]}</h2>
            <span class="toggle-icon">▼</span>
        </div>
        <div class="section-content" id="{sec["id"]}-content">
            <div class="subsection">
                {sec["html_content"]}
            </div>
        </div>
    </div>
    '''

def _parse_shap_report(report: str) -> Dict[str, Dict[str, str]]:
    """Parse SHAP report string into structured feature data"""
    shap_data = {}
    sections = report.split("\n\n")
    
    for section in sections:
        section = section.strip()
        if not section:
            continue
        
        header, _, content = section.partition('\n')
        header = header.strip().strip('*').strip()
        
        # Top features section
        if "Top features by average impact" in header:
            for line in content.split('\n'):
                if '•' in line and '`' in line:
                    parts = line.split('`')
                    if len(parts) >= 2:
                        feature = parts[1]
                        value = parts[2].split('=')[-1].strip().rstrip(')')
                        if feature not in shap_data:
                            shap_data[feature] = {}
                        shap_data[feature]['mean_shap'] = value
        
        # Beeswarm interpretation
        elif "Beeswarm interpretation" in header:
            for line in content.split('\n'):
                if '•' in line and '`' in line:
                    parts = line.split('`')
                    if len(parts) >= 2:
                        feature = parts[1]
                        interpretation = parts[2].split(':', 1)[1].split('(')[0].strip()
                        rho = line.split('ρ = ')[-1].rstrip(')') if 'ρ = ' in line else ''
                        if feature not in shap_data:
                            shap_data[feature] = {}
                        shap_data[feature]['beeswarm_interpretation'] = interpretation
                        shap_data[feature]['spearman_rho'] = rho
        
        # Dependency plot interpretations
        elif "Dependency plot interpretations" in header:
            for line in content.split('\n'):
                if '•' in line and '`' in line:
                    parts = line.split('`')
                    if len(parts) >= 2:
                        feature = parts[1]
                        relationship = line.split('shows a ')[1].split('(')[0].strip() if 'shows a ' in line else ''
                        rho = line.split('ρ = ')[-1].rstrip(')') if 'ρ = ' in line else ''
                        if feature not in shap_data:
                            shap_data[feature] = {}
                        shap_data[feature]['dependency_relationship'] = relationship
                        shap_data[feature]['dependency_rho'] = rho
        
        # Interaction summaries
        elif "Interaction summaries" in header:
            for line in content.split('\n'):
                if '•' in line and '`' in line:
                    parts = line.split('`')
                    if len(parts) >= 4:
                        feature = parts[1]
                        partner = parts[3]
                        score = line.split('mean |interaction SHAP| = ')[1].split(')')[0] if 'mean |interaction SHAP| = ' in line else ''
                        relationship = line.split('relationship: ')[1].split(' (ρ')[0] if 'relationship: ' in line else ''
                        rho_feat = line.split('ρ_feat→SHAP = ')[1].split(',')[0] if 'ρ_feat→SHAP = ' in line else ''
                        rho_partner = line.split('ρ_partner→SHAP = ')[1].split(')')[0] if 'ρ_partner→SHAP = ' in line else ''
                        
                        if feature not in shap_data:
                            shap_data[feature] = {}
                        shap_data[feature]['partner_feature'] = partner
                        shap_data[feature]['interaction_strength'] = score
                        shap_data[feature]['interaction_relationship'] = relationship
                        shap_data[feature]['rho_feature'] = rho_feat
                        shap_data[feature]['rho_partner'] = rho_partner
    return shap_data

def _aggregate_shap_data(shap_reports: Dict) -> Dict[str, Dict[str, str]]:
    """Combine SHAP reports from different models into single feature dictionary"""
    aggregated = {}
    for report in shap_reports.values():
        data = _parse_shap_report(report)
        for feature, values in data.items():
            if feature not in aggregated:
                aggregated[feature] = values
    return aggregated

def _prepare_features_table(
    features: List[Dict], 
    max_features: int,
    category: str
) -> pd.DataFrame:
    if not features:
        return pd.DataFrame({"Message": [f"No significant {category} features found"]})
    
    df = pd.DataFrame(features[:max_features])
    df = df.rename(columns={
        "feature": "Feature",
        "level": "Taxonomic Level",
        "test": "Test",
        "effect": "Effect Size",
        "p_value": "P-value",
        "effect_dir": "Direction"
    })
    
    if "faprotax_functions" in df.columns:
        df["Functions"] = df["faprotax_functions"].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else ""
        )
    
    df["Effect Size"] = df["Effect Size"].apply(lambda x: f"{x:.4f}")
    df["P-value"] = df["P-value"].apply(lambda x: f"{x:.2e}")
    
    return df[["Feature", "Taxonomic Level", "Test", "Effect Size", 
               "P-value", "Direction", "Functions"]]

def _prepare_stats_summary(stats: Dict) -> pd.DataFrame:
    summary = []
    for column, tables in stats.items():
        for table_type, levels in tables.items():
            for level, tests in levels.items():
                for test_name, df in tests.items():
                    if isinstance(df, pd.DataFrame) and "p_value" in df.columns:
                        n_sig = sum(df["p_value"] < 0.05)
                    else:
                        n_sig = 0
                    summary.append({
                        "Column": column,
                        "Table Type": table_type,
                        "Test": test_name,
                        "Level": level,
                        "Significant Features": n_sig,
                        "Total Features": len(df) if isinstance(df, pd.DataFrame) else 0
                    })
    
    return pd.DataFrame(summary)

def _prepare_ml_summary(
    models: Dict, 
    top_group_1: List[Dict], 
    top_group_2: List[Dict]
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Dict]:
    if not models:
        return None, None, {}

    metrics_summary = []
    features_summary = []
    shap_reports = {}
    
    for table_type, levels in models.items():
        for level, methods in levels.items():
            for method, result in methods.items():
                if not result:
                    continue
                
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
                
                # Capture SHAP report if available
                if "shap_report" in result:
                    key = (table_type, level, method)
                    shap_reports[key] = result["shap_report"]
    
    metrics_df = pd.DataFrame(metrics_summary) if metrics_summary else None
    features_df = pd.DataFrame(features_summary) if features_summary else None
    # Return with explicit empty checks
    metrics_df = pd.DataFrame(metrics_summary) if metrics_summary else pd.DataFrame()
    features_df = pd.DataFrame(features_summary) if features_summary else pd.DataFrame()
    return metrics_df, features_df, shap_reports

def _prepare_shap_table(shap_reports: Dict) -> pd.DataFrame:
    """Prepare comprehensive SHAP data table for ML section"""
    rows = []
    for (table_type, level, method), report in shap_reports.items():
        shap_data = _parse_shap_report(report)
        for feature, values in shap_data.items():
            row = {
                "Table Type": table_type,
                "Level": level,
                "Method": method,
                "Feature": feature,
                "Mean |SHAP|": values.get("mean_shap", ""),
                "Beeswarm Interpretation": values.get("beeswarm_interpretation", ""),
                "Spearman's ρ": values.get("spearman_rho", ""),
                "Dependency plot interpretation Relationship": values.get("dependency_relationship", ""),
                "Partner Feature": values.get("partner_feature", ""),
                "Interaction Strength": values.get("interaction_strength", ""),
                "Relationship": values.get("interaction_relationship", ""),
                "ρ (Feature)": values.get("rho_feature", ""),
                "ρ (Partner)": values.get("rho_partner", "")
            }
            rows.append(row)
    
    if not rows:
        return pd.DataFrame(columns=[
            "Table Type", "Level", "Method", "Feature", "Mean |SHAP|", 
            "Beeswarm Interpretation", "Spearman's ρ", 
            "Dependency plot interpretation Relationship", "Partner Feature", 
            "Interaction Strength", "Relationship", "ρ (Feature)", "ρ (Partner)"
        ])
    
    return pd.DataFrame(rows)

def _format_ml_section(
    ml_metrics: pd.DataFrame, 
    ml_features: pd.DataFrame,
    shap_reports: Dict
) -> str:
    if ml_metrics is None or ml_metrics.empty:
        return "<p>No ML results available</p>"
    
    ml_metrics_html = ml_metrics.to_html(index=False, classes='dynamic-table', table_id='ml-metrics-table')
    
    tooltip_map = {
        "MCC": "Balanced classifier metric (-1 to 1) that considers all confusion matrix values...",
        "ROC AUC": "Probability that random positive ranks higher than random negative...",
        "F1 Score": "Balance between precision and recall...",
        "PR AUC": "Positive-class focused metric for imbalanced data..."
    }
    ml_metrics_html = _add_header_tooltips(ml_metrics_html, tooltip_map)
    
    enhanced_metrics = f"""
    <div class="table-container" id="container-ml-metrics-table">
        {ml_metrics_html}
        <div class="table-controls">
            <div class="pagination-controls">
                <span>Rows per page:</span>
                <select class="rows-per-page" onchange="changePageSize('ml-metrics-table', this.value)">
                    <option value="5">5</option>
                    <option value="10" selected>10</option>
                    <option value="20">20</option>
                    <option value="50">50</option>
                    <option value="100">100</option>
                    <option value="-1">All</option>
                </select>
                <div class="pagination-buttons" id="pagination-ml-metrics-table"></div>
                <span class="pagination-indicator" id="indicator-ml-metrics-table"></span>
            </div>
        </div>
    </div>
    """
    
    features_html = _add_table_functionality(ml_features, 'ml-features-table') if ml_features is not None else "<p>No feature importance data available</p>"
    
    # SHAP Analysis table
    shap_html = ""
    if shap_reports:
        shap_df = _prepare_shap_table(shap_reports)
        if not shap_df.empty:
            shap_html = """
            <h3>SHAP Analysis</h3>
            <p>Comprehensive SHAP analysis for top features across all models:</p>
            """ + _add_table_functionality(shap_df, 'shap-table')
    
    return f"""
    <div class="ml-section">
        <h3>Model Performance</h3>
        {enhanced_metrics}
        
        <h3>Top Features by Importance</h3>
        {features_html}
        
        {shap_html}
    </div>
    """

def _add_header_tooltips(
    table_html: str, 
    tooltip_map: Dict[str, str]
) -> str:
    for header, tooltip_text in tooltip_map.items():
        tooltip_html = (
            f'<span class="tooltip">{header}'
            f'<span class="tooltiptext">{tooltip_text}</span>'
            f'</span>'
        )
        table_html = table_html.replace(
            f'<th>{header}</th>', 
            f'<th>{tooltip_html}</th>'
        )
    return table_html
    
def _add_table_functionality(df: pd.DataFrame, table_id: str) -> str:
    if df is None or df.empty:
        return "<p>No data available</p>"
    
    # Use unique table_id for all control elements
    container_id = f"container-{table_id}"
    pagination_id = f"pagination-{table_id}"
    indicator_id = f"indicator-{table_id}"
    
    table_html = df.to_html(index=False, classes='dynamic-table', table_id=table_id)
    
    return f"""
    <div class='table-container' id='{container_id}'>
        {table_html}
        <div class='table-controls'>
            <div class='pagination-controls'>
                <span>Rows per page:</span>
                <select class='rows-per-page' onchange='changePageSize('{table_id}', this.value)'>
                    <option value="5">5</option>
                    <option value="10" selected>10</option>
                    <option value="20">20</option>
                    <option value="50">50</option>
                    <option value="100">100</option>
                    <option value="-1">All</option>
                </select>
                <div class='pagination-buttons' id='{pagination_id}'></div>
                <span class='pagination-indicator' id='{indicator_id}'></span>
            </div>
        </div>
    </div>
    """

def generate_html_report(
    amplicon_data: "AmpliconData",
    output_path: Union[str, Path],
    include_sections: Optional[List[str]] = None,
    max_features: int = 20,
    cfg: Optional[Dict] = None
) -> None:
    if cfg:
        group_col = cfg.get("group_column", "nuclear_contamination_status")
        group_col_values = cfg.get("group_column_values", [True, False])
    else:
        group_col = "nuclear_contamination_status"
        group_col_values = [True, False]
        
    figures_dict = _extract_figures(amplicon_data)
    
    include_sections = include_sections or [
        k for k, v in figures_dict.items() if v
    ]
    if 'violin' in figures_dict and 'violin' not in include_sections:
        include_sections.append('violin')
    
    ts = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tables_html = f"""
    <div class="subsection">
        <h3>Top Features</h3>
    """
    
    # Loop through top_features
    for col, val_dict in amplicon_data.top_features.items():
        for val, features in val_dict.items():
            group_key = f"{col}={val}"
            df = _prepare_features_table(features, max_features, group_key)  # Use loop variable
            tables_html += f"""
            <h4>Features associated with {group_key}</h4>
            {_add_table_functionality(df, f'{group_key}-table')}
            """
    
    # Stats summary
    stats_df = _prepare_stats_summary(amplicon_data.stats)
    
    # ML summary (with safety checks)
    if group_col in amplicon_data.top_features:
        group_data = amplicon_data.top_features[group_col]
        group1_data = group_data.get(group_col_values[0], None)
        group2_data = group_data.get(group_col_values[1], None)
        
        ml_metrics, ml_features, shap_reports = _prepare_ml_summary(
            amplicon_data.models,
            group1_data,
            group2_data
        )
    else:
        ml_metrics = None
    if ml_metrics is not None and not ml_metrics.empty:
        ml_html = _format_ml_section(ml_metrics, ml_features, shap_reports) if ml_metrics else "<p>No ML results</p>"
    else:
        ml_html = "<p>No ML results</p>"
    # Append final sections
    tables_html += f"""
    </div>
    <div class="subsection">
        <h3>Statistical Summary</h3>
        {_add_table_functionality(stats_df, 'stats-table')}
    </div>
    <div class="subsection">
        <h3>Machine Learning Results</h3>
        {ml_html}
    </div>
    """
    
    sections = _prepare_sections(figures_dict, include_sections)
    sections_html = "\n".join(_section_html(s) for s in sections)

    nav_items = [
        ("Analysis Summary", "analysis-summary"),
        *[(sec['title'], sec['id']) for sec in sections]
    ]
    
    nav_html = """
    <div class="toc">
        <h2>Table of Contents</h2>
        <ul>
    """
    for title, section_id in nav_items:
        nav_html += f'<li><a href="#{section_id}">{title}</a></li>\n'
    nav_html += "        </ul>\n    </div>"

    try:
        plotly_ver = get_plotlyjs_version()
    except Exception:
        plotly_ver = "3.0.1"
    plotly_js_tag = (
        f'<script src="https://cdn.plot.ly/plotly-{plotly_ver}.min.js"></script>'
    )

    # Note: We no longer need to prepare a separate plot_data dictionary 
    # since the Plotly selector handles its own data internally
    
    try:
        table_js = import_js_as_str(tables_js_path)
    except Exception as e:
        logger.error(f"Error reading JavaScript file: {e}")
        table_js = ""
    
    try:
        css_content = css_path.read_text(encoding='utf-8')
        # CSS already includes all necessary Plotly selector styles
    except Exception as e:
        logger.error(f"Error reading CSS file: {e}")
        css_content = ""
        
    try:
        html_template = html_template_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.error(f"Error loading HTML template: {e}")
        html_template = """<!DOCTYPE html>
        <html>
        <head><title>Error</title></head>
        <body>Report generation failed: Missing template</body>
        </html>"""

    html = html_template.format(
        title="16S Amplicon Analysis Report",
        plotly_js_tag=plotly_js_tag,
        generated_ts=ts,
        section_list=", ".join(include_sections),
        nav_html=nav_html,
        tables_html=tables_html,
        sections_html=sections_html,
        plot_data_json="{}",  # Empty since selectors handle their own data
        table_js=table_js,
        css_content=css_content
    )
        
    output_path.write_text(html, encoding="utf-8")

# ========================== ADDITIONAL UTILITY FUNCTIONS ========================== #

def create_standalone_plotly_selector(figures_dict: Dict[str, Any], 
                                    title: str = "Interactive Plotly Dashboard",
                                    output_path: Optional[Union[str, Path]] = None) -> str:
    """
    Create a standalone HTML page with just the Plotly selector.
    Useful for testing or creating focused figure viewers.
    """
    
    selector_html = _generate_plotly_selector_html(figures_dict, "main-dashboard", title)
    
    try:
        plotly_ver = get_plotlyjs_version()
    except Exception:
        plotly_ver = "3.0.1"
    
    complete_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-{plotly_ver}.min.js"></script>
    <style>
        body {{
            margin: 20px;
            background-color: #ffffff;
            font-family: Arial, sans-serif;
        }}
        
        .plotly-selector-container {{
            width: 100%;
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        .selector-controls {{
            margin-bottom: 15px;
            padding: 10px;
            background-color: #f8f9fa;
            border-radius: 5px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .selector-label {{
            font-weight: bold;
            margin: 0;
        }}
        
        .figure-dropdown {{
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            background-color: white;
            min-width: 250px;
            font-size: 14px;
        }}
        
        .plotly-selector-plot {{
            width: 100%;
            min-height: 600px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        
        .error-message {{
            padding: 20px;
            color: #721c24;
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            border-radius: 4px;
            margin: 10px 0;
        }}
    </style>
</head>
<body>
    <h1 style="text-align: center; color: #333;">{title}</h1>
    {selector_html}
</body>
</html>
"""
    
    if output_path:
        Path(output_path).write_text(complete_html, encoding="utf-8")
        logger.info(f"Standalone Plotly selector saved to {output_path}")
    
    return complete_html
