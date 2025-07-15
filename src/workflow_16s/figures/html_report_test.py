# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import base64
import itertools
import json
import logging
import uuid
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

# Third Party Imports
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from plotly.offline import get_plotlyjs_version

# ================================== LOCAL IMPORTS =================================== #

from workflow_16s.utils.io import import_js_as_str
from workflow_16s.utils.amplicon_data import AmpliconData

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger('workflow_16s')

script_dir = Path(__file__).parent  
tables_js_path = script_dir / "tables.js"  
css_path = script_dir / "style.css"  
html_template_path = script_dir / "template.html"  

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


level_titles = {
    "color_col": "Color",
    "level": "Taxonomic Level",
    "method": "Method",
    "metric": "Metric",
    "table_type": "Table",
    "test_name": "Test",
}

section_info = {
    "stats": { 
        "title": "Statistical Testing",
        "level_1": "table_type",
        "level_2": "level",
        "level_3": "test_name",
    },
    "alpha_diversity": { 
        "title": "Alpha Diversity",
        "level_1": "table_type",
        "level_2": "level",
        "special_level_3": {
            "results": {"title": "Results"},
            "stats": {"title": "Statistics Summary"}
        }
    },
    "ordination": {
        "title": "Beta Diversity",
        "level_1": "table_type",
        "level_2": "level",
        "level_3": "method"
    },
    "ml": {
        "title": "Machine Learning",
        "level_1": "table_type",
        "level_2": "level",
        "level_3": "method"
    }
}

section_figure_info = {
    "alpha_diversity": { 
        "title": "Alpha Diversity",
        "level_1": "table_type",
        "level_2": "level",
        "level_3": "metric",
        "special_level_3": {"summary": {"title": "Summary",}}
    },
    "ordination": {},
    "map": {
        "title": "Sample Map",
        "level_1": "color_col"
    },
    "ml": { 
        "title": "Machine Learning",
        "level_1": "table_type",
        "level_2": "level",
        "level_3": "method",
        "special_level_4": {
            'shap_summary_bar': {"title": "SHAP Summary (Bar)"},
            'shap_summary_beeswarm': {"title": "SHAP Summary (Beeswarm)"},
            'shap_dependency': {"title": "SHAP Dependency"}
        }
    }
}
cols_to_rename = {
    'feature': 'Feature',
    't_statistic': 'T-statistic',
    'p_value': 'P-value', 
    'mean_difference': 'Mean Difference',
    'cohens_d': 'Cohen\'s D',
    'u_statistic': 'U-statistic',
    'median_difference': 'Median Difference',
    'effect_size_r': 'Effect Size (r)',
    'h_statistic': 'H-statistic',
    'epsilon_squared': 'ε²',
    'groups_tested': 'Groups Tested'
}

def rename_columns(df: pd.DataFrame, rename_map: dict = cols_to_rename) -> pd.DataFrame:
    return df.rename(columns=rename_map)
    
debug_mode = True    

def check_dict(amplicon_data, attr_name):
    val = getattr(amplicon_data, attr_name, None)
    exists = hasattr(amplicon_data, attr_name)
    is_non_empty_dict = isinstance(val, dict) and bool(val)
    logger.info(f"AmpliconData has non-empty {attr_name}: {is_non_empty_dict} (exists: {exists})")
    logger.info(type(getattr(amplicon_data, attr_name, None)))
    logger.info(getattr(amplicon_data, attr_name, None))


class OrdinationFigures:
    #self.ordination[table_type][level][method] = {'result': res, 'figures': figs}
    def __init__(self, amplicon_data: AmpliconData):
        self.amplicon_data = amplicon_data
        self.figures = {}
        if check_dict(self.amplicon_data, 'ordination'):
            self.fetch_figures()
        else:
            logger.warning("No ordination data found in AmpliconData")
            
    def fetch_figures(self):
        for table_type, levels in self.amplicon_data.ordination.items():
            for level, methods in levels.items():
                for method, data in methods.items():
                    if data and 'figures' in data and data['figures']:
                        if table_type not in self.figures:
                            self.figures[table_type] = {}
                        if level not in self.figures[table_type]:
                            self.figures[table_type][level] = {}
                        self.figures[table_type][level][method] = data['figures']
                    else:
                        logger.warning(f"No ordination figures found for {table_type}/{level}/{method}")
        
class AlphaDivFigures:
    #self.alpha_diversity[table_type][level]['figures'][metric] = fig
    #self.alpha_diversity[table_type][level]['figures']['summary'] = stats_fig
    #self.alpha_diversity[table_type][level]['figures']['correlations'] = corr_figures
    def __init__(self, amplicon_data: AmpliconData):
        self.amplicon_data = amplicon_data
        self.figures = {}
        if check_dict(self.amplicon_data, 'alpha_diversity'):
            self.fetch_figures()
        else:
            logger.warning("No alpha diversity data found in AmpliconData")
            
    def fetch_figures(self):
        for table_type, levels in self.amplicon_data.alpha_diversity.items():
            for level, data in levels.items():
                if data and 'figures' in data and data['figures']:
                    if table_type not in self.figures:
                        self.figures[table_type] = {}
                        self.figures[table_type][level] = data['figures']
                    else:
                        logger.warning(f"No alpha diversity figures found for {table_type}/{level}")        

class MLFigures:        
    #self.models[table_type][level][method] = model_result
    #model_result['figures']: 'eval_plots', 'shap_summary_bar', 'shap_summary_beeswarm', 'shap_dependency'
    def __init__(self, amplicon_data: AmpliconData):
        self.amplicon_data = amplicon_data
        self.figures = {}
        if check_dict(self.amplicon_data, 'models'):
            self.fetch_figures()
        else:
            logger.warning("No ML data found in AmpliconData")
            
    def fetch_figures(self):
        for table_type, levels in self.amplicon_data.models.items():
            print(table_type)
            for level, methods in levels.items():
                print(level)
                for method, data in methods.items():
                    print(method)
                    print(data)
                    if data and 'figures' in data and data['figures']:
                        print(data['figures'])
                        if table_type not in self.figures:
                            self.figures[table_type] = {}
                        if level not in self.figures[table_type]:
                            self.figures[table_type][level] = {}
                        self.figures[table_type][level][method] = data['figures']
                    else:
                        logger.warning(f"No ML figures found for {table_type}/{level}/{method}")        

class Section:
    def __init__(self, amplicon_data: AmpliconData):
        self.amplicon_data = amplicon_data
        self.figures = {}
        self._extract_figures()

    def _extract_figures(self) -> Dict[str, Any]:
        logger.info("Analyzing AmpliconData structure...")
        # Ordination figures
        logger.info("Extracting ordination figures...")
        self.figures['ordination'] = OrdinationFigures(self.amplicon_data)
        
        # Alpha diversity figures
        logger.info("Extracting alpha diversity figures...")
        self.figures['alpha_diversity'] = AlphaDivFigures(self.amplicon_data)
        
        logger.info("Extracting ML figures...")
        self.figures['models'] = MLFigures(self.amplicon_data)
        #for attr, value in vars(self.figures).items():
        #    print(f"{attr}: {value}")
        """
        # Sample maps
        logger.info("Extracting sample maps...")
        if hasattr(amplicon_data, 'maps') and amplicon_data.maps:
            logger.info(f"  Found {len(amplicon_data.maps)} sample maps")
            figures['map'] = amplicon_data.maps
        else:
            logger.warning("No sample maps found in AmpliconData")
        
        # SHAP figures
        logger.info("Extracting SHAP figures...")
        shap_figures = {}
        if hasattr(amplicon_data, 'models'):
            for table_type, levels in amplicon_data.models.items():
                logger.info(f"  Processing table_type: {table_type}")
                for level, methods in levels.items():
                    logger.info(f"    Processing level: {level}")
                    for method, model_result in methods.items():
                        logger.info(f"      Processing method: {method}")
                        if model_result and 'figures' in model_result:
                            logger.info(f"        Found figures: {list(model_result['figures'].keys())}")
                            if table_type not in shap_figures:
                                shap_figures[table_type] = {}
                            if level not in shap_figures[table_type]:
                                shap_figures[table_type][level] = {}
                            shap_figures[table_type][level][method] = model_result['figures']
                        else:
                            logger.warning(f"        No figures found for {table_type}/{level}/{method}")
        else:
            logger.warning("No models data found in AmpliconData")
        figures['shap'] = shap_figures
    
        # Violin plots
        logger.info("Extracting violin plots...")
        violin_figures = {'contaminated': {}, 'pristine': {}}
        if hasattr(amplicon_data, 'top_contaminated_features'):
            logger.info(f"  Found {len(amplicon_data.top_contaminated_features)} contaminated features")
            for feat in amplicon_data.top_contaminated_features:
                if 'violin_figure' in feat and feat['violin_figure']:
                    logger.info(f"    Violin figure found for {feat['feature']}")
                    violin_figures['contaminated'][feat['feature']] = feat['violin_figure']
        if hasattr(amplicon_data, 'top_pristine_features'):
            logger.info(f"  Found {len(amplicon_data.top_pristine_features)} pristine features")
            for feat in amplicon_data.top_pristine_features:
                if 'violin_figure' in feat and feat['violin_figure']:
                    logger.info(f"    Violin figure found for {feat['feature']}")
                    violin_figures['pristine'][feat['feature']] = feat['violin_figure']
        figures['violin'] = violin_figures
    
        # Log the final extracted figures structure
        logger.info("Extracted figures summary:")
        for section, data in figures.items():
            if isinstance(data, dict):
                logger.info(f"  {section}: {len(data)} items")
                for key, subdata in data.items():
                    if isinstance(subdata, dict):
                        logger.info(f"    {key}: {len(subdata)} sub-items")
                    else:
                        logger.info(f"    {key}: {type(subdata)}")
            else:
                logger.info(f"  {section}: {type(data)}")
        
        return figures
        """


def _prepare_stats_summary(stats: Dict) -> pd.DataFrame:
    """
    Create statistical test summary table.
    
    Args:
        stats: Nested dictionary of statistical results.
    
    Returns:
        DataFrame summarizing number of significant features per test/level.
    """
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


def generate_html_report(
    amplicon_data: AmpliconData,
    output_path: Union[str, Path],
    include_sections: Optional[List[str]] = None,
    max_features: int = 20  
) -> None:
    """
    Generate interactive HTML report for 16S analysis results.
    
    Compiles visualizations, statistical summaries, and machine learning
    results into a self-contained HTML file with interactive elements.
    
    Args:
        amplicon_data:    Analysis results container.
        output_path:      Destination path for HTML report.
        include_sections: Sections to include (default: all non-empty sections).
        max_features:     Maximum features to display in differential analysis tables.
    
    Raises:
        IOError: If report file cannot be written.
    """
    include_sections = include_sections or [
        k for k, v in amplicon_data.figures.items() if v
    ]
    if 'violin' in amplicon_data.figures and 'violin' not in include_sections:
        include_sections.append('violin')
    ts = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare tables section 
    tables_html = ""
    
    # Top features tables
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
    
    # Stats summary
    stats_df = _prepare_stats_summary(
        amplicon_data.stats
    )
    
    # ML summary
    ml_metrics, ml_features = _prepare_ml_summary(
        amplicon_data.models,
        amplicon_data.top_contaminated_features,
        amplicon_data.top_pristine_features
    )
    ml_html = _format_ml_section(ml_metrics, ml_features)
    
    # Build tables section HTML
    tables_html = f"""
    <div class="subsection">
        <h3>Top Features</h3>
        <h4>Contaminated-Associated Features</h4>
        {_add_table_functionality(contam_df, 'contam-table')}
        
        <h4>Pristine-Associated Features</h4>
        {_add_table_functionality(pristine_df, 'pristine-table')}
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

    # Prepare figures section
    id_counter = itertools.count()
    sections, plot_data = _prepare_sections(
        amplicon_data.figures, include_sections, id_counter
    )
    sections_html = "\n".join(_section_html(s) for s in sections)

    # Prepare navigation section
    nav_items = [
        ("Analysis Summary", "analysis-summary"),
        *[(sec['title'], sec['id']) for sec in sections]
    ]
    
    # Generate navigation HTML
    nav_html = """
    <div class="toc">
        <h2>Table of Contents</h2>
        <ul>
    """
    for title, section_id in nav_items:
        nav_html += f'<li><a href="#{section_id}">{title}</a></li>\n'
    nav_html += "        </ul>\n    </div>"

    # Prepare CDN tag for Plotly 
    try:
        plotly_ver = get_plotlyjs_version()
    except Exception:
        plotly_ver = "3.0.1"
    plotly_js_tag = (
        f'<script src="https://cdn.plot.ly/plotly-{plotly_ver}.min.js"></script>'
    )

    # JSON payload (escape "</" so it can never close the <script>)
    payload = json.dumps(plot_data, cls=NumpySafeJSONEncoder, ensure_ascii=False)
    payload = payload.replace("</", "<\\/")  # safety

    # Table functionality JavaScript
    try:
        table_js = import_js_as_str(tables_js_path)
    except Exception as e:
        logger.error(f"Error reading JavaScript file: {e}")
        table_js = ""
    # CSS
    # Append tooltip CSS to existing styles
    tooltip_css = """
    /* Tooltip styles */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
        border-bottom: 1px dashed #3498db;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 280px;
        background-color: #222;
        color: #fff;
        text-align: left;
        border-radius: 6px;
        padding: 12px;
        position: absolute;
        z-index: 1000;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 14px;
        line-height: 1.5;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    
    .tooltip .tooltiptext::after {
        content: "";
        position: absolute;
        top: 100%;
        left: 50%;
        margin-left: -5px;
        border-width: 5px;
        border-style: solid;
        border-color: #222 transparent transparent transparent;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    """
    
    # CSS
    try:
        css_content = css_path.read_text(encoding='utf-8')
        css_content += tooltip_css  # Append tooltip styles
    except Exception as e:
        logger.error(f"Error reading CSS file: {e}")
        css_content = tooltip_css  # Fallback to just tooltip styles
    # HTML template
    try:
        html_template = html_template_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.error(f"Error loading HTML template: {e}")
        # Fallback minimal template
        html_template = """<!DOCTYPE html>
        <html>
        <head><title>Error</title></head>
        <body>Report generation failed: Missing template</body>
        </html>"""

    # Build the full HTML
    html = html_template.format(
        plotly_js_tag=plotly_js_tag,
        generated_ts=ts,
        section_list=", ".join(include_sections),
        nav_html=nav_html,
        tables_html=tables_html,
        sections_html=sections_html,
        plot_data_json=payload,
        table_js=table_js,
        css_content=css_content
    )
    output_path.write_text(html, encoding="utf-8")
