# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import base64
import itertools
import json
import logging
import uuid
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

sections = {
    "stats": {},
    "alpha_diversity": {},
    "ordination": {},
    "models": {}
}

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
