# ===================================== IMPORTS ====================================== #

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import sys

# Add API modules to path
sys.path.append(str(Path(__file__).parent.parent))

from api.main import load_report_data, app as fastapi_app
from api.data_processor import ReportDataProcessor

# ===================================== ENHANCED REPORT GENERATION ====================================== #

def generate_modern_html_report(
    amplicon_data: "AmpliconData",
    output_path: Union[str, Path],
    include_sections: Optional[List[str]] = None,
    max_features: int = 20,
    config: Optional[Dict] = None,
    mode: str = "legacy"  # "legacy", "api", or "hybrid"
) -> None:
    """
    Enhanced HTML report generation that supports both legacy HTML and modern React/API modes.
    
    Args:
        amplicon_data: The AmpliconData object containing analysis results
        output_path: Path where the report should be saved
        include_sections: List of sections to include in the report
        max_features: Maximum number of features to display in tables
        config: Configuration dictionary
        mode: Report generation mode:
            - "legacy": Generate traditional HTML report (default)
            - "api": Generate API data files for React frontend
            - "hybrid": Generate both legacy HTML and API data
    """
    
    logger = logging.getLogger('workflow_16s')
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if mode in ["api", "hybrid"]:
        # Generate API data for React frontend
        _generate_api_data(amplicon_data, output_path, config)
    
    if mode in ["legacy", "hybrid"]:
        # Generate traditional HTML report
        from workflow_16s.figures.html_report import generate_html_report
        generate_html_report(
            amplicon_data=amplicon_data,
            output_path=output_path,
            include_sections=include_sections,
            max_features=max_features,
            config=config
        )
    
    if mode in ["api", "hybrid"]:
        # Generate React frontend index
        _generate_react_index(output_path)

def _generate_api_data(amplicon_data: "AmpliconData", output_path: Path, config: Optional[Dict] = None):
    """Generate structured data files for API consumption"""
    
    try:
        # Process the amplicon data
        report_data = load_report_data(amplicon_data, "current")
        
        # Create API data directory
        api_data_dir = output_path.parent / "api_data"
        api_data_dir.mkdir(exist_ok=True)
        
        # Save structured data as JSON files
        with open(api_data_dir / "report_metadata.json", "w") as f:
            json.dump({
                "id": "current",
                "title": report_data["metadata"]["title"],
                "generated_ts": report_data["metadata"]["generated_ts"],
                "sections": report_data["metadata"]["sections_available"],
                "config": report_data["metadata"].get("config", {})
            }, f, indent=2)
        
        # Save sections data
        sections_dir = api_data_dir / "sections"
        sections_dir.mkdir(exist_ok=True)
        
        for section in report_data["sections"]:
            section_file = sections_dir / f"{section.id}.json"
            with open(section_file, "w") as f:
                json.dump({
                    "id": section.id,
                    "title": section.title,
                    "type": section.type,
                    "has_data": section.has_data,
                    "filters": section.filters
                }, f, indent=2)
        
        # Save figures data
        figures_dir = api_data_dir / "figures"
        figures_dir.mkdir(exist_ok=True)
        
        for figure_id, figure_data in report_data["figures"].items():
            figure_file = figures_dir / f"{figure_id}.json"
            with open(figure_file, "w") as f:
                json.dump({
                    "id": figure_id,
                    "type": figure_data.type,
                    "data": figure_data.data,
                    "metadata": figure_data.metadata
                }, f, indent=2)
        
        # Save tables data
        if report_data["tables"]:
            tables_dir = api_data_dir / "tables"
            tables_dir.mkdir(exist_ok=True)
            
            for table_id, table_data in report_data["tables"].items():
                table_file = tables_dir / f"{table_id}.json"
                with open(table_file, "w") as f:
                    json.dump(table_data, f, indent=2)
        
        logging.getLogger('workflow_16s').info(f"API data generated in {api_data_dir}")
        
    except Exception as e:
        logging.getLogger('workflow_16s').error(f"Error generating API data: {e}")
        raise

def _generate_react_index(output_path: Path):
    """Generate an HTML index that loads the React application"""
    
    react_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="theme-color" content="#000000" />
    <meta name="description" content="16S rRNA Amplicon Analysis Interactive Report" />
    <title>16S Analysis Report - Interactive</title>
    <style>
        body { 
            margin: 0; 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
        }
        .loading { 
            display: flex; 
            justify-content: center; 
            align-items: center; 
            height: 100vh; 
            font-size: 1.2rem;
            color: #666;
        }
        .fallback {
            padding: 2rem;
            text-align: center;
        }
        .fallback a {
            color: #007bff;
            text-decoration: none;
        }
    </style>
</head>
<body>
    <noscript>
        <div class="fallback">
            <h1>JavaScript Required</h1>
            <p>This interactive report requires JavaScript to run.</p>
            <p><a href="./report.html">View the legacy HTML report instead</a></p>
        </div>
    </noscript>
    <div id="root">
        <div class="loading">Loading interactive report...</div>
    </div>
    
    <!-- In production, these would be built React bundle files -->
    <script>
        // Placeholder for React application
        // In a full implementation, this would load the compiled React bundle
        document.getElementById('root').innerHTML = `
            <div style="padding: 2rem; text-align: center;">
                <h1>16S Analysis Report - Interactive Version</h1>
                <p>The React frontend is not built yet. This is a placeholder.</p>
                <p><a href="./report.html" style="color: #007bff;">View the full HTML report</a></p>
                <div style="margin-top: 2rem; padding: 1rem; background: #f8f9fa; border-radius: 0.5rem;">
                    <h3>Modern Features Available:</h3>
                    <ul style="text-align: left; max-width: 500px; margin: 0 auto;">
                        <li>✓ FastAPI backend with structured data endpoints</li>
                        <li>✓ React component architecture</li>
                        <li>✓ Interactive data tables with sorting/filtering</li>
                        <li>✓ Lazy-loaded Plotly visualizations</li>
                        <li>✓ Responsive design</li>
                        <li>✓ API data export for external tools</li>
                    </ul>
                    <p style="margin-top: 1rem;"><strong>To activate:</strong> Build the React frontend and start the FastAPI server.</p>
                </div>
            </div>
        `;
    </script>
</body>
</html>"""
    
    react_index_path = output_path.parent / "interactive_report.html"
    with open(react_index_path, "w") as f:
        f.write(react_html)
    
    logging.getLogger('workflow_16s').info(f"Interactive report placeholder generated: {react_index_path}")