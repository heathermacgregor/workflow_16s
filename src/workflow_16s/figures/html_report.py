# ===================================== IMPORTS ====================================== #

import os
import webbrowser
from pathlib import Path
from typing import Dict, Union
from plotly.graph_objs import Figure

# ==================================== FUNCTIONS ===================================== #

class HTMLReportGenerator:
    def __init__(self, output_file: str = "report.html", auto_open: bool = True,
                 plotly_config: dict = None):
        """
        Initialize HTML report generator.
        
        Args:
            output_file: Path for output HTML file
            auto_open: Whether to automatically open the report after generation
            plotly_config: Plotly configuration options (e.g., staticPlot, responsive)
        """
        self.output_file = Path(output_file)
        self.auto_open = auto_open
        self.plotly_config = plotly_config or {
            'responsive': True,
            'displayModeBar': True,
        }

    def generate(self, figure_dict: Dict[str, Union[dict, Figure]]):
        """
        Generate HTML report from nested dictionary of Plotly figures.
        
        Args:
            figure_dict: Nested dictionary containing Plotly figures
        """
        content = self._generate_html_content(figure_dict)
        full_html = self._wrap_in_template(content)
        
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.output_file.write_text(full_html)
        
        if self.auto_open:
            webbrowser.open(self.output_file.as_uri())

    def _generate_html_content(self, data: Dict[str, Union[dict, Figure]], level: int = 0) -> str:
        """Recursively generate HTML content from nested dictionary"""
        html = []
        for key, value in data.items():
            if isinstance(value, dict):
                html.append(self._generate_section(key, value, level))
            elif isinstance(value, Figure):
                html.append(self._generate_figure(key, value))
            else:
                raise TypeError(f"Unsupported type {type(value)} for key {key}")
        return '\n'.join(html)

    def _generate_section(self, title: str, content: dict, level: int) -> str:
        """Generate section with nested content"""
        heading_level = min(level + 2, 6)
        return f"""
        <div class="section level-{level}">
            <h{heading_level}>{title.title()}</h{heading_level}>
            {self._generate_html_content(content, level + 1)}
        </div>
        """

    def _generate_figure(self, title: str, figure: Figure) -> str:
        """Convert Plotly figure to HTML div"""
        return f"""
        <div class="figure-container">
            <h4>{title.title()}</h4>
            {self._plotly_figure_to_html(figure)}
        </div>
        """

    def _plotly_figure_to_html(self, figure: Figure) -> str:
        """Convert Plotly figure to HTML string"""
        return figure.to_html(
            full_html=False,
            include_plotlyjs=False,
            config=self.plotly_config
        )

    def _wrap_in_template(self, content: str) -> str:
        """Wrap content in full HTML template"""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Plotly Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ 
                    font-family: Arial, sans-serif;
                    margin: 2rem;
                    color: #333;
                }}
                .figure-container {{
                    margin: 1rem 0;
                    padding: 1rem;
                    border: 1px solid #eee;
                    border-radius: 8px;
                }}
                .section {{
                    margin-left: {self.plotly_config.get('section_indent', 1)}rem;
                    margin-top: 1rem;
                }}
                h2 {{ color: #1a237e; }}
                h3 {{ color: #283593; }}
                h4 {{ color: #3949ab; }}
                h5 {{ color: #5c6bc0; }}
                h6 {{ color: #7986cb; }}
            </style>
        </head>
        <body>
            <h1>Data Analysis Report</h1>
            {content}
        </body>
        </html>
        """
