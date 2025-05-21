# ===================================== IMPORTS ====================================== #

import os
import webbrowser
from pathlib import Path
from typing import Dict, Union
from plotly.graph_objs import Figure

# ==================================== FUNCTIONS ===================================== #

import os
import webbrowser
from pathlib import Path
from typing import Dict, Union
from plotly.graph_objs import Figure

class HTMLReportGenerator:
    def __init__(self, output_file: str = "report.html", auto_open: bool = True,
                 plotly_config: dict = None):
        self.output_file = Path(output_file)
        self.auto_open = auto_open
        self.plotly_config = plotly_config or {
            'responsive': True,
            'displayModeBar': True,
        }

    def generate(self, figure_dict: Dict):
        """Generate report from strictly nested dictionary of Plotly figures"""
        self._validate_structure(figure_dict)
        content = self._generate_sections(figure_dict)
        full_html = self._wrap_in_template(content)
        
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.output_file.write_text(full_html)
        
        if self.auto_open:
            webbrowser.open(self.output_file.as_uri())

    def _validate_structure(self, data: Dict, path: str = "root"):
        """Recursively validate the figure dictionary structure"""
        for key, value in data.items():
            current_path = f"{path} -> {key}"
            if isinstance(value, dict):
                self._validate_structure(value, current_path)
            elif not isinstance(value, Figure):
                raise TypeError(
                    f"Invalid type {type(value)} at {current_path}. "
                    f"Expected only nested dictionaries terminating in Plotly Figures."
                )

    def _generate_sections(self, data: Dict, level: int = 0) -> str:
        """Generate nested HTML sections from validated structure"""
        html = []
        for key, value in data.items():
            if isinstance(value, dict):
                section_content = self._generate_sections(value, level + 1)
                html.append(self._section_template(key, section_content, level))
            else:
                html.append(self._figure_template(key, value))
        return '\n'.join(html)

    def _section_template(self, title: str, content: str, level: int) -> str:
        heading_level = min(level + 2, 6)
        return f"""
        <div class="section level-{level}">
            <h{heading_level}>{self._format_title(title)}</h{heading_level}>
            {content}
        </div>
        """

    def _figure_template(self, title: str, figure: Figure) -> str:
        return f"""
        <div class="figure-container">
            <h4>{self._format_title(title)}</h4>
            {figure.to_html(full_html=False, include_plotlyjs=False, config=self.plotly_config)}
        </div>
        """

    def _format_title(self, text: str) -> str:
        return text.replace('_', ' ').title()

    def _wrap_in_template(self, content: str) -> str:
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Analysis Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ 
                    font-family: Arial, sans-serif;
                    margin: 2rem;
                    color: #333;
                }}
                .section {{
                    margin: 1rem 0;
                    padding-left: {15 * level}px;
                }}
                .figure-container {{
                    margin: 1rem 0;
                    padding: 1rem;
                    background: #f8f9fa;
                    border-radius: 8px;
                }}
                h2 {{ color: #2c3e50; }}
                h3 {{ color: #34495e; }}
                h4 {{ color: #7f8c8d; font-size: 1.1rem; }}
            </style>
        </head>
        <body>
            <h1>Analysis Report</h1>
            {content}
        </body>
        </html>
        """
