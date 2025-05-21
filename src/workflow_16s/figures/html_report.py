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
from typing import Dict, Union, Tuple, Any
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

    def generate(self, figure_dict: Dict[str, Any]):
        """Handle complex nested structure with mixed types"""
        content = self._generate_content(figure_dict)
        full_html = self._wrap_in_template(content)
        
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.output_file.write_text(full_html)
        
        if self.auto_open:
            webbrowser.open(self.output_file.as_uri())

    def _generate_content(self, data: Any, level: int = 0) -> str:
        """Recursive content generator handling multiple types"""
        if isinstance(data, dict):
            return self._process_dict(data, level)
        if isinstance(data, tuple):
            return self._process_tuple(data, level)
        if isinstance(data, Figure):
            return self._generate_figure("", data)
        raise TypeError(f"Unsupported type {type(data)}")

    def _process_dict(self, data: Dict[str, Any], level: int) -> str:
        """Process dictionary items with automatic heading creation"""
        html = []
        for key, value in data.items():
            html.append(f"""
            <div class="section level-{level}">
                <h{min(level+2, 6)}>{self._format_key(key)}</h{min(level+2, 6)}>
                {self._generate_content(value, level+1)}
            </div>
            """)
        return '\n'.join(html)

    def _process_tuple(self, data: Tuple, level: int) -> str:
        """Process tuples as horizontal figure groups"""
        html = []
        for item in data:
            if isinstance(item, Figure):
                html.append(self._generate_figure("", item))
            elif isinstance(item, dict):
                html.append(self._process_dict(item, level))
            else:
                html.append(f"<div class='metadata'>{str(item)}</div>")
        return f"<div class='tuple-container'>{''.join(html)}</div>"

    def _generate_figure(self, title: str, figure: Figure) -> str:
        return f"""
        <div class="figure-container">
            {f'<h4>{title}</h4>' if title else ''}
            {figure.to_html(full_html=False, include_plotlyjs=False, config=self.plotly_config)}
        </div>
        """

    def _format_key(self, key: str) -> str:
        return key.replace('_', ' ').title()

    def _wrap_in_template(self, content: str) -> str:
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Microbiome Analysis Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ 
                    font-family: Arial, sans-serif;
                    margin: 2rem;
                    color: #333;
                    background-color: #fafafa;
                }}
                .section {{
                    margin: 1rem 0;
                    padding: 1rem;
                    background: white;
                    border-radius: 8px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                }}
                .tuple-container {{
                    display: grid;
                    gap: 1rem;
                    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                    margin: 1rem 0;
                }}
                .figure-container {{
                    padding: 1rem;
                    background: white;
                    border: 1px solid #eee;
                    border-radius: 8px;
                }}
                h2 {{ color: #2c3e50; border-bottom: 2px solid #eee; }}
                h3 {{ color: #34495e; }}
                h4 {{ color: #7f8c8d; font-size: 0.9em; }}
                .metadata {{ 
                    padding: 1rem;
                    background: #f8f9fa;
                    border-radius: 4px;
                    font-family: monospace;
                }}
            </style>
        </head>
        <body>
            <h1>Microbiome Analysis Report</h1>
            {content}
        </body>
        </html>
        """
