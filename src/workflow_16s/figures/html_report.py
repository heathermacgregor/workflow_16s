# ===================================== IMPORTS ====================================== #

import os
import webbrowser
from pathlib import Path
from typing import Dict, Union
from plotly.graph_objs import Figure

# ==================================== FUNCTIONS ===================================== #

import os
from pathlib import Path
from typing import Dict, Union
from plotly.graph_objs import Figure

class HTMLReportGenerator:
    def __init__(self, output_file: str = "report.html", auto_open: bool = False,  # Changed default to False
                 plotly_config: dict = None):
        self.output_file = Path(output_file)
        self.auto_open = auto_open  # Now defaults to False
        self.plotly_config = plotly_config or {
            'responsive': True,
            'displayModeBar': True,
        }

    def generate(self, figure_dict: Dict):
        """Generate report and save to file without automatic opening"""
        self._validate_structure(figure_dict)
        content = self._generate_sections(figure_dict)
        full_html = self._wrap_in_template(content)
        
        # Convert to absolute path and ensure directory exists
        self.output_file = self.output_file.resolve()
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.output_file.write_text(full_html)
        
        # Removed the auto-open functionality completely

    def _validate_structure(self, data: Dict, path: str = "root"):
        """Validation remains the same"""
        for key, value in data.items():
            current_path = f"{path} -> {key}"
            
            if isinstance(value, dict):
                self._validate_structure(value, current_path)
            elif isinstance(value, tuple):
                if not any(isinstance(item, Figure) for item in value):
                    raise TypeError(
                        f"Tuple at {current_path} contains no Plotly Figure. "
                        f"Contents: {tuple(type(x) for x in value)}"
                    )
            elif not isinstance(value, Figure):
                raise TypeError(
                    f"Invalid type {type(value)} at {current_path}. "
                    f"Expected Plotly Figure or tuple containing Figure."
                )

    def _generate_sections(self, data: Dict, level: int = 0) -> str:
        """Content generation remains the same"""
        html = []
        for key, value in data.items():
            if isinstance(value, dict):
                section_content = self._generate_sections(value, level + 1)
                html.append(self._section_template(key, section_content, level))
            elif isinstance(value, tuple):
                html.append(self._tuple_template(key, value))
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

    def _tuple_template(self, title: str, figures: tuple) -> str:
        return f"""
        <div class="tuple-container">
            <h4>{self._format_title(title)}</h4>
            <div class="figure-group">
                {"".join(self._figure_template("", fig) for fig in figures if isinstance(fig, Figure))}
            </div>
        </div>
        """

    def _figure_template(self, title: str, figure: Figure) -> str:
        return f"""
        <div class="figure-container">
            {f'<h5>{title}</h5>' if title else ''}
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
                    padding-left: 15px;
                }}
                .tuple-container {{
                    margin: 2rem 0;
                    padding: 1rem;
                    background: #f8f9fa;
                    border-radius: 8px;
                }}
                .figure-group {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                    gap: 1rem;
                }}
                .figure-container {{
                    padding: 1rem;
                    background: white;
                    border: 1px solid #eee;
                    border-radius: 8px;
                }}
                h4 {{ margin-bottom: 0.5rem; color: #444; }}
                h5 {{ font-size: 0.9em; color: #666; margin: 0.5rem 0; }}
            </style>
        </head>
        <body>
            <h1>Analysis Report</h1>
            {content}
        </body>
        </html>
        """
