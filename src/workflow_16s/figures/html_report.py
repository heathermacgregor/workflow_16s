# ===================================== IMPORTS ====================================== #

import os
from pathlib import Path
from typing import Dict, Union
from plotly.graph_objs import Figure

# ==================================== FUNCTIONS ===================================== #

html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ASV Analysis Report</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 40px;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
            }
            .section {
                background: white;
                padding: 20px;
                margin-bottom: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            h1 {
                color: #2c3e50;
                text-align: center;
                margin-bottom: 40px;
            }
            h2 {
                color: #34495e;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
                margin-top: 0;
            }
            .grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(550px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }
            .figure-container {
                padding: 15px;
                background: #f9f9f9;
                border-radius: 6px;
            }
            .figure-title {
                margin-top: 0;
                color: #3498db;
                font-size: 1.1em;
            }
            @media (max-width: 600px) {
                .grid {
                    grid-template-columns: 1fr;
                }
                body {
                    margin: 20px;
                }
            }
            h3 {
                color: #2c3e50;
                font-size: 1.2em;
                margin-top: 15px;
            }
            h4 {
                color: #34495e;
                font-size: 1.1em;
                margin-top: 10px;
            }
            .sub-section {
                margin-left: 20px;
                border-left: 3px solid #eee;
                padding-left: 15px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>ASV Analysis Report</h1>
    """

def _fig_html(fig):
    fig_html = fig.to_html(full_html=False, include_plotlyjs=False)
    return fig_html
    
class HTMLReport:
    def __init__(self, input_data, filename='/usr2/people/macgregor/report.html'):
        self.input_data = input_data
        self.filename = filename
        self._jupyter_cleanup = True  # Flag to prevent Jupyter HTML wrapping

    def write_report(self):
        for section, figures in self.input_data.items():
            html_content.append(f'<div class="section"><h2>{section}</h2>')
            for idx, fig_data in enumerate(figures, 1):
                try:
                    fig_html = _fig_html(fig_data['figure'])
                    html_content.append(f'<div class="fig-container">{fig_html}</div>')
                    print(f'<div class="fig-container">{fig_html}</div>')
                except Exception as e:
                    html_content.append(f'<div style="color: red">Error: {str(e)}</div>')
            html_content.append('</div>')

        html_content += ['</body>', '</html>']

        with open(self.filename, 'w') as f:
            f.write(html_content)
        print(f"Report generated successfully at {self.filename}")


