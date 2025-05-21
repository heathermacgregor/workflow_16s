# ===================================== IMPORTS ====================================== #

import os
from pathlib import Path
from typing import Dict, Union
from plotly.graph_objs import Figure

# ==================================== FUNCTIONS ===================================== #

class HTMLReportWriter:
    def __init__(self, input_data, filename='report.html'):
        self.input_data = input_data
        self.filename = filename
        self._jupyter_cleanup = True  # Flag to prevent Jupyter HTML wrapping

    def write_report(self):
        """Generate a clean HTML report immune to Jupyter wrapping"""
        content = [
            '<!DOCTYPE html>',
            '<html>',
            '<head>',
            '<meta charset="UTF-8">',
            '<title>Analysis Report</title>',
            '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>',
            '<style>',
            '  body { padding: 20px; font-family: Arial, sans-serif; }',
            '  .section { margin-bottom: 40px; }',
            '  .plot-container { margin: 20px 0; }',
            '</style>',
            '</head>',
            '<body>',
            '<h1>Data Analysis Report</h1>'
        ]

        for section, figures in self.input_data.items():
            content.append(f'<div class="section"><h2>{section}</h2>')
            for idx, fig_data in enumerate(figures, 1):
                try:
                    fig = fig_data['figure']
                    plot_div = f'plot_{section}_{idx}'.replace(' ', '_')
                    plot_html = fig.to_html(
                        full_html=False,
                        include_plotlyjs=False,
                        div_id=plot_div,
                        config={'responsive': True}
                    )
                    content.append(f'<div class="plot-container">{plot_html}</div>')
                except Exception as e:
                    content.append(f'<div style="color: red">Error: {str(e)}</div>')
            content.append('</div>')

        content += ['</body>', '</html>']

        with open(self.filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content))
