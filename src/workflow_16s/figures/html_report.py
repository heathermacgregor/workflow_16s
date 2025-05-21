# ===================================== IMPORTS ====================================== #

import os
from pathlib import Path
from typing import Dict, Union
from plotly.graph_objs import Figure

# ==================================== FUNCTIONS ===================================== #

class HTMLReportWriter:
    def __init__(self, input_data, filename='my_report.html'):
        self.input_data = input_data
        self.filename = filename

    def write_report(self):
        html_content = []
        html_content.append('''<!DOCTYPE html>
<html>
<head>
    <title>Data Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 2em; }
        .section { margin-bottom: 3em; padding-bottom: 2em; }
        .plotly-graph-div { width: 100% !important; height: 600px !important; }
        .figure-container { margin: 1.5em 0; padding: 1em; background: #fff; }
    </style>
</head>
<body>
    <h1>Data Analysis Report</h1>''')

        for section_title, figures_list in self.input_data.items():
            html_content.append(f'<div class="section"><h2>{section_title}</h2>')
            
            for figure_idx, fig_dict in enumerate(figures_list, 1):
                try:
                    fig = fig_dict['figure']
                    
                    # Add diagnostic metadata
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        hoverlabel=dict(bgcolor="white")
                    )
                    
                    # Force marker visibility
                    fig.update_traces(
                        marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')),
                        selector=dict(mode='markers')
                    )

                    # Generate HTML with proper container sizing
                    fig_html = fig.to_html(
                        full_html=False,
                        include_plotlyjs=False,
                        div_id=f"plot_{section_title}_{figure_idx}",
                        config={'scrollZoom': True}
                    )

                    # Improved HTML splitting
                    div_end = fig_html.find('</div>') + len('</div>')
                    div_part = fig_html[:div_end]
                    script_part = fig_html[div_end:]
                    
                    html_content.append(f'''
                        <div class="figure-container">
                            {div_part}
                            <div style="color: #666; margin-top: 0.5em;">
                                Figure {figure_idx}: {section_title}
                            </div>
                        </div>
                    ''')
                    scripts.append(script_part.strip())

                except Exception as e:
                    error_msg = f'''<div style="color: red; padding: 1em;">
                        Error: {str(e)}<br>
                        Section: {section_title}<br>
                        Figure Index: {figure_idx}
                    </div>'''
                    html_content.append(error_msg)

            html_content.append('</div>')

        html_content.append(f'''
            <script>
                // Force redraw after initial render
                setTimeout(() => {{
                    Plotly.Plots.resizeAll();
                }}, 100);
                
                // Add error reporting to console
                window.addEventListener('error', (e) => {{
                    console.error('Plotly Error:', e.error);
                }});
            </script>
            {'\n'.join(scripts)}
            </body></html>
        ''')

        with open(self.filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(html_content))
