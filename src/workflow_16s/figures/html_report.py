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
        """
        Generates an HTML report with sections and embedded Plotly figures
        """
        html_content = []
        scripts = []  # INITIALIZE SCRIPTS LIST HERE

        # HTML header with Plotly JS and basic styling
        html_content.append('''<!DOCTYPE html>
<html>
<head>
    <title>Data Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 2em; }
        .section { margin-bottom: 3em; border-bottom: 1px solid #ccc; padding-bottom: 2em; }
        h1 { color: #333; }
        h2 { color: #666; margin-top: 1.5em; }
        .figure-container { margin: 1.5em 0; padding: 1em; background: #f9f9f9; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>Data Analysis Report</h1>''')

        # Process each section
        for section_title, figures_list in self.input_data.items():
            html_content.append(f'<div class="section"><h2>{section_title}</h2>')
            
            for figure_idx, fig_dict in enumerate(figures_list, 1):
                try:
                    fig = fig_dict['figure']
                    fig_html = fig.to_html(
                        full_html=False,
                        include_plotlyjs=False,
                        div_id=f"plot_{section_title}_{figure_idx}"
                    )
                    
                    # Split into div and script parts
                    div_end = fig_html.find('</div>') + len('</div>')
                    div_part = fig_html[:div_end]
                    script_part = fig_html[div_end:]
                    
                    html_content.append(f'<div class="figure-container">{div_part}</div>')
                    scripts.append(script_part.strip())  # ADD TO SCRIPTS LIST
                
                except Exception as e:
                    error_msg = f'''<div style="color: red; padding: 1em;">
                        Error in {section_title} figure {figure_idx}: {str(e)}
                    </div>'''
                    html_content.append(error_msg)
            
            html_content.append('</div>')  # Close section div

        # Add all scripts at the end
        html_content.append('\n'.join(scripts))
        html_content.append('''</body></html>''')

        # Write to file
        with open(self.filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(html_content))
