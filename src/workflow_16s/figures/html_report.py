#!/usr/bin/env python3
"""
Generates an HTML report from AmpliconData analysis results.
"""
import os
import shutil
import html
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Union

class ReportGenerator:
    """
    Generates an HTML report from AmpliconData analysis results.
    """
    
    def __init__(self, amplicon_data, output_dir: Union[str, Path]):
        self.data = amplicon_data
        self.output_dir = Path(output_dir)
        self.report_dir = self.output_dir / "html_report"
        self.figures_dir = self.report_dir / "figures"
        self.css_dir = self.report_dir / "css"
        self.html_file = self.report_dir / "index.html"
        
    def generate_report(self) -> None:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(exist_ok=True)
        self.css_dir.mkdir(exist_ok=True)
        self._copy_css_assets()
        html_content = self._generate_html()
        with open(self.html_file, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"Report generated at: {self.html_file}")
    
    def _copy_css_assets(self) -> None:
        css_content = '''
        body { font-family: Arial, sans-serif; }
        .container { max-width: 1200px; margin: 0 auto; }
        .section { margin-bottom: 30px; border: 1px solid #ddd; border-radius: 5px; padding: 15px; }
        .section-title { background-color: #f5f5f5; padding: 10px; cursor: pointer; margin-top: 0; }
        .section-content { padding: 15px; }
        .subsection { margin-bottom: 20px; }
        .subsection-title { color: #2c3e50; border-bottom: 1px solid #eee; padding-bottom: 5px; }
        .figure-container { text-align: center; margin: 15px 0; }
        .figure { max-width: 100%; }
        .figure-caption { font-style: italic; margin-top: 5px; }
        .feature-list { list-style-type: none; padding-left: 0; }
        .feature-item { margin-bottom: 15px; border-left: 3px solid #3498db; padding-left: 10px; }
        .collapsible::after { content: "▼"; float: right; }
        .active::after { content: "▲"; }
        '''
        with open(self.css_dir / "styles.css", "w", encoding="utf-8") as f:
            f.write(css_content)
    
    def _generate_html(self) -> str:
        return f"""<!DOCTYPE html>
<html lang='en'>
<head>
    <meta charset='UTF-8'>
    <meta name='viewport' content='width=device-width, initial-scale=1.0'>
    <title>Microbiome Analysis Report</title>
    <link rel='stylesheet' href='css/styles.css'>
    <script>
        function toggleSection(element) {{
            element.classList.toggle('active');
            const content = element.nextElementSibling;
            content.style.display = content.style.display === 'none' ? 'block' : 'none';
        }}
    </script>
</head>
<body>
    <div class='container'>
        <header>
            <h1>Microbiome Analysis Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </header>
        {self._create_overview_section()}
        {self._create_sample_maps_section()}
        {self._create_alpha_diversity_section()}
        {self._create_statistics_section()}
        {self._create_ordination_section()}
        {self._create_ml_section()}
        {self._create_top_features_section()}
    </div>
</body>
</html>
"""

    def _create_overview_section(self) -> str:
        return f"""<div class='section'>
    <h2 class='section-title collapsible' onclick='toggleSection(this)'>Analysis Overview</h2>
    <div class='section-content'>
        <p><strong>Analysis Mode:</strong> {html.escape(str(self.data.mode))}</p>
        <p><strong>Samples:</strong> {self.data.meta.shape[0]}</p>
        <p><strong>Metadata Columns:</strong> {self.data.meta.shape[1]}</p>
        <p><strong>Group Column:</strong> {html.escape(str(self.data.cfg.get("group_column", "nuclear_contamination_status")))}</p>
    </div>
</div>
"""

    # (TRUNCATED FOR SPACE - continue fixing similar triple-quoted f-strings in remaining methods)

    def _create_top_features_section(self) -> str:
        return ""

    def _copy_figure(self, fig_path: Union[str, Path], subdir: str) -> str:
        if not fig_path:
            return ""
        source_path = Path(fig_path)
        if not source_path.exists():
            return ""
        dest_dir = self.figures_dir / subdir
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / source_path.name
        shutil.copy2(source_path, dest_path)
        return f"figures/{subdir}/{source_path.name}"

    def _create_sample_maps_section(self) -> str:
        """Creates the sample maps section."""
        if not self.data.maps:
            return ""
    
        content = ""
        for col, fig_path in self.data.maps.items():
            dest_path = self._copy_figure(fig_path, "sample_maps")
            if not dest_path:
                continue
    
            safe_col = html.escape(col)
            content += f"""
            <div class='subsection'>
                <h3 class='subsection-title'>Sample Map: {safe_col}</h3>
                <div class='figure-container'>
                    <img src='{html.escape(dest_path)}' alt='Sample Map: {safe_col}' class='figure'>
                </div>
            </div>
            """
    
        return f"""
        <div class='section'>
            <h2 class='section-title collapsible' onclick='toggleSection(this)'>Sample Maps</h2>
            <div class='section-content'>
                {content}
            </div>
        </div>
        """ if content else """"""

    def _create_alpha_diversity_section(self) -> str:
        """Creates the alpha diversity section."""
        if not self.data.alpha_diversity:
            return ''
    
        content = ''
        for table_type, levels in self.data.alpha_diversity.items():
            for level, results in levels.items():
                if not results.get('figures'):
                    continue
    
                safe_table = html.escape(table_type)
                safe_level = html.escape(level)
                section_content = ''
    
                for metric, fig_path in results['figures'].items():
                    if metric in ['summary', 'correlations']:
                        continue
                    dest_path = self._copy_figure(fig_path, f'alpha/{table_type}/{level}')
                    if not dest_path:
                        continue
    
                    safe_metric = html.escape(metric)
                    section_content += f"""
                    <div class='figure-container'>
                        <img src='{html.escape(dest_path)}' alt='{safe_metric} diversity' class='figure'>
                        <p class='figure-caption'>{safe_metric.replace('_', ' ').title()} Diversity</p>
                    </div>
                    """
    
                # Add summary figure
                if 'summary' in results['figures']:
                    fig_path = results['figures']['summary']
                    dest_path = self._copy_figure(fig_path, f'alpha/{table_type}/{level}')
                    if dest_path:
                        section_content += f"""
                        <div class='figure-container'>
                            <img src='{html.escape(dest_path)}' alt='Diversity Summary' class='figure'>
                            <p class='figure-caption'>Statistical Summary</p>
                        </div>
                        """
    
                content += f"""
                <div class='subsection'>
                    <h3 class='subsection-title'>{safe_table.replace('_', ' ').title()} - {safe_level.title()}</h3>
                    {section_content}
                </div>
                """
    
        return f"""
        <div class='section'>
            <h2 class='section-title collapsible' onclick='toggleSection(this)'>Alpha Diversity</h2>
            <div class='section-content'>
                {content}
            </div>
        </div>
        """ if content else ''
