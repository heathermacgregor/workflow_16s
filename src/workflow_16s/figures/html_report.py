import base64
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dominate import document
from dominate.tags import *
from dominate.util import raw

class AmpliconReportGenerator:
    """
    Generates an interactive HTML report from an AmpliconData object.
    """
    
    def __init__(self, amplicon_data, output_path: str = "amplicon_report.html"):
        """
        Initialize the report generator.
        
        Args:
            amplicon_data: The AmpliconData object containing analysis results
            output_path: Path to save the HTML report
        """
        self.data = amplicon_data
        self.output_path = Path(output_path)
        self.figures = {}
        
    def generate_report(self):
        """Generate the complete HTML report."""
        # Create document
        doc = document(title='16S Amplicon Analysis Report')
        
        # Add CSS styling
        with doc.head:
            style("""
                body {
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }
                h1, h2, h3 {
                    color: #2c3e50;
                }
                .section {
                    margin-bottom: 30px;
                    padding: 15px;
                    background-color: #f9f9f9;
                    border-radius: 5px;
                }
                .dropdown {
                    margin-bottom: 15px;
                }
                select {
                    padding: 8px;
                    border-radius: 4px;
                    border: 1px solid #ddd;
                    font-size: 16px;
                }
                .figure-container {
                    margin: 20px 0;
                    text-align: center;
                }
                .figure-title {
                    font-weight: bold;
                    margin-bottom: 10px;
                }
                .table-container {
                    overflow-x: auto;
                    margin: 20px 0;
                }
                table {
                    border-collapse: collapse;
                    width: 100%;
                    margin-bottom: 20px;
                }
                th, td {
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }
                th {
                    background-color: #f2f2f2;
                }
                tr:nth-child(even) {
                    background-color: #f9f9f9;
                }
                .nav {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 10px;
                    margin-bottom: 20px;
                    padding-bottom: 10px;
                    border-bottom: 1px solid #eee;
                }
                .nav a {
                    padding: 8px 15px;
                    background-color: #e0e0e0;
                    border-radius: 4px;
                    text-decoration: none;
                    color: #333;
                }
                .nav a:hover {
                    background-color: #d0d0d0;
                }
                .tab-content {
                    display: none;
                }
                .tab-content.active {
                    display: block;
                }
            """)
            
            # JavaScript for interactivity
            script("""
                function showSection(sectionId) {
                    // Hide all sections
                    document.querySelectorAll('.tab-content').forEach(section => {
                        section.classList.remove('active');
                    });
                    
                    // Show selected section
                    document.getElementById(sectionId).classList.add('active');
                    
                    // Update nav links
                    document.querySelectorAll('.nav a').forEach(link => {
                        if (link.getAttribute('onclick').includes(sectionId)) {
                            link.style.backgroundColor = '#d0d0d0';
                        } else {
                            link.style.backgroundColor = '#e0e0e0';
                        }
                    });
                }
                
                function updateFigure(selectElement, figureContainerId) {
                    const selectedValue = selectElement.value;
                    const figures = JSON.parse(selectElement.getAttribute('data-figures'));
                    
                    // Hide all figures in this container
                    document.querySelectorAll(`#${figureContainerId} .figure`).forEach(fig => {
                        fig.style.display = 'none';
                    });
                    
                    // Show selected figure
                    const selectedFig = document.getElementById(`figure-${selectedValue}`);
                    if (selectedFig) {
                        selectedFig.style.display = 'block';
                    }
                }
                
                // Show first section by default
                document.addEventListener('DOMContentLoaded', function() {
                    showSection('overview');
                });
            """)
        
        # Navigation menu
        with doc:
            with div(cls="nav"):
                a("Overview", onclick="showSection('overview')")
                a("Sample Maps", onclick="showSection('maps')")
                a("Alpha Diversity", onclick="showSection('alpha')")
                a("Statistical Tests", onclick="showSection('stats')")
                a("Ordination", onclick="showSection('ordination')")
                a("Machine Learning", onclick="showSection('ml')")
                a("Top Features", onclick="showSection('features')")
            
            # Overview section
            with div(id="overview", cls="tab-content"):
                self._add_overview_section()
            
            # Sample maps section
            with div(id="maps", cls="tab-content"):
                self._add_sample_maps_section()
            
            # Alpha diversity section
            with div(id="alpha", cls="tab-content"):
                self._add_alpha_diversity_section()
            
            # Statistical tests section
            with div(id="stats", cls="tab-content"):
                self._add_statistical_tests_section()
            
            # Ordination section
            with div(id="ordination", cls="tab-content"):
                self._add_ordination_section()
            
            # Machine learning section
            with div(id="ml", cls="tab-content"):
                self._add_ml_section()
            
            # Top features section
            with div(id="features", cls="tab-content"):
                self._add_top_features_section()
        
        # Save the report
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, 'w', encoding='utf-8') as f:
            f.write(doc.render())
            
        print(f"Report generated at: {self.output_path}")
    
    def _add_overview_section(self):
        """Add overview section to the report."""
        h1("16S Amplicon Analysis Report")
        hr()
        
        h2("Analysis Overview")
        
        # Basic info
        with div(cls="section"):
            h3("Dataset Information")
            p(f"Analysis mode: {self.data.mode}")
            p(f"Number of samples: {len(self.data.meta)}")
            
            if hasattr(self.data, 'table'):
                p(f"Number of features: {self.data.table.shape[0]}")
            
            # Show metadata columns
            with details():
                summary("Metadata Columns")
                ul([li(col) for col in self.data.meta.columns])
        
        # Analysis summary
        with div(cls="section"):
            h3("Analysis Summary")
            
            # Statistical tests
            if self.data.stats:
                p("Statistical tests performed:")
                ul([li(f"{table_type} ({level}): {', '.join(tests.keys())}")
                   for table_type, levels in self.data.stats.items()
                   for level, tests in levels.items()])
            
            # Ordination methods
            if self.data.ordination:
                p("Ordination methods performed:")
                ul([li(f"{table_type} ({level}): {', '.join(methods.keys())}")
                   for table_type, levels in self.data.ordination.items()
                   for level, methods in levels.items()])
            
            # Top features
            if self.data.top_contaminated_features or self.data.top_pristine_features:
                p("Significant features identified:")
                ul([
                    li(f"{len(self.data.top_contaminated_features)} features more abundant in contaminated samples"),
                    li(f"{len(self.data.top_pristine_features)} features more abundant in pristine samples")
                ])
    
    def _add_sample_maps_section(self):
        """Add sample maps section to the report."""
        h2("Sample Maps")
        hr()
        
        if not self.data.maps:
            p("No sample maps were generated in this analysis.")
            return
            
        with div(cls="section"):
            h3("Geographic Distribution of Samples")
            
            # Create dropdown for map selection
            map_options = list(self.data.maps.keys())
            with div(cls="dropdown"):
                label("Select coloring variable:", for_="map-select")
                select(
                    id="map-select",
                    onchange="updateFigure(this, 'map-figures')",
                    *[option(col, value=col) for col in map_options],
                    data_figures=map_options
                )
            
            # Container for map figures
            with div(id="map-figures"):
                for i, (col, fig) in enumerate(self.data.maps.items()):
                    fig_div = div(
                        id=f"figure-{col}",
                        cls="figure",
                        style="display: none;" if i > 0 else ""
                    )
                    with fig_div:
                        if hasattr(fig, 'to_html'):
                            raw(fig.to_html(full_html=False))
                        else:
                            p("Figure display not available")
    
    def _add_alpha_diversity_section(self):
        """Add alpha diversity section to the report."""
        h2("Alpha Diversity Analysis")
        hr()
        
        if not self.data.alpha_diversity:
            p("No alpha diversity analysis was performed.")
            return
            
        with div(cls="section"):
            h3("Alpha Diversity Results")
            
            # Create dropdowns for table type and level selection
            table_types = list(self.data.alpha_diversity.keys())
            
            with div(cls="dropdown"):
                label("Select table type:", for_="alpha-table-select")
                select(
                    id="alpha-table-select",
                    onchange="updateAlphaLevelDropdown(this.value)",
                    *[option(tt.replace('_', ' ').title(), value=tt) for tt in table_types]
                )
            
            with div(cls="dropdown"):
                label("Select taxonomic level:", for_="alpha-level-select")
                select(
                    id="alpha-level-select",
                    onchange="updateAlphaFigures(this.value)",
                )
            
            # Container for alpha diversity figures
            with div(id="alpha-figures"):
                pass
            
            # JavaScript for alpha diversity dropdowns
            script("""
                // Initialize level dropdown based on first table type
                function initAlphaDropdowns() {
                    const tableTypes = """ + str(table_types) + """;
                    if (tableTypes.length > 0) {
                        updateAlphaLevelDropdown(tableTypes[0]);
                    }
                }
                
                // Update level dropdown based on selected table type
                function updateAlphaLevelDropdown(tableType) {
                    const levels = """ + str({tt: list(levels.keys()) for tt, levels in self.data.alpha_diversity.items()}) + """[tableType] || [];
                    const levelSelect = document.getElementById('alpha-level-select');
                    
                    // Clear existing options
                    levelSelect.innerHTML = '';
                    
                    // Add new options
                    levels.forEach(level => {
                        const opt = document.createElement('option');
                        opt.value = level;
                        opt.textContent = level.charAt(0).toUpperCase() + level.slice(1);
                        levelSelect.appendChild(opt);
                    });
                    
                    // Update figures
                    if (levels.length > 0) {
                        updateAlphaFigures(levels[0]);
                    }
                }
                
                // Update displayed figures based on selected level
                function updateAlphaFigures(level) {
                    const tableType = document.getElementById('alpha-table-select').value;
                    const container = document.getElementById('alpha-figures');
                    
                    // Clear existing content
                    container.innerHTML = '';
                    
                    // Get the alpha data for this combination
                    const alphaData = """ + str({
                        (tt, lvl): data 
                        for tt, levels in self.data.alpha_diversity.items() 
                        for lvl, data in levels.items()
                    }) + """[[tableType, level]];
                    
                    if (!alphaData) return;
                    
                    // Add metric dropdown if we have multiple metrics
                    if (alphaData.figures && Object.keys(alphaData.figures).length > 1) {
                        const metrics = Object.keys(alphaData.figures).filter(k => k !== 'summary' && k !== 'correlations');
                        if (metrics.length > 1) {
                            const metricSelect = document.createElement('select');
                            metricSelect.id = 'alpha-metric-select';
                            metricSelect.onchange = function() {
                                updateAlphaMetricFigures(this.value);
                            };
                            
                            metrics.forEach(metric => {
                                const opt = document.createElement('option');
                                opt.value = metric;
                                opt.textContent = metric.replace('_', ' ').replace(/(^|\s)\S/g, l => l.toUpperCase());
                                metricSelect.appendChild(opt);
                            });
                            
                            const label = document.createElement('label');
                            label.htmlFor = 'alpha-metric-select';
                            label.textContent = 'Select alpha metric:';
                            
                            const dropdown = document.createElement('div');
                            dropdown.className = 'dropdown';
                            dropdown.appendChild(label);
                            dropdown.appendChild(metricSelect);
                            container.appendChild(dropdown);
                            
                            // Add the first metric figure by default
                            if (metrics.length > 0) {
                                updateAlphaMetricFigures(metrics[0]);
                            }
                        }
                    }
                    
                    // Add summary figure if available
                    if (alphaData.figures && alphaData.figures.summary) {
                        const summaryDiv = document.createElement('div');
                        summaryDiv.className = 'figure-container';
                        
                        const title = document.createElement('div');
                        title.className = 'figure-title';
                        title.textContent = 'Alpha Diversity Statistical Summary';
                        summaryDiv.appendChild(title);
                        
                        const figDiv = document.createElement('div');
                        figDiv.innerHTML = """ + str({
                            (tt, lvl): data['figures']['summary'].to_html(full_html=False) 
                            if 'summary' in data.get('figures', {}) and hasattr(data['figures']['summary'], 'to_html')
                            else "Figure not available"
                            for tt, levels in self.data.alpha_diversity.items() 
                            for lvl, data in levels.items()
                        }) + """[[tableType, level]];
                        summaryDiv.appendChild(figDiv);
                        
                        container.appendChild(summaryDiv);
                    }
                    
                    // Add correlation figures if available
                    if (alphaData.figures && alphaData.figures.correlations) {
                        const corrDiv = document.createElement('div');
                        corrDiv.className = 'figure-container';
                        
                        const title = document.createElement('div');
                        title.className = 'figure-title';
                        title.textContent = 'Alpha Diversity Correlations';
                        corrDiv.appendChild(title);
                        
                        const figDiv = document.createElement('div');
                        figDiv.innerHTML = """ + str({
                            (tt, lvl): data['figures']['correlations'].to_html(full_html=False) 
                            if 'correlations' in data.get('figures', {}) and hasattr(data['figures']['correlations'], 'to_html')
                            else "Figure not available"
                            for tt, levels in self.data.alpha_diversity.items() 
                            for lvl, data in levels.items()
                        }) + """[[tableType, level]];
                        corrDiv.appendChild(figDiv);
                        
                        container.appendChild(corrDiv);
                    }
                }
                
                // Update metric-specific figures
                function updateAlphaMetricFigures(metric) {
                    const tableType = document.getElementById('alpha-table-select').value;
                    const level = document.getElementById('alpha-level-select').value;
                    const container = document.getElementById('alpha-figures');
                    
                    // Find or create metric figure container
                    let metricContainer = document.getElementById('alpha-metric-figure');
                    if (!metricContainer) {
                        metricContainer = document.createElement('div');
                        metricContainer.id = 'alpha-metric-figure';
                        metricContainer.className = 'figure-container';
                        container.appendChild(metricContainer);
                    }
                    
                    // Clear existing content
                    metricContainer.innerHTML = '';
                    
                    // Add title
                    const title = document.createElement('div');
                    title.className = 'figure-title';
                    title.textContent = metric.replace('_', ' ').replace(/(^|\s)\S/g, l => l.toUpperCase()) + ' Diversity';
                    metricContainer.appendChild(title);
                    
                    // Add figure
                    const figDiv = document.createElement('div');
                    figDiv.innerHTML = """ + str({
                        (tt, lvl, m): data['figures'][m].to_html(full_html=False) 
                        if m in data.get('figures', {}) and hasattr(data['figures'][m], 'to_html')
                        else "Figure not available"
                        for tt, levels in self.data.alpha_diversity.items() 
                        for lvl, data in levels.items()
                        for m in data.get('figures', {}).keys()
                        if m not in ['summary', 'correlations']
                    }) + """[[tableType, level, metric]];
                    metricContainer.appendChild(figDiv);
                }
                
                // Initialize on load
                document.addEventListener('DOMContentLoaded', initAlphaDropdowns);
            """)
    
    def _add_statistical_tests_section(self):
        """Add statistical tests section to the report."""
        h2("Statistical Test Results")
        hr()
        
        if not self.data.stats:
            p("No statistical tests were performed.")
            return
            
        with div(cls="section"):
            h3("Differential Abundance Analysis")
            
            # Create dropdowns for table type and level selection
            table_types = list(self.data.stats.keys())
            
            with div(cls="dropdown"):
                label("Select table type:", for_="stats-table-select")
                select(
                    id="stats-table-select",
                    onchange="updateStatsLevelDropdown(this.value)",
                    *[option(tt.replace('_', ' ').title(), value=tt) for tt in table_types]
                )
            
            with div(cls="dropdown"):
                label("Select taxonomic level:", for_="stats-level-select")
                select(
                    id="stats-level-select",
                    onchange="updateStatsTestDropdown(this.value)",
                )
            
            with div(cls="dropdown"):
                label("Select statistical test:", for_="stats-test-select")
                select(
                    id="stats-test-select",
                    onchange="updateStatsResults(this.value)",
                )
            
            # Container for stats results
            with div(id="stats-results"):
                pass
            
            # JavaScript for stats dropdowns
            script("""
                // Initialize dropdowns based on first table type
                function initStatsDropdowns() {
                    const tableTypes = """ + str(table_types) + """;
                    if (tableTypes.length > 0) {
                        updateStatsLevelDropdown(tableTypes[0]);
                    }
                }
                
                // Update level dropdown based on selected table type
                function updateStatsLevelDropdown(tableType) {
                    const levels = """ + str({tt: list(levels.keys()) for tt, levels in self.data.stats.items()}) + """[tableType] || [];
                    const levelSelect = document.getElementById('stats-level-select');
                    
                    // Clear existing options
                    levelSelect.innerHTML = '';
                    
                    // Add new options
                    levels.forEach(level => {
                        const opt = document.createElement('option');
                        opt.value = level;
                        opt.textContent = level.charAt(0).toUpperCase() + level.slice(1);
                        levelSelect.appendChild(opt);
                    });
                    
                    // Update test dropdown
                    if (levels.length > 0) {
                        updateStatsTestDropdown(levels[0]);
                    }
                }
                
                // Update test dropdown based on selected level
                function updateStatsTestDropdown(level) {
                    const tableType = document.getElementById('stats-table-select').value;
                    const tests = """ + str({
                        (tt, lvl): list(tests.keys())
                        for tt, levels in self.data.stats.items()
                        for lvl, tests in levels.items()
                    }) + """[[tableType, level]] || [];
                    const testSelect = document.getElementById('stats-test-select');
                    
                    // Clear existing options
                    testSelect.innerHTML = '';
                    
                    // Add new options
                    tests.forEach(test => {
                        const opt = document.createElement('option');
                        opt.value = test;
                        
                        // Format test names for display
                        let displayName = test;
                        if (test === 'mwub') displayName = 'Mann-Whitney U (Bonferroni)';
                        else if (test === 'kwb') displayName = 'Kruskal-Wallis (Bonferroni)';
                        else if (test === 'fisher') displayName = 'Fisher Exact (Bonferroni)';
                        else if (test === 'ttest') displayName = 'Student t-test';
                        
                        opt.textContent = displayName;
                        testSelect.appendChild(opt);
                    });
                    
                    // Update results
                    if (tests.length > 0) {
                        updateStatsResults(tests[0]);
                    }
                }
                
                // Update displayed results based on selected test
                function updateStatsResults(test) {
                    const tableType = document.getElementById('stats-table-select').value;
                    const level = document.getElementById('stats-level-select').value;
                    const container = document.getElementById('stats-results');
                    
                    // Clear existing content
                    container.innerHTML = '';
                    
                    // Get the stats data for this combination
                    const statsData = """ + str({
                        (tt, lvl, t): data[t]
                        for tt, levels in self.data.stats.items()
                        for lvl, data in levels.items()
                        for t in data.keys()
                    }) + """[[tableType, level, test]];
                    
                    if (!statsData || !(statsData instanceof Object)) return;
                    
                    // Create summary stats
                    const summaryDiv = document.createElement('div');
                    summaryDiv.className = 'section';
                    
                    const sigCount = statsData[statsData['p_value'] < 0.05].shape[0];
                    const totalCount = statsData.shape[0];
                    
                    const summaryTitle = document.createElement('h4');
                    summaryTitle.textContent = 'Summary Statistics';
                    summaryDiv.appendChild(summaryTitle);
                    
                    const summaryStats = document.createElement('p');
                    summaryStats.innerHTML = `Significant features (p < 0.05): <strong>${sigCount} / ${totalCount}</strong> (${(sigCount/totalCount*100).toFixed(1)}%)`;
                    summaryDiv.appendChild(summaryStats);
                    
                    container.appendChild(summaryDiv);
                    
                    // Create table of top features
                    const tableDiv = document.createElement('div');
                    tableDiv.className = 'section';
                    
                    const tableTitle = document.createElement('h4');
                    tableTitle.textContent = 'Top Significant Features (p < 0.05)';
                    tableDiv.appendChild(tableTitle);
                    
                    // Sort by p-value and get top 20
                    const topFeatures = statsData[statsData['p_value'] < 0.05].sort_values('p_value').head(20);
                    
                    if (topFeatures.empty) {
                        const noSig = document.createElement('p');
                        noSig.textContent = 'No significant features found at p < 0.05';
                        tableDiv.appendChild(noSig);
                    } else {
                        const tableContainer = document.createElement('div');
                        tableContainer.className = 'table-container';
                        
                        // Convert DataFrame to HTML
                        const tableHtml = """ + str({
                            (tt, lvl, t): data[t].sort_values('p_value').head(20).to_html(
                                classes='dataframe',
                                border=0,
                                float_format=lambda x: f'{x:.3e}' if isinstance(x, (int, float)) else str(x)
                            )
                            for tt, levels in self.data.stats.items()
                            for lvl, data in levels.items()
                            for t in data.keys()
                        }) + """[[tableType, level, test]];
                        
                        tableContainer.innerHTML = tableHtml;
                        tableDiv.appendChild(tableContainer);
                    }
                    
                    container.appendChild(tableDiv);
                }
                
                // Initialize on load
                document.addEventListener('DOMContentLoaded', initStatsDropdowns);
            """)
    
    def _add_ordination_section(self):
        """Add ordination section to the report."""
        h2("Beta Diversity Ordination")
        hr()
        
        if not self.data.ordination:
            p("No ordination analyses were performed.")
            return
            
        with div(cls="section"):
            h3("Ordination Results")
            
            # Create dropdowns for table type and level selection
            table_types = list(self.data.ordination.keys())
            
            with div(cls="dropdown"):
                label("Select table type:", for_="ord-table-select")
                select(
                    id="ord-table-select",
                    onchange="updateOrdLevelDropdown(this.value)",
                    *[option(tt.replace('_', ' ').title(), value=tt) for tt in table_types]
                )
            
            with div(cls="dropdown"):
                label("Select taxonomic level:", for_="ord-level-select")
                select(
                    id="ord-level-select",
                    onchange="updateOrdMethodDropdown(this.value)",
                )
            
            with div(cls="dropdown"):
                label("Select ordination method:", for_="ord-method-select")
                select(
                    id="ord-method-select",
                    onchange="updateOrdColorDropdown(this.value)",
                )
            
            with div(cls="dropdown"):
                label("Select coloring variable:", for_="ord-color-select")
                select(
                    id="ord-color-select",
                    onchange="updateOrdFigure()",
                )
            
            # Container for ordination figures
            with div(id="ord-figures"):
                pass
            
            # JavaScript for ordination dropdowns
            script("""
                // Initialize dropdowns based on first table type
                function initOrdDropdowns() {
                    const tableTypes = """ + str(table_types) + """;
                    if (tableTypes.length > 0) {
                        updateOrdLevelDropdown(tableTypes[0]);
                    }
                }
                
                // Update level dropdown based on selected table type
                function updateOrdLevelDropdown(tableType) {
                    const levels = """ + str({tt: list(levels.keys()) for tt, levels in self.data.ordination.items()}) + """[tableType] || [];
                    const levelSelect = document.getElementById('ord-level-select');
                    
                    // Clear existing options
                    levelSelect.innerHTML = '';
                    
                    // Add new options
                    levels.forEach(level => {
                        const opt = document.createElement('option');
                        opt.value = level;
                        opt.textContent = level.charAt(0).toUpperCase() + level.slice(1);
                        levelSelect.appendChild(opt);
                    });
                    
                    // Update method dropdown
                    if (levels.length > 0) {
                        updateOrdMethodDropdown(levels[0]);
                    }
                }
                
                // Update method dropdown based on selected level
                function updateOrdMethodDropdown(level) {
                    const tableType = document.getElementById('ord-table-select').value;
                    const methods = """ + str({
                        (tt, lvl): list(methods.keys())
                        for tt, levels in self.data.ordination.items()
                        for lvl, methods in levels.items()
                    }) + """[[tableType, level]] || [];
                    const methodSelect = document.getElementById('ord-method-select');
                    
                    // Clear existing options
                    methodSelect.innerHTML = '';
                    
                    // Add new options
                    methods.forEach(method => {
                        const opt = document.createElement('option');
                        opt.value = method;
                        
                        // Format method names for display
                        let displayName = method;
                        if (method === 'pca') displayName = 'PCA';
                        else if (method === 'pcoa') displayName = 'PCoA';
                        else if (method === 'tsne') displayName = 't-SNE';
                        else if (method === 'umap') displayName = 'UMAP';
                        
                        opt.textContent = displayName;
                        methodSelect.appendChild(opt);
                    });
                    
                    // Update color dropdown
                    if (methods.length > 0) {
                        updateOrdColorDropdown(methods[0]);
                    }
                }
                
                // Update color dropdown based on selected method
                function updateOrdColorDropdown(method) {
                    const tableType = document.getElementById('ord-table-select').value;
                    const level = document.getElementById('ord-level-select').value;
                    const colorVars = """ + str({
                        (tt, lvl, m): list(data['figures'].keys())
                        for tt, levels in self.data.ordination.items()
                        for lvl, data in levels.items()
                        for m in data.keys()
                        if 'figures' in data
                    }) + """[[tableType, level, method]] || [];
                    const colorSelect = document.getElementById('ord-color-select');
                    
                    // Clear existing options
                    colorSelect.innerHTML = '';
                    
                    // Add new options
                    colorVars.forEach(var => {
                        const opt = document.createElement('option');
                        opt.value = var;
                        opt.textContent = var.replace('_', ' ').replace(/(^|\s)\S/g, l => l.toUpperCase());
                        colorSelect.appendChild(opt);
                    });
                    
                    // Update figure
                    if (colorVars.length > 0) {
                        updateOrdFigure();
                    }
                }
                
                // Update displayed figure
                function updateOrdFigure() {
                    const tableType = document.getElementById('ord-table-select').value;
                    const level = document.getElementById('ord-level-select').value;
                    const method = document.getElementById('ord-method-select').value;
                    const colorVar = document.getElementById('ord-color-select').value;
                    const container = document.getElementById('ord-figures');
                    
                    // Clear existing content
                    container.innerHTML = '';
                    
                    // Get the figure for this combination
                    const figureHtml = """ + str({
                        (tt, lvl, m, c): data['figures'][c].to_html(full_html=False)
                        if c in data.get('figures', {}) and hasattr(data['figures'][c], 'to_html')
                        else "Figure not available"
                        for tt, levels in self.data.ordination.items()
                        for lvl, data in levels.items()
                        for m in data.keys()
                        for c in data.get('figures', {}).keys()
                    }) + """[[tableType, level, method, colorVar]];
                    
                    if (!figureHtml) return;
                    
                    const figureContainer = document.createElement('div');
                    figureContainer.className = 'figure-container';
                    
                    const title = document.createElement('div');
                    title.className = 'figure-title';
                    title.textContent = `${method.toUpperCase()} colored by ${colorVar.replace('_', ' ')}`;
                    figureContainer.appendChild(title);
                    
                    const figDiv = document.createElement('div');
                    figDiv.innerHTML = figureHtml;
                    figureContainer.appendChild(figDiv);
                    
                    container.appendChild(figureContainer);
                }
                
                // Initialize on load
                document.addEventListener('DOMContentLoaded', initOrdDropdowns);
            """)
    
    def _add_ml_section(self):
        """Add machine learning section to the report."""
        h2("Machine Learning Feature Selection")
        hr()
        
        if not self.data.models:
            p("No machine learning analysis was performed.")
            return
            
        with div(cls="section"):
            h3("Feature Selection Results")
            
            # Create dropdowns for table type and level selection
            table_types = list(self.data.models.keys())
            
            with div(cls="dropdown"):
                label("Select table type:", for_="ml-table-select")
                select(
                    id="ml-table-select",
                    onchange="updateMlLevelDropdown(this.value)",
                    *[option(tt.replace('_', ' ').title(), value=tt) for tt in table_types]
                )
            
            with div(cls="dropdown"):
                label("Select taxonomic level:", for_="ml-level-select")
                select(
                    id="ml-level-select",
                    onchange="updateMlMethodDropdown(this.value)",
                )
            
            with div(cls="dropdown"):
                label("Select feature selection method:", for_="ml-method-select")
                select(
                    id="ml-method-select",
                    onchange="updateMlResults()",
                )
            
            # Container for ML results
            with div(id="ml-results"):
                pass
            
            # JavaScript for ML dropdowns
            script("""
                // Initialize dropdowns based on first table type
                function initMlDropdowns() {
                    const tableTypes = """ + str(table_types) + """;
                    if (tableTypes.length > 0) {
                        updateMlLevelDropdown(tableTypes[0]);
                    }
                }
                
                // Update level dropdown based on selected table type
                function updateMlLevelDropdown(tableType) {
                    const levels = """ + str({tt: list(levels.keys()) for tt, levels in self.data.models.items()}) + """[tableType] || [];
                    const levelSelect = document.getElementById('ml-level-select');
                    
                    // Clear existing options
                    levelSelect.innerHTML = '';
                    
                    // Add new options
                    levels.forEach(level => {
                        const opt = document.createElement('option');
                        opt.value = level;
                        opt.textContent = level.charAt(0).toUpperCase() + level.slice(1);
                        levelSelect.appendChild(opt);
                    });
                    
                    // Update method dropdown
                    if (levels.length > 0) {
                        updateMlMethodDropdown(levels[0]);
                    }
                }
                
                // Update method dropdown based on selected level
                function updateMlMethodDropdown(level) {
                    const tableType = document.getElementById('ml-table-select').value;
                    const methods = """ + str({
                        (tt, lvl): list(methods.keys())
                        for tt, levels in self.data.models.items()
                        for lvl, methods in levels.items()
                    }) + """[[tableType, level]] || [];
                    const methodSelect = document.getElementById('ml-method-select');
                    
                    // Clear existing options
                    methodSelect.innerHTML = '';
                    
                    // Add new options
                    methods.forEach(method => {
                        const opt = document.createElement('option');
                        opt.value = method;
                        opt.textContent = method.replace('_', ' ').replace(/(^|\s)\S/g, l => l.toUpperCase());
                        methodSelect.appendChild(opt);
                    });
                    
                    // Update results
                    if (methods.length > 0) {
                        updateMlResults();
                    }
                }
                
                // Update displayed results
                function updateMlResults() {
                    const tableType = document.getElementById('ml-table-select').value;
                    const level = document.getElementById('ml-level-select').value;
                    const method = document.getElementById('ml-method-select').value;
                    const container = document.getElementById('ml-results');
                    
                    // Clear existing content
                    container.innerHTML = '';
                    
                    // Get the ML data for this combination
                    const mlData = """ + str({
                        (tt, lvl, m): data[m]
                        for tt, levels in self.data.models.items()
                        for lvl, data in levels.items()
                        for m in data.keys()
                    }) + """[[tableType, level, method]];
                    
                    if (!mlData) return;
                    
                    // Create evaluation metrics section
                    const metricsDiv = document.createElement('div');
                    metricsDiv.className = 'section';
                    
                    const metricsTitle = document.createElement('h4');
                    metricsTitle.textContent = 'Model Evaluation Metrics';
                    metricsDiv.appendChild(metricsTitle);
                    
                    // Add metrics table
                    if (mlData.metrics) {
                        const metricsTable = document.createElement('table');
                        
                        // Create header
                        const thead = document.createElement('thead');
                        const headerRow = document.createElement('tr');
                        ['Metric', 'Value'].forEach(text => {
                            const th = document.createElement('th');
                            th.textContent = text;
                            headerRow.appendChild(th);
                        });
                        thead.appendChild(headerRow);
                        metricsTable.appendChild(thead);
                        
                        // Create body
                        const tbody = document.createElement('tbody');
                        Object.entries(mlData.metrics).forEach(([metric, value]) => {
                            const row = document.createElement('tr');
                            
                            const metricCell = document.createElement('td');
                            metricCell.textContent = metric.replace('_', ' ').replace(/(^|\s)\S/g, l => l.toUpperCase());
                            row.appendChild(metricCell);
                            
                            const valueCell = document.createElement('td');
                            valueCell.textContent = typeof value === 'number' ? value.toFixed(3) : value;
                            row.appendChild(valueCell);
                            
                            tbody.appendChild(row);
                        });
                        metricsTable.appendChild(tbody);
                        
                        metricsDiv.appendChild(metricsTable);
                    } else {
                        const noMetrics = document.createElement('p');
                        noMetrics.textContent = 'No evaluation metrics available';
                        metricsDiv.appendChild(noMetrics);
                    }
                    
                    container.appendChild(metricsDiv);
                    
                    // Add feature importance plot if available
                    if (mlData.figures && mlData.figures.shap_summary_bar) {
                        const shapDiv = document.createElement('div');
                        shapDiv.className = 'section';
                        
                        const shapTitle = document.createElement('h4');
                        shapTitle.textContent = 'Feature Importance (SHAP values)';
                        shapDiv.appendChild(shapTitle);
                        
                        const figDiv = document.createElement('div');
                        figDiv.className = 'figure-container';
                        figDiv.innerHTML = """ + str({
                            (tt, lvl, m): data[m]['figures']['shap_summary_bar'].to_html(full_html=False)
                            if 'figures' in data[m] and 'shap_summary_bar' in data[m]['figures'] and hasattr(data[m]['figures']['shap_summary_bar'], 'to_html')
                            else "Figure not available"
                            for tt, levels in self.data.models.items()
                            for lvl, data in levels.items()
                            for m in data.keys()
                        }) + """[[tableType, level, method]];
                        shapDiv.appendChild(figDiv);
                        
                        container.appendChild(shapDiv);
                    }
                    
                    // Add top features table
                    const featuresDiv = document.createElement('div');
                    featuresDiv.className = 'section';
                    
                    const featuresTitle = document.createElement('h4');
                    featuresTitle.textContent = 'Top Selected Features';
                    featuresDiv.appendChild(featuresTitle);
                    
                    if (mlData.top_features && mlData.top_features.length > 0) {
                        const tableContainer = document.createElement('div');
                        tableContainer.className = 'table-container';
                        
                        // Create table
                        const featuresTable = document.createElement('table');
                        
                        // Create header
                        const thead = document.createElement('thead');
                        const headerRow = document.createElement('tr');
                        ['Feature', 'Importance'].forEach(text => {
                            const th = document.createElement('th');
                            th.textContent = text;
                            headerRow.appendChild(th);
                        });
                        thead.appendChild(headerRow);
                        featuresTable.appendChild(thead);
                        
                        // Create body
                        const tbody = document.createElement('tbody');
                        mlData.top_features.forEach(feat => {
                            const row = document.createElement('tr');
                            
                            const featCell = document.createElement('td');
                            featCell.textContent = feat.feature || feat.name || 'N/A';
                            row.appendChild(featCell);
                            
                            const impCell = document.createElement('td');
                            impCell.textContent = typeof feat.importance === 'number' ? feat.importance.toFixed(4) : 'N/A';
                            row.appendChild(impCell);
                            
                            tbody.appendChild(row);
                        });
                        featuresTable.appendChild(tbody);
                        
                        tableContainer.appendChild(featuresTable);
                        featuresDiv.appendChild(tableContainer);
                    } else {
                        const noFeatures = document.createElement('p');
                        noFeatures.textContent = 'No feature importance data available';
                        featuresDiv.appendChild(noFeatures);
                    }
                    
                    container.appendChild(featuresDiv);
                }
                
                // Initialize on load
                document.addEventListener('DOMContentLoaded', initMlDropdowns);
            """)
    
    def _add_top_features_section(self):
        """Add top features section to the report."""
        h2("Top Differentially Abundant Features")
        hr()
        
        if not (self.data.top_contaminated_features or self.data.top_pristine_features):
            p("No significant features were identified.")
            return
            
        with div(cls="section"):
            h3("Feature Overview")
            
            # Summary stats
            p(f"Found {len(self.data.top_contaminated_features)} features more abundant in contaminated samples")
            p(f"Found {len(self.data.top_pristine_features)} features more abundant in pristine samples")
            
            # Tabs for contaminated vs pristine
            with div(cls="nav"):
                a("Contaminated", onclick="showTopFeaturesTab('contaminated')", style="background-color: #d0d0d0")
                a("Pristine", onclick="showTopFeaturesTab('pristine')")
            
            # Contaminated features tab
            with div(id="top-features-contaminated", cls="tab-content active"):
                self._add_feature_table(self.data.top_contaminated_features, "contaminated")
            
            # Pristine features tab
            with div(id="top-features-pristine", cls="tab-content"):
                self._add_feature_table(self.data.top_pristine_features, "pristine")
            
            # JavaScript for tab switching
            script("""
                function showTopFeaturesTab(tabId) {
                    // Hide all tabs
                    document.querySelectorAll('#top-features-contaminated, #top-features-pristine').forEach(tab => {
                        tab.classList.remove('active');
                    });
                    
                    // Show selected tab
                    document.getElementById(`top-features-${tabId}`).classList.add('active');
                    
                    // Update nav button styles
                    document.querySelectorAll('.nav a').forEach(link => {
                        if (link.getAttribute('onclick') === `showTopFeaturesTab('${tabId}')`) {
                            link.style.backgroundColor = '#d0d0d0';
                        } else {
                            link.style.backgroundColor = '#e0e0e0';
                        }
                    });
                }
            """)
    
    def _add_feature_table(self, features: List[Dict], feature_type: str):
        """Add a table of features with violin plots."""
        if not features:
            p(f"No {feature_type} features found.")
            return
            
        # Create table
        with div(cls="table-container"):
            table(cls="dataframe", border="0"):
                # Table header
                with thead():
                    with tr():
                        th("Rank")
                        th("Feature")
                        th("Taxonomic Level")
                        th("Table Type")
                        th("Test")
                        th("Effect Size")
                        th("p-value")
                        th("FAPROTAX Functions", style="width: 200px")
                        th("Violin Plot")
                
                # Table body
                with tbody():
                    for i, feat in enumerate(features[:50], 1):  # Show top 50
                        with tr():
                            # Rank
                            td(str(i))
                            
                            # Feature name
                            td(feat.get('feature', 'N/A'))
                            
                            # Taxonomic level
                            td(feat.get('level', 'N/A').capitalize())
                            
                            # Table type
                            td(feat.get('table_type', 'N/A').replace('_', ' ').title())
                            
                            # Test
                            td(self._format_test_name(feat.get('test', 'N/A')))
                            
                            # Effect size
                            td(f"{feat.get('effect', 'N/A'):.3f}" if isinstance(feat.get('effect'), (int, float)) else 'N/A')
                            
                            # p-value
                            td(f"{feat.get('p_value', 'N/A'):.3e}" if isinstance(feat.get('p_value'), (int, float)) else 'N/A')
                            
                            # FAPROTAX functions
                            with td():
                                if feat.get('faprotax_functions'):
                                    with details():
                                        summary(f"{len(feat['faprotax_functions'])} functions")
                                        ul([li(func) for func in feat['faprotax_functions']])
                                else:
                                    span("N/A")
                            
                            # Violin plot
                            with td():
                                if feat.get('violin_figure'):
                                    with details():
                                        summary("View Plot")
                                        with div(style="width: 400px; margin-top: 10px;"):
                                            if hasattr(feat['violin_figure'], 'to_html'):
                                                raw(feat['violin_figure'].to_html(full_html=False))
                                            else:
                                                p("Figure display not available")
                                else:
                                    span("N/A")
    
    def _format_test_name(self, test: str) -> str:
        """Format statistical test names for display."""
        test_names = {
            'mwub': 'Mann-Whitney U',
            'kwb': 'Kruskal-Wallis',
            'fisher': 'Fisher Exact',
            'ttest': 't-test'
        }
        return test_names.get(test, test.capitalize())

# Example usage:
# report = AmpliconReportGenerator(amplicon_data, "amplicon_report.html")
# report.generate_report()
