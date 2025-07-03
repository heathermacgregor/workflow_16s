# ... (previous imports remain the same) ...

# =============================================================================
# PUBLIC API - UPDATED TO HIDE LEGENDS
# =============================================================================

def generate_html_report(
    amplicon_data: "AmpliconData",
    output_path: Union[str, Path],
) -> None:
    # ... (same as before until plot_data_json) ...
    
    html = f"""<!DOCTYPE html>
<html>
<head>
  <!-- ... (same head content as before) ... -->
</head>
<body>
  <!-- ... (same body header as before) ... -->

  <script>
    // Store plot data
    const plotData = {plot_data_json};
    
    // Track initialized plots
    const initializedPlots = new Set();
    
    function renderPlot(containerId, plotId) {{
        const container = document.getElementById(containerId);
        if (!container) {{
            console.error('Container not found:', containerId);
            return;
        }}
        
        // Clear previous content
        container.innerHTML = '';
        
        // Create plot div
        const plotDiv = document.createElement('div');
        plotDiv.id = plotId;
        plotDiv.className = 'plot-container';
        container.appendChild(plotDiv);
        
        // Get plot data
        const data = plotData[plotId];
        if (!data) {{
            plotDiv.innerHTML = '<div class="error">Plot data not available</div>';
            return;
        }}
        
        try {{
            // HIDE LEGEND FOR ALL PLOTS
            if (data.layout) {{
                data.layout.showlegend = false;
            }}
            
            // Render the plot
            Plotly.newPlot(plotId, data.data, data.layout)
                .catch(error => {{
                    plotDiv.innerHTML = `<div class="error">Plot error: ${{error}}</div>`;
                    console.error('Plotly error:', error);
                }});
        }} catch (error) {{
            plotDiv.innerHTML = `<div class="error">JS error: ${{error}}</div>`;
            console.error('Rendering error:', error);
        }}
    }}
    
    // ... (rest of the JavaScript remains the same) ...
  </script>
</body>
</html>"""
    output_path.write_text(html, encoding="utf-8")

# =============================================================================
# INTERNAL HELPERS - UPDATED TO HIDE LEGENDS AT GENERATION TIME
# =============================================================================

def _prepare_figures(figures: Dict) -> tuple:
    """Prepare HTML tabs and plot data with legends hidden."""
    if not figures or "map" not in figures:
        return (
            '<div class="error">No sample maps available.</div>',
            '<div class="error">No data</div>',
            {}
        )
    
    maps = [(c, f) for c, f in figures["map"].items() if f][:2]
    if not maps:
        return (
            '<div class="error">No sample maps available.</div>',
            '<div class="error">No data</div>',
            {}
        )
    
    tabs = []
    buttons = []
    plot_data = {}
    
    for i, (col, fig) in enumerate(maps):
        tab_id = f"tab-{i}"
        plot_id = f"plot-{i}"
        
        # Add tab button
        active = "active" if i == 0 else ""
        buttons.append(
            f'<button class="tab-button {active}" data-tab="{tab_id}" '
            f'onclick="showTab(\'{tab_id}\', \'{plot_id}\')">{col}</button>'
        )
        
        # Add tab content
        tabs.append(
            f'<div id="{tab_id}" class="tab-pane" style="display:{"block" if i == 0 else "none"}" '
            f'data-plot-id="{plot_id}">'
            f'<div id="container-{plot_id}" class="plot-container"></div>'
            f'</div>'
        )
        
        # Prepare plot data with legend hidden
        if hasattr(fig, "to_plotly_json"):
            try:
                plot_json = fig.to_plotly_json()
                
                # HIDE LEGEND IN THE LAYOUT
                layout = plot_json.get("layout", {})
                layout["showlegend"] = False  # This hides the legend
                
                plot_data[plot_id] = {
                    "data": plot_json["data"],
                    "layout": layout
                }
            except Exception as e:
                logger.error(f"Error processing {col} figure: {e}")
                plot_data[plot_id] = {"error": str(e)}
        else:
            plot_data[plot_id] = {"error": f"Unsupported figure type: {type(fig)}"}
    
    return "\n".join(tabs), "\n".join(buttons), plot_data
