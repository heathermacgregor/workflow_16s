// ============================= PLOTLY UTILITIES ============================= //

/**
 * Initialize Plotly selectors
 */
function initializePlotlySelectors() {
    document.querySelectorAll('.plotly-selector-container').forEach(container => {
        const selector = container.querySelector('.figure-dropdown');
        const plotDiv = container.querySelector('.plotly-selector-plot');
        
        if (selector && selector.options.length > 0) {
            // Trigger initial plot display
            const firstOption = selector.options[0];
            if (firstOption) {
                selector.value = firstOption.value;
                // The plot will be displayed by the embedded script in each selector
            }
        }
        
        // Fix Plotly container styling after initialization
        fixPlotlyContainers();
    });
}

/**
 * Fix Plotly container styling issues
 */
function fixPlotlyContainers() {
    // Wait for Plotly to initialize, then fix container styling
    setTimeout(() => {
        document.querySelectorAll('.plotly-selector-plot .svg-container').forEach(container => {
            // Remove problematic inline styles that cause layout issues
            container.style.height = 'auto';
            container.style.minHeight = '400px'; // Set a reasonable minimum height
        });
    }, 100);
}

/**
 * Show figure in a container (legacy support)
 */
function showFigure(dropdown, containerId) {
    const plotId = dropdown.value;
    const container = document.getElementById(containerId);
    
    if (!plotId || !container) return;
    
    // Clear previous content
    container.innerHTML = `<div id="container-${plotId}" class="plot-container"></div>`;
    
    // Initialize the plot
    initializePlot(plotId);
}

/**
 * Initialize a specific plot
 */
function initializePlot(plotId) {
    const plotInfo = window.plotData[plotId];
    const container = document.getElementById(`container-${plotId}`);
    if (!plotInfo || !container) {
        console.error(`Data or container not found for plot: ${plotId}`);
        return;
    }
    
    // Clear previous content and skip if already plotted
    if (container.classList.contains('js-plotly-plot') || container.querySelector('img')) {
        return;
    }
    container.innerHTML = '';

    switch(plotInfo.type) {
        case 'plotly':
            Plotly.newPlot(container, plotInfo.data, plotInfo.layout, { 
                responsive: true,
                displayModeBar: true
            }).then(() => {
                // Fix Plotly container styling after plot is rendered
                fixPlotlyContainers();
            });
            break;
        case 'image':
            const img = document.createElement('img');
            img.src = `data:image/png;base64,${plotInfo.data}`;
            img.style.maxWidth = '100%';
            img.style.height = 'auto';
            img.alt = 'Generated plot';
            container.appendChild(img);
            break;
        case 'error':
            container.innerHTML = `<div class="error-message">Error: ${plotInfo.error}</div>`;
            break;
        default:
            container.innerHTML = `<div class="error-message">Unsupported plot type: ${plotInfo.type}</div>`;
    }
}

/**
 * Handle window resize for Plotly plots
 */
function handlePlotlyResize() {
    document.querySelectorAll('.js-plotly-plot').forEach(div => {
        if (window.Plotly) {
            Plotly.Plots.resize(div);
            // Fix container styling after resize
            setTimeout(fixPlotlyContainers, 50);
        }
    });
}

// Export functions for external access
window.PlotlyUtils = {
    initializePlotlySelectors,
    fixPlotlyContainers,
    showFigure,
    initializePlot,
    handlePlotlyResize
};