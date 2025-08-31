// ============================= MAIN INITIALIZATION AND LEGACY SUPPORT ============================= //

// Initialize everything when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Parse plot data from the embedded JSON (legacy support)
    const plotDataElement = document.getElementById('plot-data');
    window.plotData = plotDataElement ? JSON.parse(plotDataElement.textContent) : {};
    
    // Initialize legacy dropdowns (backward compatibility)
    document.querySelectorAll('.figure-dropdown:not(.plotly-selector-container .figure-dropdown)').forEach(dropdown => {
        const firstOption = dropdown.options[0];
        if (firstOption) {
            showFigure(dropdown, dropdown.dataset.containerId || dropdown.closest('.figure-container').id);
        }
    });
    
    // Initialize all dynamic tables with improved performance
    initializeTables();
    
    // Initialize Plotly selectors
    initializePlotlySelectors();
});

// Handle window resize for responsive tables and plots
window.addEventListener('resize', debounce(() => {
    // Resize Plotly plots
    handlePlotlyResize();
}, 250));

// Export main functions for backward compatibility
window.TableUtils = {
    // Table core functions
    initializeTables,
    getPageSize,
    debounce,
    
    // Table sorting functions
    makeTableSortable,
    sortTableByColumn,
    
    // Table pagination functions
    changePageSize,
    goToPage,
    updatePagination,
    
    // Table resizing functions
    makeTableResizable,
    
    // Plotly utilities
    initializePlotlySelectors,
    fixPlotlyContainers,
    showFigure,
    initializePlot,
    
    // Section utilities
    toggleSection,
    toggleAllSections
};