// ============================= SECTION UTILITIES ============================= //

/**
 * Toggle section collapse/expand
 */
function toggleSection(event) {
    const header = event.currentTarget;
    const section = header.closest('.section');
    section.classList.toggle('collapsed');
    
    // Trigger resize event for any Plotly plots in the section
    setTimeout(() => {
        const plotlyDivs = section.querySelectorAll('.js-plotly-plot');
        plotlyDivs.forEach(div => {
            if (window.Plotly) {
                Plotly.Plots.resize(div);
                // Fix container styling after resize
                setTimeout(fixPlotlyContainers, 50);
            }
        });
    }, 400); // Wait for CSS transition to complete
}

/**
 * Toggle all sections expand/collapse
 */
function toggleAllSections(expand) {
    document.querySelectorAll('.section').forEach(section => {
        if (expand) {
            section.classList.remove('collapsed');
        } else {
            section.classList.add('collapsed');
        }
    });
    
    // Trigger resize for visible plots after expansion
    if (expand) {
        setTimeout(() => {
            document.querySelectorAll('.js-plotly-plot').forEach(div => {
                if (window.Plotly) {
                    Plotly.Plots.resize(div);
                    // Fix container styling after resize
                    setTimeout(fixPlotlyContainers, 50);
                }
            });
        }, 500);
    }
}

// Export functions for external access
window.SectionUtils = {
    toggleSection,
    toggleAllSections
};