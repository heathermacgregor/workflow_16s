/**
 * Toggles the visibility of a section's content.
 * @param {string} contentId - The ID of the section content to toggle.
 * @param {HTMLElement} headerElement - The header element that was clicked.
 */
function toggleSection(contentId, headerElement) {
    const content = document.getElementById(contentId);
    const toggleIcon = headerElement.querySelector('.toggle-icon');

    if (content.style.display === 'none' || content.style.display === '') {
        content.style.display = 'block';
        toggleIcon.textContent = '▼';
    } else {
        content.style.display = 'none';
        toggleIcon.textContent = '▶';
    }
}

/**
 * Shows a specific tab and hides others within the same tab group.
 * If a plotId is provided, it will render the Plotly figure or image.
 * @param {Event} event - The click event.
 * @param {string} tabId - The ID of the tab pane to show.
 * @param {string} [plotId] - The ID of the plot data to render (optional).
 */
function showTab(event, tabId, plotId) {
    const clickedButton = event.currentTarget;
    const tabContainer = clickedButton.closest('.tabs'); // Find the parent .tabs container
    const tabButtons = tabContainer.querySelectorAll('.tab-button');
    const tabPanes = clickedButton.closest('.subsection').querySelectorAll('.tab-pane'); // Find tab panes within the same subsection

    tabButtons.forEach(button => button.classList.remove('active'));
    clickedButton.classList.add('active');

    tabPanes.forEach(pane => pane.style.display = 'none');
    const activeTab = document.getElementById(tabId);
    if (activeTab) {
        activeTab.style.display = 'block';
        if (plotId) {
            renderPlot(plotId);
        } else {
            // If the tab contains nested tabs, activate the first one
            const nestedFirstButton = activeTab.querySelector('.tabs .tab-button');
            if (nestedFirstButton) {
                nestedFirstButton.click(); // Programmatically click the first nested button
            }
        }
    }
}

/**
 * Shows a specific table pane and hides others within its group.
 * This is for the outermost level of nested tabs.
 * @param {string} tableId - The ID of the table pane to show.
 * @param {HTMLElement} clickedButton - The button that was clicked.
 */
function showTable(tableId, clickedButton) {
    const tableContainer = clickedButton.closest('.tabs[data-label="table_type"]');
    const buttons = tableContainer.querySelectorAll('.table-button');
    const panes = tableContainer.nextElementSibling.querySelectorAll('.table-pane');

    buttons.forEach(btn => btn.classList.remove('active'));
    clickedButton.classList.add('active');

    panes.forEach(pane => pane.style.display = 'none');
    const activePane = document.getElementById(tableId);
    if (activePane) {
        activePane.style.display = 'block';
        // Activate the first sub-level button within this table pane
        const firstLevelButton = activePane.querySelector('.tabs[data-label="level"] .level-button');
        if (firstLevelButton) {
            firstLevelButton.click();
        }
    }
}

/**
 * Shows a specific level pane and hides others within its group.
 * This is for the second level of nested tabs.
 * @param {string} levelId - The ID of the level pane to show.
 * @param {HTMLElement} clickedButton - The button that was clicked.
 */
function showLevel(levelId, clickedButton) {
    const levelContainer = clickedButton.closest('.tabs[data-label="level"]');
    const buttons = levelContainer.querySelectorAll('.level-button');
    const panes = levelContainer.nextElementSibling.querySelectorAll('.level-pane');

    buttons.forEach(btn => btn.classList.remove('active'));
    clickedButton.classList.add('active');

    panes.forEach(pane => pane.style.display = 'none');
    const activePane = document.getElementById(levelId);
    if (activePane) {
        activePane.style.display = 'block';
        // Activate the first sub-level button within this level pane
        const firstMethodButton = activePane.querySelector('.tabs[data-label="method"] .method-button');
        if (firstMethodButton) {
            firstMethodButton.click();
        } else {
            // If no method buttons, check for metric buttons (alpha diversity)
            const firstMetricButton = activePane.querySelector('.tabs[data-label="metric"] .tab-button');
            if (firstMetricButton) {
                firstMetricButton.click();
            } else {
                // If neither, then this level pane directly contains plot tabs.
                // Find the first plot tab and render its plot.
                const firstPlotTab = activePane.querySelector('.tab-pane[data-plot-id]');
                if (firstPlotTab) {
                    const plotId = firstPlotTab.dataset.plotId;
                    renderPlot(plotId);
                }
            }
        }
    }
}

/**
 * Shows a specific method pane and hides others within its group.
 * This is for the third level of nested tabs (e.g., SHAP).
 * @param {string} methodId - The ID of the method pane to show.
 * @param {HTMLElement} clickedButton - The button that was clicked.
 */
function showMethod(methodId, clickedButton) {
    const methodContainer = clickedButton.closest('.tabs[data-label="method"]');
    const buttons = methodContainer.querySelectorAll('.method-button');
    const panes = methodContainer.nextElementSibling.querySelectorAll('.method-pane');

    buttons.forEach(btn => btn.classList.remove('active'));
    clickedButton.classList.add('active');

    panes.forEach(pane => pane.style.display = 'none');
    const activePane = document.getElementById(methodId);
    if (activePane) {
        activePane.style.display = 'block';
        // Activate the first plot tab within this method pane
        const firstPlotButton = activePane.querySelector('.tabs .tab-button');
        if (firstPlotButton) {
            firstPlotButton.click();
        }
    }
}


/**
 * Renders a Plotly or image plot into its container.
 * Assumes 'plot_data' is a globally accessible object.
 * @param {string} plotId - The ID of the plot data to render.
 */
function renderPlot(plotId) {
    const plotContainer = document.getElementById(`container-${plotId}`);
    if (!plotContainer) {
        console.warn(`Plot container for ${plotId} not found.`);
        return;
    }

    const data = window.plot_data[plotId];
    if (!data) {
        console.warn(`No plot data found for ${plotId}.`);
        plotContainer.innerHTML = '<p>Error: Plot data not found.</p>';
        return;
    }

    // Clear previous content
    plotContainer.innerHTML = '';

    if (data.type === 'plotly' && typeof Plotly !== 'undefined') {
        const layout = { ...data.layout };
        if (data.square) {
            // Adjust layout for square plots if necessary
            // This might involve setting fixed width/height or aspect ratio
            const size = Math.min(plotContainer.clientWidth, 600); // Example size
            layout.width = size;
            layout.height = size;
            layout.autosize = false;
        } else {
            layout.autosize = true;
            layout.width = plotContainer.clientWidth; // Use container width
            layout.height = Math.min(plotContainer.clientWidth * 0.6, 600); // Example responsive height
        }

        Plotly.newPlot(plotContainer, data.data, layout, { responsive: true });
    } else if (data.type === 'image') {
        const img = document.createElement('img');
        img.src = `data:image/png;base64,${data.data}`;
        img.style.maxWidth = '100%';
        img.style.height = 'auto';
        plotContainer.appendChild(img);
    } else if (data.type === 'error') {
        plotContainer.innerHTML = `<p style="color: red;">Error rendering plot: ${data.error}</p>`;
    } else {
        plotContainer.innerHTML = `<p>Unsupported plot type or Plotly not loaded: ${data.type}</p>`;
    }
}

/**
 * Initializes dynamic table features: sorting, pagination, and column resizing.
 */
function initializeDynamicTables() {
    document.querySelectorAll('.dynamic-table').forEach(table => {
        const tableId = table.id;
        const container = document.getElementById(`container-${tableId}`);
        if (!container) {
            console.error(`Container for table ${tableId} not found.`);
            return;
        }

        // --- Column Resizing ---
        const headers = table.querySelectorAll('th');
        headers.forEach(header => {
            const resizer = document.createElement('div');
            resizer.classList.add('resizer');
            header.appendChild(resizer);

            let x, w, currentHeader;

            const mouseMoveHandler = (e) => {
                const dx = e.clientX - x;
                currentHeader.style.width = `${w + dx}px`;
            };

            const mouseUpHandler = () => {
                document.removeEventListener('mousemove', mouseMoveHandler);
                document.removeEventListener('mouseup', mouseUpHandler);
            };

            resizer.addEventListener('mousedown', (e) => {
                currentHeader = header;
                x = e.clientX;
                w = currentHeader.offsetWidth;
                document.addEventListener('mousemove', mouseMoveHandler);
                document.addEventListener('mouseup', mouseUpHandler);
            });
        });

        // --- Sorting ---
        const tbody = table.querySelector('tbody');
        if (!tbody) {
            console.warn(`No tbody found for table ${tableId}, skipping sorting.`);
            return;
        }
        const rows = Array.from(tbody.rows);

        headers.forEach((header, index) => {
            if (header.classList.contains('sortable')) {
                header.addEventListener('click', () => {
                    const isAsc = header.classList.contains('asc');
                    const isDesc = header.classList.contains('desc');

                    headers.forEach(h => {
                        if (h !== header) {
                            h.classList.remove('asc', 'desc');
                        }
                    });

                    let direction;
                    if (isAsc) {
                        header.classList.remove('asc');
                        header.classList.add('desc');
                        direction = -1;
                    } else if (isDesc) {
                        header.classList.remove('desc');
                        direction = 1; // Cycle back to ascending
                    } else {
                        header.classList.add('asc');
                        direction = 1;
                    }

                    rows.sort((rowA, rowB) => {
                        const cellA = rowA.cells[index].innerText.toLowerCase();
                        const cellB = rowB.cells[index].innerText.toLowerCase();

                        // Basic numeric check
                        const valA = parseFloat(cellA);
                        const valB = parseFloat(cellB);
                        if (!isNaN(valA) && !isNaN(valB)) {
                            return (valA - valB) * direction;
                        }

                        // String comparison
                        return cellA.localeCompare(cellB) * direction;
                    });

                    rows.forEach(row => tbody.appendChild(row));
                    // Re-apply pagination after sort
                    applyPagination(tableId);
                });
            }
        });

        // --- Pagination and Search Initialization ---
        initPagination(tableId);
        initColumnVisibility(tableId);
    });
}

// --- Pagination Functions ---
const tableStates = {}; // Stores pagination and filter states for each table

function initPagination(tableId) {
    const table = document.getElementById(tableId);
    if (!table) return;

    tableStates[tableId] = {
        currentPage: 1,
        pageSize: 10,
        allRows: Array.from(table.querySelector('tbody').rows),
        filteredRows: [],
        currentSearchTerm: ''
    };
    filterTable(tableId, ''); // Apply initial filter to set filteredRows and re-apply pagination
}

function applyPagination(tableId) {
    const state = tableStates[tableId];
    if (!state) return;

    const tbody = document.getElementById(tableId).querySelector('tbody');
    tbody.innerHTML = ''; // Clear current rows

    const start = (state.currentPage - 1) * state.pageSize;
    const end = state.pageSize === -1 ? state.filteredRows.length : start + state.pageSize;
    const paginatedRows = state.filteredRows.slice(start, end);

    paginatedRows.forEach(row => tbody.appendChild(row));

    updatePaginationControls(tableId);
}

function updatePaginationControls(tableId) {
    const state = tableStates[tableId];
    if (!state) return;

    const paginationDiv = document.getElementById(`pagination-${tableId}`);
    const indicatorSpan = document.getElementById(`indicator-${tableId}`);
    if (!paginationDiv || !indicatorSpan) return;

    paginationDiv.innerHTML = ''; // Clear existing buttons

    const totalPages = state.pageSize === -1 ? 1 : Math.ceil(state.filteredRows.length / state.pageSize);

    // Previous button
    const prevBtn = document.createElement('button');
    prevBtn.textContent = 'Previous';
    prevBtn.disabled = state.currentPage === 1;
    prevBtn.onclick = () => {
        state.currentPage--;
        applyPagination(tableId);
    };
    paginationDiv.appendChild(prevBtn);

    // Page number buttons (simplified for brevity, can be enhanced)
    for (let i = 1; i <= totalPages; i++) {
        const pageBtn = document.createElement('button');
        pageBtn.textContent = i;
        pageBtn.classList.toggle('active', i === state.currentPage);
        pageBtn.onclick = () => {
            state.currentPage = i;
            applyPagination(tableId);
        };
        paginationDiv.appendChild(pageBtn);
    }

    // Next button
    const nextBtn = document.createElement('button');
    nextBtn.textContent = 'Next';
    nextBtn.disabled = state.currentPage === totalPages;
    nextBtn.onclick = () => {
        state.currentPage++;
        applyPagination(tableId);
    };
    paginationDiv.appendChild(nextBtn);

    // Update indicator
    const startEntry = Math.min((state.currentPage - 1) * state.pageSize + 1, state.filteredRows.length);
    const endEntry = Math.min(startEntry + state.pageSize - 1, state.filteredRows.length);
    if (state.pageSize === -1 && state.filteredRows.length > 0) {
        indicatorSpan.textContent = `Showing all ${state.filteredRows.length} entries`;
    } else if (state.filteredRows.length === 0) {
        indicatorSpan.textContent = `No entries to show`;
    } else {
        indicatorSpan.textContent = `Showing ${startEntry}-${endEntry} of ${state.filteredRows.length} entries (filtered)`;
    }
}

function changePageSize(tableId, pageSize) {
    const state = tableStates[tableId];
    if (!state) return;

    state.pageSize = parseInt(pageSize, 10);
    state.currentPage = 1; // Reset to first page
    applyPagination(tableId);
}

/**
 * Filters table rows based on a search term.
 * @param {string} tableId - The ID of the table to filter.
 * @param {string} searchTerm - The text to search for.
 */
function filterTable(tableId, searchTerm) {
    const state = tableStates[tableId];
    if (!state) return;

    state.currentSearchTerm = searchTerm.toLowerCase();

    state.filteredRows = state.allRows.filter(row => {
        const cells = Array.from(row.cells);
        return cells.some(cell => cell.innerText.toLowerCase().includes(state.currentSearchTerm));
    });

    state.currentPage = 1; // Reset to the first page after filtering
    applyPagination(tableId);
}

// --- Column Visibility Functions ---
function initColumnVisibility(tableId) {
    const table = document.getElementById(tableId);
    if (!table) return;

    const headers = Array.from(table.querySelectorAll('th'));
    const dropdown = table.closest('.table-container').querySelector('.column-visibility-dropdown');
    if (!dropdown) return;

    dropdown.innerHTML = ''; // Clear previous content

    headers.forEach((header, index) => {
        const label = document.createElement('label');
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.checked = true; // Columns are visible by default
        checkbox.dataset.columnIndex = index;
        checkbox.onchange = (e) => toggleColumn(tableId, index, e.target.checked);

        label.appendChild(checkbox);
        label.appendChild(document.createTextNode(header.innerText.trim()));
        dropdown.appendChild(label);
    });
}

function toggleColumn(tableId, colIndex, show) {
    const table = document.getElementById(tableId);
    if (!table) return;

    Array.from(table.rows).forEach(row => {
        if (row.cells[colIndex]) {
            row.cells[colIndex].style.display = show ? '' : 'none';
        }
    });
}

function toggleColumnVisibilityDropdown(button) {
    const dropdown = button.nextElementSibling; // The .column-visibility-dropdown
    dropdown.style.display = dropdown.style.display === 'block' ? 'none' : 'block';
}

// Close the dropdown if the user clicks outside of it
window.addEventListener('click', (event) => {
    document.querySelectorAll('.column-visibility-dropdown').forEach(dropdown => {
        const button = dropdown.previousElementSibling;
        if (button && !button.contains(event.target) && !dropdown.contains(event.target)) {
            dropdown.style.display = 'none';
        }
    });
});


// Call initializeDynamicTables after the DOM content is fully loaded
document.addEventListener('DOMContentLoaded', () => {
    initializeDynamicTables();

    // Trigger initial display for all top-level buttons in sections
    document.querySelectorAll('.section-content').forEach(sectionContent => {
        const firstTabButton = sectionContent.querySelector('.tabs[data-label="table_type"] .table-button');
        if (firstTabButton) {
            firstTabButton.click();
        } else {
            // Fallback for sections that might only have one level of tabs (e.g., sample maps or violin)
            const firstSimpleTabButton = sectionContent.querySelector('.tabs .tab-button');
            if (firstSimpleTabButton) {
                firstSimpleTabButton.click();
            }
        }
    });
});

// Render plots for initially visible tabs
window.addEventListener('load', () => {
    // This timeout ensures Plotly.js has loaded
    setTimeout(() => {
        document.querySelectorAll('.tab-pane[style*="block"][data-plot-id]').forEach(tabPane => {
            const plotId = tabPane.dataset.plotId;
            renderPlot(plotId);
        });
    }, 100); // Small delay to ensure Plotly is ready
});
