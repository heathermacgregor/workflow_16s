/* ======================= FIGURE FUNCTIONALITY ======================= */
/* ---- data ---- */
const plotData = JSON.parse(document.getElementById('plot-data').textContent);

/* ---- state ---- */
const rendered = new Set();
const MAX_WEBGL_CONTEXTS = 6;  // Conservative limit for most browsers
const activeWebGLPlots = new Set();

/* ---- helpers ---- */
function purgePlot(plotId) {
    const plotDiv = document.getElementById(plotId);
    if (plotDiv && Plotly) {
        Plotly.purge(plotDiv);
    }
    const container = document.getElementById(`container-${plotId}`);
    if (container) container.innerHTML = '';
    rendered.delete(plotId);
    activeWebGLPlots.delete(plotId);
}

function enforceWebGLLimit() {
    while (activeWebGLPlots.size > MAX_WEBGL_CONTEXTS) {
        const oldest = activeWebGLPlots.values().next().value;
        purgePlot(oldest);
    }
}

function renderPlot(containerId, plotId) {
    const container = document.getElementById(containerId);
    if (!container) return console.error('Missing container', containerId);

    container.innerHTML = '';
    const div = document.createElement('div');
    div.id = plotId;
    div.className = 'plot-container';
    container.appendChild(div);

    const payload = plotData[plotId];
    if (!payload) {
        div.innerHTML = '<div class="error">Plot data unavailable</div>';
        return;
    }

    // Compute responsive width (min 500px, max 1000px)
    const fullWidth = container.clientWidth || window.innerWidth;
    const minWidth = fullWidth * 0.15;
    const width = Math.max(minWidth, Math.min(1000, fullWidth * 0.95));
    const height = payload.square ? width : Math.round(width * 0.6);

    const is3D = payload.data?.some(d => d.type.includes('3d'));

    try {
        if (payload.type === 'plotly') {
            if (payload.layout) {
                payload.layout.showlegend = false;
                payload.layout.width = width;
                payload.layout.height = height;

                if (is3D) {
                    payload.layout.scene = payload.layout.scene || {};
                    payload.layout.scene.aspectmode = 'data';
                    payload.layout.uirevision = 'constant';
                }
            }

            const config = {
                responsive: true,
                webglOptions: { preserveDrawingBuffer: false }
            };

            Plotly.newPlot(plotId, payload.data, payload.layout, config)
                .then(() => {
                    if (is3D) {
                        activeWebGLPlots.add(plotId);
                        enforceWebGLLimit();
                    }
                })
                .catch(err => {
                    div.innerHTML = `<div class="error">Plotly error: ${err}</div>`;
                    console.error(err);
                });
        } else if (payload.type === 'image') {
            const img = document.createElement('img');
            img.src = 'data:image/png;base64,' + payload.data;
            img.style.maxWidth = '100%';
            img.style.height = 'auto';
            div.appendChild(img);
        } else if (payload.type === 'error') {
            div.innerHTML = `<div class="error">${payload.error}</div>`;
        } else {
            div.innerHTML = '<div class="error">Unknown plot type</div>';
        }
    } catch (err) {
        div.innerHTML = `<div class="error">Rendering error: ${err}</div>`;
        console.error(err);
    }
}

/* ---- tab logic ---- */
function showTab(tabId, plotId) {
    const pane = document.getElementById(tabId);
    if (!pane) return;

    const subsection = pane.closest('.subsection');
    if (!subsection) return;

    const prevPane = subsection.querySelector('.tab-pane[style*="display: block"]');
    if (prevPane) {
        const prevPlotId = prevPane.dataset.plotId;
        if (prevPlotId && rendered.has(prevPlotId)) {
            purgePlot(prevPlotId);
        }
    }

    subsection.querySelectorAll('.tab-pane').forEach(p => p.style.display = 'none');
    subsection.querySelectorAll('.tab-button').forEach(b => b.classList.remove('active'));

    pane.style.display = 'block';
    const button = subsection.querySelector(`[data-tab="${tabId}"]`);
    if (button) button.classList.add('active');

    // Only render if plotId is provided
    if (plotId && !rendered.has(plotId)) {
        renderPlot(`container-${plotId}`, plotId);
        rendered.add(plotId);
    }
}

/* ---- nested tab management ---- */
function showTable(tableId) {
    const currentTable = document.querySelector('.table-pane[style*="display: block"]');
    if (currentTable) {
        currentTable.querySelectorAll('.tab-pane[data-plot-id]').forEach(pane => {
            const plotId = pane.dataset.plotId;
            if (rendered.has(plotId)) purgePlot(plotId);
        });
    }

    document.querySelectorAll('.table-pane').forEach(pane => {
        pane.style.display = 'none';
    });
    document.querySelectorAll('.table-button').forEach(btn => {
        btn.classList.remove('active');
    });

    const newTable = document.getElementById(tableId);
    if (newTable) {
        newTable.style.display = 'block';
        document.querySelector(`[data-table="${tableId}"]`).classList.add('active');

        const activeLevel = newTable.querySelector('.level-pane[style*="display: block"]');
        if (!activeLevel) {
            const firstLevel = newTable.querySelector('.level-pane');
            if (firstLevel) showLevel(firstLevel.id);
        }
    }
}

function showLevel(levelId) {
    const levelPane = document.getElementById(levelId);
    if (!levelPane) return;

    const tablePane = levelPane.closest('.table-pane');
    if (!tablePane) return;

    const currentLevel = tablePane.querySelector('.level-pane[style*="display: block"]');
    if (currentLevel) {
        currentLevel.querySelectorAll('.tab-pane[data-plot-id]').forEach(pane => {
            const plotId = pane.dataset.plotId;
            if (rendered.has(plotId)) purgePlot(plotId);
        });
    }

    tablePane.querySelectorAll('.level-pane').forEach(pane => {
        pane.style.display = 'none';
    });
    tablePane.querySelectorAll('.level-button').forEach(btn => {
        btn.classList.remove('active');
    });

    levelPane.style.display = 'block';
    document.querySelector(`[data-level="${levelId}"]`).classList.add('active');

    const activeMethod = levelPane.querySelector('.method-pane[style*="display: block"]');
    const activeMetric = levelPane.querySelector('.metric-pane[style*="display: block"]');

    if (!activeMethod && !activeMetric) {
        const firstMethod = levelPane.querySelector('.method-pane');
        if (firstMethod) {
            showMethod(firstMethod.id);
        } else {
            const firstMetric = levelPane.querySelector('.metric-pane');
            if (firstMetric) {
                const plotId = firstMetric.dataset.plotId;
                if (plotId) showMetric(firstMetric.id, plotId);
            }
        }
    }
}

function showMethod(methodId) {
    const methodPane = document.getElementById(methodId);
    if (!methodPane) return;

    const levelPane = methodPane.closest('.level-pane');
    if (!levelPane) return;

    const currentMethod = levelPane.querySelector('.method-pane[style*="display: block"]');
    if (currentMethod) {
        currentMethod.querySelectorAll('.tab-pane[data-plot-id]').forEach(pane => {
            const plotId = pane.dataset.plotId;
            if (rendered.has(plotId)) purgePlot(plotId);
        });
    }

    levelPane.querySelectorAll('.method-pane').forEach(pane => {
        pane.style.display = 'none';
    });
    levelPane.querySelectorAll('.method-button').forEach(btn => {
        btn.classList.remove('active');
    });

    methodPane.style.display = 'block';
    document.querySelector(`[data-method="${methodId}"]`).classList.add('active');

    const activeTab = methodPane.querySelector('.tab-pane[style*="display: block"]');
    if (!activeTab) {
        const firstTab = methodPane.querySelector('.tab-pane');
        if (firstTab) showTab(firstTab.id, firstTab.dataset.plotId);
    }
}

function showMetric(metricId, plotId) {
    const container = document.getElementById(`container-${plotId}`);
    if (container) {
        container.innerHTML = '';
    }

    const metricPane = document.getElementById(metricId);
    if (!metricPane) return;

    const levelPane = metricPane.closest('.level-pane');
    if (!levelPane) return;

    levelPane.querySelectorAll('.metric-pane').forEach(pane => {
        pane.style.display = 'none';
    });
    levelPane.querySelectorAll('.metric-button').forEach(btn => {
        btn.classList.remove('active');
    });

    metricPane.style.display = 'block';
    document.querySelector(`[data-metric="${metricId}"]`).classList.add('active');

    if (!rendered.has(plotId)) {
        renderPlot(`container-${plotId}`, plotId);
        rendered.add(plotId);
    }
}

/* ---- section toggles ---- */
function toggleAllSections(show) {
    document.querySelectorAll('.section').forEach(s => {
        s.style.display = show ? 'block' : 'none';
    });
}

/* ======================= TABLE FUNCTIONALITY ======================= */


function sortTable(tableId, columnIndex, isNumeric) {
    const table = document.getElementById(tableId);
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    const header = table.querySelectorAll('thead th')[columnIndex];
    const isAscending = !header.classList.contains('asc');

    // Clear previous sort indicators
    table.querySelectorAll('thead th').forEach(th => {
        th.classList.remove('asc', 'desc');
    });

    // Set new sort indicator
    header.classList.add(isAscending ? 'asc' : 'desc');

    rows.sort((a, b) => {
        const aVal = a.cells[columnIndex].textContent.trim();
        const bVal = b.cells[columnIndex].textContent.trim();

        if (isNumeric) {
            const numA = parseFloat(aVal) || 0;
            const numB = parseFloat(bVal) || 0;
            return isAscending ? numA - numB : numB - numA;
        }
        return isAscending ?
            aVal.localeCompare(bVal) :
            bVal.localeCompare(aVal);
    });

    // Clear and re-add sorted rows
    tbody.innerHTML = '';
    rows.forEach(row => tbody.appendChild(row));

    // Reapply pagination
    const select = table.closest('.table-container')
        .querySelector('.rows-per-page');
    changePageSize(tableId, select.value);
}

function setupTableSorting(tableId) {
    const table = document.getElementById(tableId);
    if (!table) return;

    const headers = table.querySelectorAll('thead th');

    headers.forEach((header, index) => {
        // Check if column is numeric
        const firstRow = table.querySelector('tbody tr');
        const isNumeric = firstRow && !isNaN(parseFloat(firstRow.cells[index].textContent));

        header.style.cursor = 'pointer';
        header.addEventListener('click', () => {
            sortTable(tableId, index, isNumeric);
        });
    });
}

function setupTableResizing(tableId) {
    const table = document.getElementById(tableId);
    if (!table) return;

    const headers = table.querySelectorAll('thead th');
    headers.forEach((header, index) => {
        // Skip the last header
        if (index === headers.length - 1) return;

        // Create resizable handle
        const handle = document.createElement('div');
        handle.className = 'resizable-handle';
        header.appendChild(handle);

        let startX, startWidth;

        handle.addEventListener('mousedown', (e) => {
            e.preventDefault();
            startX = e.clientX;
            startWidth = header.offsetWidth;

            const doDrag = (e) => {
                const newWidth = startWidth + (e.clientX - startX);
                if (newWidth < 10) return; // Minimum width
                header.style.width = `${newWidth}px`;
                // Adjust all cells in this column
                const cells = table.querySelectorAll(`tbody tr > :nth-child(${index+1})`);
                cells.forEach(cell => {
                    cell.style.width = `${newWidth}px`;
                });
            };

            const stopDrag = () => {
                document.removeEventListener('mousemove', doDrag);
                document.removeEventListener('mouseup', stopDrag);
            };

            document.addEventListener('mousemove', doDrag);
            document.addEventListener('mouseup', stopDrag);
        });
    });
}

function paginateTable(tableId, pageSize) {
    const table = document.getElementById(tableId);
    if (!table) return;

    const rows = table.querySelectorAll('tbody tr');
    const paginationDiv = document.getElementById(`pagination-${tableId}`);
    const indicator = document.getElementById(`indicator-${tableId}`);
    const totalPages = pageSize === -1 ? 1 : Math.ceil(rows.length / pageSize);

    // Hide all rows
    rows.forEach(row => row.style.display = 'none');

    // Show rows for first page
    const start = 0;
    const end = pageSize === -1 ? rows.length : Math.min(start + pageSize, rows.length);
    for (let i = start; i < end; i++) {
        rows[i].style.display = '';
    }

    // Generate pagination buttons
    paginationDiv.innerHTML = '';
    if (totalPages > 1) {
        const prevButton = document.createElement('button');
        prevButton.textContent = '◄';
        prevButton.classList.add('pagination-btn');
        prevButton.disabled = true;
        prevButton.addEventListener('click', () => {
            changePage(tableId, 0, pageSize); // Go to first page
        });
        paginationDiv.appendChild(prevButton);

        for (let i = 0; i < totalPages; i++) {
            const pageButton = document.createElement('button');
            pageButton.textContent = i + 1;
            pageButton.classList.add('pagination-btn');
            if (i === 0) pageButton.classList.add('active');
            pageButton.addEventListener('click', () => {
                changePage(tableId, i, pageSize);
            });
            paginationDiv.appendChild(pageButton);
        }

        const nextButton = document.createElement('button');
        nextButton.textContent = '►';
        nextButton.classList.add('pagination-btn');
        nextButton.disabled = totalPages <= 1;
        nextButton.addEventListener('click', () => {
            changePage(tableId, totalPages - 1, pageSize); // Go to last page
        });
        paginationDiv.appendChild(nextButton);
    }

    // Update indicator
    if (indicator) {
        indicator.textContent = `Page 1 of ${totalPages}`;
    }
}

function changePage(tableId, pageNumber, pageSize) {
    const table = document.getElementById(tableId);
    if (!table) return;

    const rows = table.querySelectorAll('tbody tr');
    const paginationDiv = document.getElementById(`pagination-${tableId}`);
    const indicator = document.getElementById(`indicator-${tableId}`);
    const totalPages = pageSize === -1 ? 1 : Math.ceil(rows.length / pageSize);

    // Validate page number
    pageNumber = Math.max(0, Math.min(pageNumber, totalPages - 1));

    // Hide all rows
    rows.forEach(row => row.style.display = 'none');

    // Show rows for current page
    const start = pageNumber * pageSize;
    const end = pageSize === -1 ? rows.length : Math.min(start + pageSize, rows.length);
    for (let i = start; i < end; i++) {
        rows[i].style.display = '';
    }

    // Update pagination UI
    const buttons = paginationDiv.querySelectorAll('.pagination-btn');
    buttons.forEach(button => button.classList.remove('active'));

    // Only activate current page button if it exists
    if (buttons[pageNumber + 1]) { // +1 to skip the prev button
        buttons[pageNumber + 1].classList.add('active');
    }

    // Update button states
    buttons[0].disabled = pageNumber === 0; // Prev button
    buttons[buttons.length - 1].disabled = pageNumber === totalPages - 1; // Next button

    // Update indicator
    if (indicator) {
        indicator.textContent = `Page ${pageNumber + 1} of ${totalPages}`;
    }
}

function changePageSize(tableId, newSize) {
    const pageSize = newSize === '-1' ? 10000 : parseInt(newSize);
    paginateTable(tableId, pageSize);
}

function initTables() {
    document.querySelectorAll('.dynamic-table').forEach(table => {
        const tableId = table.id;
        setupTableSorting(tableId);
        changePageSize(tableId, 10); // Initialize with 10 rows per page
        setupTableResizing(tableId);  // Add this line
    });
}

function toggleSection(contentId, header) {
    const content = document.getElementById(contentId);
    const section = header.parentElement;
    const isCollapsing = !section.classList.contains('collapsed');
    
    // Toggle collapsed class
    section.classList.toggle('collapsed');
    
    // Rotate toggle icon
    const icon = header.querySelector('.toggle-icon');
    if (isCollapsing) {
        icon.textContent = '▶';
    } else {
        icon.textContent = '▼';
    }
}

/* ---- initialization ---- */
document.addEventListener('DOMContentLoaded', () => {
    // Initialize all first-level plots in subsections
    document.querySelectorAll('.subsection').forEach(sub => {
        const firstTab = sub.querySelector('.tab-pane');
        const plotId = firstTab?.dataset.plotId;

        if (firstTab && plotId && !rendered.has(plotId)) {
            showTab(firstTab.id, plotId);
        }
    });

    // Activate first table in each section
    const activatedTables = new Set();
    document.querySelectorAll('.table-pane').forEach(pane => {
        const tableId = pane.id;
        const tableButton = document.querySelector(`[data-table="${tableId}"]`);

        if (tableButton && !activatedTables.has(tableId)) {
            showTable(tableId);
            activatedTables.add(tableId);
        }
    });

    // Fallback: If no tables visible, show first level
    document.querySelectorAll('.level-pane').forEach(pane => {
        const levelId = pane.id;
        const levelButton = document.querySelector(`[data-level="${levelId}"]`);

        if (levelButton && pane.style.display !== 'block') {
            showLevel(levelId);
        }
    });

    // Fallback: If no levels visible, show first method
    document.querySelectorAll('.method-pane').forEach(pane => {
        const methodId = pane.id;
        const methodButton = document.querySelector(`[data-method="${methodId}"]`);

        if (methodButton && pane.style.display !== 'block') {
            showMethod(methodId);
        }
    });

    // Initialize any tabular behavior
    if (typeof initTables === 'function') {
        initTables();
    } else {
        console.warn('initTables() is not defined');
    }
});
