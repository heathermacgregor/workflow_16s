// ============================= INITIALIZATION ============================= //
document.addEventListener('DOMContentLoaded', () => {
    // Parse plot data from the embedded JSON
    const plotDataElement = document.getElementById('plot-data');
    window.plotData = plotDataElement ? JSON.parse(plotDataElement.textContent) : {};

    // Initialize all active plots
    document.querySelectorAll('.tab-pane.active, .method-pane.active, .level-pane.active, .table-pane.active').forEach(activePane => {
        // Activate first child panes for initially active panes
        activateFirstChildPanes(activePane);
        initializeVisiblePlots(activePane);
    });
    
    // Make all dynamic tables sortable and resizable
    document.querySelectorAll('.dynamic-table').forEach(table => {
        makeTableSortable(table);
        makeTableResizable(table);
        
        // Initialize pagination for each table
        const tableId = table.id;
        if (tableId) {
            const container = table.closest('.table-container');
            const select = container?.querySelector('.rows-per-page');
            if (select) {
                updatePagination(tableId, parseInt(select.value, 10), 0);
            }
        }
    });
});

function initializeVisiblePlots(container) {
  // build a NodeList of panes to check:
  //  - the container itself (if it *is* a tab-, method- or level-pane)
  //  - all nested tab-panes
  const selfIsPane =
    container.matches('.tab-pane[data-plot-id].active') ||
    container.matches('.method-pane[data-plot-id].active') ||
    container.matches('.level-pane[data-plot-id].active');

  const panes = [
    ...(selfIsPane ? [container] : []),
    ...container.querySelectorAll('.tab-pane[data-plot-id].active')
  ];

  panes.forEach(pane => {
    const plotId = pane.getAttribute('data-plot-id');
    if (plotId) initializePlot(plotId);
  });
}


// ============================ PLOT RENDERING ============================ //
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
            Plotly.newPlot(container, plotInfo.data, plotInfo.layout, { responsive: true });
            break;
        case 'image':
            const img = document.createElement('img');
            img.src = `data:image/png;base64,${plotInfo.data}`;
            container.appendChild(img);
            break;
        case 'error':
            container.innerHTML = `<div class="error">Error: ${plotInfo.error}</div>`;
            break;
        default:
            container.innerHTML = `<div class="error">Unsupported plot type: ${plotInfo.type}</div>`;
    }
}

// ========================== DYNAMIC TAB/PANE LOGIC ========================== //
function showPane(event) {
    const button = event.currentTarget;
    const targetSelector = button.getAttribute('data-pane-target');
    const targetPane = document.querySelector(targetSelector);

    if (!targetPane) return;

    // Deactivate sibling buttons and panes
    const parent = button.parentElement;
    parent.querySelectorAll('button').forEach(btn => btn.classList.remove('active'));
    
    // NEW: Get pane container and deactivate direct child panes only
    const paneContainer = targetPane.parentElement;
    paneContainer.querySelectorAll('.tab-pane, .method-pane, .level-pane, .table-pane').forEach(pane => {
        pane.classList.remove('active');
    });

    // Activate clicked button and target pane
    button.classList.add('active');
    targetPane.classList.add('active');

    // Activate first child pane in each nested level
    activateFirstChildPanes(targetPane);

    // Initialize plots within the newly visible pane
    initializeVisiblePlots(targetPane);
}

function activateFirstChildPanes(container) {
    const childSelectors = ['.level-pane', '.method-pane', '.tab-pane'];
    childSelectors.forEach(selector => {
        const childPanes = container.querySelectorAll(selector);
        if (childPanes.length > 0) {
            childPanes.forEach(pane => pane.classList.remove('active'));
            const firstChild = childPanes[0];
            firstChild.classList.add('active');
            // Recurse into the activated child
            activateFirstChildPanes(firstChild);
        }
    });
}


// ========================== SECTION COLLAPSE/EXPAND ========================= //
function toggleSection(event) {
    const header = event.currentTarget;
    const section = header.closest('.section');
    section.classList.toggle('collapsed');
}

function toggleAllSections(expand) {
    document.querySelectorAll('.section').forEach(section => {
        if (expand) {
            section.classList.remove('collapsed');
        } else {
            section.classList.add('collapsed');
        }
    });
}

// ============================== TABLE SORTING =============================== //
function makeTableSortable(table) {
    const headers = table.querySelectorAll('th');
    headers.forEach((header, index) => {
        header.addEventListener('click', () => {
            const sortDir = header.classList.contains('asc') ? 'desc' : 'asc';
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));

            const sortedRows = rows.sort((a, b) => {
                const aText = a.cells[index].textContent.trim();
                const bText = b.cells[index].textContent.trim();

                const aNum = parseFloat(aText);
                const bNum = parseFloat(bText);

                if (!isNaN(aNum) && !isNaN(bNum)) {
                    return aNum > bNum ? 1 : -1;
                }
                return aText.localeCompare(bText, undefined, { numeric: true });
            });

            if (sortDir === 'desc') {
                sortedRows.reverse();
            }

            // Re-append sorted rows
            tbody.innerHTML = '';
            tbody.append(...sortedRows);
            
            // Update header classes
            headers.forEach(h => h.classList.remove('asc', 'desc'));
            header.classList.add(sortDir);
            
            // Reset pagination to first page
            const select = table.closest('.table-container').querySelector('.rows-per-page');
            updatePagination(table.id, parseInt(select.value), 0);
        });
    });
}

// ============================ TABLE COLUMN RESIZING =========================== //
function makeTableResizable(table) {
    const headers = table.querySelectorAll('th');
    headers.forEach(header => {
        const handle = document.createElement('div');
        handle.className = 'resizable-handle';
        header.appendChild(handle);

        let startX, startWidth;

        const onMouseMove = (e) => {
            const newWidth = startWidth + (e.clientX - startX);
            if (newWidth > 40) { // Minimum width
                header.style.width = `${newWidth}px`;
            }
        };

        const onMouseUp = () => {
            document.removeEventListener('mousemove', onMouseMove);
            document.removeEventListener('mouseup', onMouseUp);
        };

        handle.addEventListener('mousedown', (e) => {
            e.stopPropagation(); // Prevent sorting
            startX = e.clientX;
            startWidth = header.offsetWidth;
            document.addEventListener('mousemove', onMouseMove);
            document.addEventListener('mouseup', onMouseUp);
        });
    });
}

// ============================ TABLE PAGINATION ============================== //
function changePageSize(tableId, size) {
    updatePagination(tableId, parseInt(size, 10), 0);
}

function goToPage(tableId, pageSize, pageIndex) {
    updatePagination(tableId, pageSize, pageIndex);
}

function updatePagination(tableId, pageSize, pageIndex) {
    const table = document.getElementById(tableId);
    if (!table) return;

    const tbody = table.tBodies[0];
    const rows = Array.from(tbody.rows);
    const totalRows = rows.length;
    const isPaginated = pageSize > 0 && totalRows > pageSize;
    
    const totalPages = isPaginated ? Math.ceil(totalRows / pageSize) : 1;
    pageIndex = Math.min(pageIndex, totalPages - 1);

    const start = isPaginated ? pageIndex * pageSize : 0;
    const end = isPaginated ? start + pageSize : totalRows;

    // Show/hide rows
    rows.forEach((row, i) => {
        row.style.display = (i >= start && i < end) ? '' : 'none';
    });

    // Update controls
    const container = table.closest('.table-container');
    const paginationContainer = container.querySelector(`#pagination-${tableId}`);
    const indicator = container.querySelector(`#indicator-${tableId}`);
    
    if (!paginationContainer || !indicator) return;

    paginationContainer.innerHTML = '';
    if (isPaginated) {
        for (let i = 0; i < totalPages; i++) {
            const btn = document.createElement('button');
            btn.textContent = i + 1;
            btn.className = 'pagination-btn' + (i === pageIndex ? ' active' : '');
            btn.onclick = () => goToPage(tableId, pageSize, i);
            paginationContainer.appendChild(btn);
        }
        indicator.textContent = `Page ${pageIndex + 1} of ${totalPages}`;
    } else {
        indicator.textContent = `Showing all ${totalRows} rows`;
    }
}
