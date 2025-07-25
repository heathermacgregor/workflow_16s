// ============================= TAB MANAGER ============================= //
class TabManager {
    constructor() {
        this.tabData = {};
        this.rootContainer = null;
    }
    
    init(data, containerId) {
        this.tabData = data;
        this.rootContainer = document.getElementById(containerId);
        if (!this.rootContainer) return;
        
        this.rootContainer.innerHTML = this.generateTabHTML(this.tabData);
    }
    
    generateTabHTML(data, path = []) {
        const isLeaf = typeof data === 'object' && data.type;
        const isBranch = !isLeaf;
        const currentId = path.join('-') || 'root';
        
        if (isLeaf) {
            const plotId = `plot-${currentId}`;
            return `
                <div class="tab-pane active" id="${currentId}">
                    <div id="container-${plotId}" class="plot-container"></div>
                </div>
            `;
        }
        
        // Generate tabs for children
        const childKeys = Object.keys(data);
        const tabs = childKeys.map(key => {
            return `<button class="tab-button" data-target="${[...path, key].join('.')}">${key}</button>`;
        }).join('');
        
        const panes = childKeys.map(key => {
            const childPath = [...path, key];
            return `
                <div class="nested-pane">
                    ${this.generateTabHTML(data[key], childPath)}
                </div>
            `;
        }).join('');
        
        return `
            <div class="tabs">${tabs}</div>
            <div class="panes-container">
                ${panes}
            </div>
        `;
    }
}

// ============================= INITIALIZATION ============================= //
document.addEventListener('DOMContentLoaded', () => {
    // Parse plot data from the embedded JSON
    const plotDataElement = document.getElementById('plot-data');
    window.plotData = plotDataElement ? JSON.parse(plotDataElement.textContent) : {};

    // Initialize Tab Managers
    document.querySelectorAll('.tab-manager-container').forEach(container => {
        const tabManager = new TabManager();
        // Find section name from container ID
        const section = container.id.replace('tabs-container-', '');
        if (window.plotData[section]) {
            tabManager.init(window.plotData[section], container.id);
        }
    });
    
    // Tab click handler
    document.addEventListener('click', (e) => {
        if (e.target.classList.contains('tab-button')) {
            const path = e.target.dataset.target.split('.');
            const paneId = path.join('-');
            const pane = document.getElementById(paneId);
            if (pane) {
                // Hide all siblings
                const siblings = Array.from(pane.parentElement.children);
                siblings.forEach(sibling => {
                    if (sibling !== pane) {
                        sibling.classList.remove('active');
                        sibling.style.display = 'none';
                    }
                });
                
                // Show target pane
                pane.style.display = 'block';
                pane.classList.add('active');
                
                // Update active tab
                const tabsContainer = e.target.parentElement;
                tabsContainer.querySelectorAll('.tab-button').forEach(btn => {
                    btn.classList.remove('active');
                });
                e.target.classList.add('active');
                
                // Initialize plot if not already initialized
                const plotContainer = pane.querySelector('.plot-container');
                if (plotContainer && !plotContainer.innerHTML.trim()) {
                    const plotId = plotContainer.id.replace('container-', '');
                    initializePlot(plotId);
                }
            }
        }
    });
    
    // Initialize all plots in visible panes
    setTimeout(() => {
        document.querySelectorAll('.plot-container').forEach(container => {
            const plotId = container.id.replace('container-', '');
            if (container.offsetParent !== null && !container.innerHTML.trim()) {
                initializePlot(plotId);
            }
        });
    }, 100);
    
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

// ============================ PLOT RENDERING ============================ //
function initializePlot(plotId) {
    // Extract the path from plotId (format: plot-key1-key2-...)
    const path = plotId.replace('plot-', '').split('-');
    
    // Find the plot data by traversing the plotData structure
    let plotInfo = window.plotData;
    for (const key of path) {
        if (plotInfo[key]) {
            plotInfo = plotInfo[key];
        } else {
            console.error(`Plot data not found for path: ${path.join('.')}`);
            return;
        }
    }
    
    // Ensure we have a leaf node with plot data
    if (!plotInfo.type) {
        console.error(`No plot type found for: ${plotId}`);
        return;
    }
    
    const container = document.getElementById(`container-${plotId}`);
    if (!container) {
        console.error(`Container not found for plot: ${plotId}`);
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
            img.style.maxWidth = '100%';
            container.appendChild(img);
            break;
        case 'error':
            container.innerHTML = `<div class="error">Error: ${plotInfo.error}</div>`;
            break;
        default:
            container.innerHTML = `<div class="error">Unsupported plot type: ${plotInfo.type}</div>`;
    }
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
