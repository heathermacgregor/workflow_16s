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
        this.activateFirstPath();
    }
    
    generateTabHTML(data, path = [], depth = 0) {
        const isLeaf = typeof data === 'object' && data.type;
        const isBranch = !isLeaf;
        const currentId = path.join('-') || 'root';
        
        if (isLeaf) {
            const plotId = `plot-${currentId}`;
            return `
                <div class="tab-pane" id="${currentId}" data-path="${path.join('.')}">
                    <div id="container-${plotId}" class="plot-container"></div>
                </div>
            `;
        }
        
        // Generate tabs for children
        const childKeys = Object.keys(data);
        const tabs = childKeys.map(key => {
            return `<button class="tab-button" data-path="${[...path, key].join('.')}" data-depth="${depth}">${key}</button>`;
        }).join('');
        
        const panes = childKeys.map(key => {
            const childPath = [...path, key];
            return `
                <div class="nested-pane" id="pane-${childPath.join('-')}">
                    ${this.generateTabHTML(data[key], childPath, depth + 1)}
                </div>
            `;
        }).join('');
        
        return `
            <div class="tabs tabs-depth-${depth}">${tabs}</div>
            <div class="panes-container">
                ${panes}
            </div>
        `;
    }
    
    activateFirstPath() {
        // Activate first tab at each level
        let currentLevel = this.rootContainer;
        let depth = 0;
        
        while (true) {
            const tabs = currentLevel.querySelector('.tabs');
            if (!tabs) break;
            
            const firstTab = tabs.querySelector('.tab-button');
            if (!firstTab) break;
            
            // Activate first tab
            firstTab.classList.add('active');
            const path = firstTab.dataset.path;
            const paneId = `pane-${path.replace(/\./g, '-')}`;
            const pane = document.getElementById(paneId);
            if (pane) {
                pane.classList.add('active');
                pane.style.display = 'block';
                currentLevel = pane;
            } else {
                break;
            }
            depth++;
        }
        
        // Initialize plot for the leaf node
        const activePane = currentLevel.querySelector('.tab-pane');
        if (activePane) {
            const plotId = activePane.id;
            const plotContainer = document.querySelector(`#container-plot-${plotId}`);
            if (plotContainer) {
                initializePlot(`plot-${plotId}`);
            }
        }
    }
}

// ============================= INITIALIZATION ============================= //
document.addEventListener('DOMContentLoaded', () => {
    // Parse plot data from the embedded JSON
    const plotDataElement = document.getElementById('plot-data');
    window.plotData = plotDataElement ? JSON.parse(plotDataElement.textContent) : {};

    // Initialize Tab Managers
    document.querySelectorAll('.tab-manager-container').forEach(container => {
        const section = container.id.replace('tabs-container-', '');
        if (window.plotData[section]) {
            const tabManager = new TabManager();
            tabManager.init(window.plotData[section], container.id);
        }
    });
    
    // Tab click handler
    document.addEventListener('click', (e) => {
        if (e.target.classList.contains('tab-button')) {
            const path = e.target.dataset.path;
            const depth = parseInt(e.target.dataset.depth);
            const paneId = `pane-${path.replace(/\./g, '-')}`;
            const pane = document.getElementById(paneId);
            
            if (!pane) return;
            
            // Deactivate all tabs and panes at the same level
            const parentContainer = e.target.closest('.panes-container') || e.target.closest('.tab-manager-container');
            const tabsContainer = e.target.parentElement;
            
            // Deactivate sibling tabs
            tabsContainer.querySelectorAll('.tab-button').forEach(btn => {
                btn.classList.remove('active');
            });
            
            // Deactivate sibling panes
            parentContainer.querySelectorAll('.nested-pane').forEach(p => {
                p.classList.remove('active');
                p.style.display = 'none';
            });
            
            // Activate clicked tab and pane
            e.target.classList.add('active');
            pane.classList.add('active');
            pane.style.display = 'block';
            
            // Deeper levels: activate first child in the new pane
            let currentPane = pane;
            let currentDepth = depth + 1;
            
            while (true) {
                const tabs = currentPane.querySelector('.tabs');
                if (!tabs) break;
                
                const firstTab = tabs.querySelector('.tab-button');
                if (!firstTab) break;
                
                // Activate first tab
                firstTab.classList.add('active');
                const path = firstTab.dataset.path;
                const paneId = `pane-${path.replace(/\./g, '-')}`;
                const nextPane = document.getElementById(paneId);
                if (nextPane) {
                    // Deactivate any existing active panes at this level
                    const parent = nextPane.parentElement;
                    parent.querySelectorAll('.nested-pane').forEach(p => {
                        p.classList.remove('active');
                        p.style.display = 'none';
                    });
                    
                    nextPane.classList.add('active');
                    nextPane.style.display = 'block';
                    currentPane = nextPane;
                    currentDepth++;
                } else {
                    break;
                }
            }
            
            // Initialize plot for the leaf node
            const leafPane = currentPane.querySelector('.tab-pane');
            if (leafPane) {
                const plotId = leafPane.id;
                const plotContainer = document.querySelector(`#container-plot-${plotId}`);
                if (plotContainer && !plotContainer.innerHTML.trim()) {
                    initializePlot(`plot-${plotId}`);
                }
            }
        }
    });
    
    // Initialize all dynamic tables
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
    let current = window.plotData;
    for (const key of path) {
        if (current[key]) {
            current = current[key];
        } else {
            console.error(`Plot data not found for path: ${path.join('.')}`);
            return;
        }
    }
    
    const plotInfo = current;
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
