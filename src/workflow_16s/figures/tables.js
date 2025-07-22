// tables.js

// ================= PLOT INITIALIZATION ================= //
function initializePlot(plotId) {
    const plotInfo = window.plotData[plotId];
    if (!plotInfo) {
        console.error(`Plot data not found for: ${plotId}`);
        return;
    }
    
    const container = document.getElementById(`container-${plotId}`);
    if (!container) {
        console.error(`Container not found for: ${plotId}`);
        return;
    }
    
    // Clear previous content
    container.innerHTML = '';
    
    switch(plotInfo.type) {
        case 'plotly':
            Plotly.newPlot(container, plotInfo.data, plotInfo.layout);
            break;
            
        case 'image':
            const img = document.createElement('img');
            img.src = `data:image/png;base64,${plotInfo.data}`;
            img.style.maxWidth = '100%';
            container.appendChild(img);
            break;
            
        case 'error':
            container.innerHTML = `<div class="error">Error loading plot: ${plotInfo.error}</div>`;
            break;
            
        default:
            container.innerHTML = `<div class="error">Unsupported plot type: ${plotInfo.type}</div>`;
    }
}

// ================= TAB NAVIGATION FUNCTIONS ================= //
function showTab(tabId, plotId) {
    // Hide all tab panes in the same section
    const section = document.getElementById(tabId).closest('.section-content');
    if (!section) return;
    
    const tabPanes = section.querySelectorAll('.tab-pane');
    tabPanes.forEach(pane => {
        pane.style.display = 'none';
    });
    
    // Remove active class from all tab buttons in the same section
    const buttons = section.querySelectorAll('.tab-button');
    buttons.forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show the selected tab pane
    const selectedTab = document.getElementById(tabId);
    if (selectedTab) {
        selectedTab.style.display = 'block';
        initializePlot(plotId);
    }
    
    // Add active class to the clicked button
    const clickedButton = document.querySelector(`[data-tab="${tabId}"]`);
    if (clickedButton) {
        clickedButton.classList.add('active');
    }
}

function showTable(tableId) {
    // Hide all table panes in the same section
    const section = document.getElementById(tableId).closest('.section-content');
    if (!section) return;
    
    const tablePanes = section.querySelectorAll('.table-pane');
    tablePanes.forEach(pane => {
        pane.style.display = 'none';
    });
    
    // Remove active class from all table buttons in the same section
    const buttons = section.querySelectorAll('.table-button');
    buttons.forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show the selected table pane
    const selectedTable = document.getElementById(tableId);
    if (selectedTable) {
        selectedTable.style.display = 'block';
    }
    
    // Add active class to the clicked button
    const clickedButton = document.querySelector(`[data-table="${tableId}"]`);
    if (clickedButton) {
        clickedButton.classList.add('active');
    }
}

function showLevel(levelId) {
    // Hide all level panes in the same section
    const section = document.getElementById(levelId).closest('.section-content');
    if (!section) return;
    
    const levelPanes = section.querySelectorAll('.level-pane');
    levelPanes.forEach(pane => {
        pane.style.display = 'none';
    });
    
    // Remove active class from all level buttons in the same section
    const buttons = section.querySelectorAll('.level-button');
    buttons.forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show the selected level pane
    const selectedLevel = document.getElementById(levelId);
    if (selectedLevel) {
        selectedLevel.style.display = 'block';
    }
    
    // Add active class to the clicked button
    const clickedButton = document.querySelector(`[data-level="${levelId}"]`);
    if (clickedButton) {
        clickedButton.classList.add('active');
    }
}

function showMethod(methodId) {
    // Hide all method panes in the same section
    const section = document.getElementById(methodId).closest('.section-content');
    if (!section) return;
    
    const methodPanes = section.querySelectorAll('.method-pane');
    methodPanes.forEach(pane => {
        pane.style.display = 'none';
    });
    
    // Remove active class from all method buttons in the same section
    const buttons = section.querySelectorAll('.method-button');
    buttons.forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show the selected method pane
    const selectedMethod = document.getElementById(methodId);
    if (selectedMethod) {
        selectedMethod.style.display = 'block';
    }
    
    // Add active class to the clicked button
    const clickedButton = document.querySelector(`[data-method="${methodId}"]`);
    if (clickedButton) {
        clickedButton.classList.add('active');
    }
}

// ================= SECTION TOGGLING ================= //
function toggleSection(sectionContentId, element) {
    const content = document.getElementById(sectionContentId);
    if (!content) return;
    
    if (content.style.display === 'none' || content.style.display === '') {
        content.style.display = 'block';
        element.querySelector('.toggle-icon').textContent = '▼';
    } else {
        content.style.display = 'none';
        element.querySelector('.toggle-icon').textContent = '►';
    }
}

// ================= PAGINATION FOR TABLES ================= //
function changePageSize(tableId, size) {
    const table = document.getElementById(tableId);
    if (!table) return;
    
    const rows = table.getElementsByTagName('tbody')[0].rows;
    const totalRows = rows.length;
    const pageSize = size === '-1' ? totalRows : parseInt(size);
    
    // Hide all rows
    for (let i = 0; i < totalRows; i++) {
        rows[i].style.display = 'none';
    }
    
    // Show rows for current page
    for (let i = 0; i < pageSize && i < totalRows; i++) {
        rows[i].style.display = '';
    }
    
    // Update pagination buttons
    updatePaginationButtons(tableId, pageSize);
}

function updatePaginationButtons(tableId, pageSize) {
    const table = document.getElementById(tableId);
    if (!table) return;
    
    const totalRows = table.getElementsByTagName('tbody')[0].rows.length;
    const totalPages = Math.ceil(totalRows / pageSize);
    const paginationContainer = document.getElementById(`pagination-${tableId}`);
    const indicator = document.getElementById(`indicator-${tableId}`);
    
    if (!paginationContainer || !indicator) return;
    
    // Clear existing buttons
    paginationContainer.innerHTML = '';
    
    // Create buttons
    for (let i = 0; i < totalPages; i++) {
        const btn = document.createElement('button');
        btn.textContent = i + 1;
        btn.onclick = () => goToPage(tableId, pageSize, i);
        paginationContainer.appendChild(btn);
    }
    
    // Update indicator
    indicator.textContent = `Page 1 of ${totalPages}`;
}

function goToPage(tableId, pageSize, pageIndex) {
    const table = document.getElementById(tableId);
    if (!table) return;
    
    const rows = table.getElementsByTagName('tbody')[0].rows;
    const totalRows = rows.length;
    const start = pageIndex * pageSize;
    const end = Math.min(start + pageSize, totalRows);
    
    // Hide all rows
    for (let i = 0; i < totalRows; i++) {
        rows[i].style.display = 'none';
    }
    
    // Show rows for current page
    for (let i = start; i < end; i++) {
        rows[i].style.display = '';
    }
    
    // Update indicator
    const indicator = document.getElementById(`indicator-${tableId}`);
    if (indicator) {
        indicator.textContent = `Page ${pageIndex + 1} of ${Math.ceil(totalRows / pageSize)}`;
    }
}

// ================= INITIALIZE ON LOAD ================= //
document.addEventListener('DOMContentLoaded', () => {
    // Activate first tab in each tab container
    document.querySelectorAll('.tabs').forEach(tabContainer => {
        const firstButton = tabContainer.querySelector('.tab-button, .table-button, .level-button, .method-button');
        if (firstButton) {
            firstButton.classList.add('active');
            
            if (firstButton.classList.contains('tab-button')) {
                const tabId = firstButton.getAttribute('data-tab');
                const plotId = firstButton.getAttribute('data-plot-id');
                showTab(tabId, plotId);
            } else if (firstButton.classList.contains('table-button')) {
                const tableId = firstButton.getAttribute('data-table');
                showTable(tableId);
            } else if (firstButton.classList.contains('level-button')) {
                const levelId = firstButton.getAttribute('data-level');
                showLevel(levelId);
            } else if (firstButton.classList.contains('method-button')) {
                const methodId = firstButton.getAttribute('data-method');
                showMethod(methodId);
            }
        }
    });
    
    // Initialize tables pagination
    document.querySelectorAll('.dynamic-table').forEach(table => {
        const tableId = table.getAttribute('id');
        if (tableId) {
            const select = table.closest('.table-container')?.querySelector('.rows-per-page');
            if (select) {
                changePageSize(tableId, select.value);
            }
        }
    });
    
    // Initialize first plots
    document.querySelectorAll('.tab-pane[style*="block"]').forEach(pane => {
        const plotId = pane.getAttribute('data-plot-id');
        if (plotId) initializePlot(plotId);
    });
});
