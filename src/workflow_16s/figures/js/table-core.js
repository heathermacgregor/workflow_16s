// ============================= TABLE CORE INITIALIZATION ============================= //

/**
 * Initialize all dynamic tables with performance optimization
 */
function initializeTables() {
    const tables = document.querySelectorAll('.dynamic-table');
    
    // Use requestAnimationFrame to prevent blocking
    const initTable = (index) => {
        if (index >= tables.length) return;
        
        const table = tables[index];
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
        
        // Process next table asynchronously
        requestAnimationFrame(() => initTable(index + 1));
    };
    
    initTable(0);
}

/**
 * Get page size for a specific table
 */
function getPageSize(tableId) {
    const container = document.getElementById(tableId)?.closest('.table-container');
    const select = container?.querySelector('.rows-per-page');
    return select ? parseInt(select.value, 10) : 10;
}

/**
 * Debounce function for performance optimization
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Export functions for external access
window.TableCore = {
    initializeTables,
    getPageSize,
    debounce
};