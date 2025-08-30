// ============================= TABLE SORTING FUNCTIONALITY ============================= //

/**
 * Make a table sortable by adding click handlers to headers
 */
function makeTableSortable(table) {
    const headers = table.querySelectorAll('th');
    
    headers.forEach((header, index) => {
        // Skip if header already has click listener
        if (header.dataset.sortable === 'true') return;
        header.dataset.sortable = 'true';
        
        header.addEventListener('click', (e) => {
            // Don't sort if clicking on resize handle
            if (e.target.classList.contains('resizable-handle')) return;
            
            const currentSort = header.dataset.sortDirection || 'none';
            const newSort = currentSort === 'asc' ? 'desc' : 'asc';
            
            sortTableByColumn(table, index, newSort);
            
            // Update header indicators
            headers.forEach(h => {
                h.classList.remove('asc', 'desc');
                h.dataset.sortDirection = 'none';
            });
            
            header.classList.add(newSort);
            header.dataset.sortDirection = newSort;
            
            // Reset pagination to first page after sorting
            const container = table.closest('.table-container');
            const select = container?.querySelector('.rows-per-page');
            if (select && table.id) {
                updatePagination(table.id, parseInt(select.value), 0);
            }
        });
    });
}

/**
 * Sort table by column with intelligent type detection
 */
function sortTableByColumn(table, columnIndex, direction) {
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    
    const sortedRows = rows.sort((a, b) => {
        const aCell = a.cells[columnIndex];
        const bCell = b.cells[columnIndex];
        
        if (!aCell || !bCell) return 0;
        
        const aText = aCell.textContent.trim();
        const bText = bCell.textContent.trim();
        
        // Try numeric comparison first
        const aNum = parseFloat(aText);
        const bNum = parseFloat(bText);
        
        if (!isNaN(aNum) && !isNaN(bNum)) {
            return direction === 'asc' ? aNum - bNum : bNum - aNum;
        }
        
        // Fall back to string comparison
        const result = aText.localeCompare(bText, undefined, { 
            numeric: true, 
            sensitivity: 'base' 
        });
        
        return direction === 'asc' ? result : -result;
    });
    
    // Use DocumentFragment for efficient DOM manipulation
    const fragment = document.createDocumentFragment();
    sortedRows.forEach(row => fragment.appendChild(row));
    tbody.appendChild(fragment);
}

// Export functions for external access
window.TableSorting = {
    makeTableSortable,
    sortTableByColumn
};