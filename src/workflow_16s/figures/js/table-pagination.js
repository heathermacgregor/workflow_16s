// ============================= TABLE PAGINATION FUNCTIONALITY ============================= //

/**
 * Change page size for a table
 */
function changePageSize(tableId, size) {
    const sizeInt = parseInt(size, 10);
    updatePagination(tableId, sizeInt, 0);
    
    // Store preference in localStorage if available
    try {
        localStorage.setItem(`pageSize_${tableId}`, size);
    } catch (e) {
        // Ignore localStorage errors
    }
}

/**
 * Navigate to a specific page
 */
function goToPage(tableId, pageSize, pageIndex) {
    updatePagination(tableId, pageSize, pageIndex);
}

/**
 * Update pagination for a table
 */
function updatePagination(tableId, pageSize, pageIndex) {
    const table = document.getElementById(tableId);
    if (!table) return;

    const tbody = table.tBodies[0];
    const rows = Array.from(tbody.rows);
    const totalRows = rows.length;
    const isPaginated = pageSize > 0 && totalRows > pageSize;
    
    const totalPages = isPaginated ? Math.ceil(totalRows / pageSize) : 1;
    const currentPage = Math.min(Math.max(0, pageIndex), totalPages - 1);

    const start = isPaginated ? currentPage * pageSize : 0;
    const end = isPaginated ? Math.min(start + pageSize, totalRows) : totalRows;

    // Show/hide rows efficiently
    rows.forEach((row, i) => {
        row.style.display = (i >= start && i < end) ? '' : 'none';
    });

    // Update controls
    updatePaginationControls(tableId, totalPages, currentPage, totalRows, start, end);
}

/**
 * Update pagination controls
 */
function updatePaginationControls(tableId, totalPages, currentPage, totalRows, start, end) {
    // Escape special characters in tableId
    const escapedTableId = CSS.escape(tableId);
    
    const container = document.getElementById(escapedTableId)?.closest('.table-container');
    if (!container) return;
    
    const paginationContainer = container.querySelector(`#pagination-${CSS.escape(tableId)}`);
    const indicator = container.querySelector(`#indicator-${CSS.escape(tableId)}`);
    
    if (!paginationContainer || !indicator) return;

    // Clear previous buttons
    paginationContainer.innerHTML = '';
    
    if (totalPages > 1) {
        // Create pagination buttons with smart truncation
        createPaginationButtons(paginationContainer, tableId, totalPages, currentPage);
        indicator.textContent = `Showing ${start + 1}-${end} of ${totalRows} rows`;
    } else {
        indicator.textContent = `Showing all ${totalRows} rows`;
    }
}

/**
 * Create pagination buttons with smart truncation
 */
function createPaginationButtons(container, tableId, totalPages, currentPage) {
    const maxVisibleButtons = 7;
    
    // Previous button
    if (currentPage > 0) {
        const prevBtn = createPaginationButton('‹', () => goToPage(tableId, getPageSize(tableId), currentPage - 1));
        prevBtn.title = 'Previous page';
        container.appendChild(prevBtn);
    }
    
    // Calculate visible page range
    let startPage = Math.max(0, currentPage - Math.floor(maxVisibleButtons / 2));
    let endPage = Math.min(totalPages - 1, startPage + maxVisibleButtons - 1);
    
    // Adjust start if we're near the end
    if (endPage - startPage < maxVisibleButtons - 1) {
        startPage = Math.max(0, endPage - maxVisibleButtons + 1);
    }
    
    // First page + ellipsis
    if (startPage > 0) {
        container.appendChild(createPaginationButton(1, () => goToPage(tableId, getPageSize(tableId), 0)));
        if (startPage > 1) {
            const ellipsis = document.createElement('span');
            ellipsis.textContent = '...';
            ellipsis.className = 'pagination-ellipsis';
            container.appendChild(ellipsis);
        }
    }
    
    // Page buttons
    for (let i = startPage; i <= endPage; i++) {
        const btn = createPaginationButton(i + 1, () => goToPage(tableId, getPageSize(tableId), i));
        if (i === currentPage) {
            btn.classList.add('active');
        }
        container.appendChild(btn);
    }
    
    // Ellipsis + last page
    if (endPage < totalPages - 1) {
        if (endPage < totalPages - 2) {
            const ellipsis = document.createElement('span');
            ellipsis.textContent = '...';
            ellipsis.className = 'pagination-ellipsis';
            container.appendChild(ellipsis);
        }
        container.appendChild(createPaginationButton(totalPages, () => goToPage(tableId, getPageSize(tableId), totalPages - 1)));
    }
    
    // Next button
    if (currentPage < totalPages - 1) {
        const nextBtn = createPaginationButton('›', () => goToPage(tableId, getPageSize(tableId), currentPage + 1));
        nextBtn.title = 'Next page';
        container.appendChild(nextBtn);
    }
}

/**
 * Create a pagination button
 */
function createPaginationButton(text, onClick) {
    const btn = document.createElement('button');
    btn.textContent = text;
    btn.className = 'pagination-btn';
    btn.onclick = onClick;
    return btn;
}

// Export functions for external access
window.TablePagination = {
    changePageSize,
    goToPage,
    updatePagination,
    updatePaginationControls,
    createPaginationButtons,
    createPaginationButton
};