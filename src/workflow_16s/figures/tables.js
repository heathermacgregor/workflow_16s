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
    });
}
