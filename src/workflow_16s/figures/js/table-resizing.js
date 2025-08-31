// ============================= TABLE COLUMN RESIZING FUNCTIONALITY ============================= //

/**
 * Make table columns resizable
 */
function makeTableResizable(table) {
    const headers = table.querySelectorAll('th');
    
    headers.forEach(header => {
        // Skip if already has resize handle
        if (header.querySelector('.resizable-handle')) return;
        
        const handle = document.createElement('div');
        handle.className = 'resizable-handle';
        header.appendChild(handle);

        let startX, startWidth, isResizing = false;

        const onMouseMove = (e) => {
            if (!isResizing) return;
            
            const newWidth = Math.max(50, startWidth + (e.clientX - startX));
            header.style.width = `${newWidth}px`;
            
            // Prevent text selection during resize
            e.preventDefault();
        };

        const onMouseUp = () => {
            if (!isResizing) return;
            
            isResizing = false;
            document.removeEventListener('mousemove', onMouseMove);
            document.removeEventListener('mouseup', onMouseUp);
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
        };

        handle.addEventListener('mousedown', (e) => {
            e.stopPropagation(); // Prevent sorting
            e.preventDefault();
            
            isResizing = true;
            startX = e.clientX;
            startWidth = header.offsetWidth;
            
            document.body.style.cursor = 'col-resize';
            document.body.style.userSelect = 'none';
            
            document.addEventListener('mousemove', onMouseMove);
            document.addEventListener('mouseup', onMouseUp);
        });
    });
}

// Export functions for external access
window.TableResizing = {
    makeTableResizable
};