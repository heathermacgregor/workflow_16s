# Example: Using the Modern Report System

This example demonstrates how to integrate the new React/FastAPI architecture with the existing 16S workflow.

## Quick Start

1. **Generate a modern report from existing AmpliconData:**

```python
from workflow_16s.html_report.modern_report import generate_modern_html_report

# Your existing workflow code...
amplicon_data = load_your_amplicon_data()

# Generate modern interactive report
generate_modern_html_report(
    amplicon_data=amplicon_data,
    output_path="output/modern_report.html",
    mode="hybrid"  # Generates both legacy HTML and API data
)
```

2. **Start the development server:**

```bash
# From the workflow_16s directory
./start_dev_server.sh
```

3. **Access the reports:**
   - **Legacy HTML**: `output/modern_report.html`
   - **Interactive React**: `http://localhost:3000`
   - **API endpoints**: `http://localhost:8000/docs`

## API Endpoints

The new FastAPI backend provides structured data access:

- `GET /api/reports/{report_id}` - Report metadata
- `GET /api/reports/{report_id}/sections/{section_id}` - Section data with filtering
- `GET /api/reports/{report_id}/figures/{figure_id}` - Individual figure data

## React Components

The React frontend provides interactive components:

- **ReportDashboard**: Main container with sidebar navigation
- **SectionViewer**: Displays section content with filter controls
- **PlotlyWrapper**: Interactive Plotly visualizations with lazy loading
- **DataTable**: Sortable, searchable tables with pagination

## Migration Strategy

1. **Phase 1**: Use `mode="hybrid"` to generate both reports
2. **Phase 2**: Test React frontend with your data
3. **Phase 3**: Switch to `mode="api"` for production

## Benefits

- ✅ **Real-time filtering** without page reloads
- ✅ **Lazy loading** of figures and large datasets
- ✅ **Responsive design** that works on all devices
- ✅ **API access** for external tools and integrations
- ✅ **Component isolation** for easy maintenance
- ✅ **Modern development workflow** with hot reloading