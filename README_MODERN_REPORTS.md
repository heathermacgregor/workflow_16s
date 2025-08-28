# Modern 16S Analysis Report System

This directory contains the implementation of a modern React/FastAPI architecture for interactive 16S amplicon analysis reports, as specified in the requirements.

## Architecture Overview

The system consists of three main components:

### 1. FastAPI Backend (`api/`)
- **`main.py`**: RESTful API with endpoints for reports, sections, and figures
- **`data_processor.py`**: Extracts structured data from existing AmpliconData objects
- **`requirements.txt`**: Backend dependencies

**Key Features:**
- JSON-based data serving instead of HTML string interpolation
- Lazy loading of figures and large datasets
- Real-time filtering support
- CORS enabled for development

### 2. React Frontend (`frontend/`)
- **Component-based architecture** with separation of concerns
- **React Query** for efficient data fetching and caching
- **Interactive Plotly** visualizations with responsive design
- **Modern data tables** with sorting, filtering, and pagination

**Key Components:**
- `ReportDashboard.jsx`: Main container with sidebar navigation
- `SectionViewer.jsx`: Section content with dynamic filtering
- `PlotlyWrapper.jsx`: Lazy-loaded interactive visualizations
- `DataTable.jsx`: Enhanced tables with search and pagination

### 3. Enhanced Report Generation (`src/workflow_16s/html_report/`)
- **`modern_report.py`**: Unified report generation supporting multiple modes
- **Backward compatibility** with existing HTML generation
- **Hybrid mode** generates both legacy HTML and modern API data

## Usage

### Basic Integration
```python
from workflow_16s.html_report.modern_report import generate_modern_html_report

# Generate hybrid report (legacy + modern)
generate_modern_html_report(
    amplicon_data=your_amplicon_data,
    output_path="output/report.html",
    mode="hybrid"
)
```

### Development Server
```bash
# Start both API and React development servers
./start_dev_server.sh
```

### API-Only Mode
```python
# Generate only API data for React frontend
generate_modern_html_report(
    amplicon_data=your_amplicon_data,
    output_path="output/report.html", 
    mode="api"
)
```

## Benefits Over Current System

### Performance
- **Lazy loading** of figures reduces initial load time
- **Virtualized tables** handle large datasets efficiently
- **React Query caching** eliminates redundant API calls
- **Responsive design** optimized for all screen sizes

### User Experience
- **Real-time filtering** without page reloads
- **Interactive visualizations** with zoom, pan, and selection
- **Keyboard navigation** and accessibility support
- **Mobile-friendly** responsive interface

### Maintainability
- **Component isolation** replaces monolithic HTML generation
- **Type safety** with Pydantic models and TypeScript (future)
- **Testable components** with Jest/React Testing Library
- **Modern development workflow** with hot reloading

### Extensibility
- **Plugin architecture** for new analysis types
- **Theme system** for different report styles
- **Export capabilities** (PDF, PowerPoint, etc.)
- **API access** for external tools and integrations

## File Structure

```
workflow_16s/
├── api/                              # FastAPI backend
│   ├── main.py                       # API application and endpoints
│   ├── data_processor.py             # Data extraction from AmpliconData
│   └── requirements.txt              # Backend dependencies
├── frontend/                         # React application
│   ├── src/components/               # React components
│   ├── src/hooks/                    # Custom hooks
│   ├── package.json                  # Frontend dependencies
│   └── public/index.html             # HTML template
├── src/workflow_16s/html_report/     # Enhanced report generation
│   └── modern_report.py              # Unified report generation
├── start_dev_server.sh               # Development server script
└── MODERN_REPORT_EXAMPLE.md          # Usage examples
```

## API Endpoints

- `GET /` - Health check
- `GET /api/reports/{report_id}` - Report metadata and sections list
- `GET /api/reports/{report_id}/sections/{section_id}` - Section data with filtering
- `GET /api/reports/{report_id}/figures/{figure_id}` - Individual figure data

## Migration Strategy

1. **Phase 1**: Use `mode="hybrid"` to generate both legacy and modern reports
2. **Phase 2**: Test React frontend with your specific data and requirements
3. **Phase 3**: Switch to `mode="api"` for production when ready

The system maintains full backward compatibility while providing a path to modern interactive reporting.