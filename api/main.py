# ===================================== IMPORTS ====================================== #

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import json
import logging
from pathlib import Path
import sys

# Add the src directory to Python path to import workflow_16s modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

from api.data_processor import ReportDataProcessor

# ===================================== MODELS ====================================== #

class ReportMetadata(BaseModel):
    id: str
    title: str
    generated_ts: str
    sections: List[str]
    config: Dict[str, Any]

class SectionData(BaseModel):
    id: str
    title: str
    type: str
    figures: List[Dict[str, Any]]
    tables: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class FigureData(BaseModel):
    id: str
    type: str  # 'plotly' | 'image' | 'table'
    data: Dict[str, Any]
    layout: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any]

# ===================================== APP SETUP ====================================== #

app = FastAPI(title="16S Analysis API", version="1.0.0")

# Add CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up logging
logger = logging.getLogger("workflow_16s_api")

# Global storage for report data (in production, this would be a database)
REPORT_STORE = {}

# ===================================== HELPER FUNCTIONS ====================================== #

def load_report_data(amplicon_data, report_id: str = "current"):
    """Load and process AmpliconData into structured format"""
    try:
        processor = ReportDataProcessor(amplicon_data)
        structured_data = processor.extract_structured_data()
        
        # Store in global cache
        REPORT_STORE[report_id] = structured_data
        return structured_data
    except Exception as e:
        logger.error(f"Error processing amplicon data: {e}")
        raise

def get_stored_report(report_id: str):
    """Get cached report data"""
    if report_id not in REPORT_STORE:
        raise HTTPException(status_code=404, detail=f"Report {report_id} not found")
    return REPORT_STORE[report_id]

# ===================================== ENDPOINTS ====================================== #

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "16S Analysis API is running"}

@app.get("/api/reports/{report_id}", response_model=ReportMetadata)
async def get_report(report_id: str) -> ReportMetadata:
    """Get report metadata and structure"""
    try:
        report_data = get_stored_report(report_id)
        metadata = report_data["metadata"]
        
        return ReportMetadata(
            id=report_id,
            title=metadata["title"],
            generated_ts=metadata["generated_ts"],
            sections=metadata["sections_available"],
            config=metadata.get("config", {})
        )
    except Exception as e:
        logger.error(f"Error loading report {report_id}: {e}")
        raise HTTPException(status_code=404, detail=f"Report {report_id} not found")

@app.get("/api/reports/{report_id}/sections/{section_id}", response_model=SectionData)
async def get_section_data(
    report_id: str, 
    section_id: str,
    table_type: Optional[str] = "raw",
    level: Optional[str] = "genus"
) -> SectionData:
    """Get specific section data with filtering"""
    try:
        report_data = get_stored_report(report_id)
        sections = report_data["sections"]
        
        # Find the matching section
        section_info = None
        for section in sections:
            if section.id == section_id:
                section_info = section
                break
        
        if not section_info:
            raise HTTPException(status_code=404, detail=f"Section {section_id} not found")
        
        # Get figures for this section
        figures = []
        all_figures = report_data.get("figures", {})
        for figure_id, figure_data in all_figures.items():
            if figure_data.metadata.get("section") == section_id:
                # Check if filters match
                if (figure_data.metadata.get("table_type", table_type) == table_type and 
                    figure_data.metadata.get("level", level) == level):
                    figures.append({
                        "id": figure_id,
                        "type": figure_data.type,
                        "metadata": figure_data.metadata
                    })
        
        # Get tables for this section
        tables = []
        all_tables = report_data.get("tables", {})
        for table_id, table_data in all_tables.items():
            if section_id in table_id:
                tables.append({
                    "id": table_id,
                    "data": table_data,
                    "title": table_id.replace("_", " ").title()
                })
        
        return SectionData(
            id=section_id,
            title=section_info.title,
            type=section_info.type,
            figures=figures,
            tables=tables,
            metadata={
                "table_type": table_type, 
                "level": level,
                "filters": section_info.filters
            }
        )
    except Exception as e:
        logger.error(f"Error loading section {section_id} for report {report_id}: {e}")
        raise HTTPException(status_code=404, detail=f"Section {section_id} not found")

@app.get("/api/reports/{report_id}/figures/{figure_id}", response_model=FigureData)
async def get_figure_data(report_id: str, figure_id: str) -> FigureData:
    """Get individual figure data for lazy loading"""
    try:
        report_data = get_stored_report(report_id)
        all_figures = report_data.get("figures", {})
        
        if figure_id not in all_figures:
            raise HTTPException(status_code=404, detail=f"Figure {figure_id} not found")
        
        figure_data = all_figures[figure_id]
        
        return FigureData(
            id=figure_id,
            type=figure_data.type,
            data=figure_data.data,
            layout=figure_data.data.get("layout") if figure_data.type == "plotly" else None,
            metadata=figure_data.metadata
        )
    except Exception as e:
        logger.error(f"Error loading figure {figure_id} for report {report_id}: {e}")
        raise HTTPException(status_code=404, detail=f"Figure {figure_id} not found")

# Mount static files for serving the React frontend
# app.mount("/static", StaticFiles(directory="frontend/build/static"), name="static")
# app.mount("/", StaticFiles(directory="frontend/build", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
