# ===================================== IMPORTS ====================================== #

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import json
import uuid
import logging
from pathlib import Path

# ===================================== TYPES ====================================== #

@dataclass
class FigureData:
    id: str
    type: str  # 'plotly' | 'image' | 'table'
    data: Dict[str, Any]
    metadata: Dict[str, Any]

@dataclass
class SectionInfo:
    id: str
    title: str
    type: str
    has_data: bool
    filters: Dict[str, List[str]]

# ===================================== PROCESSOR ====================================== #

class ReportDataProcessor:
    """
    Replace the current HTML generation with structured data extraction.
    This class extracts data from AmpliconData and converts it to JSON-serializable format.
    """
    
    def __init__(self, amplicon_data):
        self.amplicon_data = amplicon_data
        self.logger = logging.getLogger("workflow_16s_api")
    
    def extract_structured_data(self) -> Dict[str, Any]:
        """
        Extract all data from AmpliconData in a structured format
        that can be served by the API endpoints.
        """
        return {
            "metadata": self._extract_metadata(),
            "sections": self._extract_sections(),
            "figures": self._extract_all_figures(),
            "tables": self._extract_all_tables()
        }
    
    def _extract_metadata(self) -> Dict[str, Any]:
        """Extract report metadata"""
        import pandas as pd
        
        return {
            "generated_ts": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "title": "16S rRNA Amplicon Analysis Report",
            "sections_available": self._get_available_sections(),
            "config": getattr(self.amplicon_data, 'config', {})
        }
    
    def _extract_sections(self) -> List[SectionInfo]:
        """Extract section structure without HTML generation"""
        sections = []
        
        # Map section names to their data attributes
        section_mapping = {
            'alpha_diversity': 'alpha_diversity',
            'beta_diversity': 'ordination', 
            'stats': 'stats',
            'ml': 'models',
            'maps': 'maps'
        }
        
        for section_name, data_attr in section_mapping.items():
            if hasattr(self.amplicon_data, data_attr):
                section_data = getattr(self.amplicon_data, data_attr)
                sections.append(SectionInfo(
                    id=section_name,
                    title=section_name.replace('_', ' ').title(),
                    type=self._get_section_type(section_name),
                    has_data=bool(section_data),
                    filters=self._get_available_filters(section_data, section_name)
                ))
        
        return sections
    
    def _extract_all_figures(self) -> Dict[str, FigureData]:
        """Extract all figures with metadata for lazy loading"""
        figures = {}
        
        # Alpha diversity figures
        if hasattr(self.amplicon_data, 'alpha_diversity'):
            alpha_figures = self._process_alpha_diversity_figures()
            figures.update(alpha_figures)
        
        # Beta diversity figures  
        if hasattr(self.amplicon_data, 'ordination'):
            beta_figures = self._process_beta_diversity_figures()
            figures.update(beta_figures)
        
        # Machine learning figures
        if hasattr(self.amplicon_data, 'models'):
            ml_figures = self._process_ml_figures()
            figures.update(ml_figures)
        
        # Map figures
        if hasattr(self.amplicon_data, 'maps'):
            map_figures = self._process_map_figures()
            figures.update(map_figures)
            
        return figures
    
    def _extract_all_tables(self) -> Dict[str, Any]:
        """Extract all table data"""
        tables = {}
        
        # Statistical test results
        if hasattr(self.amplicon_data, 'stats') and self.amplicon_data.stats:
            stats_tables = self._process_stats_tables()
            tables.update(stats_tables)
        
        # Machine learning results
        if hasattr(self.amplicon_data, 'models'):
            ml_tables = self._process_ml_tables()
            tables.update(ml_tables)
        
        return tables
    
    # Helper methods
    
    def _get_available_sections(self) -> List[str]:
        """Get list of sections that have data"""
        available = []
        
        if hasattr(self.amplicon_data, 'alpha_diversity') and self.amplicon_data.alpha_diversity:
            available.append('alpha_diversity')
        if hasattr(self.amplicon_data, 'ordination') and self.amplicon_data.ordination:
            available.append('beta_diversity')
        if hasattr(self.amplicon_data, 'stats') and self.amplicon_data.stats:
            available.append('stats')
        if hasattr(self.amplicon_data, 'models') and self.amplicon_data.models:
            available.append('ml')
        if hasattr(self.amplicon_data, 'maps') and self.amplicon_data.maps:
            available.append('maps')
            
        return available
    
    def _get_section_type(self, section_name: str) -> str:
        """Get the type of section for UI rendering"""
        type_mapping = {
            'alpha_diversity': 'metrics',
            'beta_diversity': 'ordination',
            'stats': 'statistical_tests',
            'ml': 'machine_learning',
            'maps': 'geospatial'
        }
        return type_mapping.get(section_name, 'analysis')
    
    def _get_available_filters(self, section_data: Any, section_name: str) -> Dict[str, List[str]]:
        """Get available filter options for a section"""
        filters = {}
        
        if not section_data:
            return filters
            
        try:
            if section_name in ['alpha_diversity', 'beta_diversity', 'stats', 'ml']:
                if isinstance(section_data, dict):
                    # Extract table types
                    if 'raw' in section_data or 'normalized' in section_data or 'clr' in section_data:
                        filters['table_type'] = [k for k in section_data.keys() if isinstance(section_data[k], dict)]
                    
                    # Extract taxonomic levels
                    for table_type_data in section_data.values():
                        if isinstance(table_type_data, dict):
                            levels = list(table_type_data.keys())
                            if levels:
                                filters['level'] = levels
                                break
                                
        except Exception as e:
            self.logger.warning(f"Error extracting filters for {section_name}: {e}")
            
        return filters
    
    def _process_alpha_diversity_figures(self) -> Dict[str, FigureData]:
        """Process alpha diversity figures"""
        figures = {}
        
        try:
            for table_type, levels in self.amplicon_data.alpha_diversity.items():
                for level, data in levels.items():
                    if 'figures' in data and data['figures']:
                        for metric, figure_data in data['figures'].items():
                            figure_id = f"alpha_{table_type}_{level}_{metric}"
                            figures[figure_id] = FigureData(
                                id=figure_id,
                                type='plotly',
                                data=figure_data,
                                metadata={
                                    'section': 'alpha_diversity',
                                    'table_type': table_type,
                                    'level': level,
                                    'metric': metric
                                }
                            )
        except Exception as e:
            self.logger.error(f"Error processing alpha diversity figures: {e}")
            
        return figures
    
    def _process_beta_diversity_figures(self) -> Dict[str, FigureData]:
        """Process beta diversity/ordination figures"""
        figures = {}
        
        try:
            for table_type, levels in self.amplicon_data.ordination.items():
                for level, methods in levels.items():
                    for method, data in methods.items():
                        if data and 'figures' in data and data['figures']:
                            for figure_type, figure_data in data['figures'].items():
                                figure_id = f"beta_{table_type}_{level}_{method}_{figure_type}"
                                figures[figure_id] = FigureData(
                                    id=figure_id,
                                    type='plotly',
                                    data=figure_data,
                                    metadata={
                                        'section': 'beta_diversity',
                                        'table_type': table_type,
                                        'level': level,
                                        'method': method,
                                        'figure_type': figure_type
                                    }
                                )
        except Exception as e:
            self.logger.error(f"Error processing beta diversity figures: {e}")
            
        return figures
    
    def _process_ml_figures(self) -> Dict[str, FigureData]:
        """Process machine learning figures"""
        figures = {}
        
        try:
            for table_type, levels in self.amplicon_data.models.items():
                for level, methods in levels.items():
                    for method, model_result in methods.items():
                        if model_result and 'figures' in model_result:
                            for figure_type, figure_data in model_result['figures'].items():
                                figure_id = f"ml_{table_type}_{level}_{method}_{figure_type}"
                                figures[figure_id] = FigureData(
                                    id=figure_id,
                                    type='plotly',
                                    data=figure_data,
                                    metadata={
                                        'section': 'ml',
                                        'table_type': table_type,
                                        'level': level,
                                        'method': method,
                                        'figure_type': figure_type
                                    }
                                )
        except Exception as e:
            self.logger.error(f"Error processing ML figures: {e}")
            
        return figures
    
    def _process_map_figures(self) -> Dict[str, FigureData]:
        """Process map figures"""
        figures = {}
        
        try:
            if isinstance(self.amplicon_data.maps, dict):
                for map_type, figure_data in self.amplicon_data.maps.items():
                    figure_id = f"map_{map_type}"
                    figures[figure_id] = FigureData(
                        id=figure_id,
                        type='plotly',
                        data=figure_data,
                        metadata={
                            'section': 'maps',
                            'map_type': map_type
                        }
                    )
        except Exception as e:
            self.logger.error(f"Error processing map figures: {e}")
            
        return figures
    
    def _process_stats_tables(self) -> Dict[str, Any]:
        """Process statistical test result tables"""
        tables = {}
        
        try:
            if isinstance(self.amplicon_data.stats, dict):
                # Overall summary
                if 'summary' in self.amplicon_data.stats:
                    tables['stats_summary'] = self.amplicon_data.stats['summary']
                
                # Top features
                if 'top_features' in self.amplicon_data.stats:
                    tables['top_features'] = self.amplicon_data.stats['top_features'].to_dict('records') \
                        if hasattr(self.amplicon_data.stats['top_features'], 'to_dict') \
                        else self.amplicon_data.stats['top_features']
                
                # Individual test results
                for key, value in self.amplicon_data.stats.items():
                    if key not in ['summary', 'top_features', 'recommendations']:
                        tables[f'stats_{key}'] = value
                        
        except Exception as e:
            self.logger.error(f"Error processing stats tables: {e}")
            
        return tables
    
    def _process_ml_tables(self) -> Dict[str, Any]:
        """Process machine learning result tables"""
        tables = {}
        
        try:
            for table_type, levels in self.amplicon_data.models.items():
                for level, methods in levels.items():
                    for method, model_result in methods.items():
                        if model_result:
                            table_id = f"ml_{table_type}_{level}_{method}"
                            
                            # Extract non-figure data
                            table_data = {k: v for k, v in model_result.items() if k != 'figures'}
                            tables[table_id] = table_data
                            
        except Exception as e:
            self.logger.error(f"Error processing ML tables: {e}")
            
        return tables