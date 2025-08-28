// src/components/reports/SectionViewer.jsx
import React from 'react';
import { useSectionData } from '../../hooks/useReportData';
import PlotlyWrapper from './PlotlyWrapper';
import DataTable from './DataTable';
import './SectionViewer.css';

const SectionViewer = ({ reportId, sectionId, filters, onFiltersChange }) => {
  const { data: sectionData, loading, error } = useSectionData(reportId, sectionId, filters);

  if (loading) return <div className="section-loading">Loading section...</div>;
  if (error) return <div className="section-error">Error: {error.message}</div>;

  const handleFilterChange = (filterName, value) => {
    onFiltersChange({
      ...filters,
      [filterName]: value
    });
  };

  return (
    <div className="section-viewer">
      <header className="section-header">
        <h2>{sectionData?.title || sectionId}</h2>
        
        {/* Filter Controls */}
        <div className="filter-controls">
          {sectionData?.metadata?.filters?.table_type && (
            <select 
              value={filters.tableType}
              onChange={(e) => handleFilterChange('tableType', e.target.value)}
            >
              {sectionData.metadata.filters.table_type.map(type => (
                <option key={type} value={type}>{type}</option>
              ))}
            </select>
          )}
          
          {sectionData?.metadata?.filters?.level && (
            <select 
              value={filters.level}
              onChange={(e) => handleFilterChange('level', e.target.value)}
            >
              {sectionData.metadata.filters.level.map(level => (
                <option key={level} value={level}>{level}</option>
              ))}
            </select>
          )}
        </div>
      </header>

      <div className="section-content">
        {/* Figures */}
        {sectionData?.figures?.map((figure, index) => (
          <div key={figure.id || index} className="figure-container">
            <PlotlyWrapper 
              figureId={figure.id}
              filters={filters}
              className="section-plot"
            />
          </div>
        ))}

        {/* Tables */}
        {sectionData?.tables?.map((table, index) => (
          <div key={table.id || index} className="table-container">
            <DataTable 
              data={table.data || []}
              columns={table.columns || []}
              title={table.title}
            />
          </div>
        ))}
      </div>
    </div>
  );
};

export default SectionViewer;