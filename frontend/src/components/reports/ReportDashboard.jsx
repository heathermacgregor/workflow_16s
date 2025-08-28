// src/components/reports/ReportDashboard.jsx
import React, { useState, useEffect } from 'react';
import { useReportData } from '../../hooks/useReportData';
import SectionViewer from './SectionViewer';
import Sidebar from '../navigation/Sidebar';
import './ReportDashboard.css';

const ReportDashboard = ({ reportId }) => {
  const { data, loading, error } = useReportData(reportId);
  const [activeSection, setActiveSection] = useState('analysis-summary');
  const [filters, setFilters] = useState({
    tableType: 'raw',
    level: 'genus',
    method: 'braycurtis'
  });

  if (loading) return <div className="spinner">Loading report...</div>;
  if (error) return <div className="error">Error: {error.message}</div>;

  return (
    <div className="report-dashboard">
      <Sidebar 
        sections={data?.sections || []}
        activeSection={activeSection}
        onSectionChange={setActiveSection}
      />
      <main className="report-content">
        <SectionViewer 
          reportId={reportId}
          sectionId={activeSection}
          filters={filters}
          onFiltersChange={setFilters}
        />
      </main>
    </div>
  );
};

export default ReportDashboard;