// src/components/navigation/Sidebar.jsx
import React from 'react';
import './Sidebar.css';

const Sidebar = ({ sections, activeSection, onSectionChange }) => {
  return (
    <nav className="sidebar">
      <div className="sidebar-header">
        <h2>16S Analysis Report</h2>
      </div>
      
      <ul className="section-list">
        <li className={activeSection === 'analysis-summary' ? 'active' : ''}>
          <button 
            onClick={() => onSectionChange('analysis-summary')}
            className="section-button"
          >
            Analysis Summary
          </button>
        </li>
        
        {sections.map(section => (
          <li key={section.id} className={activeSection === section.id ? 'active' : ''}>
            <button 
              onClick={() => onSectionChange(section.id)}
              className="section-button"
              disabled={!section.has_data}
            >
              {section.title}
              {!section.has_data && <span className="no-data">(No data)</span>}
            </button>
          </li>
        ))}
      </ul>
      
      <div className="sidebar-footer">
        <div className="expand-controls">
          <button className="expand-button">Expand All</button>
          <button className="collapse-button">Collapse All</button>
        </div>
      </div>
    </nav>
  );
};

export default Sidebar;