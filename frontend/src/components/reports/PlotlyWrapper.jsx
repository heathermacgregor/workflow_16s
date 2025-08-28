// src/components/reports/PlotlyWrapper.jsx
import React, { useEffect, useRef, useState } from 'react';
import Plotly from 'plotly.js-dist';

const PlotlyWrapper = ({ figureId, filters, className }) => {
  const plotRef = useRef();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const loadFigure = async () => {
      try {
        setLoading(true);
        setError(null);
        
        // Fetch figure data from API
        const response = await fetch(`/api/reports/current/figures/${figureId}?${new URLSearchParams(filters)}`);
        if (!response.ok) {
          throw new Error(`Failed to load figure: ${response.statusText}`);
        }
        
        const figureData = await response.json();
        
        if (figureData.type === 'plotly') {
          await Plotly.newPlot(
            plotRef.current,
            figureData.data,
            figureData.layout || {},
            { 
              responsive: true, 
              displayModeBar: true,
              modeBarButtonsToRemove: ['toImage']
            }
          );
        } else if (figureData.type === 'image') {
          plotRef.current.innerHTML = `<img src="data:image/png;base64,${figureData.data}" style="max-width: 100%;" />`;
        }
        
        setLoading(false);
      } catch (err) {
        setError(err.message);
        setLoading(false);
      }
    };

    if (figureId) {
      loadFigure();
    }

    // Cleanup
    return () => {
      if (plotRef.current && window.Plotly) {
        Plotly.purge(plotRef.current);
      }
    };
  }, [figureId, filters]);

  if (loading) return <div className="plot-loading">Loading plot...</div>;
  if (error) return <div className="plot-error">Error: {error}</div>;

  return <div ref={plotRef} className={`plotly-container ${className || ''}`} />;
};

export default PlotlyWrapper;