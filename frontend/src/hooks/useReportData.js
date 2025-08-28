// src/hooks/useReportData.js
import { useQuery } from 'react-query';

const API_BASE = process.env.REACT_APP_API_BASE || '';

const fetchReportData = async (reportId) => {
  const response = await fetch(`${API_BASE}/api/reports/${reportId}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch report: ${response.statusText}`);
  }
  return response.json();
};

const fetchSectionData = async (reportId, sectionId, filters) => {
  const params = new URLSearchParams(filters);
  const response = await fetch(`${API_BASE}/api/reports/${reportId}/sections/${sectionId}?${params}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch section: ${response.statusText}`);
  }
  return response.json();
};

const fetchFigureData = async (reportId, figureId, filters = {}) => {
  const params = new URLSearchParams(filters);
  const response = await fetch(`${API_BASE}/api/reports/${reportId}/figures/${figureId}?${params}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch figure: ${response.statusText}`);
  }
  return response.json();
};

export const useReportData = (reportId) => {
  return useQuery(
    ['report', reportId],
    () => fetchReportData(reportId),
    {
      staleTime: 5 * 60 * 1000, // 5 minutes
      cacheTime: 10 * 60 * 1000, // 10 minutes
      enabled: !!reportId
    }
  );
};

export const useSectionData = (reportId, sectionId, filters = {}) => {
  return useQuery(
    ['section', reportId, sectionId, filters],
    () => fetchSectionData(reportId, sectionId, filters),
    {
      keepPreviousData: true,
      staleTime: 2 * 60 * 1000, // 2 minutes
      enabled: !!(reportId && sectionId)
    }
  );
};

export const useFigureData = (reportId, figureId, filters = {}) => {
  return useQuery(
    ['figure', reportId, figureId, filters],
    () => fetchFigureData(reportId, figureId, filters),
    {
      staleTime: 10 * 60 * 1000, // 10 minutes
      enabled: !!(reportId && figureId)
    }
  );
};