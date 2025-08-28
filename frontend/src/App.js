// src/App.js
import React from 'react';
import { QueryClient, QueryClientProvider } from 'react-query';
import ReportDashboard from './components/reports/ReportDashboard';
import './App.css';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 2,
      refetchOnWindowFocus: false,
    },
  },
});

function App() {
  // For now, use a fixed report ID
  // In production, this would come from URL routing
  const reportId = 'current';

  return (
    <QueryClientProvider client={queryClient}>
      <div className="App">
        <ReportDashboard reportId={reportId} />
      </div>
    </QueryClientProvider>
  );
}

export default App;