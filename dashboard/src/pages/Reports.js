import React, { useState, useEffect } from 'react';
import ReportPanel from '../components/ReportPanel';
import { getReports } from '../utils/api';

const Reports = () => {
  const [reports, setReports] = useState([]);

  useEffect(() => {
    getReports().then((data) => setReports(data.reports || [])).catch(console.error);
  }, []);

  return (
    <div style={{ padding: '24px' }}>
      <h2>Detailed Environmental Compliance Reports</h2>
      {reports.map((report) => (
        <ReportPanel key={report.report_id} report={report} />
      ))}
    </div>
  );
};

export default Reports;

