import React, { useState, useEffect } from 'react';
import IncidentCard from '../components/IncidentCard';
import DomainFilter from '../components/DomainFilter';
import { getReports } from '../utils/api';

const Dashboard = () => {
  const [reports, setReports] = useState([]);
  const [selectedDomain, setSelectedDomain] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchData();
  }, [selectedDomain]);

  const fetchData = async () => {
    try {
      setLoading(true);
      const data = await getReports(selectedDomain);
      setReports(data.reports || []);
    } catch (err) {
      console.error('Error fetching reports:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: '24px', maxWidth: '1200px', margin: '0 auto' }}>
      <h1>City Guardian: Municipal Authority Dashboard</h1>
      <DomainFilter selected={selectedDomain} onSelect={setSelectedDomain} />
      {loading ? (
        <p>Loading incident reports...</p>
      ) : reports.length === 0 ? (
        <p>No active incidents found for this filter.</p>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(360px, 1fr))', gap: '16px', marginTop: '20px' }}>
          {reports.map((report) => (
            <IncidentCard key={report.report_id} report={report} />
          ))}
        </div>
      )}
    </div>
  );
};

export default Dashboard;

