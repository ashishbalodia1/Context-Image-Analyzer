import React from 'react';
import MapView from '../components/MapView';

const IncidentMap = () => {
  return (
    <div style={{ padding: '24px' }}>
      <h2>Geospatial Pollution Hotspots & Incidents</h2>
      <MapView />
    </div>
  );
};

export default IncidentMap;

