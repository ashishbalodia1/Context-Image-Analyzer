import React, { useEffect, useState } from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup } from 'react-leaflet';
import { getIncidentGeoJSON } from '../utils/api';
import 'leaflet/dist/leaflet.css';

const SEVERITY_COLORS = {
  CRITICAL: '#FF0000',
  HIGH: '#FF8C00',
  MEDIUM: '#FFD700',
  LOW: '#32CD32',
  NONE: '#808080',
};

function MapView({ domainFilter }) {
  const [features, setFeatures] = useState([]);
  const defaultCenter = [28.6139, 77.2090]; // Delhi

  useEffect(() => {
    getIncidentGeoJSON(domainFilter)
      .then((data) => setFeatures(data.features || []))
      .catch(console.error);
  }, [domainFilter]);

  return (
    <MapContainer center={defaultCenter} zoom={11} style={{ height: '500px', width: '100%' }}>
      <TileLayer
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        attribution='© OpenStreetMap contributors'
      />
      {features.map((feature, idx) => {
        const [lon, lat] = feature.geometry.coordinates;
        const props = feature.properties;
        const color = SEVERITY_COLORS[props.severity_level] || '#808080';
        return (
          <CircleMarker key={idx} center={[lat, lon]} radius={10} color={color} fillOpacity={0.7}>
            <Popup>
              <strong>{props.domain_name}</strong><br />
              <span>{props.priority_label}</span><br />
              <span>{props.address}</span><br />
              <small>ID: {props.report_id}</small>
            </Popup>
          </CircleMarker>
        );
      })}
    </MapContainer>
  );
}

export default MapView;
