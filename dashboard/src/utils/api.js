import axios from 'axios';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
});

// Upload an image for analysis
export const uploadImage = async (file, domain = null, lat = null, lon = null) => {
  const formData = new FormData();
  formData.append('file', file);
  if (domain) formData.append('domain', domain);
  if (lat) formData.append('lat', lat);
  if (lon) formData.append('lon', lon);

  const response = await api.post('/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
};

// Get all reports
export const getReports = async (domain = null, severity = null) => {
  const params = {};
  if (domain) params.domain = domain;
  if (severity) params.severity = severity;
  const response = await api.get('/reports', { params });
  return response.data;
};

// Get a specific report
export const getReport = async (reportId) => {
  const response = await api.get(`/reports/${reportId}`);
  return response.data;
};

// Update report status
export const updateReportStatus = async (reportId, status) => {
  const response = await api.patch(`/reports/${reportId}/status`, { status });
  return response.data;
};

// Get GeoJSON incidents
export const getIncidentGeoJSON = async (domain = null) => {
  const params = domain ? { domain } : {};
  const response = await api.get('/map/incidents', { params });
  return response.data;
};

// Get hotspots
export const getHotspots = async (domain = null, minIncidents = 3) => {
  const params = { min_incidents: minIncidents };
  if (domain) params.domain = domain;
  const response = await api.get('/map/hotspots', { params });
  return response.data;
};

// Health check
export const healthCheck = async () => {
  const response = await api.get('/health');
  return response.data;
};

export default api;
