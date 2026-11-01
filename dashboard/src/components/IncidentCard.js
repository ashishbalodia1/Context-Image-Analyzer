import React from 'react';

const SEVERITY_COLORS = {
  CRITICAL: { bg: '#FFEBEE', border: '#FF0000', text: '#B71C1C' },
  HIGH: { bg: '#FFF3E0', border: '#FF8C00', text: '#E65100' },
  MEDIUM: { bg: '#FFFDE7', border: '#FFD700', text: '#F57F17' },
  LOW: { bg: '#E8F5E9', border: '#32CD32', text: '#1B5E20' },
  NONE: { bg: '#F5F5F5', border: '#9E9E9E', text: '#424242' },
};

function IncidentCard({ report }) {
  const severity = report.severity?.level || 'LOW';
  const colors = SEVERITY_COLORS[severity] || SEVERITY_COLORS.LOW;

  return (
    <div style={{
      border: `2px solid ${colors.border}`,
      backgroundColor: colors.bg,
      borderRadius: '8px',
      padding: '16px',
      marginBottom: '12px',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h3 style={{ margin: 0, color: colors.text }}>{report.domain_name}</h3>
        <SeverityBadge level={severity} />
      </div>
      <p style={{ margin: '8px 0 4px' }}><strong>Location:</strong> {report.location?.address || 'N/A'}</p>
      <p style={{ margin: '4px 0' }}><strong>Detected:</strong> {report.detection_summary?.detected_objects?.join(', ') || 'None'}</p>
      <p style={{ margin: '4px 0' }}><strong>Department:</strong> {report.response?.responding_department || 'N/A'}</p>
      <p style={{ margin: '4px 0', fontSize: '12px', color: '#757575' }}>
        {report.report_id} · {new Date(report.timestamp).toLocaleString()}
      </p>
    </div>
  );
}

function SeverityBadge({ level }) {
  const colors = SEVERITY_COLORS[level] || SEVERITY_COLORS.LOW;
  return (
    <span style={{
      backgroundColor: colors.border,
      color: '#fff',
      padding: '4px 10px',
      borderRadius: '12px',
      fontSize: '12px',
      fontWeight: 'bold',
    }}>
      {level}
    </span>
  );
}

export default IncidentCard;
