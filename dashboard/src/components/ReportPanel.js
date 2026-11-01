import React from 'react';

function ReportPanel({ report, onClose }) {
  if (!report) return null;

  return (
    <div style={{
      position: 'fixed', right: 0, top: 0, bottom: 0,
      width: '420px', backgroundColor: '#fff',
      boxShadow: '-4px 0 16px rgba(0,0,0,0.15)',
      overflowY: 'auto', padding: '24px', zIndex: 1000,
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2 style={{ margin: 0 }}>{report.domain_name}</h2>
        <button onClick={onClose} style={{ border: 'none', background: 'none', fontSize: '20px', cursor: 'pointer' }}>✕</button>
      </div>

      <hr />

      <section>
        <h3>📍 Location</h3>
        <p>{report.location?.address || 'N/A'}</p>
        <p><strong>Ward:</strong> {report.location?.ward || 'N/A'} | <strong>City:</strong> {report.location?.city || 'N/A'}</p>
      </section>

      <section>
        <h3>🔍 Detection Summary</h3>
        <p><strong>Objects Detected:</strong> {report.detection_summary?.detected_objects?.join(', ') || 'None'}</p>
        <p><strong>Total Detections:</strong> {report.detection_summary?.total_objects_detected}</p>
        <p><strong>Confidence Range:</strong> {report.detection_summary?.confidence_range}</p>
      </section>

      <section>
        <h3>📋 AI Context</h3>
        <p style={{ lineHeight: 1.6, color: '#424242' }}>{report.ai_context}</p>
      </section>

      <section>
        <h3>🏛️ Response</h3>
        <p><strong>Department:</strong> {report.response?.responding_department}</p>
        <p><strong>Action:</strong> {report.response?.recommended_action}</p>
        <p><strong>SLA:</strong> {report.response?.sla_hours} hours</p>
      </section>

      <p style={{ fontSize: '12px', color: '#9E9E9E' }}>
        Report ID: {report.report_id} | {new Date(report.timestamp).toLocaleString()}
      </p>
    </div>
  );
}

export default ReportPanel;
