import React from 'react';

const SEVERITY_COLORS = {
  CRITICAL: '#FF0000',
  HIGH: '#FF8C00',
  MEDIUM: '#FFD700',
  LOW: '#32CD32',
  NONE: '#9E9E9E',
};

function SeverityBadge({ level, score }) {
  const color = SEVERITY_COLORS[level] || '#9E9E9E';
  return (
    <div style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}>
      <span style={{
        backgroundColor: color,
        color: '#fff',
        padding: '4px 12px',
        borderRadius: '12px',
        fontSize: '13px',
        fontWeight: 'bold',
      }}>
        {level}
      </span>
      {score !== undefined && (
        <span style={{ fontSize: '12px', color: '#757575' }}>
          ({(score * 100).toFixed(0)}%)
        </span>
      )}
    </div>
  );
}

export default SeverityBadge;
