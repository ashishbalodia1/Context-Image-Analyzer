import React from 'react';

const DOMAINS = [
  { key: null, label: 'All Domains', icon: '🌍' },
  { key: 'air_pollution', label: 'Air Pollution', icon: '🏭' },
  { key: 'water_quality', label: 'Water Quality', icon: '💧' },
  { key: 'sewage_drainage', label: 'Sewage & Drainage', icon: '🚰' },
  { key: 'mining_dust', label: 'Mining Dust', icon: '⛏️' },
  { key: 'solid_waste', label: 'Solid Waste', icon: '🗑️' },
];

function DomainFilter({ selected, onChange }) {
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginBottom: '16px' }}>
      {DOMAINS.map((domain) => (
        <button
          key={domain.key || 'all'}
          onClick={() => onChange(domain.key)}
          style={{
            padding: '6px 14px',
            borderRadius: '20px',
            border: '2px solid',
            borderColor: selected === domain.key ? '#1976D2' : '#BDBDBD',
            backgroundColor: selected === domain.key ? '#1976D2' : '#fff',
            color: selected === domain.key ? '#fff' : '#424242',
            cursor: 'pointer',
            fontSize: '13px',
            fontWeight: selected === domain.key ? 'bold' : 'normal',
          }}
        >
          {domain.icon} {domain.label}
        </button>
      ))}
    </div>
  );
}

export default DomainFilter;
