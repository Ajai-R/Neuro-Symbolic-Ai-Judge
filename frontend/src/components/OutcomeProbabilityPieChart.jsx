import React, { useState } from 'react';
import { OUTCOME_META } from '../services/pipelineEngine';

const OUTCOME_CONFIG = {
  INFORMATION_DIRECTED: {
    label: 'Information Directed',
    shortLabel: 'Full Disclosure',
    statute: 'Sec 7(1)',
    color: '#059669',
    hoverColor: '#10B981',
    lightBg: '#ECFDF5',
    border: '#047857'
  },
  APPEAL_DISMISSED: {
    label: 'Appeal Dismissed',
    shortLabel: 'Exemption Upheld',
    statute: 'Sec 8(1) / 24',
    color: '#4338CA',
    hoverColor: '#6366F1',
    lightBg: '#EEF2FF',
    border: '#3730A3'
  },
  PENALTY_IMPOSED: {
    label: 'Penalty Imposed',
    shortLabel: 'Sec 20 Show-Cause',
    statute: 'Sec 20(1) & (2)',
    color: '#DC2626',
    hoverColor: '#EF4444',
    lightBg: '#FEF2F2',
    border: '#B91C1C'
  },
  PARTIAL_RELIEF: {
    label: 'Partial Relief',
    shortLabel: 'Severability',
    statute: 'Sec 10(1)',
    color: '#D97706',
    hoverColor: '#F59E0B',
    lightBg: '#FFFBEB',
    border: '#B45309'
  }
};

/**
 * Calculates SVG arc path for a Donut/Pie slice
 */
function getDonutSlicePath(cx, cy, rInner, rOuter, startAngleDeg, endAngleDeg) {
  // Clamp angle delta
  const delta = endAngleDeg - startAngleDeg;
  if (delta >= 359.99) {
    // Almost full circle: return two 180 deg arcs
    const midAngle = startAngleDeg + 180;
    return `${getDonutSlicePath(cx, cy, rInner, rOuter, startAngleDeg, midAngle)} ${getDonutSlicePath(cx, cy, rInner, rOuter, midAngle, endAngleDeg)}`;
  }

  const toRad = deg => (deg * Math.PI) / 180;
  const startRad = toRad(startAngleDeg);
  const endRad = toRad(endAngleDeg);

  const x1 = cx + rOuter * Math.cos(startRad);
  const y1 = cy + rOuter * Math.sin(startRad);
  const x2 = cx + rOuter * Math.cos(endRad);
  const y2 = cy + rOuter * Math.sin(endRad);

  const x3 = cx + rInner * Math.cos(endRad);
  const y3 = cy + rInner * Math.sin(endRad);
  const x4 = cx + rInner * Math.cos(startRad);
  const y4 = cy + rInner * Math.sin(startRad);

  const largeArcFlag = delta > 180 ? 1 : 0;

  return `M ${x1} ${y1} A ${rOuter} ${rOuter} 0 ${largeArcFlag} 1 ${x2} ${y2} L ${x3} ${y3} A ${rInner} ${rInner} 0 ${largeArcFlag} 0 ${x4} ${y4} Z`;
}

export default function OutcomeProbabilityPieChart({
  probabilities = {},
  finalOutcome = 'INFORMATION_DIRECTED'
}) {
  const [hoveredKey, setHoveredKey] = useState(null);

  // 1. Extract and sanitize probabilities
  const classes = ['INFORMATION_DIRECTED', 'APPEAL_DISMISSED', 'PENALTY_IMPOSED', 'PARTIAL_RELIEF'];
  const rawList = classes.map(key => ({
    key,
    prob: typeof probabilities[key] === 'number' ? probabilities[key] : (key === finalOutcome ? 0.70 : 0.10)
  }));

  const rawSum = rawList.reduce((acc, it) => acc + it.prob, 0);
  const denom = rawSum > 0 ? rawSum : 1.0;

  // 2. Normalize so sum is guaranteed to equal 1.00 (100.0%)
  let cumulativeAngle = -90; // Start at 12 o'clock
  const slices = rawList.map(it => {
    const normProb = it.prob / denom;
    const pct = Math.round(normProb * 1000) / 10; // 1 decimal place
    const angle = normProb * 360;
    const startAngle = cumulativeAngle;
    const endAngle = cumulativeAngle + angle;
    cumulativeAngle = endAngle;

    const config = OUTCOME_CONFIG[it.key] || {
      label: it.key,
      shortLabel: it.key,
      statute: 'RTI Act',
      color: '#64748B',
      hoverColor: '#475569',
      lightBg: '#F8FAFC'
    };

    return {
      key: it.key,
      prob: normProb,
      pct,
      startAngle,
      endAngle,
      angle,
      config,
      isWinner: it.key === finalOutcome
    };
  }).sort((a, b) => b.prob - a.prob);

  // SVG dimensions
  const size = 200;
  const cx = size / 2;
  const cy = size / 2;
  const rOuter = 82;
  const rInner = 50;

  // Active slice for center display
  const activeSlice = hoveredKey 
    ? slices.find(s => s.key === hoveredKey) 
    : slices.find(s => s.isWinner) || slices[0];

  return (
    <div className="outcome-pie-card">
      <div className="pie-card-header">
        <div className="pie-card-title-group">
          <div className="pie-card-title">
            Prospective Adjudication Likelihood
          </div>
          <div className="pie-card-badge">
            Σ P = 1.00 (100%)
          </div>
        </div>
        <p className="pie-card-subtitle">
          Advisory scenario breakdown estimating appellate rulings under the RTI Act, 2005.
        </p>
      </div>

      <div className="pie-chart-body">
        {/* SVG Donut Visual */}
        <div className="pie-svg-wrapper">
          <svg 
            width={size} 
            height={size} 
            viewBox={`0 0 ${size} ${size}`} 
            className="donut-svg"
          >
            <defs>
              <filter id="pie-glow" x="-20%" y="-20%" width="140%" height="140%">
                <feDropShadow dx="0" dy="2" stdDeviation="3" floodOpacity="0.25" />
              </filter>
            </defs>

            {/* Slices */}
            {slices.map(slice => {
              if (slice.angle <= 0.1) return null;
              const isHovered = hoveredKey === slice.key;
              const pathData = getDonutSlicePath(cx, cy, rInner, rOuter, slice.startAngle, slice.endAngle);

              return (
                <path
                  key={slice.key}
                  d={pathData}
                  fill={isHovered ? slice.config.hoverColor : slice.config.color}
                  stroke="#FFFFFF"
                  strokeWidth={slices.filter(s => s.angle > 0.1).length > 1 ? 2 : 0}
                  className="donut-slice-path"
                  style={{
                    cursor: 'pointer',
                    transition: 'all 0.25s cubic-bezier(0.4, 0, 0.2, 1)',
                    opacity: hoveredKey && !isHovered ? 0.45 : 1.0,
                    filter: isHovered ? 'url(#pie-glow)' : 'none'
                  }}
                  onMouseEnter={() => setHoveredKey(slice.key)}
                  onMouseLeave={() => setHoveredKey(null)}
                />
              );
            })}

            {/* Central Donut Readout */}
            <circle cx={cx} cy={cy} r={rInner - 3} fill="var(--bg-surface)" />
            <text 
              x={cx} 
              y={cy - 4} 
              textAnchor="middle" 
              className="donut-center-pct"
              fill={activeSlice ? activeSlice.config.color : 'var(--text-primary)'}
            >
              {activeSlice ? `${activeSlice.pct.toFixed(1)}%` : '100%'}
            </text>
            <text 
              x={cx} 
              y={cy + 13} 
              textAnchor="middle" 
              className="donut-center-label"
              fill="var(--text-secondary)"
            >
              {activeSlice?.isWinner ? 'Top Precedent' : (activeSlice?.config?.shortLabel || 'Probability')}
            </text>
          </svg>
        </div>

        {/* Legend & Breakdown Table */}
        <div className="pie-legend-container">
          {slices.map(slice => {
            const isHovered = hoveredKey === slice.key;
            return (
              <div
                key={slice.key}
                className={`pie-legend-item ${slice.isWinner ? 'winner-legend' : ''} ${isHovered ? 'hovered-legend' : ''}`}
                onMouseEnter={() => setHoveredKey(slice.key)}
                onMouseLeave={() => setHoveredKey(null)}
              >
                <div className="pie-legend-left">
                  <span 
                    className="pie-legend-dot" 
                    style={{ backgroundColor: slice.config.color }}
                  />
                  <div className="pie-legend-names">
                    <span className="pie-legend-label">
                      {slice.config.label}
                    </span>
                    <span className="pie-legend-statute">
                      {slice.config.statute}
                    </span>
                  </div>
                </div>

                <div className="pie-legend-right">
                  <span className="pie-legend-prob-val">
                    {slice.pct.toFixed(1)}%
                  </span>
                  <span className="pie-legend-prob-decimal">
                    P = {slice.prob.toFixed(3)}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Advisory Legal Disclaimer */}
      <div className="pie-advisory-footer">
        <span className="advisory-icon">⚖</span>
        <span className="advisory-text">
          <strong>Decision Support Notice:</strong> This distribution visualizes prospective case outcomes based on multi-signal consensus. It serves as an informative pre-litigation guide for citizens and CPIOs, not an authoritative judicial order.
        </span>
      </div>
    </div>
  );
}
