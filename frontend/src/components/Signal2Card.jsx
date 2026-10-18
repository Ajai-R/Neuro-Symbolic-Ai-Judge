import React from 'react';

export default function Signal2Card({ signal2 }) {
  if (!signal2) return null;

  const {
    totalItems = 0,
    addressedCount = 0,
    partialCount = 0,
    missingCount = 0,
    coverageRatio = 0,
    complianceVerdict = 'UNKNOWN',
    items = [],
    judgeModel,
    embeddingModel
  } = signal2;

  const compliancePercent = Math.round((coverageRatio || 0) * 100);

  return (
    <div className="signal-detail-card">
      {/* Header Row */}
      <div className="signal-meta-row">
        <div className="signal-title-wrap">
          <h3 className="signal-card-heading">Signal 2: Substantive Coverage Verification & LLM Judicial Auditor</h3>
        </div>
        <span className="signal-chip">
          {judgeModel || embeddingModel || 'LLM Judicial Auditor (Nemotron 3 Ultra 550B)'}
        </span>
      </div>

      {/* High-Density Compliance Ledger Strip (Replacing 4 generic SaaS KPI tiles) */}
      <div className="compliance-ledger-strip">
        <div className="ledger-metric-item">
          <span>Total Requisitions:</span>
          <strong>{totalItems}</strong>
        </div>
        <div className="ledger-metric-item">
          <span>Fully Disclosed:</span>
          <strong style={{ color: 'var(--state-green-text)' }}>{addressedCount}</strong>
        </div>
        <div className="ledger-metric-item">
          <span>Partially Disclosed:</span>
          <strong style={{ color: 'var(--state-amber-text)' }}>{partialCount}</strong>
        </div>
        <div className="ledger-metric-item">
          <span>Withheld / Evaded:</span>
          <strong style={{ color: 'var(--state-red-text)' }}>{missingCount}</strong>
        </div>
        <div className="ledger-metric-item">
          <span>Aggregate Substantive Score:</span>
          <strong style={{ color: 'var(--navy-primary)' }}>{compliancePercent}%</strong>
        </div>
      </div>

      {/* Itemized Audit Section Heading */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.65rem', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '0.45rem' }}>
        <span style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          Requisition Disclosure Status
        </span>
      </div>

      {/* Itemized Checklist */}
      <div className="checklist-container">
        {items.map((item, idx) => {
          const score = item.complianceScore ?? item.cosineSimilarity ?? 0.0;
          
          let badgeClass = 'badge-missing';
          let statusText = 'Not Answered / Omitted';

          if (score >= 0.85 || item.status === 'ADDRESSED' || item.status === 'FURNISHED') {
            badgeClass = 'badge-addressed';
            statusText = 'Answered / Disclosed';
          } else if (score >= 0.4 || item.status === 'PARTIALLY_PROVIDED' || item.status === 'PARTIALLY_ADDRESSED' || item.status === 'PARTIALLY_FURNISHED') {
            badgeClass = 'badge-partial';
            statusText = 'Partially Answered';
          } else if (item.status === 'WITHHELD' || item.status === 'WITHHELD_WITH_STATUTORY_CLAIM') {
            badgeClass = 'badge-withheld';
            statusText = 'Withheld / Refused';
          }

          return (
            <div key={item.queryIndex || idx} className="checklist-card">
              {/* Question & Status Row */}
              <div className="checklist-top-row">
                <div style={{ flex: 1, paddingRight: '0.75rem' }}>
                  <p className="checklist-query-text" style={{ fontSize: '0.85rem', color: 'var(--text-primary)', margin: 0 }}>
                    <strong>{item.queryText}</strong>
                  </p>
                </div>

                <div style={{ flexShrink: 0 }}>
                  <span className={`status-badge ${badgeClass}`} style={{ fontSize: '0.75rem', padding: '0.25rem 0.6rem' }}>
                    {statusText}
                  </span>
                </div>
              </div>

              {/* Matched Reply Context */}
              {item.matchedReplyContext && (
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', background: 'var(--bg-secondary)', border: '1px solid var(--border-subtle)', padding: '0.45rem 0.7rem', borderRadius: 'var(--radius-xs)', margin: '0.45rem 0' }}>
                  <span style={{ color: 'var(--navy-primary)', fontWeight: 600 }}>PIO Response Excerpt: </span>
                  <span>"{item.matchedReplyContext}"</span>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
