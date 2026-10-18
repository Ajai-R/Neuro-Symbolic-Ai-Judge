import React from 'react';

export default function Signal3Card({ signal3 }) {
  if (!signal3) return null;

  const {
    detectedSections = [],
    flags = {},
    statutoryFramework = "RTI Act, 2005 (Formal Legal Ontology)",
    precedentsMatched = []
  } = signal3;

  const boundaryChecks = [
    {
      label: 'Section 24 Jurisdictional Immunity Bar',
      provision: 'Sec 24 • Second Schedule',
      triggered: Boolean(flags.is_sec24_claimed),
      activeText: 'EXEMPT AGENCY BAR (JURISDICTION PRECLUDED)',
      inactiveText: 'Standard Public Authority (Jurisdiction Sustained)',
      legalEffect: 'Renders the entire application non-maintainable unless corruption or human rights violations are established.'
    },
    {
      label: 'Section 8(1) Exemption Plea',
      provision: 'Sec 8(1)(a)-(j)',
      triggered: Boolean(flags.is_sec8_claimed),
      activeText: 'EXEMPTION DEFENSE RAISED',
      inactiveText: 'No Exemption Pleaded (Mandatory Disclosure Applies)',
      legalEffect: 'Places burden of establishing harm/commercial injury squarely upon the public authority.'
    },
    {
      label: 'Section 20 Show-Cause Penalty Sanctions',
      provision: 'Sec 20(1) & 20(2)',
      triggered: Boolean(flags.is_sec20_penalty_risk),
      activeText: 'PENAL INQUIRY RISK TRIGGERED (>30D DELAY)',
      inactiveText: 'Within Timelines / No Mala Fide Refusal',
      legalEffect: 'Imposes statutory liability of Rs 250/day up to Rs 25,000 against delinquent PIO for unexcused delay.'
    },
    {
      label: 'Section 10 Severability Principle',
      provision: 'Sec 10(1)',
      triggered: Boolean(flags.is_severability_indicated),
      activeText: 'SEVERANCE REQUIRED',
      inactiveText: 'Unified Document Disposition',
      legalEffect: 'Directs supply of non-exempt factual portions while withholding protected clauses.'
    }
  ];

  return (
    <div className="signal-detail-card">
      {/* Header Row */}
      <div className="signal-meta-row">
        <div className="signal-title-wrap">
          <h3 className="signal-card-heading">Signal 3: Declarative Statutory Ontology & Boundary Scanner</h3>
        </div>
        <span className="signal-chip">RTI Act, 2005 Declarative Rules</span>
      </div>

      {/* Statutory Boundary & Exemption Analysis Ledger (Replacing 4 multi-colored SaaS boxes) */}
      <table className="statutory-status-table">
        <thead>
          <tr>
            <th style={{ width: '32%' }}>Statutory Gate / Boundary</th>
            <th style={{ width: '20%' }}>Governing Section</th>
            <th style={{ width: '22%' }}>Evaluation Status</th>
            <th style={{ width: '26%' }}>Statutory Legal Effect</th>
          </tr>
        </thead>
        <tbody>
          {boundaryChecks.map((gate, i) => (
            <tr key={i}>
              <td style={{ fontWeight: 600, color: 'var(--navy-primary)' }}>
                {gate.label}
              </td>
              <td style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                {gate.provision}
              </td>
              <td>
                {gate.triggered ? (
                  <span style={{ 
                    fontFamily: 'var(--font-mono)', 
                    fontSize: '0.675rem', 
                    fontWeight: 700, 
                    color: 'var(--state-amber-text)', 
                    background: 'var(--state-amber-bg)', 
                    border: '1px solid var(--state-amber-border)', 
                    padding: '0.1rem 0.4rem', 
                    borderRadius: 'var(--radius-xs)' 
                  }}>
                    {gate.activeText}
                  </span>
                ) : (
                  <span style={{ 
                    fontFamily: 'var(--font-mono)', 
                    fontSize: '0.675rem', 
                    color: 'var(--text-muted)' 
                  }}>
                    {gate.inactiveText}
                  </span>
                )}
              </td>
              <td style={{ fontSize: '0.725rem', color: 'var(--text-secondary)', lineHeight: 1.4 }}>
                {gate.legalEffect}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {/* Identified Statutory Provisions List */}
      <span style={{ fontSize: '0.7rem', fontWeight: 700, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.06em', display: 'block', marginBottom: '0.65rem' }}>
        Identified Statutory Provisions & Cited Grounds
      </span>

      {detectedSections && detectedSections.length > 0 ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', marginBottom: '1.15rem' }}>
          {detectedSections.map((sec, idx) => (
            <div key={idx} style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-xs)', padding: '0.65rem 0.85rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', marginBottom: '0.2rem' }}>
                <span className="step-index-badge">
                  {sec.section}
                </span>
                <span style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--navy-primary)' }}>
                  {sec.title}
                </span>
              </div>
              <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', margin: 0, lineHeight: 1.45 }}>
                {sec.detail}
              </p>
            </div>
          ))}
        </div>
      ) : (
        <div style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-xs)', padding: '0.75rem 1rem', color: 'var(--text-secondary)', fontSize: '0.775rem', marginBottom: '1.15rem' }}>
          No statutory exemptions or Section 20 red flags detected in respondent submissions. The general statutory presumption of disclosure under Section 7(1) remains fully operative.
        </div>
      )}

      {/* Jurisprudential Precedent Alignment (Supreme Court Headnotes) */}
      {precedentsMatched && precedentsMatched.length > 0 && (
        <div style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-xs)', padding: '0.85rem 1rem' }}>
          <div style={{ fontSize: '0.7rem', fontWeight: 700, color: 'var(--navy-primary)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '0.55rem' }}>
            Binding Jurisprudential Precedent Grounding
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.55rem' }}>
            {precedentsMatched.map((prec, i) => {
              const citation = typeof prec === 'object' && prec !== null ? prec.citation : String(prec);
              const principle = typeof prec === 'object' && prec !== null ? prec.principle : null;

              return (
                <div key={i} style={{ fontSize: '0.75rem', borderLeft: '2px solid var(--navy-primary)', paddingLeft: '0.65rem' }}>
                  <div style={{ fontWeight: 700, color: 'var(--navy-primary)', fontFamily: 'var(--font-serif)' }}>
                    {citation}
                  </div>
                  {principle && (
                    <div style={{ color: 'var(--text-secondary)', marginTop: '0.1rem', lineHeight: 1.4 }}>
                      {principle}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
