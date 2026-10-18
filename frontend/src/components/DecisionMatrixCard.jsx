import React from 'react';
import { OUTCOME_META } from '../services/pipelineEngine';
import OutcomeProbabilityPieChart from './OutcomeProbabilityPieChart';

const OUTCOME_DESCRIPTIONS = {
  PARTIAL_RELIEF: 'The Doctrine of Severability under Section 10(1) is invoked. Non-exempt factual information is ordered for immediate disclosure, while portions qualifying under Section 8(1) remain protected.',
  INFORMATION_DIRECTED: 'Complete disclosure directed pursuant to Section 7(1). The respondent public authority failed to discharge the statutory burden of establishing exemptions under Section 8(1) or Section 24.',
  APPEAL_DISMISSED: 'The administrative refusal by the public authority is sustained pursuant to Section 8(1) non-disclosure provisions or Section 24 intelligence/security jurisdictional immunity.',
  PENALTY_IMPOSED: 'Sanction and show-cause proceedings initiated under Section 20(1) and Section 20(2) for unexcused statutory delay, bad-faith obstruction, or misleading claims.'
};

const STATUTORY_GROUNDS = {
  PARTIAL_RELIEF: 'Section 10(1) — Doctrine of Severability',
  INFORMATION_DIRECTED: 'Section 7(1) — Mandatory Duty to Disclose',
  APPEAL_DISMISSED: 'Section 8(1) / Section 24 — Non-Disclosure Immunity',
  PENALTY_IMPOSED: 'Section 20(1) & 20(2) — Penalties & Disciplinary Action'
};

export default function DecisionMatrixCard({ stage4 }) {
  if (!stage4) return null;

  const {
    finalOutcome,
    outcomeMeta,
    overrideApplied,
    ruleTriggered,
    certaintyScore,
    legalReasoningTrace = [],
    judicialDirections = [],
    fusedProbabilities = {},
    governingClause,
    rawSignalVotes = {},
    latency,
    caseFinding
  } = stage4;

  const meta = outcomeMeta || OUTCOME_META[finalOutcome] || {};
  const todayStr = new Date().toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' });
  const rawFindingText = caseFinding || 'LLM not working';
  const displayFinding = rawFindingText.replace(/^Case Finding:\s*/i, '');

  // Format probabilities for analytical table
  const outcomeEntries = Object.entries(fusedProbabilities).map(([cls, prob]) => {
    const pct = Math.round(prob * 100);
    const label = OUTCOME_META[cls]?.label || cls.replace(/_/g, ' ');
    const isWinner = cls === finalOutcome;
    const statGround = STATUTORY_GROUNDS[cls] || 'Statutory Consensus';

    return {
      cls,
      label,
      pct,
      prob,
      isWinner,
      statGround
    };
  }).sort((a, b) => b.prob - a.prob);

  return (
    <div>
      {/* =========================================================================
          PART I: PRIMARY JUDICIAL OPINION & OPERATIVE DIRECTIVES
          ========================================================================= */}
      
      {/* Primary Ruling & Prospective Case Probability Grid (Directly Near Output) */}
      <div className="verdict-primary-grid">
        {/* 1. Official Appellate Order Dossier */}
        <div className={`hero-verdict-card outcome-${(finalOutcome || '').toLowerCase()}`}>
          <div className="hero-verdict-top">
            <div className="verdict-bench-label">
              Appellate Advisory Dossier • Docket Ref: CIC/ADV/2026/0491
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem' }}>
              <span style={{ fontSize: '0.725rem', color: 'var(--text-muted)' }}>
                Adjudicated: {todayStr}
              </span>
              <div className="certainty-badge">
                <span style={{ color: 'var(--text-muted)' }}>Certainty:</span>
                <span className="certainty-val">{certaintyScore}%</span>
              </div>
              {latency && (
                <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                  {latency}s latency
                </span>
              )}
            </div>
          </div>

          {/* Authoritative Serif Ruling */}
          <h2 className="verdict-ruling-title">
            {meta.label || finalOutcome.replace(/_/g, ' ')}
          </h2>

          {/* Governing Statutory Authority */}
          {governingClause && (
            <div className="statutory-authority-row" style={{ marginTop: '0.75rem' }}>
              <span>Governing Statutory Authority:</span>
              <strong>{governingClause}</strong>
            </div>
          )}

          {/* Plain-English Judicial Reasoning & Finding */}
          {displayFinding && (
            <p className="verdict-description" style={{ marginTop: '0.85rem' }}>
              <span className="case-finding-prefix">Judicial Reasoning:</span>
              {displayFinding}
            </p>
          )}

          {/* Guardrail Enforcement Notice */}
          {overrideApplied && (
            <div className="override-alert-banner">
              <div>
                <strong>Statutory Legal Guardrail Enforced: </strong>
                <span>
                  Rule <code>{ruleTriggered}</code> intervened pursuant to the Hierarchy of Legal Precedence under the RTI Act, 2005 to supersede mathematical consensus and enforce mandatory statutory disclosure standards.
                </span>
              </div>
            </div>
          )}
        </div>

        {/* 2. Prospective Scenario Probability Pie Chart (Directly Near Output) */}
        <OutcomeProbabilityPieChart 
          probabilities={fusedProbabilities}
          finalOutcome={finalOutcome}
        />
      </div>

      {/* Operative Judicial Mandate & Directive */}
      {judicialDirections && judicialDirections.length > 0 && (
        <div className="directive-callout-card" style={{ marginTop: '1rem', marginBottom: '1.25rem' }}>
          <div className="directive-header">
            <span>⚖</span>
            <span>Operative Judicial Directive & Mandate</span>
          </div>
          <div className="directive-body-text">
            {judicialDirections[0]}
          </div>
        </div>
      )}

      {/* =========================================================================
          PART II: OUTCOME PROBABILITY DISTRIBUTION (MULTI-SIGNAL CONSENSUS)
          ========================================================================= */}
      <div className="analysis-section-card">
        <div className="section-card-header">
          <div className="section-card-title">
            Outcome Probability Distribution (Multi-Signal Consensus)
          </div>
          <span className="section-card-meta">
            Weights: InLegalBERT (0.50) + Coverage (0.30) + Statutory (0.20)
          </span>
        </div>

        <table className="analytical-table">
          <thead>
            <tr>
              <th style={{ width: '35%' }}>Advisory Outcome Class</th>
              <th style={{ width: '35%' }}>Governing Statutory Basis</th>
              <th style={{ width: '15%' }}>Consensus Score</th>
              <th style={{ width: '15%' }}>Distribution</th>
            </tr>
          </thead>
          <tbody>
            {outcomeEntries.map(item => (
              <tr key={item.cls} className={item.isWinner ? 'winner-row' : ''}>
                <td>
                  <span>{item.label}</span>
                  {item.isWinner && <span className="winner-tag">Governing Outcome</span>}
                </td>
                <td style={{ color: 'var(--text-secondary)', fontSize: '0.75rem' }}>
                  {item.statGround}
                </td>
                <td style={{ fontFamily: 'var(--font-mono)', fontWeight: item.isWinner ? 700 : 500 }}>
                  {(item.prob * 100).toFixed(1)}%
                </td>
                <td>
                  <div className="prob-meter-wrapper">
                    <div className={`prob-meter-track ${item.isWinner ? '' : 'muted'}`}>
                      <div 
                        className="prob-meter-fill" 
                        style={{ width: `${item.pct}%` }} 
                      />
                    </div>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
