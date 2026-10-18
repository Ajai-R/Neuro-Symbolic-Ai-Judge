import React from 'react';
import { OUTCOME_META } from '../services/pipelineEngine';

export default function Signal1Card({ signal1 }) {
  if (!signal1) return null;

  const {
    predictedClass,
    confidence,
    probabilities = {},
    modelName,
    tokensProcessed,
    headTokens,
    tailTokens,
    isLiveModel,
    device
  } = signal1;

  const meta = OUTCOME_META[predictedClass] || {};

  const probEntries = Object.entries(probabilities).map(([k, v]) => ({
    key: k,
    label: OUTCOME_META[k]?.label || k.replace(/_/g, ' '),
    val: v,
    pct: Math.round(v * 100),
    isTop: k === predictedClass
  })).sort((a, b) => b.val - a.val);

  return (
    <div className="signal-detail-card">
      {/* Header Row */}
      <div className="signal-meta-row">
        <div className="signal-title-wrap">
          <h3 className="signal-card-heading">Signal 1: InLegalBERT Domain Neural Classifier</h3>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
          {isLiveModel && (
            <span style={{ fontSize: '0.675rem', fontWeight: 600, color: 'var(--state-green-text)', background: 'var(--state-green-bg)', border: '1px solid var(--state-green-border)', padding: '0.1rem 0.45rem', borderRadius: 'var(--radius-xs)' }}>
              PyTorch Inference ({device?.toUpperCase() || 'CUDA'})
            </span>
          )}
          <span className="signal-chip">{modelName || 'InLegalBERT (Fine-Tuned Prior)'}</span>
        </div>
      </div>

      {/* Summary Score Strip */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', background: 'var(--bg-secondary)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-xs)', padding: '0.75rem 1rem', marginBottom: '1.15rem' }}>
        <div>
          <span style={{ fontSize: '0.675rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600 }}>
            Predicted Outcome Prior
          </span>
          <div style={{ fontFamily: 'var(--font-serif)', fontSize: '1.1rem', fontWeight: 700, color: 'var(--navy-primary)', marginTop: '0.15rem' }}>
            {meta.label || predictedClass}
          </div>
        </div>
        <div style={{ textAlign: 'right' }}>
          <span style={{ fontSize: '0.675rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600 }}>
            Empirical Confidence
          </span>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '1.15rem', fontWeight: 700, color: 'var(--navy-primary)', marginTop: '0.15rem' }}>
            {confidence}%
          </div>
        </div>
      </div>

      {/* Probability Distribution Table */}
      <div style={{ marginBottom: '1.15rem' }}>
        <span style={{ fontSize: '0.7rem', fontWeight: 700, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.06em', display: 'block', marginBottom: '0.55rem' }}>
          Softmax Output Probability Distribution
        </span>

        <table className="analytical-table">
          <thead>
            <tr>
              <th style={{ width: '45%' }}>Advisory Class Label</th>
              <th style={{ width: '20%' }}>Softmax Probability</th>
              <th style={{ width: '35%' }}>Relative Density</th>
            </tr>
          </thead>
          <tbody>
            {probEntries.map(item => (
              <tr key={item.key} className={item.isTop ? 'winner-row' : ''}>
                <td>
                  <span>{item.label}</span>
                  {item.isTop && <span className="winner-tag">Top Prior</span>}
                </td>
                <td style={{ fontFamily: 'var(--font-mono)', fontWeight: item.isTop ? 700 : 500 }}>
                  {(item.val * 100).toFixed(1)}%
                </td>
                <td>
                  <div className="prob-meter-wrapper">
                    <div className={`prob-meter-track ${item.isTop ? '' : 'muted'}`}>
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

      {/* Head + Tail Truncation Technical Note */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--text-muted)', background: 'var(--bg-secondary)', padding: '0.45rem 0.75rem', borderRadius: 'var(--radius-xs)', border: '1px solid var(--border-subtle)' }}>
        <span>
          Head (160) + Tail (350) Legal Token Retention Architecture
        </span>
        <span style={{ fontFamily: 'var(--font-mono)' }}>
          Tokens Processed: ~{tokensProcessed || 120}/512 Tokens
        </span>
      </div>
    </div>
  );
}
