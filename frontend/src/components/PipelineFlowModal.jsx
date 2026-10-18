import React from 'react';
import { X } from 'lucide-react';

export default function PipelineFlowModal({ isOpen, onClose }) {
  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <button className="modal-close-btn" onClick={onClose} aria-label="Close modal">
          <X size={16} />
        </button>

        <div style={{ marginBottom: '1.25rem', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '0.85rem' }}>
          <div style={{ fontSize: '0.675rem', fontWeight: 700, color: 'var(--teal-accent)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: '0.2rem' }}>
            Technical Methodology Specifications
          </div>
          <h2 style={{ fontFamily: 'var(--font-serif)', fontSize: '1.3rem', fontWeight: 700, color: 'var(--navy-primary)' }}>
            Neuro-Symbolic Pipeline Architecture
          </h2>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.15rem' }}>
            Continuous multi-signal probabilistic fusion gated by deterministic statutory guardrails under the Right to Information Act, 2005.
          </p>
        </div>

        {/* 4-Stage Specifications Grid */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: '0.85rem', marginBottom: '1.25rem' }}>
          {/* Signal 1 */}
          <div style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-xs)', padding: '1rem' }}>
            <h3 style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--navy-primary)', marginBottom: '0.35rem' }}>
              Signal 1: InLegalBERT Neural Classifier
            </h3>
            <p style={{ fontSize: '0.775rem', color: 'var(--text-secondary)', lineHeight: 1.5, marginBottom: '0.55rem' }}>
              Domain-specific transformer pre-trained on Indian legal corpora and fine-tuned on Central Information Commission appellate orders. Employs Head (160) + Tail (350) legal token preservation.
            </p>
            <div style={{ fontSize: '0.7rem', color: 'var(--navy-primary)', fontWeight: 600, background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)', padding: '0.25rem 0.5rem', borderRadius: 'var(--radius-xs)', fontFamily: 'var(--font-mono)' }}>
              Blind Test Accuracy: 73.11% (Macro F1: 73.35%)
            </div>
          </div>

          {/* Signal 2 */}
          <div style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-xs)', padding: '1rem' }}>
            <h3 style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--navy-primary)', marginBottom: '0.35rem' }}>
              Signal 2: Substantive Coverage Verification
            </h3>
            <p style={{ fontSize: '0.775rem', color: 'var(--text-secondary)', lineHeight: 1.5, marginBottom: '0.55rem' }}>
              Eliminates strategic administrative evasion. Itemizes citizen demands ($Q_1, Q_2, \dots$) and audits substantive disclosure using an LLM-as-a-Judge with dynamic statutory weighting (Core: 65% vs Auxiliary: 35%).
            </p>
            <div style={{ fontSize: '0.7rem', color: 'var(--navy-primary)', fontWeight: 600, background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)', padding: '0.25rem 0.5rem', borderRadius: 'var(--radius-xs)', fontFamily: 'var(--font-mono)' }}>
              Substantive Metric: Disclosed (1.0) | Partial (0.5) | Evaded/Withheld (0.0)
            </div>
          </div>

          {/* Signal 3 */}
          <div style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-xs)', padding: '1rem' }}>
            <h3 style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--navy-primary)', marginBottom: '0.35rem' }}>
              Signal 3: Declarative Statutory Knowledge Base
            </h3>
            <p style={{ fontSize: '0.775rem', color: 'var(--text-secondary)', lineHeight: 1.5, marginBottom: '0.55rem' }}>
              Formal legal ontology encoding Section 8(1)(a)-(j) exemptions, Section 24 intelligence agency bars, Section 20 show-cause penalty criteria (&gt;30-day delay), and Section 10 severability rules.
            </p>
            <div style={{ fontSize: '0.7rem', color: 'var(--navy-primary)', fontWeight: 600, background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)', padding: '0.25rem 0.5rem', borderRadius: 'var(--radius-xs)', fontFamily: 'var(--font-mono)' }}>
              Deterministic Legal Boundaries & Precedent Grounding
            </div>
          </div>

          {/* Stage 4 */}
          <div style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-xs)', padding: '1rem' }}>
            <h3 style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--navy-primary)', marginBottom: '0.35rem' }}>
              Stage 4: Master Neuro-Symbolic Decision Engine
            </h3>
            <p style={{ fontSize: '0.775rem', color: 'var(--text-secondary)', lineHeight: 1.5, marginBottom: '0.55rem' }}>
              Continuous weighted multi-signal fusion ($w_1=0.50, w_2=0.30, w_3=0.20$) strictly governed by the Hierarchy of Statutory Precedence to prevent AI hallucinations and enforce statutory mandates.
            </p>
            <div style={{ fontSize: '0.7rem', color: 'var(--navy-primary)', fontWeight: 600, background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)', padding: '0.25rem 0.5rem', borderRadius: 'var(--radius-xs)', fontFamily: 'var(--font-mono)' }}>
              Human-Auditable Evidence Traces & Official Directives
            </div>
          </div>
        </div>

        {/* Hierarchy of Statutory Precedence */}
        <div style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-xs)', padding: '1rem' }}>
          <h4 style={{ fontSize: '0.725rem', fontWeight: 700, color: 'var(--navy-primary)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '0.65rem' }}>
            Hierarchy of Statutory Legal Guardrails (Conflict Resolution Precedence)
          </h4>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.55rem', fontSize: '0.75rem' }}>
            <div style={{ borderLeft: '2px solid var(--navy-primary)', paddingLeft: '0.65rem' }}>
              <strong style={{ color: 'var(--navy-primary)' }}>Level 1 — Section 24 Jurisdictional Immunity Bar (Veto): </strong>
              <span style={{ color: 'var(--text-secondary)' }}>Mandatory dismissal if respondent is an exempt intelligence agency (Second Schedule), depriving the Commission of jurisdiction.</span>
            </div>

            <div style={{ borderLeft: '2px solid var(--navy-primary)', paddingLeft: '0.65rem' }}>
              <strong style={{ color: 'var(--navy-primary)' }}>Level 2 — Section 20 Penal Sanctions: </strong>
              <span style={{ color: 'var(--text-secondary)' }}>Enforces mandatory show-cause notice where unexplained delay exceeds statutory limits or records are untraceable.</span>
            </div>

            <div style={{ borderLeft: '2px solid var(--navy-primary)', paddingLeft: '0.65rem' }}>
              <strong style={{ color: 'var(--navy-primary)' }}>Level 3 — Section 10 Severability Principle: </strong>
              <span style={{ color: 'var(--text-secondary)' }}>Directs partial relief when disclosable factual parts can be severed from exempt records.</span>
            </div>

            <div style={{ borderLeft: '2px solid var(--navy-primary)', paddingLeft: '0.65rem' }}>
              <strong style={{ color: 'var(--navy-primary)' }}>Level 4 — Section 7(1) Duty to Disclose: </strong>
              <span style={{ color: 'var(--text-secondary)' }}>Overrules administrative refusal when requested records are omitted without invoking valid Section 8(1) exemptions.</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
