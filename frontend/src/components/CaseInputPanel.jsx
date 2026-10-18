import React from 'react';

export default function CaseInputPanel({
  formData,
  onChangeField,
  onSubmit,
  isAnalyzing,
  backendInfo,
  hasResults,
  onViewResults
}) {
  const {
    citizenQuery,
    pioReply,
    publicAuthority,
    exemptionsCited,
    daysElapsed
  } = formData;

  const isFormValid = citizenQuery.trim().length > 5 && pioReply.trim().length > 5;
  const isOverdue = daysElapsed > 30;

  return (
    <div className="case-intake-wrapper">
      {/* Existing advisory opinion alert */}
      {hasResults && !isAnalyzing && (
        <div className="existing-results-alert">
          <div>
            <strong>Active Advisory Opinion on File: </strong>
            <span>A synthesized judicial opinion already exists for these submitted pleadings.</span>
          </div>
          <button 
            type="button" 
            className="btn-view-advisory-pill"
            onClick={onViewResults}
          >
            Inspect Advisory Opinion →
          </button>
        </div>
      )}

      {/* Institutional Case Header */}
      <div className="intake-institutional-header">
        <div className="intake-kicker">Form A • Statutory Dispute Lodgement</div>
        <h2 className="intake-title">RTI Pleading Intake Dossier</h2>
        <p className="intake-lead">
          Lodge the citizen's itemized RTI demands alongside the respondent Public Information Officer's (CPIO) written submissions to evaluate statutory compliance and synthesize an authoritative judicial advisory opinion under the Right to Information Act, 2005.
        </p>
      </div>

      <form onSubmit={(e) => { e.preventDefault(); if (isFormValid && !isAnalyzing) onSubmit(); }}>
        {/* Dual Column Legal Pleading Grid */}
        <div className="dual-quarters-grid">
          
          {/* =========================================================================
              LEFT CARD: CITIZEN DEMANDS
              ========================================================================= */}
          <div className="quarter-card quarter-citizen">
            <div className="quarter-header">
              <h3 className="quarter-title">01. Citizen's RTI Query / Information Sought</h3>
              <p className="quarter-subtitle">Enter the query asked by the citizen (plain text, paragraphs, or queries)</p>
            </div>

            <div className="form-group" style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
              <label className="form-label" htmlFor="citizenQuery">
                <span>Enter the Query Asked</span>
                <span className="form-label-hint">Plain text, paragraphs, or queries accepted</span>
              </label>
              <textarea
                id="citizenQuery"
                className="textarea-legal"
                style={{ minHeight: '220px', flex: 1 }}
                value={citizenQuery}
                onChange={(e) => onChangeField('citizenQuery', e.target.value)}
                placeholder="Enter the query asked by the citizen (e.g. Please provide certified copies of the departmental inquiry report dated 14/05/2023 along with details of total expenditure sanctioned and disbursed for highway maintenance and file notings approving tender allocations)..."
                required
              />
            </div>

            {/* Respondent Public Authority */}
            <div className="quarter-footer-field">
              <label className="form-label" htmlFor="publicAuthority">
                <span>Respondent Public Authority</span>
                <span className="form-label-hint">Ministry, Department, or Public Sector Undertaking</span>
              </label>
              <input
                id="publicAuthority"
                type="text"
                className="input-text"
                value={publicAuthority}
                onChange={(e) => onChangeField('publicAuthority', e.target.value)}
                placeholder="e.g. Central Public Works Department (CPWD), Ministry of Railways"
              />
            </div>
          </div>

          {/* =========================================================================
              RIGHT CARD: RESPONDENT CPIO DEFENSE
              ========================================================================= */}
          <div className="quarter-card quarter-cpio">
            <div className="quarter-header">
              <h3 className="quarter-title">02. CPIO Written Response / Case Narrative</h3>
              <p className="quarter-subtitle">Enter the reply given by the RTI / CPIO</p>
            </div>

            <div className="form-group" style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
              <label className="form-label" htmlFor="pioReply">
                <span>Enter the Reply Given by the RTI / CPIO</span>
                <span className="form-label-hint">Official statement, records supplied, or grounds for withholding</span>
              </label>
              <textarea
                id="pioReply"
                className="textarea-legal"
                style={{ minHeight: '220px', flex: 1 }}
                value={pioReply}
                onChange={(e) => onChangeField('pioReply', e.target.value)}
                placeholder="Enter the reply given by the RTI (e.g. The inquiry report is provided herewith. Regarding expenditure records, the same involve commercial confidence withheld under Section 8(1)(d). File notings cannot be disclosed as they contain personal evaluations exempt under Section 8(1)(j))..."
                required
              />
            </div>

            {/* Statutory Delay & Exemption Section */}
            <div className="quarter-footer-grid">
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.35rem' }}>
                  <label className="form-label" htmlFor="daysElapsed" style={{ margin: 0 }}>
                    Statutory Limitation Window
                  </label>
                  <span style={{ 
                    fontFamily: 'var(--font-mono)', 
                    fontSize: '0.725rem', 
                    fontWeight: 700, 
                    color: isOverdue ? 'var(--state-red-text)' : 'var(--state-green-text)',
                    background: isOverdue ? 'var(--state-red-bg)' : 'var(--state-green-bg)',
                    padding: '0.1rem 0.4rem',
                    borderRadius: 'var(--radius-xs)',
                    border: `1px solid ${isOverdue ? 'var(--state-red-border)' : 'var(--state-green-border)'}`
                  }}>
                    {daysElapsed} Days Elapsed
                  </span>
                </div>
                <input
                  id="daysElapsed"
                  type="range"
                  min="1"
                  max="120"
                  value={daysElapsed}
                  onChange={(e) => onChangeField('daysElapsed', parseInt(e.target.value, 10))}
                  className="slider-input"
                />
                <div className="delay-description-text" style={{ color: isOverdue ? 'var(--state-red-text)' : 'var(--text-muted)' }}>
                  {isOverdue 
                    ? 'Exceeds 30-day statutory mandate pursuant to Section 7(1); Section 20 review triggered.'
                    : 'Within 30-day statutory window pursuant to Section 7(1).'}
                </div>
              </div>

              <div>
                <label className="form-label" htmlFor="exemptionsCited">
                  <span>Claimed Exemption Clauses</span>
                  <span className="form-label-hint">Section 8(1) / Section 24</span>
                </label>
                <input
                  id="exemptionsCited"
                  type="text"
                  className="input-text"
                  value={exemptionsCited}
                  onChange={(e) => onChangeField('exemptionsCited', e.target.value)}
                  placeholder="e.g. 8(1)(j), 8(1)(d), 24"
                />
              </div>
            </div>
          </div>
        </div>

        {/* Primary Action Button */}
        <div className="action-center-container">
          <button
            type="submit"
            className="btn-run-model"
            disabled={!isFormValid || isAnalyzing}
          >
            {isAnalyzing ? (
              <>
                <div className="spinner-pulse" />
                <span>Synthesizing Legal Evidence & Precedents...</span>
              </>
            ) : (
              <span>Generate Judicial Advisory Opinion →</span>
            )}
          </button>

          <div className="model-integration-tag">
            <span 
              className="integration-dot" 
              style={{ background: backendInfo?.model_loaded ? 'var(--state-green-text)' : 'var(--state-amber-text)' }} 
            />
            <span>
              {backendInfo?.model_loaded 
                ? `Adjudication Framework Ready • InLegalBERT Prior + LLM Judicial Auditor + Statutory Guardrails (${backendInfo.device?.toUpperCase() || 'GPU'})` 
                : 'Decision Engine Server Online • Ready for Pleading Evaluation'}
            </span>
          </div>
        </div>
      </form>
    </div>
  );
}
