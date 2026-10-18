import React from 'react';
import { Scale } from 'lucide-react';

export default function Header({ 
  currentView, 
  onSwitchView, 
  hasResults, 
  analysisResult, 
  onOpenMethodology, 
  onReset, 
  isAnalyzing 
}) {
  return (
    <header className="header-wrapper">
      <div className="brand-section">
        <div className="brand-crest" aria-hidden="true">
          <Scale size={18} strokeWidth={2.2} />
        </div>
        <div>
          <h1 className="brand-title">RTI Neuro-Symbolic Judicial Advisory System</h1>
          <p className="brand-subtitle">
            <span className="subtitle-tag">Statutory Decision Support Framework</span>
            <span className="subtitle-dot">•</span>
            <span className="subtitle-statute">Right to Information Act, 2005</span>
          </p>
        </div>
      </div>

      {/* Central View Switcher */}
      <nav className="header-view-nav" aria-label="Adjudication Workflow Navigation">
        <button
          type="button"
          className={`nav-tab-btn ${currentView === 'input' ? 'active' : ''}`}
          onClick={() => onSwitchView('input')}
        >
          Case Intake Dossier
        </button>

        <button
          type="button"
          className={`nav-tab-btn ${currentView === 'results' ? 'active' : ''} ${!hasResults && !isAnalyzing ? 'disabled' : ''}`}
          onClick={() => (hasResults || isAnalyzing) && onSwitchView('results')}
          disabled={!hasResults && !isAnalyzing}
          title={hasResults ? "View Judicial Advisory Finding & Forensic Audit" : "Submit a case dispute to synthesize an advisory opinion"}
        >
          <span>Judicial Advisory Opinion</span>
          {hasResults && (
            <span className="nav-status-indicator">
              Ready
            </span>
          )}
        </button>
      </nav>

      {/* Header Actions */}
      <div className="header-actions">
        <button 
          type="button"
          className="btn-secondary" 
          onClick={onOpenMethodology}
          title="Inspect 4-Stage Methodology & Architecture Specifications"
        >
          Methodology
        </button>

        <button 
          type="button"
          className="btn-secondary" 
          onClick={onReset}
          disabled={isAnalyzing}
          title="Reset Inputs & Open New Case File"
        >
          New Case
        </button>
      </div>
    </header>
  );
}
