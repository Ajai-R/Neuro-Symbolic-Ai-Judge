import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import CaseInputPanel from './components/CaseInputPanel';
import Signal1Card from './components/Signal1Card';
import Signal2Card from './components/Signal2Card';
import Signal3Card from './components/Signal3Card';
import DecisionMatrixCard from './components/DecisionMatrixCard';
import PipelineFlowModal from './components/PipelineFlowModal';

import { predictRTIOutcome, checkBackendHealth } from './services/apiService';

export default function App() {
  const [currentView, setCurrentView] = useState('input'); // 'input' or 'results'
  const [formData, setFormData] = useState({
    citizenQuery: '',
    pioReply: '',
    publicAuthority: '',
    exemptionsCited: '',
    daysElapsed: 15
  });

  const [backendInfo, setBackendInfo] = useState({ status: 'checking', model_loaded: false });
  const [analysisResult, setAnalysisResult] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [activeSignalTab, setActiveSignalTab] = useState('stage4');
  const [isMethodologyOpen, setIsMethodologyOpen] = useState(false);
  const [apiError, setApiError] = useState(null);

  // Check backend health on mount
  useEffect(() => {
    checkBackendHealth().then(info => {
      setBackendInfo(info);
      console.log("[App] Backend connection status:", info);
    });
  }, []);

  // Trigger analysis
  const runAnalysis = async (customData) => {
    const dataToProcess = customData || formData;
    if (!dataToProcess.citizenQuery?.trim() || !dataToProcess.pioReply?.trim()) {
      return;
    }
    setIsAnalyzing(true);
    setApiError(null);
    setCurrentView('results');

    try {
      const result = await predictRTIOutcome(dataToProcess);
      setAnalysisResult(result);
      setActiveSignalTab('stage4');
    } catch (err) {
      console.error("Pipeline analysis failure:", err);
      setApiError(err.message || "Backend server is offline or unreachable at http://127.0.0.1:8000. Please start backend_server.py.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleChangeField = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const handleReset = () => {
    setFormData({
      citizenQuery: '',
      pioReply: '',
      publicAuthority: '',
      exemptionsCited: '',
      daysElapsed: 15
    });
    setAnalysisResult(null);
    setApiError(null);
    setCurrentView('input');
  };

  return (
    <div className="app-container">
      {/* Top Masthead with Institutional Brand and Adjudication Navigation */}
      <Header 
        currentView={currentView}
        onSwitchView={(view) => setCurrentView(view)}
        hasResults={Boolean(analysisResult)}
        analysisResult={analysisResult}
        onOpenMethodology={() => setIsMethodologyOpen(true)}
        onReset={handleReset}
        isAnalyzing={isAnalyzing}
      />

      {/* Main Viewport */}
      <main className="main-viewport">
        {currentView === 'input' ? (
          /* =========================================================================
              VIEW 1: CASE INTAKE & DISPUTED RECORD PLEADINGS
              ========================================================================= */
          <CaseInputPanel 
            formData={formData}
            onChangeField={handleChangeField}
            onSubmit={() => runAnalysis(formData)}
            isAnalyzing={isAnalyzing}
            backendInfo={backendInfo}
            hasResults={Boolean(analysisResult)}
            onViewResults={() => setCurrentView('results')}
          />
        ) : (
          /* =========================================================================
              VIEW 2: JUDICIAL ADVISORY OPINION & AUDIT
              ========================================================================= */
          <section className="advisory-results-view">
            {/* Docket Action Bar Above Results */}
            <div className="results-top-nav">
              <button 
                type="button" 
                className="btn-back-to-input"
                onClick={() => setCurrentView('input')}
              >
                ← Return to Case Pleadings
              </button>

              <div className="case-mini-summary">
                <span className="summary-docket-item">
                  <strong>Respondent:</strong> {formData.publicAuthority || 'Public Authority'}
                </span>
                <span className="summary-docket-item">
                  <strong>Response Window:</strong> {formData.daysElapsed} Days
                </span>
                {analysisResult?.stage4?.latency && (
                  <span className="summary-docket-item" style={{ fontFamily: 'var(--font-mono)' }}>
                    <strong>Computation:</strong> {analysisResult.stage4.latency}s
                  </span>
                )}
              </div>

              <div>
                <button 
                  type="button"
                  className="btn-secondary"
                  onClick={handleReset}
                  title="Clear current docket and lodge a new case file"
                >
                  New Case Docket
                </button>
              </div>
            </div>

            {apiError ? (
              <div className="empty-state-box" style={{ borderColor: '#FCA5A5', background: '#FEF2F2', padding: '2rem' }}>
                <div className="empty-icon-circle" style={{ color: '#DC2626', borderColor: '#FCA5A5' }}>
                  ⚠️
                </div>
                <h3 className="empty-title" style={{ color: '#991B1B' }}>Backend Service Offline</h3>
                <p className="empty-desc" style={{ color: '#B91C1C', maxWidth: '520px', margin: '0 auto' }}>
                  {apiError}
                </p>
                <div style={{ marginTop: '1.25rem', display: 'flex', gap: '0.75rem', justifyContent: 'center' }}>
                  <button 
                    type="button" 
                    className="btn-secondary" 
                    onClick={() => setCurrentView('input')}
                  >
                    ← Return to Case Pleadings
                  </button>
                  <button 
                    type="button" 
                    className="btn-primary" 
                    onClick={() => runAnalysis()}
                  >
                    ↻ Retry Connection
                  </button>
                </div>
              </div>
            ) : isAnalyzing ? (
              <div className="analyzing-state-box">
                <div className="spinner-large" />
                <h3 className="analyzing-title">Synthesizing Neuro-Symbolic Judicial Evidence...</h3>
                <p className="analyzing-desc">
                  Evaluating Signal 1 (Neural InLegalBERT Prior), Signal 2 (LLM Substantive Compliance Audit), and Signal 3 (RTI Statutory Guardrails) pursuant to the Hierarchy of Legal Precedence.
                </p>
              </div>
            ) : analysisResult ? (
              <div>
                {/* Forensic & Evidence Navigation Tabs */}
                <div className="tabs-navigation">
                  <button 
                    className={`tab-btn ${activeSignalTab === 'stage4' ? 'active' : ''}`}
                    onClick={() => setActiveSignalTab('stage4')}
                  >
                    Advisory Opinion & Directive
                  </button>

                  <button 
                    className={`tab-btn ${activeSignalTab === 'signal1' ? 'active' : ''}`}
                    onClick={() => setActiveSignalTab('signal1')}
                  >
                    Signal 1: InLegalBERT Prior
                  </button>

                  <button 
                    className={`tab-btn ${activeSignalTab === 'signal2' ? 'active' : ''}`}
                    onClick={() => setActiveSignalTab('signal2')}
                  >
                    Signal 2: Substantive Coverage Audit
                  </button>

                  <button 
                    className={`tab-btn ${activeSignalTab === 'signal3' ? 'active' : ''}`}
                    onClick={() => setActiveSignalTab('signal3')}
                  >
                    Signal 3: Statutory Scanner
                  </button>

                </div>

                {/* Tab Content Views */}
                <div className="tab-content-container">
                  {activeSignalTab === 'stage4' && (
                    <DecisionMatrixCard 
                      stage4={analysisResult.stage4} 
                    />
                  )}

                  {activeSignalTab === 'signal1' && (
                    <Signal1Card signal1={analysisResult.signal1} />
                  )}

                  {activeSignalTab === 'signal2' && (
                    <Signal2Card signal2={analysisResult.signal2} />
                  )}

                  {activeSignalTab === 'signal3' && (
                    <Signal3Card signal3={analysisResult.signal3} />
                  )}
                </div>
              </div>
            ) : (
              <div className="empty-state-box">
                <div className="empty-icon-circle">
                  ⚖
                </div>
                <h3 className="empty-title">No Case Dispute Lodged</h3>
                <p className="empty-desc">
                  Please return to the Case Intake Dossier to submit citizen demands and CPIO submissions for adjudication.
                </p>
                <button 
                  type="button" 
                  className="btn-run-model" 
                  style={{ marginTop: '1.25rem' }}
                  onClick={() => setCurrentView('input')}
                >
                  Go to Case Intake Dossier →
                </button>
              </div>
            )}
          </section>
        )}
      </main>

      {/* Technical Methodology Architecture Modal */}
      <PipelineFlowModal 
        isOpen={isMethodologyOpen}
        onClose={() => setIsMethodologyOpen(false)}
      />
    </div>
  );
}
