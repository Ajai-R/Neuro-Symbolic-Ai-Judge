"""
Signal 3: Master End-to-End Neuro-Symbolic Advisory System Pipeline
Project: Explainable Neuro-Symbolic AI Advisory System for RTI Decision Outcome Prediction
Batch ID: 30
"""

import os
import sys
import time
from typing import Dict, Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from signal_1_inlegalbert.predict import InLegalBERTOutcomePredictor
from signal_2_coverage.pipeline import CoverageVerificationPipeline
from signal_3.scanner import StatutoryScanner
from signal_3.engine import NeuroSymbolicDecisionEngine, AdvisoryVerdict


class NeuroSymbolicAdvisorySystem:
    """
    Master Advisory System integrating:
    - Signal 1: Neural Outcome Classifier (InLegalBERT, 73.11% Acc / 73.35% F1)
    - Signal 2: Semantic Coverage Verification (FLAN-T5 + Sentence Transformers)
    - Signal 3: Declarative Statutory Scanner & Gated Hierarchy Decision Engine
    """
    def __init__(self, use_gpu: bool = True):
        print("=" * 80)
        print(" INITIALIZING MASTER EXPLAINABLE NEURO-SYMBOLIC ADVISORY SYSTEM")
        print("=" * 80)

        # 1. Initialize Signal 1 Predictor
        print("[System Init] Loading Signal 1: InLegalBERT Outcome Predictor...")
        self.signal1 = InLegalBERTOutcomePredictor()

        # 2. Initialize Signal 2 Coverage Pipeline
        print("[System Init] Loading Signal 2: Semantic Information Coverage Pipeline...")
        self.signal2 = CoverageVerificationPipeline()

        # 3. Initialize Signal 3 Statutory Scanner & Engine
        print("[System Init] Loading Signal 3: Statutory Scanner & Symbolic Fusion Engine...")
        self.scanner = StatutoryScanner()
        self.engine = NeuroSymbolicDecisionEngine()

        print("[System Init] All 3 Signals Successfully Initialized & Synced.")
        print("=" * 80 + "\n")

    def advise_case(
        self,
        case_narrative: str,
        information_sought: Optional[str] = None,
        public_authority: Optional[str] = None,
        exemptions_cited: Optional[str] = None,
        case_id: Optional[str] = None
    ) -> Dict:
        """
        Runs the complete 3-Signal Neuro-Symbolic pipeline on a single RTI case.
        """
        start_time = time.time()
        c_id = case_id or "RTI_ADVISORY_CASE"

        # 1. Execute Signal 1 (Neural Prior)
        s1_out = self.signal1.predict(
            case_narrative=case_narrative,
            information_sought=information_sought,
            exemptions_cited=exemptions_cited,
            public_authority=public_authority
        )
        s1_probs = s1_out["probabilities"]

        # 2. Execute Signal 2 (Factual Coverage Audit)
        s2_out = self.signal2.analyze_case(
            information_sought=information_sought,
            case_narrative=case_narrative,
            case_id=c_id
        )

        # 3. Execute Signal 3 (Statutory Fact Scanning)
        s3_facts = self.scanner.scan_case(
            case_narrative=case_narrative,
            information_sought=information_sought,
            public_authority=public_authority,
            exemptions_cited=exemptions_cited,
            case_id=c_id
        )

        # 4. Synthesize via Priority-Based Neuro-Symbolic Decision Engine
        verdict: AdvisoryVerdict = self.engine.synthesize(
            signal1_probs=s1_probs,
            signal2_result=s2_out,
            signal3_facts=s3_facts,
            case_id=c_id
        )

        total_time = round(time.time() - start_time, 3)

        return {
            "case_id": c_id,
            "total_execution_time_sec": total_time,
            "final_verdict": {
                "predicted_outcome": verdict.final_predicted_outcome,
                "confidence": verdict.confidence,
                "is_statutory_override": verdict.is_statutory_override,
                "governing_clause": verdict.governing_statutory_clause,
                "rule_triggered": verdict.symbolic_rule_triggered,
                "actionable_judicial_directive": verdict.actionable_judicial_directive
            },
            "signals": {
                "signal_1_neural": {
                    "raw_prediction": s1_out["predicted_class"],
                    "confidence": s1_out["confidence"],
                    "probabilities": s1_probs
                },
                "signal_2_coverage": {
                    "coverage_ratio": s2_out["coverage_metrics"]["coverage_ratio"],
                    "compliance_verdict": s2_out["coverage_metrics"]["compliance_verdict"],
                    "total_items": s2_out["coverage_metrics"]["total_items"],
                    "addressed_count": s2_out["coverage_metrics"]["addressed_count"],
                    "missing_count": s2_out["coverage_metrics"]["missing_count"],
                    "audit_table": s2_out["audit_table"]
                },
                "signal_3_statutory": {
                    "exemptions_detected": s3_facts.exemptions_detected,
                    "is_sec24_organization": s3_facts.is_sec24_organization,
                    "sec24_agency_name": s3_facts.sec24_agency_name,
                    "sec20_penalty_triggered": s3_facts.sec20_penalty_triggered,
                    "statutory_summary": s3_facts.statutory_summary
                }
            },
            "explainable_reasoning_trace": verdict.reasoning_audit_trace
        }

    def format_advisory_report(self, result: Dict) -> str:
        """
        Formats a clean, human-readable terminal / web advisory report.
        """
        v = result["final_verdict"]
        s1 = result["signals"]["signal_1_neural"]
        s2 = result["signals"]["signal_2_coverage"]
        s3 = result["signals"]["signal_3_statutory"]

        lines = [
            "=" * 85,
            "          EXPLAINABLE NEURO-SYMBOLIC AI ADVISORY VERDICT & REPORT",
            "=" * 85,
            f"Case Identifier:       {result['case_id']}",
            f"FINAL PREDICTED VERDICT: {v['predicted_outcome']}",
            f"Advisory Confidence:   {v['confidence']*100:.1f}%",
            f"Statutory Override:    {'YES (Legal Guardrail Enforced)' if v['is_statutory_override'] else 'NO (Neural-Factual Consensus)'}",
            f"Governing Legal Rule:  {v['governing_clause']}",
            "-" * 85,
            "ACTIONABLE JUDICIAL DIRECTIVE:",
            f"  {v['actionable_judicial_directive']}",
            "=" * 85,
            "[SIGNAL 1: INLEGALBERT NEURAL OUTCOME PRIOR]",
            f"  Raw Neural Prediction: {s1['raw_prediction']} ({s1['confidence']*100:.1f}%)",
            f"  Class Distribution:    INFO={s1['probabilities']['INFORMATION_DIRECTED']:.3f} | "
            f"DISMISSED={s1['probabilities']['APPEAL_DISMISSED']:.3f} | "
            f"PENALTY={s1['probabilities']['PENALTY_IMPOSED']:.3f} | "
            f"PARTIAL={s1['probabilities']['PARTIAL_RELIEF']:.3f}",
            "-" * 85,
            "[SIGNAL 2: FACTUAL INFORMATION COVERAGE AUDIT]",
            f"  Factual Compliance:    {s2['compliance_verdict']} (Coverage Ratio: {s2['coverage_ratio']*100:.1f}%)",
            f"  Items Breakdown:       {s2['addressed_count']}/{s2['total_items']} Addressed, {s2['missing_count']} Missing",
            "-" * 85,
            "[SIGNAL 3: STATUTORY LAW & BOUNDARY ANALYSIS]",
            f"  Statutory Summary:     {s3['statutory_summary']}",
            "=" * 85,
            "[NEURO-SYMBOLIC REASONING & AUDIT PROOF TRACE]"
        ]
        for step in result["explainable_reasoning_trace"]:
            lines.append(f"  • {step}")
        lines.append("=" * 85)
        lines.append(f"Processing Latency: {result['total_execution_time_sec']}s")
        lines.append("=" * 85)

        return "\n".join(lines)
