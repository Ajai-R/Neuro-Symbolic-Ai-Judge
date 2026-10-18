"""
Master Neuro-Symbolic Decision Making Engine
Project: Explainable Neuro-Symbolic AI Advisory System for RTI Decision Outcome Prediction
Batch ID: 30
"""

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from signal_1_inlegalbert.predict import InLegalBERTOutcomePredictor
from signal_2_coverage.pipeline import CoverageVerificationPipeline
from signal_3.scanner import StatutoryScanner
from .weights_config import FusionWeightsConfig
from .signal_adapters import SignalAdapters
from .weighted_fusion import WeightedSignalFuser
from .symbolic_guardrails import StatutoryGuardrails


@dataclass
class DecisionVerdict:
    case_id: str
    final_predicted_outcome: str
    confidence: float
    is_statutory_override: bool
    governing_statutory_clause: str
    symbolic_rule_triggered: str
    actionable_judicial_directive: str
    fused_probabilities: Dict[str, float]
    raw_signal_predictions: Dict[str, str]
    contributions_by_signal: Dict[str, Dict[str, float]]
    reasoning_audit_trace: List[str]
    execution_time_sec: float
    raw_signal1_result: Optional[Dict] = None
    raw_signal2_coverage: Optional[Dict] = None
    raw_signal3_facts: Optional[object] = None


class NeuroSymbolicDecisionEngine:
    """
    Master Decision Making Engine:
    Combines Signal 1 (Neural Prior) + Signal 2 (Coverage Matcher) + Signal 3 (Statutory Scanner)
    via Weighted Mathematical Fusion and Symbolic Legal Guardrails.
    """
    def __init__(self, config: Optional[FusionWeightsConfig] = None):
        self.config = config or FusionWeightsConfig()
        print("=" * 80)
        print(" INITIALIZING MASTER NEURO-SYMBOLIC DECISION MAKING ENGINE")
        print("=" * 80)

        print("[Engine] Loading Signal 1 (InLegalBERT Classifier)...")
        self.signal1 = InLegalBERTOutcomePredictor()

        print("[Engine] Loading Signal 2 (Factual Coverage Matcher)...")
        self.signal2 = CoverageVerificationPipeline()

        print("[Engine] Loading Signal 3 (Statutory Scanner)...")
        self.scanner = StatutoryScanner()

        self.fuser = WeightedSignalFuser(self.config)
        self.guardrails = StatutoryGuardrails()

        print("[Engine] All 3 Signals Synced with Weighted Fusion & Legal Guardrails.")
        print(f"[Engine] Active Weights: w1={self.config.w1_signal1_neural:.2f} (Neural), "
              f"w2={self.config.w2_signal2_coverage:.2f} (Coverage), "
              f"w3={self.config.w3_signal3_statutory:.2f} (Statutory)")
        print("=" * 80 + "\n")

    def decide(
        self,
        case_narrative: str,
        information_sought: Optional[str] = None,
        public_authority: Optional[str] = None,
        exemptions_cited: Optional[str] = None,
        case_id: Optional[str] = None
    ) -> DecisionVerdict:
        """
        Executes the full Decision Making Engine on an RTI case.
        """
        start_time = time.time()
        cid = case_id or "RTI_CASE"
        trace = []

        # ---------------------------------------------------------------------
        # 1. Execute Signal 1: Neural Outcome Prior (InLegalBERT)
        # ---------------------------------------------------------------------
        s1_raw = self.signal1.predict(
            case_narrative=case_narrative,
            information_sought=information_sought,
            exemptions_cited=exemptions_cited,
            public_authority=public_authority
        )
        p_s1 = SignalAdapters.adapt_signal1(s1_raw)
        trace.append(f"Signal 1 (InLegalBERT): Predicted '{s1_raw['predicted_class']}' ({s1_raw['confidence']*100:.1f}%)")

        # ---------------------------------------------------------------------
        # 2. Execute Signal 2: Factual Information Coverage (Sentence Transformers)
        # ---------------------------------------------------------------------
        s2_raw = self.signal2.analyze_case(
            information_sought=information_sought,
            case_narrative=case_narrative,
            case_id=cid
        )
        p_s2 = SignalAdapters.adapt_signal2(s2_raw)
        s2_metrics = s2_raw.get("coverage_metrics", {})
        trace.append(
            f"Signal 2 (Coverage): {s2_metrics.get('addressed_count', 0)}/{s2_metrics.get('total_items', 1)} Addressed, "
            f"{s2_metrics.get('missing_count', 0)} Missing ({s2_metrics.get('compliance_verdict', 'UNKNOWN')})"
        )

        # ---------------------------------------------------------------------
        # 3. Execute Signal 3: Statutory Fact & Boundary Scanner
        # ---------------------------------------------------------------------
        s3_facts = self.scanner.scan_case(
            case_narrative=case_narrative,
            information_sought=information_sought,
            public_authority=public_authority,
            exemptions_cited=exemptions_cited,
            case_id=cid
        )
        p_s3 = SignalAdapters.adapt_signal3(s3_facts)
        trace.append(f"Signal 3 (Statutory): {s3_facts.statutory_summary}")

        # ---------------------------------------------------------------------
        # 4. Step 1: Weighted Probabilistic Fusion
        # ---------------------------------------------------------------------
        fusion_result = self.fuser.fuse(p_s1, p_s2, p_s3)
        fused_pred = fusion_result["fused_predicted_class"]
        fused_conf = fusion_result["fused_confidence"]
        fused_probs = fusion_result["fused_probabilities"]
        trace.append(f"Step 1 (Weighted Consensus): Top fused class = '{fused_pred}' ({fused_conf*100:.1f}%)")

        # ---------------------------------------------------------------------
        # 5. Step 2: Symbolic Legal Guardrails (Hierarchy of Precedence)
        # ---------------------------------------------------------------------
        guardrail_result = self.guardrails.apply_guardrails(
            fused_prediction=fused_pred,
            fused_confidence=fused_conf,
            signal2_coverage=s2_raw,
            signal3_facts=s3_facts
        )
        trace.extend(guardrail_result["reasoning_steps"])

        elapsed = round(time.time() - start_time, 3)

        raw_predictions = {
            "signal_1_neural": s1_raw["predicted_class"],
            "signal_2_coverage_verdict": s2_metrics.get("compliance_verdict", "UNKNOWN"),
            "signal_3_statutory_detected": ", ".join(s3_facts.exemptions_detected) or "None",
            "weighted_fusion_top": fused_pred,
            "final_governing_outcome": guardrail_result["final_outcome"]
        }

        return DecisionVerdict(
            case_id=cid,
            final_predicted_outcome=guardrail_result["final_outcome"],
            confidence=guardrail_result["confidence"],
            is_statutory_override=guardrail_result["is_override"],
            governing_statutory_clause=guardrail_result["governing_clause"],
            symbolic_rule_triggered=guardrail_result["rule_triggered"],
            actionable_judicial_directive=guardrail_result["directive"],
            fused_probabilities=fused_probs,
            raw_signal_predictions=raw_predictions,
            contributions_by_signal=fusion_result["contributions_by_class"],
            reasoning_audit_trace=trace,
            execution_time_sec=elapsed,
            raw_signal1_result=s1_raw,
            raw_signal2_coverage=s2_raw,
            raw_signal3_facts=s3_facts
        )

    def format_verdict_report(self, verdict: DecisionVerdict) -> str:
        """
        Formats a clean, judicial advisory report.
        """
        lines = [
            "=" * 85,
            "       MASTER NEURO-SYMBOLIC DECISION MAKING ENGINE: ADVISORY VERDICT",
            "=" * 85,
            f"Case Identifier:         {verdict.case_id}",
            f"FINAL PREDICTED VERDICT: {verdict.final_predicted_outcome}",
            f"Advisory Confidence:     {verdict.confidence*100:.1f}%",
            f"Statutory Override:      {'YES (Legal Guardrail Enforced)' if verdict.is_statutory_override else 'NO (Weighted Consensus Accepted)'}",
            f"Governing Legal Rule:    {verdict.governing_statutory_clause} ({verdict.symbolic_rule_triggered})",
            "-" * 85,
            "ACTIONABLE JUDICIAL DIRECTIVE:",
            f"  {verdict.actionable_judicial_directive}",
            "=" * 85,
            "[MULTI-SIGNAL PROBABILISTIC FUSION DISTRIBUTION]",
            f"  • INFORMATION_DIRECTED: {verdict.fused_probabilities.get('INFORMATION_DIRECTED', 0.0):.3f}",
            f"  • APPEAL_DISMISSED:     {verdict.fused_probabilities.get('APPEAL_DISMISSED', 0.0):.3f}",
            f"  • PENALTY_IMPOSED:      {verdict.fused_probabilities.get('PENALTY_IMPOSED', 0.0):.3f}",
            f"  • PARTIAL_RELIEF:       {verdict.fused_probabilities.get('PARTIAL_RELIEF', 0.0):.3f}",
            "-" * 85,
            "[INDIVIDUAL SIGNAL VOTES & PREDICTIONS]",
            f"  • Signal 1 (InLegalBERT Prior):   {verdict.raw_signal_predictions['signal_1_neural']}",
            f"  • Signal 2 (Coverage Compliance): {verdict.raw_signal_predictions['signal_2_coverage_verdict']}",
            f"  • Signal 3 (Statutory Citations): {verdict.raw_signal_predictions['signal_3_statutory_detected']}",
            f"  • Weighted Fusion Winner:         {verdict.raw_signal_predictions['weighted_fusion_top']}",
            "=" * 85,
            "[EXPLAINABLE REASONING & AUDIT PROOF TRACE]"
        ]
        for step in verdict.reasoning_audit_trace:
            lines.append(f"  • {step}")
        lines.append("=" * 85)
        lines.append(f"Total Decision Latency: {verdict.execution_time_sec}s")
        lines.append("=" * 85)
        return "\n".join(lines)
