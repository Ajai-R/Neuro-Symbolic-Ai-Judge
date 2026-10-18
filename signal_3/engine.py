"""
Signal 3: Neuro-Symbolic Synthesis & Priority Decision Engine
Project: Explainable Neuro-Symbolic AI Advisory System for RTI Decision Outcome Prediction
Batch ID: 30
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from .scanner import StatutoryFacts


@dataclass
class AdvisoryVerdict:
    final_predicted_outcome: str
    confidence: float
    is_statutory_override: bool
    governing_statutory_clause: str
    symbolic_rule_triggered: Optional[str]
    actionable_judicial_directive: str
    reasoning_audit_trace: List[str]
    signal1_neural_prior: Dict[str, float]
    signal2_coverage_summary: Dict
    signal3_statutory_facts: Dict


class NeuroSymbolicDecisionEngine:
    """
    The Executive Gavel of the System.
    Synthesizes Neural Intuition (Signal 1) + Factual Coverage (Signal 2) + Statutory Law (Signal 3)
    using a Priority-Based Hierarchy of Norms grounded in the RTI Act, 2005.
    """
    def __init__(self, partial_relief_threshold: float = 0.50):
        self.partial_relief_threshold = partial_relief_threshold

    def synthesize(
        self,
        signal1_probs: Dict[str, float],
        signal2_result: Dict,
        signal3_facts: StatutoryFacts,
        case_id: Optional[str] = None
    ) -> AdvisoryVerdict:
        """
        Executes the Gated Neuro-Symbolic Fusion Algorithm.
        """
        trace = []
        rule_triggered = None
        governing_clause = "RTI Act, 2005"
        is_override = False

        # Extract Signal 1 components
        s1_pred = max(signal1_probs, key=signal1_probs.get)
        s1_conf = signal1_probs[s1_pred]
        trace.append(f"[Signal 1 Neural Prior]: Initial statistical prediction = '{s1_pred}' (Confidence: {s1_conf*100:.1f}%)")

        # Extract Signal 2 components
        s2_metrics = signal2_result.get("coverage_metrics", {})
        total_items = s2_metrics.get("total_items", 1)
        addressed_count = s2_metrics.get("addressed_count", 0)
        missing_count = s2_metrics.get("missing_count", 0)
        partial_count = s2_metrics.get("partial_count", 0)
        coverage_ratio = s2_metrics.get("coverage_ratio", 0.0)
        compliance_verdict = s2_metrics.get("compliance_verdict", "UNKNOWN")

        trace.append(
            f"[Signal 2 Coverage Audit]: {addressed_count}/{total_items} items ADDRESSED, "
            f"{missing_count} MISSING, {partial_count} PARTIAL (Coverage: {coverage_ratio*100:.1f}%, Verdict: {compliance_verdict})"
        )

        # Extract Signal 3 components
        trace.append(f"[Signal 3 Statutory Facts]: {signal3_facts.statutory_summary}")

        # =========================================================================
        # PRIORITY LEVEL 1: JURISDICTIONAL IMMUNITY (Section 24 Second Schedule)
        # =========================================================================
        if signal3_facts.is_sec24_organization:
            final_outcome = "APPEAL_DISMISSED"
            confidence = 0.99
            is_override = (s1_pred != final_outcome)
            rule_triggered = "RULE_1_SEC24_JURISDICTIONAL_IMMUNITY"
            governing_clause = "Section 24(1) - Second Schedule Exempt Security Organization"
            trace.append(
                f"[Symbolic Override -> Level 1]: Public Authority is an exempt intelligence/security agency ({signal3_facts.sec24_agency_name}). "
                f"Under Section 24(1), Commission lacks legal jurisdiction. Mandatory dismissal enforced."
            )
            directive = (
                f"Appeal dismissed for lack of jurisdiction under Section 24 of the RTI Act, 2005. "
                f"Respondent ({signal3_facts.sec24_agency_name}) is exempt from the purview of the Act."
            )

        # =========================================================================
        # PRIORITY LEVEL 2: STATUTORY PENAL SANCTION (Section 20 Triggers)
        # =========================================================================
        elif signal3_facts.sec20_penalty_triggered:
            final_outcome = "PENALTY_IMPOSED"
            confidence = 0.94
            is_override = (s1_pred != final_outcome)
            rule_triggered = "RULE_2_SEC20_STATUTORY_PENALTY"
            governing_clause = "Section 20(1) & 20(2) - Penalties and Disciplinary Action"
            trace.append(
                f"[Symbolic Override -> Level 2]: Statutory grounds for penalty confirmed ({', '.join(signal3_facts.sec20_indicators[:2])}). "
                f"Enforcing Section 20 penal proceedings."
            )
            directive = (
                "Issue Show-Cause Notice under Section 20(1) of the RTI Act, 2005 to designated CPIO to show cause "
                "why penalty of Rs. 250/- per day (not exceeding Rs. 25,000/-) should not be levied for unjustified default."
            )

        # =========================================================================
        # PRIORITY LEVEL 3: SEVERABILITY PRINCIPLE (Section 10 Mandate)
        # =========================================================================
        elif (addressed_count > 0 and (missing_count > 0 or partial_count > 0)) or \
             (total_items > 1 and 0.20 <= coverage_ratio <= 0.80) or \
             (signal3_facts.sec10_severability_indicated):
            final_outcome = "PARTIAL_RELIEF"
            confidence = 0.91
            is_override = (s1_pred != final_outcome)
            rule_triggered = "RULE_3_SEC10_SEVERABILITY_PRINCIPLE"
            governing_clause = "Section 10(1) - Severability of Non-Exempt Information"
            trace.append(
                f"[Symbolic Override -> Level 3]: Factual audit confirms mixed compliance ({addressed_count} furnished, {missing_count} unfulfilled). "
                f"Under Section 10(1), disclosable portions must be severed. Enforcing PARTIAL_RELIEF."
            )
            directive = (
                "Grant Partial Relief pursuant to Section 10 of the RTI Act, 2005: Direct CPIO to furnish verified copies "
                "of unaddressed items within 15 days of this order, while upholding closure on items already furnished."
            )

        # =========================================================================
        # PRIORITY LEVEL 4: UNJUSTIFIED EVASION VS ZERO EXEMPTIONS
        # =========================================================================
        elif (missing_count > 0 or coverage_ratio < 0.35) and (not signal3_facts.has_valid_sec8_claim):
            final_outcome = "INFORMATION_DIRECTED"
            confidence = 0.88
            is_override = (s1_pred != final_outcome)
            rule_triggered = "RULE_4_UNJUSTIFIED_EVASION_OVERRULED"
            governing_clause = "Section 7(1) - Duty to Disclose Information"
            trace.append(
                f"[Symbolic Override -> Level 4]: PIO failed to furnish requested records while citing zero statutory exemptions "
                f"under Section 8(1). Overruling bureaucratic denial -> Directing Disclosure."
            )
            directive = (
                "Direct CPIO to furnish complete, point-wise certified information on all requested items to the appellant "
                "free of charge within 15 days of receipt of this order under Section 7(6)."
            )

        # =========================================================================
        # PRIORITY LEVEL 5: LEGITIMATE SECTION 8 EXEMPTION SUSTAINED
        # =========================================================================
        elif signal3_facts.has_valid_sec8_claim and (s1_pred == "APPEAL_DISMISSED" or coverage_ratio < 0.30):
            final_outcome = "APPEAL_DISMISSED"
            confidence = 0.89
            is_override = (s1_pred != final_outcome)
            rule_triggered = "RULE_5_SEC8_EXEMPTION_SUSTAINED"
            exempt_str = ", ".join(signal3_facts.exemptions_detected)
            governing_clause = f"Section 8(1) - Statutory Exemption ({exempt_str})"
            trace.append(
                f"[Symbolic Verification -> Level 5]: Exemption claim under Section {exempt_str} is legally sustained. "
                f"No overriding public interest under Section 8(2) established. Upholding CPIO refusal."
            )
            directive = (
                f"Appeal dismissed: Refusal by CPIO is upheld under Section(s) {exempt_str} of the RTI Act, 2005. "
                f"Requested records are protected from disclosure."
            )

        # =========================================================================
        # PRIORITY LEVEL 6: SOFT GATED FUSION & NEURAL BAYESIAN DEFAULT
        # =========================================================================
        else:
            final_outcome = s1_pred
            confidence = s1_conf
            is_override = False
            rule_triggered = "RULE_6_NEURAL_BAYESIAN_CONSENSUS"
            governing_clause = "Historical CIC Case Precedent & Procedural Regularity"
            trace.append(
                f"[Symbolic Consensus -> Level 6]: No statutory boundary violated. Case decided in accordance with "
                f"calibrated neural prior '{final_outcome}' ({confidence*100:.1f}% confidence)."
            )
            if final_outcome == "INFORMATION_DIRECTED":
                directive = "Direct CPIO to supply information sought within 15 days."
            elif final_outcome == "APPEAL_DISMISSED":
                directive = "No further intervention warranted. Appeal is dismissed."
            elif final_outcome == "PARTIAL_RELIEF":
                directive = "Direct partial disclosure under Section 10 within 15 days."
            else:
                directive = "Issue notice to designated officer under Section 20."

        return AdvisoryVerdict(
            final_predicted_outcome=final_outcome,
            confidence=round(confidence, 4),
            is_statutory_override=is_override,
            governing_statutory_clause=governing_clause,
            symbolic_rule_triggered=rule_triggered,
            actionable_judicial_directive=directive,
            reasoning_audit_trace=trace,
            signal1_neural_prior=signal1_probs,
            signal2_coverage_summary={
                "total_items": total_items,
                "addressed_count": addressed_count,
                "missing_count": missing_count,
                "coverage_ratio": coverage_ratio,
                "compliance_verdict": compliance_verdict
            },
            signal3_statutory_facts={
                "exemptions_detected": signal3_facts.exemptions_detected,
                "is_sec24_organization": signal3_facts.is_sec24_organization,
                "sec20_penalty_triggered": signal3_facts.sec20_penalty_triggered,
                "sec10_severability_indicated": signal3_facts.sec10_severability_indicated,
                "statutory_summary": signal3_facts.statutory_summary
            }
        )
