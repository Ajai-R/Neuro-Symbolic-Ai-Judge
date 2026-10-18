"""
Statutory Legal Guardrails & Hierarchy of Precedence
Project: Explainable Neuro-Symbolic AI Advisory System for RTI Decision Outcome Prediction
Batch ID: 30
"""

from typing import Dict, List, Optional, Tuple


class StatutoryGuardrails:
    """
    Enforces non-negotiable statutory rules under the RTI Act, 2005.
    Acts as the judicial sanity layer that guards against ungrounded probabilistic guesses.
    """

    @classmethod
    def apply_guardrails(
        cls,
        fused_prediction: str,
        fused_confidence: float,
        signal2_coverage: Dict,
        signal3_facts
    ) -> Dict:
        """
        Applies the Hierarchy of Legal Precedence to the weighted mathematical prediction.
        """
        metrics = signal2_coverage.get("coverage_metrics", {})
        total_items = max(metrics.get("total_items", 1), 1)
        addressed_count = metrics.get("addressed_count", 0)
        missing_count = metrics.get("missing_count", 0)
        partial_count = metrics.get("partial_count", 0)
        coverage_ratio = metrics.get("coverage_ratio", 0.0)

        is_sec24 = getattr(signal3_facts, "is_sec24_organization", False)
        sec24_name = getattr(signal3_facts, "sec24_agency_name", "Exempt Security Agency")
        sec20_penalty = getattr(signal3_facts, "sec20_penalty_triggered", False)
        sec20_indicators = getattr(signal3_facts, "sec20_indicators", [])
        sec10_severability = getattr(signal3_facts, "sec10_severability_indicated", False)
        has_sec8 = getattr(signal3_facts, "has_valid_sec8_claim", False)
        exemptions = getattr(signal3_facts, "exemptions_detected", [])

        steps = []
        is_override = False
        final_outcome = fused_prediction
        confidence = fused_confidence
        rule_name = "WEIGHTED_PROBABILISTIC_CONSENSUS"
        governing_clause = "Continuous Multi-Signal Weighted Consensus"
        directive = "Dispose appeal in accordance with the multi-signal weighted finding."

        # =========================================================================
        # GUARDRAIL 1: SECTION 24 JURISDICTIONAL IMMUNITY (VETO)
        # =========================================================================
        if is_sec24 and fused_prediction != "PENALTY_IMPOSED":
            final_outcome = "APPEAL_DISMISSED"
            confidence = 0.99
            is_override = (fused_prediction != final_outcome)
            rule_name = "SEC24_JURISDICTIONAL_IMMUNITY_VETO"
            governing_clause = "Section 24(1) - Second Schedule Exempt Security Organization"
            steps.append(
                f"[GUARDRAIL 1 TRIGGERED - Section 24]: Public Authority is an exempt intelligence agency ({sec24_name}). "
                f"Statute strictly deprives the Commission of jurisdiction. Enforcing mandatory dismissal."
            )
            directive = (
                f"Appeal dismissed: Commission lacks jurisdiction under Section 24 of the RTI Act, 2005. "
                f"Respondent ({sec24_name}) is exempt from the purview of the Act."
            )

        # =========================================================================
        # GUARDRAIL 2: SECTION 20 STATUTORY PENAL SANCTION (MANDATORY PENALTY)
        # =========================================================================
        elif sec20_penalty:
            final_outcome = "PENALTY_IMPOSED"
            confidence = 0.95
            is_override = (fused_prediction != final_outcome)
            rule_name = "SEC20_STATUTORY_PENAL_SANCTION"
            governing_clause = "Section 20(1) & 20(2) - Penalties and Disciplinary Action"
            steps.append(
                f"[GUARDRAIL 2 TRIGGERED - Section 20]: Established grounds for penalty ({', '.join(sec20_indicators[:2])}). "
                f"Enforcing Section 20 penal show-cause proceeding."
            )
            directive = (
                "Issue Show-Cause Notice under Section 20(1) of the RTI Act, 2005 to designated CPIO to show cause "
                "why maximum penalty of Rs. 25,000/- should not be levied for unjustified default."
            )

        # =========================================================================
        # GUARDRAIL 3: SECTION 10 SEVERABILITY PRINCIPLE (MANDATORY PARTIAL RELIEF)
        # Applies when disclosable parts can be severed from exempt records.
        # Does not override APPEAL_DISMISSED unless severability was explicitly argued.
        # =========================================================================
        elif fused_prediction != "PENALTY_IMPOSED" and has_sec8 and (
            (sec10_severability and missing_count > 0) or
            (fused_prediction != "APPEAL_DISMISSED" and addressed_count > 0 and (missing_count > 0 or partial_count > 0))
        ):
            final_outcome = "PARTIAL_RELIEF"
            confidence = 0.92
            is_override = (fused_prediction != final_outcome)
            rule_name = "SEC10_SEVERABILITY_MANDATE"
            governing_clause = "Section 10(1) - Severability Principle"
            steps.append(
                f"[GUARDRAIL 3 TRIGGERED - Section 10]: Factual audit proves mixed compliance ({addressed_count} furnished, {missing_count} omitted) with partial exemptions. "
                f"Under Section 10(1), disclosable portions must be severed. Enforcing PARTIAL_RELIEF."
            )
            directive = (
                "Grant Partial Relief pursuant to Section 10 of the RTI Act, 2005: Direct CPIO to furnish verified copies "
                "of unaddressed items within 15 days of this order, while upholding closure on items already furnished."
            )

        # =========================================================================
        # GUARDRAIL 4: UNJUSTIFIED EVASION OVERRULE (DISCLOSURE MANDATE)
        # Never downgrade an existing penalty or partial relief verdict
        # =========================================================================
        elif fused_prediction not in ["PENALTY_IMPOSED", "PARTIAL_RELIEF"] and (missing_count > 0 or coverage_ratio < 0.35) and (not has_sec8):
            final_outcome = "INFORMATION_DIRECTED"
            confidence = 0.89
            is_override = (fused_prediction != final_outcome)
            rule_name = "UNJUSTIFIED_EVASION_OVERRULE"
            governing_clause = "Section 7(1) - Mandatory Duty to Disclose"
            steps.append(
                f"[GUARDRAIL 4 TRIGGERED - Section 7]: PIO failed to furnish requested records while citing ZERO statutory exemptions "
                f"under Section 8(1). Administrative refusal is overruled. Directing complete disclosure."
            )
            directive = (
                "Direct CPIO to furnish complete, point-wise certified information on all requested items to the appellant "
                "free of charge within 15 days under Section 7(6) of the RTI Act, 2005."
            )

        # =========================================================================
        # GUARDRAIL 5: LEGITIMATE SECTION 8 EXEMPTION SUSTAINED
        # Sustains dismissal when valid exemption covers the requested record
        # Never overrides an existing penalty or partial relief verdict
        # =========================================================================
        elif fused_prediction not in ["PENALTY_IMPOSED", "PARTIAL_RELIEF"] and has_sec8 and (fused_prediction == "APPEAL_DISMISSED" or coverage_ratio >= 0.90):
            final_outcome = "APPEAL_DISMISSED"
            confidence = 0.88
            is_override = (fused_prediction != final_outcome)
            rule_name = "SEC8_EXEMPTION_SUSTAINED"
            ex_str = ", ".join(exemptions)
            governing_clause = f"Section 8(1) Exemption ({ex_str})"
            steps.append(
                f"[GUARDRAIL 5 TRIGGERED - Section 8]: Statutory exemption under Section {ex_str} is legally sustained. "
                f"Requested records are protected from public disclosure."
            )
            directive = (
                f"Appeal dismissed: Refusal by CPIO is upheld under Section(s) {ex_str} of the RTI Act, 2005."
            )

        # =========================================================================
        # LEVEL 6: NO STATUTORY BOUNDARY VIOLATED -> ACCEPT WEIGHTED PREDICTION
        # =========================================================================
        else:
            steps.append(
                f"[NO GUARDRAIL VIOLATION]: Case falls within standard judicial discretion. "
                f"Accepting weighted multi-signal consensus '{final_outcome}' ({confidence*100:.1f}% confidence)."
            )
            if final_outcome == "INFORMATION_DIRECTED":
                directive = "Direct CPIO to supply information sought within 15 days."
            elif final_outcome == "APPEAL_DISMISSED":
                directive = "No further intervention warranted. Appeal is dismissed."
            elif final_outcome == "PARTIAL_RELIEF":
                directive = "Direct partial disclosure under Section 10 within 15 days."
            else:
                directive = "Issue notice to designated officer under Section 20."

        return {
            "final_outcome": final_outcome,
            "confidence": round(confidence, 4),
            "is_override": is_override,
            "rule_triggered": rule_name,
            "governing_clause": governing_clause,
            "directive": directive,
            "reasoning_steps": steps
        }
