"""
Signal 2: Explainability & Audit Trail Generation Module
Project: Explainable Neuro-Symbolic AI Advisory System for RTI Decision Outcome Prediction
Batch ID: 30
"""

from typing import Dict, List


class ExplainabilityGenerator:
    """
    Transforms dense similarity scores and matched text evidence into
    auditable human-readable checklists and plain-English legal rationales.
    """

    @staticmethod
    def generate_audit_table(coverage_result: Dict) -> str:
        """
        Creates an ASCII table for judicial audit inspection.
        """
        lines = []
        lines.append("=" * 95)
        lines.append(f"{'Item':<6} | {'Information Requested':<35} | {'Matched Evidence Snippet':<32} | {'Score':<6} | {'Status'}")
        lines.append("-" * 95)

        for rec in coverage_result["item_breakdown"]:
            item_id = rec["item_id"]
            q_text = rec["question_text"]
            if len(q_text) > 33:
                q_text = q_text[:30] + "..."
            
            ev_text = rec["best_match_text"]
            if len(ev_text) > 30:
                ev_text = ev_text[:27] + "..."
            
            score = f"{rec['similarity_score']:.2f}"
            status = rec["status"]
            lines.append(f"{item_id:<6} | {q_text:<35} | {ev_text:<32} | {score:<6} | {status}")

        lines.append("=" * 95)
        lines.append(
            f"SUMMARY: Total Items: {coverage_result['total_items']} | "
            f"Addressed: {coverage_result['addressed_count']} | "
            f"Partial: {coverage_result['partial_count']} | "
            f"Missing: {coverage_result['missing_count']} | "
            f"Coverage Ratio: {coverage_result['coverage_ratio']*100:.1f}% ({coverage_result['compliance_verdict']})"
        )
        return "\n".join(lines)

    @staticmethod
    def generate_plain_english_rationale(coverage_result: Dict) -> str:
        """
        Synthesizes an explainable legal audit paragraph.
        """
        total = coverage_result["total_items"]
        addressed = coverage_result["addressed_count"]
        partial = coverage_result["partial_count"]
        missing = coverage_result["missing_count"]
        verdict = coverage_result["compliance_verdict"]
        ratio_pct = coverage_result["coverage_ratio"] * 100

        evaded_items = [
            f"'{r['question_text']}'" for r in coverage_result["item_breakdown"]
            if r["status"] in ["MISSING", "PARTIALLY_ADDRESSED"]
        ]

        if verdict == "FULL_COMPLIANCE":
            rationale = (
                f"Full Statutory Compliance Verified ({ratio_pct:.1f}% coverage): The PIO provided direct, verifiable "
                f"factual disclosures matching all {total} item(s) requested by the citizen without omission. "
                f"Factual records show zero unaddressed queries."
            )
        elif verdict == "PARTIAL_COMPLIANCE":
            evaded_str = ", ".join(evaded_items[:3])
            rationale = (
                f"Partial Compliance Detected ({ratio_pct:.1f}% coverage): The PIO addressed {addressed} of {total} "
                f"requested items, but failed to provide complete disclosure on {len(evaded_items)} item(s), notably {evaded_str}. "
                f"This engages Section 10 of the RTI Act (Severability Principle), supporting partial relief or a directed disclosure."
            )
        else:  # TOTAL_EVASION
            rationale = (
                f"Administrative Non-Compliance / Evasion Detected ({ratio_pct:.1f}% coverage): The PIO failed to furnish "
                f"substantive information on all {total} requested item(s). The submissions show either total omission or "
                f"conclusory evasion without demonstrable factual supply."
            )

        return rationale

    @staticmethod
    def generate_neuro_symbolic_features(coverage_result: Dict) -> Dict:
        """
        Produces numerical and boolean features specifically formatted
        for the Signal 3 Deterministic Switch Matrix.
        """
        return {
            "signal2_coverage_ratio": coverage_result["coverage_ratio"],
            "signal2_total_items": coverage_result["total_items"],
            "signal2_addressed_count": coverage_result["addressed_count"],
            "signal2_missing_count": coverage_result["missing_count"],
            "signal2_has_missing_items": coverage_result["missing_count"] > 0,
            "signal2_is_full_compliance": coverage_result["compliance_verdict"] == "FULL_COMPLIANCE",
            "signal2_is_partial_compliance": coverage_result["compliance_verdict"] == "PARTIAL_COMPLIANCE",
            "signal2_is_total_evasion": coverage_result["compliance_verdict"] == "TOTAL_EVASION"
        }
