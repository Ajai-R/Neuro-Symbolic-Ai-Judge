"""
Signal 2: Substantive Coverage Verification Pipeline (LLM-as-a-Judge)
Project: Explainable Neuro-Symbolic AI Advisory System for RTI Decision Outcome Prediction
Batch ID: 30
"""

import time
from typing import Dict, List, Optional
import pandas as pd

from .config import CoverageConfig
from .llm_judge import LLMJudicialAuditor


class CoverageVerificationPipeline:
    """
    Signal 2 Substantive Compliance Audit Pipeline:
    1. Itemizes discrete queries requested by the citizen (Q_1 ... Q_N).
    2. Dynamic legal weighting: Core operative demands (weight 0.65) vs. Auxiliary inquiries (weight 0.35).
    3. Evaluates substantive furnishing (Furnished, Partially Furnished, Withheld with Exemption, Evaded).
    4. Computes weighted factual compliance score in [0.0, 1.0].
    5. Emits explainable judicial audit rationale with grounding.
    """
    def __init__(
        self,
        config: Optional[CoverageConfig] = None,
        use_neural_extractor: bool = False,
        model_name: str = "nvidia/nemotron-3-ultra-550b-a55b:free"
    ):
        self.config = config or CoverageConfig()
        self.model_name = model_name
        self.llm_judge = LLMJudicialAuditor(model_name=self.model_name)

    def analyze_case(
        self,
        information_sought: str,
        case_narrative: str,
        case_id: Optional[str] = None
    ) -> Dict:
        """
        Audits an individual RTI dispute through Signal 2 (LLM-as-a-Judge).
        """
        audit = self.llm_judge.audit_case(
            information_sought=information_sought,
            case_narrative=case_narrative,
            case_id=case_id
        )

        metrics = {
            "total_items": audit["total_items"],
            "addressed_count": audit["addressed_count"],
            "partial_count": audit["partial_count"],
            "missing_count": audit["missing_count"],
            "coverage_ratio": audit["coverage_ratio"],
            "compliance_verdict": audit["compliance_verdict"]
        }

        total_items = max(audit["total_items"], 1)

        return {
            "case_id": audit["case_id"],
            "execution_time_sec": audit["execution_time_sec"],
            "judge_model": audit.get("judge_model", self.model_name),
            "questions_extracted": [it["question_text"] for it in audit["item_breakdown"]],
            "num_questions": audit["total_items"],
            "num_response_sentences": len(audit["item_breakdown"]),
            "coverage_metrics": metrics,
            "item_breakdown": audit["item_breakdown"],
            "audit_table": audit.get("audit_summary", ""),
            "explainable_rationale": audit["explainable_rationale"],
            "neuro_symbolic_features": {
                "coverage_ratio": audit["coverage_ratio"],
                "missing_ratio": round(audit["missing_count"] / total_items, 4),
                "has_evasion": audit["missing_count"] > 0,
                "is_full_compliance": audit["compliance_verdict"] == "COMPLETE_COMPLIANCE"
            }
        }

    def generate_case_finding(
        self,
        citizen_query: str,
        pio_reply: str,
        final_outcome: str,
        governing_clause: Optional[str] = None,
        days_elapsed: int = 15,
        exemptions_cited: str = "",
        item_breakdown: Optional[List[Dict]] = None,
        rule_triggered: Optional[str] = None
    ) -> str:
        """
        Synthesizes an explainable judicial finding explaining why the Main Engine's
        predicted outcome applies, grounded strictly in the citizen's query and PIO reply.
        """
        return self.llm_judge.generate_case_finding(
            citizen_query=citizen_query,
            pio_reply=pio_reply,
            days_elapsed=days_elapsed,
            exemptions_cited=exemptions_cited,
            final_outcome=final_outcome,
            governing_clause=governing_clause or "RTI Statutory Provisions",
            item_breakdown=item_breakdown,
            rule_triggered=rule_triggered
        )

    def batch_evaluate(self, df: pd.DataFrame, max_cases: Optional[int] = None) -> pd.DataFrame:
        """
        Runs Signal 2 over a dataset and returns aggregated coverage statistics.
        """
        records = []
        subset = df if max_cases is None else df.head(max_cases)
        total = len(subset)
        print(f"[Signal 2 Batch] Auditing {total} cases with LLM Judge...")

        for idx, row in subset.iterrows():
            case_id = row.get("case_id", f"CASE_{idx}")
            query = row.get("information_sought", "")
            narrative = row.get("case_narrative", "")
            actual_outcome = row.get("outcome", "UNKNOWN")

            result = self.analyze_case(query, narrative, case_id=case_id)
            metrics = result["coverage_metrics"]

            records.append({
                "case_id": case_id,
                "actual_outcome": actual_outcome,
                "total_items": metrics["total_items"],
                "addressed_items": metrics["addressed_count"],
                "partial_items": metrics["partial_count"],
                "missing_items": metrics["missing_count"],
                "coverage_ratio": metrics["coverage_ratio"],
                "compliance_verdict": metrics["compliance_verdict"],
                "execution_time_sec": result["execution_time_sec"]
            })

            if (len(records) % 25 == 0) or (len(records) == total):
                print(f"  Processed {len(records)}/{total} cases...")

        return pd.DataFrame(records)

    def generate_case_finding(
        self,
        citizen_query: str,
        pio_reply: str,
        days_elapsed: int,
        exemptions_cited: str,
        final_outcome: str,
        governing_clause: str,
        item_breakdown: Optional[List[Dict]] = None,
        rule_triggered: Optional[str] = None
    ) -> str:
        """
        Generates context-bounded judicial case finding via LLM Judicial Auditor.
        """
        return self.llm_judge.generate_case_finding(
            citizen_query=citizen_query,
            pio_reply=pio_reply,
            days_elapsed=days_elapsed,
            exemptions_cited=exemptions_cited,
            final_outcome=final_outcome,
            governing_clause=governing_clause,
            item_breakdown=item_breakdown,
            rule_triggered=rule_triggered
        )

