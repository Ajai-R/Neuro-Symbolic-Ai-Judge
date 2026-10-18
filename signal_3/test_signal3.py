"""
Signal 3: Unit Test Suite & End-to-End Advisory Verification
Project: Explainable Neuro-Symbolic AI Advisory System for RTI Decision Outcome Prediction
Batch ID: 30
"""

import os
import sys
import json
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from signal_3.scanner import StatutoryScanner
from signal_3.engine import NeuroSymbolicDecisionEngine
from signal_3.pipeline import NeuroSymbolicAdvisorySystem


def test_statutory_overrides():
    print("=" * 80)
    print("TEST 1: VERIFYING STATUTORY OVERRIDES ON SYNTHETIC BENCHMARK CASES")
    print("=" * 80)

    scanner = StatutoryScanner()
    engine = NeuroSymbolicDecisionEngine()

    # Scenario 1: Section 24 Jurisdictional Bar (CBI / RAW)
    print("\n--- Scenario 1: Section 24 Intelligence Agency Veto ---")
    facts_sec24 = scanner.scan_case(
        case_narrative="The applicant sought internal case investigation diary and records from the Central Bureau of Investigation (CBI).",
        public_authority="Central Bureau of Investigation",
        information_sought="Case investigation file no 24/2021"
    )
    assert facts_sec24.is_sec24_organization, "Failed to detect Section 24 exempt organization!"
    
    # Even if Signal 1 neural model predicts INFORMATION_DIRECTED with 80%
    fake_s1 = {"INFORMATION_DIRECTED": 0.80, "APPEAL_DISMISSED": 0.15, "PENALTY_IMPOSED": 0.03, "PARTIAL_RELIEF": 0.02}
    fake_s2 = {"coverage_metrics": {"total_items": 1, "addressed_count": 0, "missing_count": 1, "coverage_ratio": 0.0}}
    verdict_sec24 = engine.synthesize(fake_s1, fake_s2, facts_sec24)
    print(f"Result: {verdict_sec24.final_predicted_outcome} (Rule: {verdict_sec24.symbolic_rule_triggered})")
    assert verdict_sec24.final_predicted_outcome == "APPEAL_DISMISSED", "Section 24 failed to override to APPEAL_DISMISSED!"
    print("[PASS] Scenario 1: Section 24 Jurisdictional Bar correctly overrode neural prior.")

    # Scenario 2: Section 10 Severability (Partial Relief)
    print("\n--- Scenario 2: Section 10 Severability Mandate ---")
    facts_sec10 = scanner.scan_case(
        case_narrative="CPIO supplied item 1 but withheld item 2 citing confidentiality.",
        information_sought="1. Tender copy. 2. Internal audit sheet."
    )
    fake_s1 = {"APPEAL_DISMISSED": 0.65, "INFORMATION_DIRECTED": 0.25, "PENALTY_IMPOSED": 0.05, "PARTIAL_RELIEF": 0.05}
    fake_s2 = {"coverage_metrics": {"total_items": 2, "addressed_count": 1, "missing_count": 1, "coverage_ratio": 0.50}}
    verdict_sec10 = engine.synthesize(fake_s1, fake_s2, facts_sec10)
    print(f"Result: {verdict_sec10.final_predicted_outcome} (Rule: {verdict_sec10.symbolic_rule_triggered})")
    assert verdict_sec10.final_predicted_outcome == "PARTIAL_RELIEF", "Failed to enforce Section 10 PARTIAL_RELIEF!"
    print("[PASS] Scenario 2: Section 10 correctly enforced PARTIAL_RELIEF.")

    # Scenario 3: Section 20 Statutory Penalty
    print("\n--- Scenario 3: Section 20 Penalty Enforcement ---")
    facts_sec20 = scanner.scan_case(
        case_narrative="The Commission issued a show cause notice to the CPIO for malafide denial and deliberate delay of 90 days.",
        information_sought="Service verification records"
    )
    assert facts_sec20.sec20_penalty_triggered, "Failed to detect Section 20 penalty triggers!"
    fake_s1 = {"INFORMATION_DIRECTED": 0.60, "APPEAL_DISMISSED": 0.20, "PENALTY_IMPOSED": 0.15, "PARTIAL_RELIEF": 0.05}
    fake_s2 = {"coverage_metrics": {"total_items": 1, "addressed_count": 0, "missing_count": 1, "coverage_ratio": 0.0}}
    verdict_sec20 = engine.synthesize(fake_s1, fake_s2, facts_sec20)
    print(f"Result: {verdict_sec20.final_predicted_outcome} (Rule: {verdict_sec20.symbolic_rule_triggered})")
    assert verdict_sec20.final_predicted_outcome == "PENALTY_IMPOSED", "Failed to enforce Section 20 PENALTY_IMPOSED!"
    print("[PASS] Scenario 3: Section 20 penalty correctly triggered.")


def test_end_to_end_advisory():
    print("\n" + "=" * 80)
    print("TEST 2: END-TO-END ADVISORY PIPELINE ON REAL CASE FROM TEST SET")
    print("=" * 80)

    test_csv = os.path.join(project_root, "Dataset", "test.csv")
    df = pd.read_csv(test_csv)
    sample = df.iloc[0]

    system = NeuroSymbolicAdvisorySystem()
    result = system.advise_case(
        case_narrative=sample["case_narrative"],
        information_sought=sample["information_sought"],
        public_authority=sample["public_authority"],
        exemptions_cited=sample["exemptions_cited"],
        case_id=sample["case_id"]
    )

    report = system.format_advisory_report(result)
    print("\n" + report)
    print(f"\nGround-Truth Outcome in Dataset: {sample['outcome']}")
    print("\n[SUCCESS] Test 2: Master Neuro-Symbolic Advisory System successfully executed!")


if __name__ == "__main__":
    test_statutory_overrides()
    test_end_to_end_advisory()
