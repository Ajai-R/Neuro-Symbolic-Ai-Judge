"""
Signal 2: Verification and Unit Test Suite
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

from signal_2_coverage.pipeline import CoverageVerificationPipeline
from signal_2_coverage.config import CoverageConfig


def test_out_of_order_alignment():
    print("=" * 80)
    print("TEST 1: VERIFYING OUT-OF-ORDER ALIGNMENT (Q1 in R3, Q2 in R2, Q3 MISSING)")
    print("=" * 80)

    # Citizen asks for 3 distinct items
    query = (
        "1. Tender quotation documents submitted by all bidders.\n"
        "2. Approved structural design drawings.\n"
        "3. Logbook of site inspection and quality testing."
    )

    # Narrative answers Q2 first, then Q1, and remains completely silent on Q3
    narrative = (
        "The hearing was conducted before the Information Commissioner. "
        "The CPIO stated that certified copies of the approved structural design drawings were dispatched on 15.01.2023. "
        "Furthermore, tender quotation documents submitted by all bidders were formally furnished via registered post. "
        "The respondent offered no response or documentation concerning any other departmental records."
    )

    pipeline = CoverageVerificationPipeline()
    result = pipeline.analyze_case(query, narrative, case_id="SYNTHETIC_TEST_01")

    print("\n" + result["audit_table"])
    print("\nPlain-English Rationale:")
    print(result["explainable_rationale"])
    print("\nNeuro-Symbolic Features for Signal 3:")
    print(json.dumps(result["neuro_symbolic_features"], indent=2))

    breakdown = result["item_breakdown"]
    assert len(breakdown) == 3, f"Expected 3 items, got {len(breakdown)}"
    
    # Q1 should match sentence about tender quotations
    assert "tender" in breakdown[0]["best_match_text"].lower(), "Q1 did not match tender sentence!"
    assert breakdown[0]["status"] == "ADDRESSED", f"Q1 status should be ADDRESSED, got {breakdown[0]['status']}"
    
    # Q2 should match structural design drawings
    assert "structural" in breakdown[1]["best_match_text"].lower(), "Q2 did not match structural sentence!"
    assert breakdown[1]["status"] == "ADDRESSED", f"Q2 status should be ADDRESSED, got {breakdown[1]['status']}"

    # Q3 should be missing (logbook)
    assert breakdown[2]["status"] in ["MISSING", "PARTIALLY_ADDRESSED"], "Q3 should be MISSING or PARTIAL"

    print("\n[SUCCESS] Test 1: Out-of-order alignment successfully verified!")


def test_real_dataset_case():
    print("\n" + "=" * 80)
    print("TEST 2: EVALUATING REAL RTI CASE FROM BLIND TEST SET (Dataset/test.csv)")
    print("=" * 80)

    test_csv = os.path.join(project_root, "Dataset", "test.csv")
    if not os.path.exists(test_csv):
        print(f"Skipping Test 2: {test_csv} not found.")
        return

    df = pd.read_csv(test_csv)
    # Pick a sample case
    sample_row = df.iloc[0]
    case_id = sample_row["case_id"]
    query = sample_row["information_sought"]
    narrative = sample_row["case_narrative"]
    actual = sample_row["outcome"]

    print(f"Case ID: {case_id}")
    print(f"Ground-Truth Outcome: {actual}")
    print(f"Information Sought: {query[:120]}...")

    pipeline = CoverageVerificationPipeline()
    result = pipeline.analyze_case(query, narrative, case_id=case_id)

    print("\n" + result["audit_table"])
    print("\nExplainable Rationale:")
    print(result["explainable_rationale"])
    print(f"\nExecution Time: {result['execution_time_sec']}s")

    print("\n[SUCCESS] Test 2: Real dataset case successfully analyzed!")


if __name__ == "__main__":
    test_out_of_order_alignment()
    test_real_dataset_case()
