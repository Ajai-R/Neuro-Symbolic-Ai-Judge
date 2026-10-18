"""
Unit Test Suite for Master Neuro-Symbolic Decision Making Engine
Project: Explainable Neuro-Symbolic AI Advisory System for RTI Decision Outcome Prediction
Batch ID: 30
"""

import os
import sys
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from decision_engine.weights_config import FusionWeightsConfig
from decision_engine.weighted_fusion import WeightedSignalFuser
from decision_engine.master_engine import NeuroSymbolicDecisionEngine


def test_pure_weighted_fusion():
    print("=" * 80)
    print("TEST 1: VERIFYING WEIGHTED PROBABILISTIC FUSION (w1=0.5, w2=0.3, w3=0.2)")
    print("=" * 80)

    fuser = WeightedSignalFuser()

    # Case: General dispute where Signal 1 and Signal 2 point to Information Directed
    s1_probs = {"INFORMATION_DIRECTED": 0.60, "APPEAL_DISMISSED": 0.30, "PENALTY_IMPOSED": 0.05, "PARTIAL_RELIEF": 0.05}
    s2_probs = {"INFORMATION_DIRECTED": 0.70, "APPEAL_DISMISSED": 0.15, "PENALTY_IMPOSED": 0.05, "PARTIAL_RELIEF": 0.10}
    s3_probs = {"INFORMATION_DIRECTED": 0.65, "APPEAL_DISMISSED": 0.20, "PENALTY_IMPOSED": 0.05, "PARTIAL_RELIEF": 0.10}

    result = fuser.fuse(s1_probs, s2_probs, s3_probs)
    print("Fused Probabilities:", result["fused_probabilities"])
    print("Top Fused Class:", result["fused_predicted_class"])
    print("Confidence:", result["fused_confidence"])

    assert result["fused_predicted_class"] == "INFORMATION_DIRECTED"
    assert round(sum(result["fused_probabilities"].values()), 2) == 1.00
    print("[PASS] Test 1: Mathematical weighted fusion verified.")


def test_master_engine_on_real_case():
    print("\n" + "=" * 80)
    print("TEST 2: MASTER DECISION MAKING ENGINE ON REAL TEST CASE (Dataset/test.csv)")
    print("=" * 80)

    test_csv = os.path.join(project_root, "Dataset", "test.csv")
    df = pd.read_csv(test_csv)
    sample = df.iloc[0]

    engine = NeuroSymbolicDecisionEngine()
    verdict = engine.decide(
        case_narrative=sample["case_narrative"],
        information_sought=sample["information_sought"],
        public_authority=sample["public_authority"],
        exemptions_cited=sample["exemptions_cited"],
        case_id=sample["case_id"]
    )

    report = engine.format_verdict_report(verdict)
    print("\n" + report)
    print(f"\nGround-Truth Outcome in Dataset: {sample['outcome']}")
    print("\n[PASS] Test 2: Master Decision Making Engine successfully executed!")


if __name__ == "__main__":
    test_pure_weighted_fusion()
    test_master_engine_on_real_case()
