"""
Full Benchmark Evaluation of Master Decision Making Engine on 424 Blind Test Cases
Project: Explainable Neuro-Symbolic AI Advisory System for RTI Decision Outcome Prediction
Batch ID: 30
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from decision_engine.master_engine import NeuroSymbolicDecisionEngine


def run_benchmark():
    test_csv = os.path.join(project_root, "Dataset", "test.csv")
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test set not found at {test_csv}")

    df = pd.read_csv(test_csv)
    total_cases = len(df)
    print("=" * 85)
    print(f" BENCHMARKING MASTER DECISION MAKING ENGINE ON UNSEEN BLIND TEST SET (N={total_cases})")
    print("=" * 85)

    engine = NeuroSymbolicDecisionEngine()

    preds = []
    actuals = []
    overrides = []
    records = []

    start_time = time.time()

    for idx, row in df.iterrows():
        case_id = row.get("case_id", f"CASE_{idx}")
        narrative = row.get("case_narrative", "")
        query = row.get("information_sought", "")
        authority = row.get("public_authority", "")
        exemptions = row.get("exemptions_cited", "")
        actual = row.get("outcome", "")

        verdict = engine.decide(
            case_narrative=narrative,
            information_sought=query,
            public_authority=authority,
            exemptions_cited=exemptions,
            case_id=case_id
        )

        pred = verdict.final_predicted_outcome
        preds.append(pred)
        actuals.append(actual)
        overrides.append(verdict.is_statutory_override)

        records.append({
            "case_id": case_id,
            "actual_outcome": actual,
            "predicted_outcome": pred,
            "confidence": verdict.confidence,
            "is_statutory_override": verdict.is_statutory_override,
            "governing_clause": verdict.governing_statutory_clause,
            "rule_triggered": verdict.symbolic_rule_triggered,
            "signal1_raw": verdict.raw_signal_predictions["signal_1_neural"],
            "signal2_coverage": verdict.raw_signal_predictions["signal_2_coverage_verdict"],
            "signal3_statutory": verdict.raw_signal_predictions["signal_3_statutory_detected"],
            "weighted_fusion_top": verdict.raw_signal_predictions["weighted_fusion_top"],
            "p_INFO": verdict.fused_probabilities.get("INFORMATION_DIRECTED", 0.0),
            "p_DISMISSED": verdict.fused_probabilities.get("APPEAL_DISMISSED", 0.0),
            "p_PENALTY": verdict.fused_probabilities.get("PENALTY_IMPOSED", 0.0),
            "p_PARTIAL": verdict.fused_probabilities.get("PARTIAL_RELIEF", 0.0),
            "latency_sec": verdict.execution_time_sec
        })

        if (idx + 1) % 50 == 0 or (idx + 1) == total_cases:
            print(f"  Processed {idx + 1}/{total_cases} test cases... (Override rate so far: {sum(overrides)/(idx+1)*100:.1f}%)")

    total_time = round(time.time() - start_time, 2)
    classes = [
        "INFORMATION_DIRECTED",
        "APPEAL_DISMISSED",
        "PENALTY_IMPOSED",
        "PARTIAL_RELIEF"
    ]

    acc = accuracy_score(actuals, preds)
    prec_m, rec_m, macro_f1, _ = precision_recall_fscore_support(actuals, preds, labels=classes, average="macro", zero_division=0)
    prec_w, rec_w, weighted_f1, _ = precision_recall_fscore_support(actuals, preds, labels=classes, average="weighted", zero_division=0)
    cm = confusion_matrix(actuals, preds, labels=classes)

    p_c, r_c, f_c, sup_c = precision_recall_fscore_support(actuals, preds, labels=classes, average=None, zero_division=0)
    class_report = {}
    for i, c in enumerate(classes):
        class_report[c] = {
            "precision": round(float(p_c[i]), 4),
            "recall": round(float(r_c[i]), 4),
            "f1-score": round(float(f_c[i]), 4),
            "support": int(sup_c[i])
        }

    # Print Summary
    print("\n" + "=" * 85)
    print(" MASTER NEURO-SYMBOLIC DECISION MAKING ENGINE: BLIND TEST BENCHMARK RESULTS")
    print("=" * 85)
    print(f"Overall Accuracy:  {acc*100:.2f}%")
    print(f"Macro F1-Score:    {macro_f1*100:.2f}%")
    print(f"Weighted F1-Score: {weighted_f1*100:.2f}%")
    print(f"Total Test Cases:  {total_cases} (106 per class, exactly balanced)")
    print(f"Statutory Override Rate: {sum(overrides)/total_cases*100:.2f}% ({sum(overrides)}/{total_cases} cases)")
    print(f"Total Benchmark Time:    {total_time:.1f}s (Avg {total_time/total_cases:.3f}s/case)")
    print("\nPer-Class Breakdown:")
    for c in classes:
        cr = class_report[c]
        print(f"  • {c:<22}: Precision={cr['precision']:.3f} | Recall={cr['recall']:.3f} | F1={cr['f1-score']:.3f}")

    print("\nConfusion Matrix (Rows: Actual, Columns: Predicted):")
    print(f"{'Actual / Pred':<22} | {'INFO':<8} | {'DISMISSED':<10} | {'PENALTY':<8} | {'PARTIAL':<8}")
    print("-" * 65)
    for i, c in enumerate(classes):
        print(f"{c:<22} | {cm[i][0]:<8} | {cm[i][1]:<10} | {cm[i][2]:<8} | {cm[i][3]:<8}")

    # Save outputs
    pred_df = pd.DataFrame(records)
    pred_csv_path = os.path.join(current_dir, "test_predictions.csv")
    pred_df.to_csv(pred_csv_path, index=False)
    print(f"\nDetailed predictions saved to: {pred_csv_path}")

    report_path = os.path.join(current_dir, "test_evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump({
            "test_accuracy": round(float(acc), 4),
            "macro_f1": round(float(macro_f1), 4),
            "weighted_f1": round(float(weighted_f1), 4),
            "override_rate": round(sum(overrides) / total_cases, 4),
            "total_time_sec": total_time,
            "classification_report": class_report,
            "confusion_matrix": cm.tolist(),
            "target_names": classes
        }, f, indent=2)
    print(f"Evaluation report saved to: {report_path}")
    print("=" * 85)


if __name__ == "__main__":
    run_benchmark()
