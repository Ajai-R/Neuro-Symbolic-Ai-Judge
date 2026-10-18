"""
Signal 1: Test Set Evaluation & Benchmark Metrics
Project: Explainable Neuro-Symbolic AI Advisory System for RTI Decision Outcome Prediction
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# Ensure UTF-8 output encoding on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from signal_1_inlegalbert.predict import InLegalBERTOutcomePredictor
from signal_1_inlegalbert.dataset import LABEL2ID, ID2LABEL


def evaluate_test_set():
    test_path = os.path.join(project_root, "Dataset", "test.csv")
    print(f"Loading Unseen Blind Test Set from: {test_path}")
    test_df = pd.read_csv(test_path)
    print(f"Total Test Cases: {len(test_df)}")

    predictor = InLegalBERTOutcomePredictor()
    print("Model loaded successfully. Running inference on test set...")

    predictions = []
    probabilities_list = []

    for idx, row in test_df.iterrows():
        res = predictor.predict(
            case_narrative=row['case_narrative'],
            information_sought=row.get('information_sought'),
            exemptions_cited=row.get('exemptions_cited'),
            public_authority=row.get('public_authority')
        )
        predictions.append(res['predicted_class'])
        probabilities_list.append(res['probabilities'])
        if (idx + 1) % 50 == 0 or (idx + 1) == len(test_df):
            print(f"  Processed {idx + 1}/{len(test_df)} cases...")

    test_df['predicted_outcome'] = predictions
    test_df['predicted_probabilities'] = [json.dumps(p) for p in probabilities_list]

    y_true = test_df['outcome'].tolist()
    y_pred = predictions
    target_names = [ID2LABEL[i] for i in range(len(ID2LABEL))]

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=target_names, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, labels=target_names, average='weighted')
    report = classification_report(y_true, y_pred, labels=target_names, target_names=target_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=target_names)

    print("\n" + "=" * 70)
    print(" SIGNAL 1: TEST BENCHMARK EVALUATION RESULTS (20% BLIND TEST SET)")
    print("=" * 70)
    print(f"Overall Accuracy:  {acc * 100:.2f}%")
    print(f"Macro F1-Score:    {macro_f1 * 100:.2f}%")
    print(f"Weighted F1-Score: {weighted_f1 * 100:.2f}%\n")

    print(classification_report(y_true, y_pred, labels=target_names, target_names=target_names, zero_division=0))

    print("\nConfusion Matrix (Rows: Actual, Columns: Predicted):")
    print(f"{'':<24} " + " ".join([f"{name[:8]:>8}" for name in target_names]))
    for idx, row in enumerate(cm):
        row_str = " ".join([f"{val:>8}" for val in row])
        print(f"{target_names[idx]:<24} {row_str}")

    output_dir = os.path.join(current_dir, "checkpoints")
    report_file = os.path.join(output_dir, "test_evaluation_report.json")
    with open(report_file, "w") as f:
        json.dump({
            "test_accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "target_names": target_names
        }, f, indent=2)

    predictions_out = os.path.join(output_dir, "test_predictions.csv")
    test_df.to_csv(predictions_out, index=False)
    print(f"\nSaved test predictions to: {predictions_out}")
    print(f"Saved evaluation metrics to: {report_file}")


if __name__ == "__main__":
    evaluate_test_set()
