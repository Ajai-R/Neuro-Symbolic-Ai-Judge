"""
Dataset Harmonization & Balanced Splitting Pipeline
Combines dataset_cleaned.csv and newcases.csv into Final_dataset.csv,
and generates stratified 80/20 train/test splits with balanced classes.
Project: Explainable Neuro-Symbolic AI Advisory System for RTI Decision Outcome Prediction
Batch ID: 30
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Ensure UTF-8 output encoding on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, "Dataset")

# The 4 target outcome classes
TARGET_CLASSES = [
    'INFORMATION_DIRECTED',
    'APPEAL_DISMISSED',
    'PENALTY_IMPOSED',
    'PARTIAL_RELIEF'
]

REQUIRED_COLUMNS = [
    'case_id',
    'source',
    'case_narrative',
    'information_sought',
    'exemptions_cited',
    'outcome',
    'ruling_text',
    'public_authority'
]


def load_and_validate_dataset(file_path: str, name: str) -> pd.DataFrame:
    """Loads a CSV dataset and validates schema integrity."""
    print(f"Loading {name} from: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)
    print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")

    # Check required columns
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in {name}: {missing_cols}")

    # Reorder columns to standard schema
    df = df[REQUIRED_COLUMNS].copy()

    # Check for null values
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print(f"  [WARNING] Null values detected in {name}:\n{null_counts[null_counts > 0]}")
        # Standardize empty text
        df['information_sought'] = df['information_sought'].fillna("Information sought under RTI application")
        df['public_authority'] = df['public_authority'].fillna("Not Specified")
        df['exemptions_cited'] = df['exemptions_cited'].fillna("[]")
        df = df.dropna(subset=['case_narrative', 'outcome']).reset_index(drop=True)

    # Validate exemption format
    invalid_ex = 0
    for ex in df['exemptions_cited']:
        try:
            p = json.loads(str(ex))
            if not isinstance(p, list):
                invalid_ex += 1
        except Exception:
            invalid_ex += 1
    if invalid_ex > 0:
        print(f"  [WARNING] {invalid_ex} records in {name} had unparsed exemptions_cited; normalising to '[]'")
        df['exemptions_cited'] = df['exemptions_cited'].apply(
            lambda x: x if (isinstance(x, str) and x.startswith('[')) else '[]'
        )

    # Class distribution breakdown
    print(f"  Outcome counts in {name}:")
    for outcome, cnt in df['outcome'].value_counts().items():
        print(f"    - {outcome:<22}: {cnt}")

    return df


def combine_and_split(random_state: int = 42, test_size: float = 0.20):
    """
    Merges dataset_cleaned.csv and newcases.csv,
    shuffles thoroughly to break sequential label ordering,
    saves Final_dataset.csv, and generates stratified balanced 80/20 train/test splits.
    """
    clean_path = os.path.join(DATASET_DIR, "dataset_cleaned.csv")
    new_path = os.path.join(DATASET_DIR, "newcases.csv")
    final_path = os.path.join(DATASET_DIR, "Final_dataset.csv")
    train_path = os.path.join(DATASET_DIR, "train.csv")
    test_path = os.path.join(DATASET_DIR, "test.csv")

    print("=" * 80)
    print(" STEP 1: LOAD & VALIDATE INDIVIDUAL DATASETS")
    print("=" * 80)
    df_clean = load_and_validate_dataset(clean_path, "dataset_cleaned.csv")
    df_new = load_and_validate_dataset(new_path, "newcases.csv")

    print("\n" + "=" * 80)
    print(" STEP 2: CONCATENATE & SHUFFLE COMBINED DATASET")
    print("=" * 80)
    df_combined = pd.concat([df_clean, df_new], ignore_index=True)
    print(f"Total concatenated records: {len(df_combined)}")

    # Check for duplicate case_ids or duplicate narratives
    dup_ids = df_combined.duplicated(subset=['case_id'], keep='first')
    if dup_ids.sum() > 0:
        print(f"  [NOTICE] Found {dup_ids.sum()} duplicate case_ids across sources; making IDs unique...")
        df_combined['case_id'] = [
            f"{cid}_dup{i}" if is_dup else cid
            for i, (cid, is_dup) in enumerate(zip(df_combined['case_id'], dup_ids))
        ]

    # Filter strictly to target 4 classes
    df_combined = df_combined[df_combined['outcome'].isin(TARGET_CLASSES)].copy()

    # Thorough random shuffle to ensure labels from newcases are not sequential
    df_final = df_combined.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    print("\nOutcome distribution in Final_dataset:")
    for outcome, cnt in df_final['outcome'].value_counts().items():
        pct = (cnt / len(df_final)) * 100
        print(f"  - {outcome:<22}: {cnt:>5} ({pct:>5.1f}%)")

    # Save Final_dataset.csv
    df_final.to_csv(final_path, index=False)
    print(f"\n[SAVED] Combined dataset saved to: {final_path}")

    # Also save a copy at project root for convenience
    root_final_path = os.path.join(PROJECT_ROOT, "Final_dataset.csv")
    df_final.to_csv(root_final_path, index=False)
    print(f"[SAVED] Copy saved to: {root_final_path}")

    print("\n" + "=" * 80)
    print(f" STEP 3: GENERATE STRATIFIED {int((1-test_size)*100)}/{int(test_size*100)} BALANCED SPLITS")
    print("=" * 80)

    train_df, test_df = train_test_split(
        df_final,
        test_size=test_size,
        stratify=df_final['outcome'],
        random_state=random_state,
        shuffle=True
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Zero leakage assertion
    train_ids = set(train_df['case_id'])
    test_ids = set(test_df['case_id'])
    overlap_ids = train_ids.intersection(test_ids)
    assert len(overlap_ids) == 0, f"Error: Leakage detected! {len(overlap_ids)} overlapping case_ids!"

    train_narratives = set(train_df['case_narrative'])
    test_narratives = set(test_df['case_narrative'])
    overlap_narratives = train_narratives.intersection(test_narratives)
    assert len(overlap_narratives) == 0, f"Error: Leakage detected! {len(overlap_narratives)} overlapping narratives!"

    print("Leakage verification passed: 0 overlapping case IDs and 0 overlapping narratives.")

    # Save train.csv and test.csv
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"[SAVED] Training set ({len(train_df)} rows) saved to: {train_path}")
    print(f"[SAVED] Blind test set ({len(test_df)} rows) saved to: {test_path}")

    print("\n" + "=" * 80)
    print(" STRATIFIED CLASS BALANCE SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Outcome Class':<24} | {'Total Count':<11} | {'Train Count':<11} | {'Train %':<8} | {'Test Count':<10} | {'Test %'}")
    print("-" * 80)
    for outcome in TARGET_CLASSES:
        tot = (df_final['outcome'] == outcome).sum()
        trn = (train_df['outcome'] == outcome).sum()
        tst = (test_df['outcome'] == outcome).sum()
        trn_pct = (trn / len(train_df)) * 100
        tst_pct = (tst / len(test_df)) * 100
        print(f"{outcome:<24} | {tot:>11} | {trn:>11} | {trn_pct:>7.2f}% | {tst:>10} | {tst_pct:>6.2f}%")
    print("-" * 80)
    print(f"{'TOTAL':<24} | {len(df_final):>11} | {len(train_df):>11} | 100.00% | {len(test_df):>10} | 100.00%")
    print("=" * 80)

    return df_final, train_df, test_df


if __name__ == "__main__":
    combine_and_split(random_state=42, test_size=0.20)
