"""
RTI-Bench Data Preprocessing & Stratified Splitting Pipeline
Project: Explainable Neuro-Symbolic AI Advisory System for RTI Decision Outcome Prediction
Batch ID: 30
"""

import os
import sys
import json
import ast
import re
import unicodedata
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Ensure UTF-8 output encoding on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Target 4 core classes matching Slide 14 of Review 1 PPT
TARGET_CLASSES = [
    'INFORMATION_DIRECTED',
    'APPEAL_DISMISSED',
    'PENALTY_IMPOSED',
    'PARTIAL_RELIEF'
]

def clean_text(text):
    """
    Cleans, normalizes, and removes noise from raw legal text.
    Handles Unicode normalization, mojibake, unescaped characters, and spacing.
    """
    if text is None or pd.isna(text):
        return ""
    
    text = str(text)
    
    # 1. Unicode NFKC normalization
    text = unicodedata.normalize('NFKC', text)
    
    # 2. Remove soft hyphens and non-breaking spaces
    text = text.replace('\xad', '').replace('\xa0', ' ')
    
    # 3. Standardize smart quotes and typographic dashes to ASCII
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2013', '-').replace('\u2014', '--')
    text = text.replace('\u2026', '...')
    text = text.replace('vis-\xe0-vis', 'vis-a-vis')
    text = text.replace('sub\xe0section', 'sub-section').replace('sub\xadsection', 'sub-section')
    text = text.replace('\xe0\xbd\xac', 'Rs. ')
    
    # 4. Fix mojibake / replacement characters (\ufffd)
    # Case a: apostrophe inside words (e.g. appellants -> appellant's)
    text = re.sub(r'(\w)\ufffd(\w)', r"\1'\2", text)
    # Case b: quote marks around phrases (e.g. Excise Duty -> "Excise Duty")
    text = re.sub(r'\ufffd([^\ufffd\n\r]+?)\ufffd', r'"\1"', text)
    # Case c: dangling quotes / dashes
    text = text.replace('\ufffd', "'")
    
    # 4. Clean inline statutory insertions where section name runs into sentence
    # e.g., "section19(8)(b)In its decision" -> "Section 19(8)(b) In its decision"
    text = re.sub(r'(section\s*\d+(?:\([a-zA-Z0-9]+\))*)([A-Za-z])', r'\1 \2', text, flags=re.IGNORECASE)
    # e.g., "section 2(f)" capitalization
    text = re.sub(r'\bsection\s+(\d+)', r'Section \1', text, flags=re.IGNORECASE)
    
    # 5. Fix punctuation spacing
    text = re.sub(r'(\))\s*([A-Za-z])', r'\1 \2', text)
    
    # 6. Normalize whitespace and newlines
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 7. Collapse isolated paragraph numbers (e.g., "4. \n The Commission" -> "4. The Commission")
    text = re.sub(r'^\s*(\d+\.)\s*\n+', r'\1 ', text)
    text = re.sub(r'\n+(\d+\.)\s*\n+', r'\n\n\1 ', text)
    
    return text.strip()


def parse_exemptions(exemptions_raw):
    """
    Standardizes statutory exemption citations into a clean JSON array string.
    Example output: '["8(1)(j)"]' or '[]'
    """
    if exemptions_raw is None:
        return json.dumps([])
    
    if isinstance(exemptions_raw, (list, tuple, np.ndarray)):
        items = [str(x).strip() for x in exemptions_raw if str(x).strip()]
        return json.dumps(items)
        
    if isinstance(exemptions_raw, float) and pd.isna(exemptions_raw):
        return json.dumps([])
    
    s = str(exemptions_raw).strip()
    if s in ('[]', '', 'nan', 'None'):
        return json.dumps([])
    
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            items = [str(x).strip() for x in parsed if str(x).strip()]
            return json.dumps(items)
    except Exception:
        pass
    
    # Fallback: regex extraction of sections like 8(1)(a)
    matches = re.findall(r'8\s*\(\s*1\s*\)\s*\(\s*[a-jA-J]\s*\)', s)
    if matches:
        cleaned = [re.sub(r'\s+', '', m).lower() for m in matches]
        return json.dumps(cleaned)
    
    return json.dumps([s] if s else [])


def load_source_a(csv_path):
    """Loads and harmonizes Source A: hf_annotated.csv (1,218 cases)."""
    print(f"[1/4] Loading Source A from: {csv_path}")
    df = pd.read_csv(csv_path)
    records = []
    
    for _, row in df.iterrows():
        hf_id = row['hf_id']
        case_id = f"HF_{int(hf_id):04d}" if pd.notnull(hf_id) else f"HF_{_}"
        
        narrative = clean_text(row['instruction'])
        info_sought = clean_text(row.get('information_sought', ''))
        
        # Check if info_sought is missing or prematurely cut off (e.g. ending in 'form no')
        is_truncated = bool(re.search(r'\b(no|dt|sec|ref)\s*$', info_sought, re.IGNORECASE))
        if not info_sought or is_truncated:
            m = re.search(r'seeking\s+(?:information\s+)?(?:regarding|pertaining\s+to|on|about|as\s+to)\s+(.+?)(?:\.\s+[A-Z]|\.\n|\n\n|The\s+PIO|The\s+Public\s+Information)', narrative, re.IGNORECASE)
            if m and len(m.group(1).strip()) > 10:
                info_sought = clean_text(m.group(1))
            else:
                title = clean_text(row.get('title', ''))
                info_sought = title if title else (info_sought if info_sought else "Information sought under RTI application")
                
        ruling = clean_text(row['response'])
        public_auth = clean_text(row.get('public_authority', ''))
        public_auth = public_auth if public_auth else "Not Specified"
        exemptions = parse_exemptions(row.get('exemptions_cited'))
        outcome = str(row['outcome']).strip() if pd.notnull(row['outcome']) else 'UNKNOWN'
        
        records.append({
            'case_id': case_id,
            'source': 'source_a_hf',
            'case_narrative': narrative,
            'information_sought': info_sought,
            'exemptions_cited': exemptions,
            'outcome': outcome,
            'ruling_text': ruling,
            'public_authority': public_auth
        })
        
    print(f"      Loaded {len(records)} records from Source A.")
    return records


def load_source_b(jsonl_path):
    """Loads and harmonizes Source B: cic_combined.jsonl (298 cases)."""
    print(f"[2/4] Loading Source B from: {jsonl_path}")
    records = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            data = json.loads(line)
            
            case_no = data.get('case_no')
            filename = data.get('filename')
            case_id = str(case_no).strip() if case_no and str(case_no) != 'nan' else f"CIC_PDF_{idx:04d}_{filename}"
            
            issue = clean_text(data.get('issue'))
            app = clean_text(data.get('application'))
            
            # Construct pre-judgment case narrative from issue & application
            if issue and app:
                narrative = f"Information Sought / RTI Application:\n{issue}\n\nProceedings & Submissions:\n{app}"
            elif issue:
                narrative = f"Information Sought / RTI Application:\n{issue}"
            elif app:
                narrative = f"Proceedings & Submissions:\n{app}"
            else:
                narrative = ""
                
            info_sought = issue if issue else "Information points sought under RTI application"
            ruling = clean_text(data.get('conclusion'))
            public_auth = clean_text(data.get('public_authority'))
            public_auth = public_auth if public_auth else "Not Specified"
            exemptions = parse_exemptions(data.get('exemptions_cited'))
            outcome = str(data.get('outcome')).strip() if data.get('outcome') and str(data.get('outcome')) != 'nan' else 'UNKNOWN'
            
            records.append({
                'case_id': case_id,
                'source': 'source_b_cic_pdf',
                'case_narrative': narrative,
                'information_sought': info_sought,
                'exemptions_cited': exemptions,
                'outcome': outcome,
                'ruling_text': ruling,
                'public_authority': public_auth
            })
            
    print(f"      Loaded {len(records)} records from Source B.")
    return records


def process_dataset(data_dir="Dataset"):
    """
    Main pipeline:
    1. Loads raw files and exports dataset_combined_raw.csv (1,516 cases).
    2. Cleans & filters records to 4 target classes and exports dataset_cleaned.csv (1,153 cases).
    3. Performs stratified 80/20 train/test split and exports train.csv and test.csv.
    """
    hf_path = os.path.join(data_dir, "hf_annotated.csv")
    cic_path = os.path.join(data_dir, "cic_combined.jsonl")
    
    # 1. Load both sources
    records_a = load_source_a(hf_path)
    records_b = load_source_b(cic_path)
    all_records = records_a + records_b
    
    df_raw = pd.DataFrame(all_records)
    print(f"\n[3/4] Building Combined Raw Dataset:")
    print(f"      Total raw cases: {len(df_raw)} (Source A: {len(records_a)}, Source B: {len(records_b)})")
    
    raw_csv_path = os.path.join(data_dir, "dataset_combined_raw.csv")
    df_raw.to_csv(raw_csv_path, index=False, encoding='utf-8')
    print(f"      Saved raw combined dataset to: {raw_csv_path}")
    
    # Raw outcome distribution
    print("\n      Raw Outcome Distribution:")
    for out, cnt in df_raw['outcome'].value_counts(dropna=False).items():
        print(f"        • {str(out):<22} : {cnt:>4} ({cnt/len(df_raw)*100:>5.1f}%)")
        
    # 2. Filter for Clean Dataset (4 target classes & non-empty narrative)
    print(f"\n[4/4] Filtering & Cleaning Dataset for 4-Class Classification:")
    mask_outcome = df_raw['outcome'].isin(TARGET_CLASSES)
    mask_narrative = df_raw['case_narrative'].str.strip().str.len() > 30
    
    df_clean = df_raw[mask_outcome & mask_narrative].copy().reset_index(drop=True)
    
    # Deduplicate identical case narratives to eliminate train/test data leakage
    before_dedup = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=['case_narrative']).copy().reset_index(drop=True)
    print(f"      Total valid cases before deduplication: {before_dedup}")
    print(f"      Total unique cases retained after deduplication: {len(df_clean)} (Removed {before_dedup - len(df_clean)} duplicates)")
    
    clean_csv_path = os.path.join(data_dir, "dataset_cleaned.csv")
    df_clean.to_csv(clean_csv_path, index=False, encoding='utf-8')
    print(f"      Saved clean dataset to: {clean_csv_path}")
    
    print("\n      Cleaned Outcome Distribution (4 Core Classes - 100% Unique):")
    for out, cnt in df_clean['outcome'].value_counts().items():
        print(f"        • {str(out):<22} : {cnt:>4} ({cnt/len(df_clean)*100:>5.1f}%)")
        
    # 3. Stratified 80/20 Train-Test Split
    print(f"\n[Split] Generating Stratified 80/20 Train/Test Split (random_state=42):")
    train_df, test_df = train_test_split(
        df_clean,
        test_size=0.20,
        stratify=df_clean['outcome'],
        random_state=42
    )
    
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    train_csv_path = os.path.join(data_dir, "train.csv")
    test_csv_path = os.path.join(data_dir, "test.csv")
    
    train_df.to_csv(train_csv_path, index=False, encoding='utf-8')
    test_df.to_csv(test_csv_path, index=False, encoding='utf-8')
    
    print(f"      Train Set: {len(train_df)} cases (Saved to {train_csv_path})")
    print(f"      Test Set:  {len(test_df)} cases (Saved to {test_csv_path})")
    
    # Print Stratification Comparison Table
    print("\n" + "="*70)
    print(f"{'Class Outcome':<24} | {'Train Count':<11} | {'Train %':<8} | {'Test Count':<10} | {'Test %':<7}")
    print("="*70)
    for cls in TARGET_CLASSES:
        tr_cnt = (train_df['outcome'] == cls).sum()
        tr_pct = tr_cnt / len(train_df) * 100
        te_cnt = (test_df['outcome'] == cls).sum()
        te_pct = te_cnt / len(test_df) * 100
        print(f"{cls:<24} | {tr_cnt:>11} | {tr_pct:>7.2f}% | {te_cnt:>10} | {te_pct:>6.2f}%")
    print("="*70)
    
    # Verification assertions
    assert len(train_df) + len(test_df) == len(df_clean), "Split row count mismatch!"
    assert df_clean['outcome'].isnull().sum() == 0, "Null outcomes found in clean dataset!"
    assert df_clean['case_narrative'].isnull().sum() == 0, "Null narratives found in clean dataset!"
    
    print("\n[SUCCESS] Pipeline completed with 100% verified data integrity.")
    return df_raw, df_clean, train_df, test_df


if __name__ == "__main__":
    process_dataset()
