"""
Signal 1: Dataset & Tokenization Module for InLegalBERT
Project: Explainable Neuro-Symbolic AI Advisory System for RTI Decision Outcome Prediction
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from sklearn.utils.class_weight import compute_class_weight

LABEL2ID = {
    'INFORMATION_DIRECTED': 0,
    'APPEAL_DISMISSED': 1,
    'PENALTY_IMPOSED': 2,
    'PARTIAL_RELIEF': 3
}

ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_CLASSES = len(LABEL2ID)


def format_rti_input(row) -> str:
    """
    Constructs a rich, multi-field legal prompt combining:
    - Citizen's Query (information_sought)
    - Statutory Exemptions Claimed (exemptions_cited)
    - Public Authority Department (public_authority)
    - Case Narrative (Background + Hearing Proceedings)
    """
    query = str(row.get('information_sought', '')).strip()
    exemptions = str(row.get('exemptions_cited', '')).strip()
    authority = str(row.get('public_authority', '')).strip()
    narrative = str(row.get('case_narrative', '')).strip()

    try:
        import json
        ex_list = json.loads(exemptions)
        if isinstance(ex_list, list) and ex_list:
            exemptions_str = ", ".join(ex_list)
        else:
            exemptions_str = "None cited"
    except Exception:
        exemptions_str = exemptions if (exemptions and exemptions != '[]') else "None cited"

    if not query or query == 'nan':
        query = "Not explicitly stated"
    if not authority or authority == 'nan' or authority == 'Not Specified':
        authority = "Public Authority"

    prompt = (
        f"[CITIZEN QUERY]: {query}\n"
        f"[EXEMPTIONS CITED]: {exemptions_str}\n"
        f"[PUBLIC AUTHORITY]: {authority}\n"
        f"[CASE NARRATIVE & PROCEEDINGS]:\n{narrative}"
    )
    return prompt


class RTIDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: AutoTokenizer, max_length: int = 512, head_len: int = 160):
        self.texts = [format_rti_input(row) for _, row in df.iterrows()]
        self.labels = [LABEL2ID[outcome] for outcome in df['outcome']]
        self.case_ids = df['case_id'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.head_len = head_len
        self.tail_len = max_length - 2 - head_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        case_id = self.case_ids[idx]

        tokens = self.tokenizer.encode(text, add_special_tokens=False, verbose=False)
        # Head + Tail truncation: Preserves Query & Exemptions (Head: 160 tokens) + Hearing Arguments (Tail: 350 tokens)
        if len(tokens) > (self.max_length - 2):
            tokens = tokens[:self.head_len] + tokens[-self.tail_len:]

        input_ids = [self.tokenizer.cls_token_id] + tokens + [self.tokenizer.sep_token_id]
        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            attention_mask = [1] * len(input_ids) + [0] * pad_len
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
        else:
            input_ids = input_ids[:self.max_length]
            attention_mask = [1] * self.max_length

        return {
            'case_id': case_id,
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }


def get_class_weights(df: pd.DataFrame, device: torch.device, smooth_power: float = 0.5) -> torch.Tensor:
    """
    Computes smoothed class weights to counteract class imbalance without destabilizing training.
    raw_weights ** 0.5 prevents extreme scaling while boosting minority class gradients.
    """
    labels = [LABEL2ID[outcome] for outcome in df['outcome']]
    classes = np.array(list(LABEL2ID.values()))
    raw_weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=labels
    )
    smoothed = raw_weights ** smooth_power
    smoothed = smoothed / smoothed.mean()
    return torch.tensor(smoothed, dtype=torch.float, device=device)
