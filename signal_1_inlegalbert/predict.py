"""
Signal 1: Inference & Probability Prediction Module
Project: Explainable Neuro-Symbolic AI Advisory System for RTI Decision Outcome Prediction
"""

import os
import sys
import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Ensure UTF-8 output encoding on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from signal_1_inlegalbert.dataset import ID2LABEL, LABEL2ID


class InLegalBERTOutcomePredictor:
    def __init__(self, model_dir: str = None, device: str = None):
        if model_dir is None:
            model_dir = os.path.join(current_dir, "checkpoints", "best_model")

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model checkpoint directory not found at: {model_dir}. Please run train.py first.")

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

    def predict(
        self,
        case_narrative: str,
        information_sought: str = None,
        exemptions_cited: str = None,
        public_authority: str = None,
        max_length: int = 512,
        head_len: int = 160
    ) -> dict:
        """
        Takes case narrative and optional structured metadata, returns class probabilities and prediction.
        """
        if information_sought is not None or exemptions_cited is not None:
            from signal_1_inlegalbert.dataset import format_rti_input
            prompt = format_rti_input({
                'information_sought': information_sought,
                'exemptions_cited': exemptions_cited,
                'public_authority': public_authority,
                'case_narrative': case_narrative
            })
        else:
            prompt = str(case_narrative)

        tokens = self.tokenizer.encode(prompt, add_special_tokens=False, verbose=False)
        tail_len = max_length - 2 - head_len
        if len(tokens) > (max_length - 2):
            tokens = tokens[:head_len] + tokens[-tail_len:]

        input_ids = [self.tokenizer.cls_token_id] + tokens + [self.tokenizer.sep_token_id]
        pad_len = max_length - len(input_ids)
        if pad_len > 0:
            attention_mask = [1] * len(input_ids) + [0] * pad_len
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
        else:
            input_ids = input_ids[:max_length]
            attention_mask = [1] * max_length

        input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        attention_mask = torch.tensor([attention_mask], dtype=torch.long, device=self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        pred_idx = int(torch.argmax(logits, dim=-1).item())
        predicted_class = ID2LABEL[pred_idx]
        confidence = float(probs[pred_idx])

        probabilities = {
            ID2LABEL[i]: float(probs[i]) for i in range(len(ID2LABEL))
        }

        return {
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in probabilities.items()}
        }


# Alias for backward compatibility
RobertaOutcomePredictor = InLegalBERTOutcomePredictor


if __name__ == "__main__":
    predictor = InLegalBERTOutcomePredictor()
    sample_text = (
        "The appellant filed an RTI application seeking copies of tender documents and bills. "
        "The PIO refused to provide the information claiming it was exempt. "
        "During the hearing, the PIO failed to show any valid exemption under Section 8."
    )
    result = predictor.predict(sample_text)
    print("\nSample Prediction Output:")
    print(json.dumps(result, indent=2))
