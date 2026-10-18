"""
Signal 1: InLegalBERT Classification Model Architecture
Project: Explainable Neuro-Symbolic AI Advisory System for RTI Decision Outcome Prediction
"""

import torch
import torch.nn as nn
from signal_1_inlegalbert.dataset import LABEL2ID, ID2LABEL, NUM_CLASSES


class InLegalBERTForRTIClassification(nn.Module):
    def __init__(self, pretrained_model_name: str = "law-ai/InLegalBERT", class_weights: torch.Tensor = None):
        super().__init__()
        from transformers import AutoModelForSequenceClassification, AutoConfig
        self.config = AutoConfig.from_pretrained(
            pretrained_model_name,
            num_labels=NUM_CLASSES,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            local_files_only=True
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name,
            config=self.config,
            local_files_only=True
        )
        self.class_weights = class_weights
        if class_weights is not None:
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {
            'loss': loss,
            'logits': logits
        }


# Alias for backward compatibility
RobertaForRTIClassification = InLegalBERTForRTIClassification
