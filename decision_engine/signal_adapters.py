"""
Signal Output Adapters for Decision Making Engine
Project: Explainable Neuro-Symbolic AI Advisory System for RTI Decision Outcome Prediction
Batch ID: 30
"""

from typing import Dict, List
import numpy as np


class SignalAdapters:
    """
    Standardizes heterogeneous outputs from Signal 1 (Neural Probabilities),
    Signal 2 (Coverage Checklist), and Signal 3 (Statutory Citations) into
    calibrated 4-class probability distributions.
    """
    CLASSES = [
        "INFORMATION_DIRECTED",
        "APPEAL_DISMISSED",
        "PENALTY_IMPOSED",
        "PARTIAL_RELIEF"
    ]

    @classmethod
    def adapt_signal1(cls, s1_output: Dict) -> Dict[str, float]:
        """
        Extracts and normalizes probabilities from Signal 1 (InLegalBERT).
        """
        raw_probs = s1_output.get("probabilities", {})
        # Ensure all 4 classes are present
        probs = {c: float(raw_probs.get(c, 0.0)) for c in cls.CLASSES}
        total = sum(probs.values())
        if total > 0:
            return {c: p / total for c, p in probs.items()}
        return {c: 0.25 for c in cls.CLASSES}

    @classmethod
    def adapt_signal2(cls, s2_output: Dict) -> Dict[str, float]:
        """
        Converts factual coverage checklist metrics (Signal 2) into a 4-class distribution:
        - Mixed addressed & missing -> boosts PARTIAL_RELIEF
        - High missing / evasion -> boosts INFORMATION_DIRECTED
        - Full addressed -> boosts APPEAL_DISMISSED (already supplied)
        """
        metrics = s2_output.get("coverage_metrics", {})
        total_items = max(metrics.get("total_items", 1), 1)
        addressed = metrics.get("addressed_count", 0)
        missing = metrics.get("missing_count", 0)
        partial = metrics.get("partial_count", 0)
        ratio = metrics.get("coverage_ratio", 0.0)

        scores = {c: 0.05 for c in cls.CLASSES}  # laplace baseline smoothing

        # 1. Mixed Coverage -> Strong Partial Relief
        if (addressed > 0 and (missing > 0 or partial > 0)) or (total_items > 1 and 0.20 <= ratio <= 0.80):
            scores["PARTIAL_RELIEF"] += 0.70
            scores["INFORMATION_DIRECTED"] += 0.15
            scores["APPEAL_DISMISSED"] += 0.10

        # 2. Total Evasion / Substantial Omission -> Strong Information Directed
        elif missing > 0 or ratio < 0.35:
            scores["INFORMATION_DIRECTED"] += 0.75
            scores["APPEAL_DISMISSED"] += 0.10
            scores["PENALTY_IMPOSED"] += 0.10

        # 3. Full Compliance (All items provided) -> Strong Appeal Dismissed (records already furnished)
        elif addressed == total_items or ratio >= 0.85:
            scores["APPEAL_DISMISSED"] += 0.80
            scores["INFORMATION_DIRECTED"] += 0.10

        # Normalize to probability distribution
        total = sum(scores.values())
        return {c: round(scores[c] / total, 4) for c in cls.CLASSES}

    @classmethod
    def adapt_signal3(cls, s3_facts) -> Dict[str, float]:
        """
        Converts statutory citations, Section 24 jurisdiction, and Section 20 penalty triggers
        into a legal probability distribution.
        """
        scores = {c: 0.05 for c in cls.CLASSES}

        # 1. Section 24 Security Agency -> Overwhelming Appeal Dismissed
        if getattr(s3_facts, "is_sec24_organization", False):
            scores["APPEAL_DISMISSED"] += 0.90

        # 2. Section 20 Penalty Triggered -> Overwhelming Penalty Imposed
        elif getattr(s3_facts, "sec20_penalty_triggered", False):
            scores["PENALTY_IMPOSED"] += 0.85
            scores["INFORMATION_DIRECTED"] += 0.10

        # 3. Section 10 Severability Indicated -> Strong Partial Relief
        elif getattr(s3_facts, "sec10_severability_indicated", False):
            scores["PARTIAL_RELIEF"] += 0.75
            scores["INFORMATION_DIRECTED"] += 0.15

        # 4. Valid Section 8(1) Exemption Cited -> Strong Appeal Dismissed
        elif getattr(s3_facts, "has_valid_sec8_claim", False):
            scores["APPEAL_DISMISSED"] += 0.70
            scores["PARTIAL_RELIEF"] += 0.15

        # 5. Zero Section 8 Exemptions Cited -> Strong Information Directed (Duty to disclose under Sec 7)
        else:
            scores["INFORMATION_DIRECTED"] += 0.65
            scores["APPEAL_DISMISSED"] += 0.20

        total = sum(scores.values())
        return {c: round(scores[c] / total, 4) for c in cls.CLASSES}
