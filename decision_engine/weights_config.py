"""
Weights & Hyperparameters Configuration for Decision Making Engine
Project: Explainable Neuro-Symbolic AI Advisory System for RTI Decision Outcome Prediction
Batch ID: 30
"""

from dataclasses import dataclass
from typing import List


@dataclass
class FusionWeightsConfig:
    # 1. Signal Probabilistic Weights (Sum = 1.0)
    # w1: Signal 1 (InLegalBERT Neural Prior) - captures semantic & dialectic narrative patterns
    # w2: Signal 2 (Factual Information Coverage) - captures whether records were actually provided or omitted
    # w3: Signal 3 (Statutory Rules & Citations) - captures explicit sections cited under the RTI Act, 2005
    w1_signal1_neural: float = 0.50
    w2_signal2_coverage: float = 0.30
    w3_signal3_statutory: float = 0.20

    # 2. Target Classes
    classes: List[str] = None

    # 3. Decision Sensitivity Thresholds
    min_confidence_for_neural_pass: float = 0.70
    severability_partial_min_ratio: float = 0.20
    severability_partial_max_ratio: float = 0.80

    def __post_init__(self):
        if self.classes is None:
            self.classes = [
                "INFORMATION_DIRECTED",
                "APPEAL_DISMISSED",
                "PENALTY_IMPOSED",
                "PARTIAL_RELIEF"
            ]
        # Normalize weights to sum strictly to 1.0
        total = self.w1_signal1_neural + self.w2_signal2_coverage + self.w3_signal3_statutory
        self.w1_signal1_neural /= total
        self.w2_signal2_coverage /= total
        self.w3_signal3_statutory /= total
