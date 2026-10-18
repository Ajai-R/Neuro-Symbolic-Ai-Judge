"""
Weighted Probabilistic Signal Fusion Module
Project: Explainable Neuro-Symbolic AI Advisory System for RTI Decision Outcome Prediction
Batch ID: 30
"""

from typing import Dict, List, Tuple
import numpy as np

from .weights_config import FusionWeightsConfig
from .signal_adapters import SignalAdapters


class WeightedSignalFuser:
    """
    Computes a continuous mathematical consensus across Signal 1, Signal 2, and Signal 3:
    P_fused(c) = w1 * P_S1(c) + w2 * P_S2(c) + w3 * P_S3(c)
    """
    def __init__(self, config: FusionWeightsConfig = None):
        self.config = config or FusionWeightsConfig()

    def fuse(
        self,
        p_signal1: Dict[str, float],
        p_signal2: Dict[str, float],
        p_signal3: Dict[str, float]
    ) -> Dict:
        """
        Combines the 3 probability vectors using calibrated weights.
        Returns the fused probability distribution and full contribution breakdown.
        """
        w1 = self.config.w1_signal1_neural
        w2 = self.config.w2_signal2_coverage
        w3 = self.config.w3_signal3_statutory

        fused_probs = {}
        contributions = {}

        for c in self.config.classes:
            s1_c = p_signal1.get(c, 0.0)
            s2_c = p_signal2.get(c, 0.0)
            s3_c = p_signal3.get(c, 0.0)

            weighted_score = (w1 * s1_c) + (w2 * s2_c) + (w3 * s3_c)
            fused_probs[c] = round(weighted_score, 4)

            contributions[c] = {
                "signal_1_neural": round(w1 * s1_c, 4),
                "signal_2_coverage": round(w2 * s2_c, 4),
                "signal_3_statutory": round(w3 * s3_c, 4),
                "total": round(weighted_score, 4)
            }

        # Normalize to guarantee exact sum = 1.0
        total_sum = sum(fused_probs.values())
        if total_sum > 0:
            fused_probs = {c: round(v / total_sum, 4) for c, v in fused_probs.items()}

        top_class = max(fused_probs, key=fused_probs.get)
        confidence = fused_probs[top_class]

        return {
            "fused_probabilities": fused_probs,
            "fused_predicted_class": top_class,
            "fused_confidence": confidence,
            "weights_used": {
                "w1_signal1": round(w1, 3),
                "w2_signal2": round(w2, 3),
                "w3_signal3": round(w3, 3)
            },
            "contributions_by_class": contributions
        }
