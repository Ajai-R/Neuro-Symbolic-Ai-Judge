"""
Unit tests for Continuous Multi-Signal Weighted Probabilistic Fusion.
"""

import unittest
from decision_engine.weights_config import FusionWeightsConfig
from decision_engine.weighted_fusion import WeightedSignalFuser


class TestWeightedFusion(unittest.TestCase):
    def setUp(self):
        self.config = FusionWeightsConfig(
            w1_signal1_neural=0.50,
            w2_signal2_coverage=0.30,
            w3_signal3_statutory=0.20
        )
        self.fuser = WeightedSignalFuser(self.config)

    def test_weight_normalization(self):
        total = (
            self.config.w1_signal1_neural +
            self.config.w2_signal2_coverage +
            self.config.w3_signal3_statutory
        )
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_weighted_consensus_information_directed(self):
        s1 = {"INFORMATION_DIRECTED": 0.60, "APPEAL_DISMISSED": 0.30, "PENALTY_IMPOSED": 0.05, "PARTIAL_RELIEF": 0.05}
        s2 = {"INFORMATION_DIRECTED": 0.70, "APPEAL_DISMISSED": 0.15, "PENALTY_IMPOSED": 0.05, "PARTIAL_RELIEF": 0.10}
        s3 = {"INFORMATION_DIRECTED": 0.65, "APPEAL_DISMISSED": 0.20, "PENALTY_IMPOSED": 0.05, "PARTIAL_RELIEF": 0.10}

        result = self.fuser.fuse(s1, s2, s3)
        self.assertEqual(result["fused_predicted_class"], "INFORMATION_DIRECTED")
        self.assertGreater(result["fused_confidence"], 0.60)
        self.assertAlmostEqual(sum(result["fused_probabilities"].values()), 1.0, places=4)

    def test_weighted_consensus_penalty_imposed(self):
        s1 = {"INFORMATION_DIRECTED": 0.10, "APPEAL_DISMISSED": 0.10, "PENALTY_IMPOSED": 0.75, "PARTIAL_RELIEF": 0.05}
        s2 = {"INFORMATION_DIRECTED": 0.05, "APPEAL_DISMISSED": 0.05, "PENALTY_IMPOSED": 0.85, "PARTIAL_RELIEF": 0.05}
        s3 = {"INFORMATION_DIRECTED": 0.05, "APPEAL_DISMISSED": 0.05, "PENALTY_IMPOSED": 0.80, "PARTIAL_RELIEF": 0.10}

        result = self.fuser.fuse(s1, s2, s3)
        self.assertEqual(result["fused_predicted_class"], "PENALTY_IMPOSED")
        self.assertGreater(result["fused_confidence"], 0.75)


if __name__ == "__main__":
    unittest.main()
