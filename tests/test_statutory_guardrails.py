"""
Unit tests for Statutory Fact Scanner and 5-Tier Legal Guardrails Hierarchy.
"""

import unittest
from signal_3.scanner import StatutoryScanner, StatutoryFacts
from decision_engine.symbolic_guardrails import StatutoryGuardrails


class TestStatutoryGuardrails(unittest.TestCase):
    def setUp(self):
        self.scanner = StatutoryScanner()
        self.guardrails = StatutoryGuardrails()

    def test_guardrail1_sec24_jurisdictional_veto(self):
        # Case involving exempt security organization (e.g. CBI)
        facts = self.scanner.scan_case(
            case_narrative="Appellant sought investigation records from CBI under RTI.",
            public_authority="Central Bureau of Investigation",
            information_sought="Preliminary enquiry report"
        )
        self.assertTrue(facts.is_sec24_organization)

        # Even if neural consensus predicts INFORMATION_DIRECTED with 85%
        res = self.guardrails.apply_guardrails(
            fused_prediction="INFORMATION_DIRECTED",
            fused_confidence=0.85,
            signal2_coverage={"coverage_metrics": {"total_items": 1, "addressed_count": 0, "missing_count": 1, "coverage_ratio": 0.0}},
            signal3_facts=facts
        )
        self.assertEqual(res["final_outcome"], "APPEAL_DISMISSED")
        self.assertTrue(res["is_override"])
        self.assertEqual(res["rule_triggered"], "SEC24_JURISDICTIONAL_IMMUNITY_VETO")

    def test_guardrail2_sec20_penalty_sanction(self):
        facts = self.scanner.scan_case(
            case_narrative="The Commission issued a show cause notice to the CPIO for malafide denial and deliberate delay.",
            information_sought="File notings"
        )
        self.assertTrue(facts.sec20_penalty_triggered)

        res = self.guardrails.apply_guardrails(
            fused_prediction="INFORMATION_DIRECTED",
            fused_confidence=0.70,
            signal2_coverage={"coverage_metrics": {"total_items": 1, "addressed_count": 0, "missing_count": 1, "coverage_ratio": 0.0}},
            signal3_facts=facts
        )
        self.assertEqual(res["final_outcome"], "PENALTY_IMPOSED")
        self.assertTrue(res["is_override"])
        self.assertEqual(res["rule_triggered"], "SEC20_STATUTORY_PENAL_SANCTION")

    def test_guardrail3_sec10_severability_partial_relief(self):
        facts = StatutoryFacts(
            case_id="TEST_CASE_3",
            exemptions_detected=["8(1)(j)"],
            has_valid_sec8_claim=True,
            sec10_severability_indicated=True
        )
        res = self.guardrails.apply_guardrails(
            fused_prediction="INFORMATION_DIRECTED",
            fused_confidence=0.65,
            signal2_coverage={"coverage_metrics": {"total_items": 2, "addressed_count": 1, "missing_count": 1, "partial_count": 0, "coverage_ratio": 0.50}},
            signal3_facts=facts
        )
        self.assertEqual(res["final_outcome"], "PARTIAL_RELIEF")
        self.assertEqual(res["rule_triggered"], "SEC10_SEVERABILITY_MANDATE")

    def test_guardrail4_unjustified_evasion_overrule(self):
        # Neural model predicted APPEAL_DISMISSED, but records missing and NO Section 8 claimed
        facts = StatutoryFacts(
            case_id="TEST_CASE_4",
            exemptions_detected=[],
            has_valid_sec8_claim=False
        )
        res = self.guardrails.apply_guardrails(
            fused_prediction="APPEAL_DISMISSED",
            fused_confidence=0.68,
            signal2_coverage={"coverage_metrics": {"total_items": 2, "addressed_count": 0, "missing_count": 2, "partial_count": 0, "coverage_ratio": 0.0}},
            signal3_facts=facts
        )
        self.assertEqual(res["final_outcome"], "INFORMATION_DIRECTED")
        self.assertTrue(res["is_override"])
        self.assertEqual(res["rule_triggered"], "UNJUSTIFIED_EVASION_OVERRULE")

    def test_guardrail5_legitimate_sec8_sustained(self):
        facts = StatutoryFacts(
            case_id="TEST_CASE_5",
            exemptions_detected=["8(1)(j)"],
            has_valid_sec8_claim=True
        )
        res = self.guardrails.apply_guardrails(
            fused_prediction="APPEAL_DISMISSED",
            fused_confidence=0.82,
            signal2_coverage={"coverage_metrics": {"total_items": 1, "addressed_count": 0, "missing_count": 1, "partial_count": 0, "coverage_ratio": 0.0}},
            signal3_facts=facts
        )
        self.assertEqual(res["final_outcome"], "APPEAL_DISMISSED")
        self.assertEqual(res["rule_triggered"], "SEC8_EXEMPTION_SUSTAINED")


if __name__ == "__main__":
    unittest.main()
