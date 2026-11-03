import os
import unittest
from unittest.mock import patch

with patch.dict(os.environ, {"OPENAI_API_KEY": "test", "LANGSMITH_TRACING": "false"}):
    from gtm_agent.gtm_agent import score_prospect


class ScoreProspectTests(unittest.TestCase):
    def setUp(self):
        self.profile = {"annual_revenue": 100, "tech_stack": ["AWS", "Okta", "Kafka"]}
        self.offering = {
            "required_tech_stack": ["Snowflake", "AWS", "Okta", "Kafka"],
            "min_annual_revenue": 100,
            "target_segment": "Enterprise",
        }

    def score(self):
        return score_prospect.invoke({"prospect_profile": self.profile, "offering": self.offering})

    def test_three_of_four_and_identical_inputs(self):
        result = self.score()
        self.assertEqual(result["rubric_breakdown"]["tech_stack_match"], 75)
        self.assertEqual(result["rubric_breakdown"]["revenue_fit"], 100)
        self.assertEqual(result["score"], 87.5)
        self.assertIn("Required technologies present: AWS, Okta, Kafka.", result["justification"])
        self.assertIn("Missing: Snowflake.", result["justification"])
        self.assertNotIn("segment_fit", result["rubric_breakdown"])
        self.assertNotIn("Enterprise", result["justification"])
        for _ in range(3):
            self.assertEqual(self.score(), result)

    def test_revenue_ratio_and_cap(self):
        for revenue, fit, overall in [(200, 100, 87.5), (33.333, 33.33, 54.16), (0, 0, 37.5)]:
            with self.subTest(revenue=revenue):
                self.profile["annual_revenue"] = revenue
                result = self.score()
                self.assertEqual(result["rubric_breakdown"]["revenue_fit"], fit)
                self.assertEqual(result["score"], overall)
                self.assertIn(f"Revenue fit: {float(fit)}/100.", result["justification"])

    def test_invalid_revenue_leaves_component_and_overall_unavailable(self):
        for field, record, values in [
            ("annual_revenue", self.profile, [None, "unknown", -1, True, float("nan"), float("inf")]),
            ("min_annual_revenue", self.offering, [None, "unknown", 0, -1, True, float("inf")]),
        ]:
            for value in values:
                with self.subTest(field=field, value=value):
                    original = record[field]
                    record[field] = value
                    result = self.score()
                    record[field] = original
                    self.assertIsNone(result["rubric_breakdown"]["revenue_fit"])
                    self.assertEqual(result["rubric_breakdown"]["tech_stack_match"], 75)
                    self.assertIsNone(result["score"])
                    self.assertIn("Overall score unavailable", result["justification"])
        del self.profile["annual_revenue"]
        self.assertIsNone(self.score()["score"])

    def test_empty_requirements_and_unavailable_technology_data(self):
        self.offering["required_tech_stack"] = []
        self.assertEqual(self.score()["score"], 100)
        self.offering["required_tech_stack"] = ["AWS"]
        self.profile["tech_stack"] = []
        self.assertEqual(self.score()["rubric_breakdown"]["tech_stack_match"], 0)
        for value in [None, "AWS", [None]]:
            self.profile["tech_stack"] = value
            result = self.score()
            self.assertIsNone(result["rubric_breakdown"]["tech_stack_match"])
            self.assertIsNone(result["score"])
        del self.offering["required_tech_stack"]
        self.assertIsNone(self.score()["score"])

    def test_saved_technology_stack_overrides_supplied_stack(self):
        self.profile["prospect_id"] = "LEAD-90001"
        self.profile["tech_stack"] = self.offering["required_tech_stack"]
        with patch("gtm_agent.data_service.fetch_tech_stack", return_value=["AWS", "Okta", "Kafka"]):
            result = self.score()
        self.assertEqual(result["score"], 87.5)
        self.assertIn("Missing: Snowflake.", result["justification"])

    def test_no_offering_returns_error(self):
        self.offering = None
        self.assertEqual(self.score(), {"score": None, "error": "Cannot score without a valid offering."})


if __name__ == "__main__":
    unittest.main()
