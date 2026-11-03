import json
import os
import unittest
from unittest.mock import patch

from gtm_agent import data_service

with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}):
    from gtm_agent import gtm_agent


class UpdateProspectInfoTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}))
        self.enterContext(patch.dict(data_service.PROSPECTS, {
            "lead": {
                "tech_stack": ["Snowflake"],
                "engagement_history": [],
                "account_details": [],
            },
        }, clear=True))
        self.enterContext(patch.dict(data_service._PROFILES, {}, clear=True))

    def test_update_persists_and_preserves_result_shape(self):
        result = gtm_agent.update_prospect_info.invoke({
            "prospect_id": "lead", "technology": "Kafka",
        })
        self.assertEqual(result, {
            "prospect_id": "lead",
            "technology": "Kafka",
            "updated": True,
            "found": True,
            "tech_stack": ["Snowflake", "Kafka"],
        })
        self.assertEqual(data_service.fetch_tech_stack("lead"), result["tech_stack"])
        self.assertEqual(data_service.get_prospect_record("lead")["tech_stack"],
                         result["tech_stack"])

    def test_duplicate_update_does_not_duplicate_technology(self):
        result = data_service.update_prospect_info("lead", "Snowflake")
        self.assertEqual(result, {
            "updated": True, "found": True, "tech_stack": ["Snowflake"],
        })
        self.assertEqual(data_service.fetch_tech_stack("lead"), ["Snowflake"])

    def test_missing_prospect_preserves_result_shape(self):
        self.assertEqual(data_service.update_prospect_info("missing", "Kafka"), {
            "updated": False, "found": False,
        })
        self.assertIsNone(data_service.get_prospect_record("missing"))

    def test_update_rebuilds_cached_profile(self):
        original = gtm_agent.build_prospect_profile.invoke({"prospect_id": "lead"})
        other_profile = {"tech_stack": ["Redis"]}
        data_service.save_profile_to_db("other", other_profile)

        data_service.update_prospect_info("lead", "Kafka")

        rebuilt = gtm_agent.build_prospect_profile.invoke({"prospect_id": "lead"})
        self.assertEqual(original["prospect_profile"]["tech_stack"], ["Snowflake"])
        self.assertEqual(rebuilt["prospect_profile"]["tech_stack"], ["Snowflake", "Kafka"])
        self.assertEqual(data_service.get_profile_from_db("lead"), {
            "prospect_profile": rebuilt["prospect_profile"],
        })
        self.assertEqual(data_service.get_profile_from_db("other"), {
            "prospect_profile": other_profile,
        })

    def test_scoring_uses_saved_stack_even_with_old_profile(self):
        original = gtm_agent.build_prospect_profile.invoke({"prospect_id": "lead"})
        data_service.update_prospect_info("lead", "Kafka")
        offering = {
            "required_tech_stack": ["Kafka"],
            "min_annual_revenue": 100,
            "description": "Streaming platform",
        }
        with patch.object(gtm_agent, "_scoring_llm") as scoring_llm:
            gtm_agent.score_prospect.invoke({
                "prospect_profile": original["prospect_profile"],
                "offering": offering,
            })

        messages = scoring_llm.invoke.call_args.args[0]
        scored_profile = json.loads(messages[1]["content"].split("Prospect profile:\n")[1])
        self.assertEqual(scored_profile["tech_stack"], ["Snowflake", "Kafka"])


if __name__ == "__main__":
    unittest.main()
