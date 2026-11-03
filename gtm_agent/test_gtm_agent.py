import importlib
import json
import os
import unittest
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage

with patch.dict(os.environ, OPENAI_API_KEY="test-key", LANGSMITH_TRACING="false"):
    agent = importlib.import_module("gtm_agent.gtm_agent")


class ScriptedToolModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class ScoreProspectTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, LANGSMITH_TRACING="false")
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.cache = patch.dict(agent.data_service._PROFILES, {}, clear=True)
        self.cache.start()
        self.addCleanup(self.cache.stop)
        self.scorer_patch = patch.object(agent, "_scoring_llm")
        self.scorer = self.scorer_patch.start()
        self.addCleanup(self.scorer_patch.stop)
        self.result = {"score": 75, "justification": "Good fit"}
        self.scorer.invoke.return_value.model_dump.return_value = self.result
        self.prospect_id = "LEAD-71001"
        self.offering_id = "OFFER-10005"

    def score(self):
        return agent.score_prospect.invoke({
            "prospect_id": self.prospect_id,
            "offering_id": self.offering_id,
        })

    def scoring_records(self):
        messages = self.scorer.invoke.call_args.args[0]
        self.assertEqual(messages[0], {"role": "system", "content": agent.SCORING_PROMPT})
        offering, profile = messages[1]["content"].split("\n\nProspect profile:\n")
        return json.loads(offering.removeprefix("Offering:\n")), json.loads(profile)

    def test_score_fetches_records_from_ids(self):
        self.assertEqual(self.score(), self.result)
        offering, profile = self.scoring_records()
        self.assertEqual(offering, agent.data_service.get_offering(self.offering_id))
        self.assertEqual(profile, {
            "prospect_id": self.prospect_id,
            **agent.data_service.get_prospect_record(self.prospect_id),
        })
        self.scorer.invoke.assert_called_once()

    def test_missing_offering_does_not_score(self):
        self.offering_id = "missing"
        self.assertEqual(self.score(), {
            "score": None, "error": "Cannot score without a valid offering.",
        })
        self.scorer.invoke.assert_not_called()

    def test_incomplete_offering_does_not_score(self):
        offering = agent.data_service.get_offering(self.offering_id)
        for field in ("required_tech_stack", "min_annual_revenue", "description"):
            with self.subTest(field=field):
                incomplete = {key: value for key, value in offering.items() if key != field}
                with patch.dict(agent.data_service.OFFERINGS, {self.offering_id: incomplete}):
                    self.assertEqual(self.score(), {
                        "score": None, "error": "Cannot score without a valid offering.",
                    })
        self.scorer.invoke.assert_not_called()

    def test_missing_prospect_does_not_score(self):
        self.prospect_id = "missing"
        self.assertEqual(self.score(), {
            "score": None, "error": "Cannot score without a valid prospect.",
        })
        self.scorer.invoke.assert_not_called()

    def test_record_tools_remain_available(self):
        profile = agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        self.assertTrue(profile["found"])
        self.assertEqual(profile, agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id}))
        self.assertEqual(agent.build_prospect_profile.invoke({"prospect_id": "missing"}), {
            "prospect_profile": None, "found": False,
        })
        self.assertEqual(agent.lookup_offering.invoke({"offering_id": self.offering_id}), {
            "offering": agent.data_service.get_offering(self.offering_id), "found": True,
        })
        self.assertEqual(agent.lookup_offering.invoke({"offering_id": "missing"}), {
            "offering": None, "found": False,
        })
        self.scorer.invoke.assert_not_called()

    def test_agent_can_update_then_score_with_only_ids(self):
        model = ScriptedToolModel(responses=[
            AIMessage(content="", tool_calls=[{
                "name": "update_prospect_info", "id": "update", "args": {
                    "prospect_id": self.prospect_id, "technology": "Kafka",
                },
            }]),
            AIMessage(content="", tool_calls=[{
                "name": "score_prospect", "id": "score", "args": {
                    "prospect_id": self.prospect_id, "offering_id": self.offering_id,
                },
            }]),
            AIMessage(content="Scored the prospect."),
        ])
        graph = agent.create_deep_agent(
            model=model,
            tools=[agent.update_prospect_info, agent.score_prospect],
            system_prompt=agent.SYSTEM_PROMPT,
        )
        result = graph.invoke({"messages": [{
            "role": "user", "content": "Add Kafka to LEAD-71001, then score against OFFER-10005.",
        }]})
        results = [message for message in result["messages"] if isinstance(message, ToolMessage)]
        self.assertEqual([message.name for message in results], ["update_prospect_info", "score_prospect"])
        self.assertTrue(json.loads(results[0].content)["updated"])
        self.assertEqual(json.loads(results[1].content), self.result)
        self.scorer.invoke.assert_called_once()

    def test_scoring_refreshes_cached_tech_stack_after_update(self):
        profile = agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        original_stack = list(profile["prospect_profile"]["tech_stack"])
        agent.update_prospect_info.invoke({"prospect_id": self.prospect_id, "technology": "Kafka"})
        with patch.object(agent.data_service, "fetch_tech_stack", return_value=["Fresh technology"]):
            self.assertEqual(self.score(), self.result)
        _, scored_profile = self.scoring_records()
        self.assertEqual(scored_profile["tech_stack"], ["Fresh technology"])
        self.assertEqual(profile["prospect_profile"]["tech_stack"], original_stack)


if __name__ == "__main__":
    unittest.main()
