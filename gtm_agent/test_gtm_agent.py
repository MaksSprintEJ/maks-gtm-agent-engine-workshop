import importlib
import json
import os
import unittest
from unittest.mock import patch

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langsmith import tracing_context

with patch("dotenv.load_dotenv"), patch.dict(os.environ, {
    "OPENAI_API_KEY": "test-key",
    "LANGSMITH_TRACING": "false",
}):
    agent = importlib.import_module("gtm_agent.gtm_agent")


class ScriptedModel(BaseChatModel):
    prospect_id: str
    intent: str

    @property
    def _llm_type(self):
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        results = {
            message.name: json.loads(message.content)
            for message in messages if isinstance(message, ToolMessage)
        }
        if "get_current_rep" not in results:
            name, args = "get_current_rep", {}
        elif self.intent == "email" and "get_prospect" not in results:
            name, args = "get_prospect", {"prospect_id": self.prospect_id}
        elif self.intent == "email" and "send_prospect_email" not in results:
            name, args = "send_prospect_email", {
                "prospect": results["get_prospect"]["prospect"],
                "subject": "Follow up", "body": "Let's connect.",
            }
        elif self.intent == "score" and "build_prospect_profile" not in results:
            name, args = "build_prospect_profile", {"prospect_id": self.prospect_id}
        elif self.intent == "score" and "lookup_offering" not in results:
            name, args = "lookup_offering", {"offering_id": "OFFER-10001"}
        elif self.intent == "score" and "score_prospect" not in results:
            name, args = "score_prospect", {
                "prospect_profile": results["build_prospect_profile"]["prospect_profile"],
                "offering": results["lookup_offering"]["offering"],
            }
        else:
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Done"))])
        message = AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": name}])
        return ChatResult(generations=[ChatGeneration(message=message)])


class ProspectPrivacyTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(tracing_context(enabled=False))
        self.enterContext(patch.dict(agent.data_service._PROFILES, {}, clear=True))
        self.pid = "LEAD-50001"
        self.record = agent.data_service.get_prospect_record(self.pid)
        self.profile = {
            "prospect_id": self.pid,
            **{k: v for k, v in self.record.items() if k != "billing_qualification"},
        }
        self.score = agent.ProspectScore(
            score=75, justification="Revenue and technology fit",
            rubric_breakdown=agent.RubricBreakdown(
                revenue_fit=90, tech_stack_match=60, segment_fit=75,
            ),
        )
        self.llm = self.enterContext(patch.object(agent, "_scoring_llm"))
        self.llm.invoke.return_value = self.score

    def assert_private(self, value):
        serialized = json.dumps(value)
        self.assertNotIn("billing_qualification", serialized)
        for sensitive in self.record["billing_qualification"].values():
            self.assertNotIn(sensitive, serialized)

    def test_contact_fields(self):
        self.assertEqual(agent.get_prospect.invoke({"prospect_id": self.pid}), {
            "found": True, "prospect": {
                "prospect_id": self.pid,
                **{k: self.record[k] for k in ("name", "email", "disqualified")},
            },
        })

    def test_new_profile(self):
        result = agent.build_prospect_profile.invoke({"prospect_id": self.pid})
        self.assertEqual(result, {"found": True, "prospect_profile": self.profile})
        self.assertEqual(agent.data_service._PROFILES[self.pid], self.profile)
        self.assertIn("billing_qualification", self.record)

    def test_cached_profile(self):
        agent.data_service._PROFILES[self.pid] = {"prospect_id": self.pid, **self.record}
        self.assertEqual(agent.build_prospect_profile.invoke({"prospect_id": self.pid}), {
            "found": True, "prospect_profile": self.profile,
        })

    def test_missing_prospect(self):
        self.assertEqual(agent.get_prospect.invoke({"prospect_id": "missing"}), {
            "found": False, "prospect": None,
        })
        self.assertEqual(agent.build_prospect_profile.invoke({"prospect_id": "missing"}), {
            "found": False, "prospect_profile": None,
        })

    def test_scoring_prompt(self):
        profile = {"prospect_id": self.pid, **self.record, "tech_stack": ["invented"]}
        result = agent.score_prospect.invoke({
            "prospect_profile": profile,
            "offering": agent.data_service.get_offering("OFFER-10001"),
        })
        self.assertEqual(result, self.score.model_dump())
        messages = self.llm.invoke.call_args.args[0]
        self.assert_private(messages)
        sent_profile = json.loads(messages[1]["content"].split("Prospect profile:\n")[1])
        self.assertEqual(sent_profile, self.profile)
        self.assertIn("billing_qualification", profile)

    def test_agent_email_and_scoring_flows(self):
        for intent in ("email", "score"):
            for pid in agent.data_service.PROSPECTS:
                with self.subTest(intent=intent, prospect_id=pid):
                    graph = agent.create_deep_agent(
                        model=ScriptedModel(prospect_id=pid, intent=intent),
                        tools=[agent.get_current_rep, agent.get_prospect,
                               agent.send_prospect_email, agent.build_prospect_profile,
                               agent.lookup_offering, agent.score_prospect],
                        system_prompt=agent.SYSTEM_PROMPT,
                    )
                    result = graph.invoke(
                        {"messages": [{"role": "user", "content": f"{intent} {pid}"}]},
                        config={"metadata": {"user_id": "rep_amills"}},
                    )
                    for message in result["messages"]:
                        self.assert_private(message.model_dump(mode="json"))
                    if intent == "email":
                        sent = next(m for m in result["messages"]
                                    if isinstance(m, ToolMessage) and m.name == "send_prospect_email")
                        self.assertEqual(json.loads(sent.content)["status"], "sent")
                    else:
                        self.assert_private(self.llm.invoke.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
