"""Email tool tests and optional live agent checks (run from the repo root)."""

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from dotenv import load_dotenv

load_dotenv()
HAS_MODEL_KEY = bool(os.environ.get("OPENAI_API_KEY"))
with patch.dict(os.environ, {
    "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY") or "test-key",
    "LANGSMITH_TRACING": "false",
}):
    from gtm_agent.gtm_agent import data_service, run_agent, send_prospect_email


class SendProspectEmailTests(unittest.TestCase):
    def setUp(self):
        self.runtime = SimpleNamespace(config={"metadata": {"user_id": "rep_amills"}})

    def send(self, prospect, **kwargs):
        return send_prospect_email.func(
            prospect=prospect, subject="Demo", body="Let's book a demo.",
            runtime=self.runtime, **kwargs,
        )

    def test_stored_disqualification_blocks_forged_eligible_contact(self):
        result = self.send({
            "prospect_id": "LEAD-50001", "disqualified": False,
            "email": "other@example.com",
        })
        self.assertEqual(result, {
            "status": "blocked",
            "error": "Prospect is disqualified; email not sent.",
        })

    def test_unknown_or_missing_id_cannot_send(self):
        for prospect in ({"prospect_id": "unknown"}, {}):
            with self.subTest(prospect=prospect):
                result = self.send({**prospect, "email": "other@example.com"})
                self.assertEqual(result, {"status": "failed", "error": "Unknown prospect."})

    def test_eligible_send_uses_stored_contact(self):
        stored = data_service.get_prospect_record("LEAD-70001")
        result = self.send({
            "prospect_id": "LEAD-70001", "name": "Other person",
            "email": "other@example.com", "disqualified": True,
        })
        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["to"], stored["email"])
        self.assertEqual(result["to_name"], stored["name"])
        self.assertEqual(result["from"], data_service.get_rep("rep_amills")["email"])
        self.assertEqual(result["subject"], "Demo")
        self.assertEqual(result["body"], "Let's book a demo.")
        self.assertTrue(result["message_id"].startswith("msg-"))

    def test_existing_eligible_send_preserves_explicit_sender(self):
        prospect = {"prospect_id": "LEAD-70001", **data_service.get_prospect_record("LEAD-70001")}
        rep = data_service.get_rep("rep_jchen")
        result = self.send(prospect, from_rep=rep)
        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["to"], prospect["email"])
        self.assertEqual(result["from"], rep["email"])
        self.assertEqual(result["from_name"], rep["name"])

    def test_missing_stored_email_cannot_be_overridden(self):
        for email in (None, ""):
            with self.subTest(email=email), patch.dict(data_service.PROSPECTS, {
                "test-no-email": {"name": "No email", "email": email, "disqualified": False},
            }):
                result = self.send({"prospect_id": "test-no-email", "email": "other@example.com"})
                self.assertEqual(result, {
                    "status": "failed", "error": "Prospect record has no email address.",
                })


@unittest.skipUnless(HAS_MODEL_KEY, "Live agent checks require OPENAI_API_KEY")
class EmailAgentTests(unittest.TestCase):
    def request_email(self, prospect_id):
        with patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}):
            return run_agent(
                f"Email {prospect_id} with subject Demo and body: Let's book a demo.",
                user_id="rep_amills", environment="test",
            )

    def test_blocked_send_is_reported_without_retry(self):
        result = self.request_email("LEAD-50001")
        sends = [
            json.loads(message.content) for message in result["messages"]
            if message.type == "tool" and message.name == "send_prospect_email"
        ]
        self.assertEqual(sends, [{
            "status": "blocked", "error": "Prospect is disqualified; email not sent.",
        }])
        reply = result["reply"].lower()
        self.assertIn("disqualified", reply)
        self.assertRegex(reply, r"not sent|did not send|didn't send|could not send|couldn't send|wasn't sent")

    def test_eligible_send_still_succeeds(self):
        result = self.request_email("LEAD-70001")
        sends = [
            json.loads(message.content) for message in result["messages"]
            if message.type == "tool" and message.name == "send_prospect_email"
        ]
        self.assertEqual(len(sends), 1)
        self.assertEqual(sends[0]["status"], "sent")
        self.assertEqual(sends[0]["to"], data_service.get_prospect_record("LEAD-70001")["email"])


if __name__ == "__main__":
    unittest.main()
