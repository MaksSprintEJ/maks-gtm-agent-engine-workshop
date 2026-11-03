import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

with patch.dict(os.environ, {"OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY") or "test-key"}):
    from gtm_agent.gtm_agent import run_agent, send_prospect_email


class SendProspectEmailTests(unittest.TestCase):
    def setUp(self):
        self.prospect = {"name": "Prospect", "email": "prospect@example.com"}
        self.runtime = SimpleNamespace(config={"metadata": {"user_id": "rep_sbrown"}})

    def send(self, body):
        return send_prospect_email.func(
            prospect=self.prospect,
            subject="Following up",
            body=body,
            runtime=self.runtime,
        )

    def test_attachment_claim_is_rejected(self):
        for body in ("The pricing deck is attached.", "Please see the ATTACHED deck."):
            with self.subTest(body=body):
                self.assertEqual(self.send(body), {
                    "status": "failed",
                    "error": "Attachments are not supported; remove the attachment claim.",
                })

    def test_plain_text_email_is_sent(self):
        body = "Would you like to discuss pricing next week?"
        result = self.send(body)
        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["body"], body)
        self.assertEqual(result["to"], self.prospect["email"])
        self.assertEqual(result["from"], "sara.brown@northpoint.com")

    def test_collateral_link_is_sent(self):
        body = "View the pricing deck: https://example.com/pricing-deck.pdf"
        result = self.send(body)
        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["body"], body)


@unittest.skipUnless(os.environ.get("OPENAI_API_KEY"), "Requires model credentials")
class PricingDeckRequestTests(unittest.TestCase):
    def run_request(self, request):
        with patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}):
            return run_agent(request, user_id="rep_sbrown")

    def sent_emails(self, result):
        emails = []
        for message in result["messages"]:
            if message.type == "tool" and message.name == "send_prospect_email":
                payload = json.loads(message.content)
                if payload["status"] == "sent":
                    emails.append(payload)
        return emails

    def test_pricing_deck_without_link_is_reported_as_missing(self):
        result = self.run_request("Email LEAD-50003 to send over our pricing deck.")
        for email in self.sent_emails(result):
            self.assertNotIn("attached", email["body"].lower())
        reply = result["reply"].lower()
        self.assertIn("deck", reply)
        self.assertRegex(reply, r"cannot|can't|could not|couldn't|not included|not sent|unable|yourself|manually")

    def test_pricing_deck_link_is_included(self):
        link = "https://example.com/pricing-deck.pdf"
        result = self.run_request(
            "Email LEAD-50003 our pricing deck using this link: " + link
        )
        emails = self.sent_emails(result)
        self.assertTrue(emails)
        self.assertTrue(any(link in email["body"] for email in emails))
        for email in emails:
            self.assertNotIn("attached", email["body"].lower())


if __name__ == "__main__":
    unittest.main()
