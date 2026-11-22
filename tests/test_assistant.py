"""Verify the Gemini API contract locally without credentials or billed requests."""
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

from reddit_insights.assistant import Claim, Insight, explain


class AssistantTests(unittest.TestCase):
    def test_missing_key_fails_before_request(self):
        with self.assertRaisesRegex(ValueError, "GEMINI_API_KEY"):
            explain("Summarize", {"evidence": []}, "", "model")

    def fake_gemini(self, answer):
        google = ModuleType("google")
        google.__path__ = []
        genai = ModuleType("google.genai")
        genai.__path__ = []
        types = ModuleType("google.genai.types")
        types.GenerateContentConfig = MagicMock(side_effect=lambda **kwargs: kwargs)
        response = SimpleNamespace(parsed=answer, text=answer.model_dump_json() if answer else None)
        client = MagicMock()
        client.__enter__.return_value = client
        client.models.generate_content.return_value = response
        genai.Client = MagicMock(return_value=client)
        google.genai = genai
        modules = {"google": google, "google.genai": genai, "google.genai.types": types}
        return modules, client, genai, types

    def test_grounded_structured_output_and_no_tools(self):
        answer = Insight(answer="Observed sentiment changed.",
                         findings=[Claim(text="The selected archive shifted.", references=["trends"])],
                         limitations=["No causal evidence."], follow_up_questions=[])
        modules, client, genai, types = self.fake_gemini(answer)
        with patch.dict(sys.modules, modules):
            result = explain("Summarize", {"evidence": []}, "test-only", "model")
        self.assertEqual(result["findings"][0]["references"], ["trends"])
        genai.Client.assert_called_once_with(api_key="test-only")
        config = types.GenerateContentConfig.call_args.kwargs
        self.assertEqual(config["response_mime_type"], "application/json")
        self.assertIs(config["response_schema"], Insight)
        self.assertNotIn("tools", config)
        client.models.generate_content.assert_called_once()

    def test_invalid_references_and_incomplete_responses_are_rejected(self):
        invalid = Insight(answer="Unsupported", findings=[Claim(text="Claim", references=["invented:123"])],
                          limitations=[], follow_up_questions=[])
        for response in [None, invalid]:
            modules, _, _, _ = self.fake_gemini(response)
            with patch.dict(sys.modules, modules), self.assertRaises(ValueError):
                explain("Summarize", {"evidence": []}, "test-only", "model")

    def test_json_text_is_validated_when_sdk_has_no_parsed_value(self):
        answer = Insight(answer="Observed.", findings=[Claim(text="Archive changed.", references=["overview"])],
                         limitations=[], follow_up_questions=[])
        modules, client, _, _ = self.fake_gemini(None)
        client.models.generate_content.return_value = SimpleNamespace(parsed=None, text=answer.model_dump_json())
        with patch.dict(sys.modules, modules):
            result = explain("Summarize", {"evidence": []}, "test-only", "model")
        self.assertEqual(result["answer"], "Observed.")


if __name__ == "__main__":
    unittest.main()
