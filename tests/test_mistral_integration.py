import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ENGINE_DIR = Path(__file__).resolve().parents[1] / "src" / "education_engine"
sys.path.insert(0, str(ENGINE_DIR))

from ai_providers import choose_provider, provider_status  # noqa: E402
from generate_similar_subject import generate_with_mistral  # noqa: E402


class FakeMistralClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(complete=self.complete)
        self.request = None

    def complete(self, **kwargs):
        self.request = kwargs
        message = SimpleNamespace(content="Sujet ancre dans le contexte fourni.")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class MistralIntegrationTests(unittest.TestCase):
    def test_mistral_is_selected_for_generation(self):
        env = {
            "MISTRAL_API_KEY": "test-key",
            "EDUCATION_AI_MISTRAL_ENABLED": "true",
            "EDUCATION_AI_MISTRAL_GENERATION_ENABLED": "true",
            "EDUCATION_AI_DEFAULT_PROVIDER": "mistral",
            "EDUCATION_AI_FALLBACK_PROVIDER": "openai",
        }

        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(choose_provider("generation"), "mistral")
            self.assertEqual(choose_provider("quality"), "none")
            self.assertTrue(provider_status()["mistral_generation_available"])

    def test_mistral_generation_uses_configured_model_and_prompt(self):
        client = FakeMistralClient()

        result = generate_with_mistral(
            client=client,
            model="mistral-small-latest",
            prompt="Contexte pedagogique controle",
        )

        self.assertEqual(result, "Sujet ancre dans le contexte fourni.")
        self.assertEqual(client.request["model"], "mistral-small-latest")
        self.assertEqual(
            client.request["messages"][1]["content"],
            "Contexte pedagogique controle",
        )

    def test_openai_remains_the_configured_fallback(self):
        env = {
            "OPENAI_API_KEY": "test-key",
            "ASKCI_OPENAI_ENABLED": "true",
            "ASKCI_OPENAI_GENERATION_ENABLED": "true",
            "EDUCATION_AI_DEFAULT_PROVIDER": "mistral",
            "EDUCATION_AI_FALLBACK_PROVIDER": "openai",
        }

        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(choose_provider("generation"), "openai")


if __name__ == "__main__":
    unittest.main()
