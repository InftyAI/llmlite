import unittest

from llms.llama import LlamaChat
from llms.llama_hf import LlamaHFChat
from apis.api import (
    fetch_llm,
    UnavailableModelException,
)


class TestAPI(unittest.TestCase):
    def test_fetch_llm(self):
        test_cases = [
            {
                "name": "local model with llama 2",
                "model": "/models/Llama-2-7b-chat-hf",
                "host": "local",
                "chat": LlamaChat,
                "wantException": None,
            },
            {
                "name": "hosting model with llama 2",
                "model": "meta-llama/Llama-2-7b-chat-hf",
                "host": "api",
                "chat": LlamaHFChat,
                "wantException": None,
            },
            {
                "name": "model with non-exist model",
                "model": "non-exists-model",
                "host": "api",
                "chat": None,
                "wantException": UnavailableModelException,
            },
        ]

        for test in test_cases:
            model = test["model"]
            host = test["host"]

            if test["wantException"] is None:
                llm = fetch_llm(model, host)
                self.assertEqual(llm, test["chat"])
            else:
                with self.assertRaises(UnavailableModelException):
                    fetch_llm(model, host)
