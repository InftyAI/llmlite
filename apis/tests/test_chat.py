import unittest
from llms.chatglm import ChatGLMChat

from llms.llama import LlamaChat
from apis.chat import (
    fetch_llm,
    UnavailableModelException,
)


class TestChat(unittest.TestCase):
    def test_fetch_llm(self):
        test_cases = [
            {
                "name": "local llama-2 model",
                "model": "/models/Llama-2-7b-chat-hf",
                "chat": LlamaChat,
            },
            {
                "name": "huggingface llama-2 model",
                "model": "meta-llama/Llama-2-7b-chat-hf",
                "chat": LlamaChat,
            },
            {
                "name": "local chatglm2 model",
                "model": "THUDM/chatglm2-6b",
                "chat": ChatGLMChat,
            },
            {
                "name": "model with non-exist model",
                "model": "non-exists-model",
                "wantException": UnavailableModelException,
            },
        ]

        for tc in test_cases:
            model = tc["model"]

            if "wantException" not in tc:
                llm = fetch_llm(model)
                self.assertEqual(
                    llm,
                    tc["chat"],
                    "test case '{case}' not passed".format(case=tc["name"]),
                )
            else:
                with self.assertRaises(UnavailableModelException):
                    fetch_llm(model)
