from apis.api import ChatLLM
import unittest

class TestOutput(unittest.TestCase):
    def test_ChatLLM(self):
        test_cases = [
            {
                "name": "local model with llama 2",
                "model": "/models/llama-2-7b-chat-hf",
                "task": "text-generation",
                "host": "local",
                "prompt": "How many people are there in China",
                "system_prompt": None,
                "chat": ChatLLM,
            },
            {
                "name": "local model with chatglm2",
                "model": "/data/models/chatglm2-6b",
                "task": "text-generation",
                "host": "local",
                "prompt": "How many people are there in China",
                "system_prompt": None,
                "chat": ChatLLM,
            },

        ]
        for test in test_cases:
            chat = ChatLLM(
                model_name_or_path=test["model"], task=test["task"], host=test["host"],
            )
            result = chat.completion(prompt=test["prompt"], system_prompt=test["system_prompt"])
            self.assertNotEqual(len(result),0)


