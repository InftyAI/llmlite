import unittest

from apis.api import ChatLLM


class TestChat(unittest.TestCase):
    def test_chat_with_llms(self):
        test_cases = [
            {
                "name": "local model with llama 2",
                "model": "meta-llama/Llama-2-7b-chat-hf",
                "task": "text-generation",
                "host": "local",
                "prompt": "How many people are there in China",
                "system_prompt": "Please be more detailed",
            },
            {
                "name": "local model with chatglm2",
                "model": "THUDM/chatglm2-6b",
                "task": "text-generation",
                "host": "local",
                "prompt": "中国共有多少人口",
                "system_prompt": None,
            },
        ]

        for test in test_cases:
            chat = ChatLLM(
                model_name_or_path=test["model"],
                task=test["task"],
                host=test["host"],
            )
            result = chat.completion(
                prompt=test["prompt"], system_prompt=test["system_prompt"]
            )
            self.assertNotEqual(len(result), 0)
