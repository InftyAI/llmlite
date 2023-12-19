from llmlite.apis import ChatLLM, ChatMessage
from tests.integration_tests.util import build_model


class TestChat:
    def test_llama(self):
        test_cases = [
            {
                "name": "local model with llama 2",
                "model": build_model("meta-llama/Llama-2-7b-chat-hf"),
                "task": "text-generation",
                "messages": [
                    ChatMessage(
                        role="system", content="You're an agent based on fact."
                    ),
                    ChatMessage(
                        role="user", content="How many people are there in China?"
                    ),
                ],
            },
        ]

        for tc in test_cases:
            chat = ChatLLM(
                model_name_or_path=tc["model"],
                task=tc["task"],
                temperature=0.2,
                max_length=2048,
                do_sample=True,
                top_p=0.7,
                top_k=3,
            )
            result = chat.completion(messages=tc["messages"])
            assert len(result) > 0, "got result: " + result

    def test_chatglm(self):
        test_cases = [
            {
                "name": "local model with chatglm2",
                "model": build_model("THUDM/chatglm2-6b"),
                "messages": [
                    ChatMessage(role="user", content="中国共有多少人口？"),
                ],
            },
        ]

        for tc in test_cases:
            chat = ChatLLM(
                model_name_or_path=tc["model"],
                temperature=0.2,
                max_length=2048,
                do_sample=True,
                top_p=0.7,
                top_k=3,
            )
            result = chat.completion(messages=tc["messages"])
            assert len(result) > 0, "got result: " + result

    def test_chatgpt(self):
        test_cases = [
            {
                "name": "test with gpt-3.5-turbo",
                "model": "gpt-3.5-turbo",
                "messages": [
                    ChatMessage(
                        role="system", content="You're an agent based on fact."
                    ),
                    ChatMessage(
                        role="user", content="How many people are there in China?"
                    ),
                ],
            },
            {
                "name": "test with gpt-4",
                "model": "gpt-4",
                "messages": [
                    ChatMessage(
                        role="system", content="You're an agent based on fact."
                    ),
                    ChatMessage(
                        role="user", content="How many people are there in China?"
                    ),
                ],
            },
        ]

        for tc in test_cases:
            chat = ChatLLM(
                model_name_or_path=tc["model"],
            )
            result = chat.completion(messages=tc["messages"])
            assert len(result) > 0, "got result: " + result

    def test_codellama(self):
        test_cases = [
            {
                "name": "test with only user prompt",
                "messages": [
                    ChatMessage(role="user", content="def fibonacci("),
                ],
            },
            {
                "name": "test with system prompt exists",
                "messages": [
                    ChatMessage(role="system", content="Provide answers in golang"),
                    ChatMessage(
                        role="user",
                        content="Write a function that computes the sum of a given list.",
                    ),
                ],
            },
        ]

        chat = ChatLLM(
            model_name_or_path=build_model("codellama/CodeLlama-13b-instruct-hf"),
            task="text-generation",
        )
        for tc in test_cases:
            result = chat.completion(
                messages=tc["messages"],
                max_length=2048,
            )
            assert len(result) > 0, "got result: " + result

    def test_baichuan(self):
        test_cases = [
            {
                "name": "enable system prompt",
                "messages": [
                    ChatMessage(role="system", content="请用诗词形容"),
                    ChatMessage(role="user", content="中国共有多少人口？"),
                ],
            },
        ]

        chat = ChatLLM(model_name_or_path=build_model("baichuan-inc/Baichuan2-7B-Chat"))
        for tc in test_cases:
            result = chat.completion(
                messages=tc["messages"],
                temperature=0.9,
                max_length=2048,
                do_sample=True,
                top_p=0.7,
                top_k=3,
            )
            assert len(result) > 0, "got result: " + result
