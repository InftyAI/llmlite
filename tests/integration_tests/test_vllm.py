from llmlite.apis import ChatLLM, ChatMessage
from tests.integration_tests.util import build_model


class TestVLLM:
    def test_llama_with_vllm(self):
        test_cases = [
            {
                "name": "local model with llama 2",
                "model": build_model("meta-llama/Llama-2-7b-chat-hf"),
                "messages": [
                    [
                        ChatMessage(
                            role="system", content="You're an agent based on fact."
                        ),
                        ChatMessage(
                            role="user", content="How many people are there in China?"
                        ),
                    ],
                    [
                        ChatMessage(
                            role="user",
                            content="How many foreigners are there in China?",
                        ),
                    ],
                ],
            },
        ]

        for tc in test_cases:
            chat = ChatLLM(
                backend="vllm",
                model_name_or_path=tc["model"],
            )
            result = chat.completion(
                messages=tc["messages"],
                max_tokens=2048,
            )
            assert len(result) == len(tc["messages"])

    def test_chatglm_with_vllm(self):
        test_cases = [
            {
                "name": "local model with chatglm2",
                "model": build_model("THUDM/chatglm2-6b"),
                "messages": [
                    [
                        ChatMessage(role="user", content="中国共有多少人口？"),
                    ],
                    [
                        ChatMessage(role="user", content="中国有多少外国人？"),
                    ],
                ],
            },
        ]

        for tc in test_cases:
            chat = ChatLLM(
                backend="vllm",
                model_name_or_path=tc["model"],
            )
            result = chat.completion(
                messages=tc["messages"],
                max_tokens=2048,
            )
            assert len(result) == len(tc["messages"])
