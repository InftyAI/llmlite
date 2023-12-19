import pytest

from llmlite.llms.messages import ChatMessage
from llmlite.llms.model import Model
from llmlite import consts


class TestModel:
    def test_validation(self):
        test_cases = [
            {
                "name": "validation passed",
                "messages": [
                    ChatMessage(role="system", content="You are a intelligent agent"),
                    ChatMessage(role="user", content="Who you are"),
                ],
                "support_system_prompt": True,
                "succeed": True,
            },
            {
                "name": "messages should not be empty",
                "messages": [],
                "support_system_prompt": True,
            },
            {
                "name": "last message should be user prompt",
                "messages": [
                    ChatMessage(role="assistant", content="Who you are"),
                ],
                "support_system_prompt": False,
            },
            {
                "name": "system prompt should be in the first role",
                "messages": [
                    ChatMessage(role="assistant", content="Who you are"),
                    ChatMessage(role="system", content="You are an intelligent agent"),
                    ChatMessage(role="user", content="Who you are"),
                ],
                "support_system_prompt": True,
            },
            {
                "name": "system prompt not supported",
                "messages": [
                    ChatMessage(role="system", content="You are a intelligent agent"),
                    ChatMessage(role="user", content="Who you are"),
                ],
                "support_system_prompt": False,
            },
            {
                "name": "only user prompt support system_prompt",
                "messages": [
                    ChatMessage(role="user", content="Who you are"),
                ],
                "support_system_prompt": True,
                "succeed": True,
            },
            {
                "name": "batch messages all passed",
                "messages": [
                    [
                        ChatMessage(role="user", content="Who you are"),
                    ],
                    [
                        ChatMessage(
                            role="system", content="You are a intelligent agent"
                        ),
                        ChatMessage(role="user", content="Who you are"),
                    ],
                ],
                "backend": "vllm",
                "support_system_prompt": True,
                "succeed": True,
            },
            {
                "name": "messages should not be empty",
                "messages": [
                    [
                        ChatMessage(role="user", content="Who you are"),
                    ],
                    [],
                ],
                "backend": "vllm",
                "support_system_prompt": True,
            },
            {
                "name": "system prompt not supported",
                "messages": [
                    [
                        ChatMessage(role="user", content="Who you are"),
                    ],
                    [
                        ChatMessage(
                            role="system", content="You are a intelligent agent"
                        ),
                        ChatMessage(role="user", content="Who you are"),
                    ],
                ],
                "backend": "vllm",
                "support_system_prompt": False,
            },
            {
                "name": "vLLM only supports batch inference",
                "backend": "vllm",
                "messages": [
                    ChatMessage(role="user", content="Who you are"),
                ],
                "support_system_prompt": False,
            },
            {
                "name": "batch inference only supports with vLLM backend",
                "messages": [
                    [
                        ChatMessage(role="user", content="Who you are"),
                    ],
                    [
                        ChatMessage(
                            role="system", content="You are a intelligent agent"
                        ),
                        ChatMessage(role="user", content="Who you are"),
                    ],
                ],
                "support_system_prompt": True,
            },
        ]

        for tc in test_cases:
            backend = tc.get("backend", consts.BACKEND_HF)
            model = Model("llama-2", backend=backend)
            model.__config__.update(
                {"support_system_prompt": tc["support_system_prompt"]},
            )
            if tc.get("succeed", False):
                model.validation(tc["messages"])
            else:
                with pytest.raises(AssertionError, match=tc["name"]):
                    model.validation(tc["messages"])
