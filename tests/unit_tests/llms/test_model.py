import pytest

from llmlite.llms.messages import ChatMessage
from llmlite.llms.model import Model


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
                "expected": True,
            },
            {
                "name": "messages should not be empty",
                "messages": [],
                "support_system_prompt": True,
                "expected": False,
            },
            {
                "name": "last message should be user prompt",
                "messages": [
                    ChatMessage(role="assistant", content="Who you are"),
                ],
                "support_system_prompt": False,
                "expected": False,
            },
            {
                "name": "system prompt should be in the first role",
                "messages": [
                    ChatMessage(role="assistant", content="Who you are"),
                    ChatMessage(role="system", content="You are an intelligent agent"),
                    ChatMessage(role="user", content="Who you are"),
                ],
                "support_system_prompt": True,
                "expected": False,
            },
            {
                "name": "system prompt not supported",
                "messages": [
                    ChatMessage(role="system", content="You are a intelligent agent"),
                    ChatMessage(role="user", content="Who you are"),
                ],
                "support_system_prompt": False,
                "expected": False,
            },
            {
                "name": "only user prompt support system_prompt",
                "messages": [
                    ChatMessage(role="user", content="Who you are"),
                ],
                "support_system_prompt": True,
                "expected": True,
            },
        ]

        for tc in test_cases:
            model = Model("llama-2")
            model.__config__.update(
                {"support_system_prompt": tc["support_system_prompt"]},
            )
            print(model.__config__)
            if tc["expected"]:
                assert model.validation(tc["messages"])
            else:
                with pytest.raises(AssertionError, match=tc["name"]):
                    model.validation(tc["messages"])
