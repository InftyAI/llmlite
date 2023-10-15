from llmlite.utils.validation import general_validations
from llmlite.llms.messages import ChatMessage


class TestUtil:
    def test_general_validations(self):
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
                "name": "don't support system_prompt",
                "messages": [
                    ChatMessage(role="system", content="You are a intelligent agent"),
                    ChatMessage(role="user", content="Who you are"),
                ],
                "support_system_prompt": False,
                "expected": False,
            },
            {
                "name": "no messages",
                "messages": [],
                "support_system_prompt": True,
                "expected": False,
            },
            {
                "name": "user prompt support system_prompt",
                "messages": [
                    ChatMessage(role="user", content="Who you are"),
                ],
                "support_system_prompt": True,
                "expected": True,
            },
            {
                "name": "user prompt don't support system_prompt",
                "messages": [
                    ChatMessage(role="user", content="Who you are"),
                ],
                "support_system_prompt": False,
                "expected": True,
            },
        ]

        for tc in test_cases:
            got = general_validations(tc["messages"], tc["support_system_prompt"])
            assert got == tc["expected"]
