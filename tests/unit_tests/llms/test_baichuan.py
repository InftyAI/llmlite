from llmlite.llms.baichuan import Baichuan
from llmlite import consts
from llmlite.llms.messages import ChatMessage


class TestBaichuan:
    def test_baichuan_prompt(self):
        test_cases = [
            {
                "name": "prompt with both system_prompt and user_prompt",
                "messages": [
                    ChatMessage(
                        role=consts.SYSTEM_PROMPT,
                        content="You are an intelligent agent",
                    ),
                    ChatMessage(role=consts.USER_PROMPT, content="Who you are"),
                ],
                "expected": """You are an intelligent agent
<reserved_106>Who you are<reserved_107>""",
            },
            {
                "name": "continuous conversation with system_prompt exists",
                "messages": [
                    ChatMessage(
                        role=consts.SYSTEM_PROMPT,
                        content="You are an intelligent agent",
                    ),
                    ChatMessage(role=consts.USER_PROMPT, content="Who you are"),
                    ChatMessage(role=consts.ASSISTANT_PROMPT, content="I'm an agent"),
                    ChatMessage(role=consts.USER_PROMPT, content="You're so clever"),
                ],
                "expected": """You are an intelligent agent
<reserved_106>Who you are<reserved_107>I'm an agent<reserved_106>You're so clever<reserved_107>""",
            },
            {
                "name": "prompt with only user_prompt",
                "messages": [
                    ChatMessage(role=consts.USER_PROMPT, content="Who you are"),
                ],
                "expected": """<reserved_106>Who you are<reserved_107>""",
            },
            {
                "name": "continuous conversation with no system_prompt",
                "messages": [
                    ChatMessage(role=consts.USER_PROMPT, content="Who you are"),
                    ChatMessage(role=consts.ASSISTANT_PROMPT, content="I'm an agent"),
                    ChatMessage(role=consts.USER_PROMPT, content="You're so clever"),
                ],
                "expected": """<reserved_106>Who you are<reserved_107>I'm an agent<reserved_106>You're so clever<reserved_107>""",
            },
        ]

        for tc in test_cases:
            got = Baichuan.prompt("fake-model-path", tc["messages"])
            assert got == tc["expected"], "failed in case: " + tc["name"]
