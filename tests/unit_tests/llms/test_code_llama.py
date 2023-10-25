from llmlite.llms.codellama import CodeLlamaChat
from llmlite.llms.chat import ASSISTANT_PROMPT, SYSTEM_PROMPT, USER_PROMPT
from llmlite.llms.messages import ChatMessage


class TestCodeLlama:
    def test_instruct_prompt(self):
        test_cases = [
            {
                "name": "prompt with both system_prompt and user_prompt",
                "messages": [
                    ChatMessage(
                        role=SYSTEM_PROMPT, content="You are an intelligent agent"
                    ),
                    ChatMessage(role=USER_PROMPT, content="Who you are"),
                ],
                "expected": """<s>[INST] <<SYS>>
You are an intelligent agent
<</SYS>>

Who you are [/INST] """,
            },
            {
                "name": "prompt with both system_prompt user_prompt and assistant_prompt",
                "messages": [
                    ChatMessage(
                        role=SYSTEM_PROMPT, content="You are an intelligent agent"
                    ),
                    ChatMessage(role=USER_PROMPT, content="Who you are"),
                    ChatMessage(role=ASSISTANT_PROMPT, content="I'm an agent"),
                ],
                "expected": """<s>[INST] <<SYS>>
You are an intelligent agent
<</SYS>>

Who you are [/INST] I'm an agent </s>""",
            },
            {
                "name": "continuous conversation with system_prompt exists",
                "messages": [
                    ChatMessage(
                        role=SYSTEM_PROMPT, content="You are an intelligent agent"
                    ),
                    ChatMessage(role=USER_PROMPT, content="Who you are"),
                    ChatMessage(role=ASSISTANT_PROMPT, content="I'm an agent"),
                    ChatMessage(role=USER_PROMPT, content="You're so clever"),
                    ChatMessage(role=ASSISTANT_PROMPT, content="Thanks"),
                ],
                "expected": """<s>[INST] <<SYS>>
You are an intelligent agent
<</SYS>>

Who you are [/INST] I'm an agent </s><s>[INST] You're so clever [/INST] Thanks </s>""",
            },
            {
                "name": "prompt with only user_prompt",
                "messages": [
                    ChatMessage(role=USER_PROMPT, content="Who you are"),
                ],
                "expected": """<s>[INST] Who you are [/INST] """,
            },
            {
                "name": "prompt with user_prompt and assistant_prompt",
                "messages": [
                    ChatMessage(role=USER_PROMPT, content="Who you are"),
                    ChatMessage(role=ASSISTANT_PROMPT, content="I'm an agent"),
                ],
                "expected": """<s>[INST] Who you are [/INST] I'm an agent </s>""",
            },
            {
                "name": "continuous conversation with no system_prompt",
                "messages": [
                    ChatMessage(role=USER_PROMPT, content="Who you are"),
                    ChatMessage(role=ASSISTANT_PROMPT, content="I'm an agent"),
                    ChatMessage(role=USER_PROMPT, content="You're so clever"),
                    ChatMessage(role=ASSISTANT_PROMPT, content="Thanks"),
                ],
                "expected": """<s>[INST] Who you are [/INST] I'm an agent </s><s>[INST] You're so clever [/INST] Thanks </s>""",
            },
        ]

        for tc in test_cases:
            got = CodeLlamaChat.prompt(tc["messages"], mode="instruct")
            assert got == tc["expected"]
