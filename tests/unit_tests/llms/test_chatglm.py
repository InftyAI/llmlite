from llmlite.llms.chatglm2 import ChatGLMChat, build_history
from llmlite.llms.chat import ASSISTANT_PROMPT, USER_PROMPT
from llmlite.llms.messages import ChatMessage


class TestChatGLM:
    def test_build_history(self):
        test_cases = [
            {
                "messages": [
                    ChatMessage(role=USER_PROMPT, content="你是谁"),
                ],
                "want": ("你是谁", []),
            },
            {
                "messages": [
                    ChatMessage(role=USER_PROMPT, content="你是谁"),
                    ChatMessage(role=ASSISTANT_PROMPT, content="我是一个聪明的机器人"),
                    ChatMessage(role=USER_PROMPT, content="你可以帮我做作业吗"),
                ],
                "want": ("你可以帮我做作业吗", [("你是谁", "我是一个聪明的机器人")]),
            },
        ]

        for tc in test_cases:
            got = build_history(tc["messages"])
            assert got == tc["want"]

    def test_prompt(self):
        test_cases = [
            {
                "name": "prompt with user_prompt",
                "messages": [
                    ChatMessage(role=USER_PROMPT, content="你是谁"),
                ],
                "expected": """[Round 1]

问：你是谁

答：""",
            },
            {
                "name": "prompt with another round conservation",
                "messages": [
                    ChatMessage(role=USER_PROMPT, content="你是谁"),
                    ChatMessage(role=ASSISTANT_PROMPT, content="我是一个智能机器人"),
                    ChatMessage(role=USER_PROMPT, content="你能给我写作业吗"),
                ],
                "expected": """[Round 1]

问：你是谁

答：我是一个智能机器人

[Round 2]

问：你能给我写作业吗

答：""",
            },
        ]

        for tc in test_cases:
            got = ChatGLMChat.prompt(tc["messages"])
            assert got == tc["expected"]
