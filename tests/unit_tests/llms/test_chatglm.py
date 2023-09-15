import unittest

from llmlite.llms.chatglm2 import ChatGLMChat
from llmlite.llms.chat import ASSISTANT_PROMPT, SYSTEM_PROMPT, USER_PROMPT
from llmlite.apis.messages import ChatMessage


class TestChatGLM(unittest.TestCase):
    def test_prompt(self):
        test_cases = [
            {
                "name": "prompt with both system_prompt and user_prompt",
                "messages": [
                    ChatMessage(role=SYSTEM_PROMPT, content="你是一个聪明的机器人"),
                    ChatMessage(role=USER_PROMPT, content="你是谁"),
                ],
                "expected": """问：你是谁
""",
            },
            {
                "name": "prompt with both system_prompt, user_prompt and assistant_prompt",
                "messages": [
                    ChatMessage(role=SYSTEM_PROMPT, content="你是一个聪明的机器人"),
                    ChatMessage(role=USER_PROMPT, content="你是谁"),
                    ChatMessage(role=ASSISTANT_PROMPT, content="我是一个智能机器人"),
                ],
                "expected": """问：你是谁
答：我是一个智能机器人
""",
            },
            {
                "name": "continuous conversation with system_prompt exists",
                "messages": [
                    ChatMessage(role=SYSTEM_PROMPT, content="你是一个聪明的机器人"),
                    ChatMessage(role=USER_PROMPT, content="你是谁"),
                    ChatMessage(role=ASSISTANT_PROMPT, content="我是一个智能机器人"),
                    ChatMessage(role=USER_PROMPT, content="你会做什么"),
                ],
                "expected": """问：你是谁
答：我是一个智能机器人
问：你会做什么
""",
            },
            {
                "name": "prompt with only user_prompt",
                "messages": [
                    ChatMessage(role=USER_PROMPT, content="你是谁"),
                ],
                "expected": """问：你是谁
""",
            },
            {
                "name": "prompt with user_prompt and assistant_prompt, no system_prompt",
                "messages": [
                    ChatMessage(role=USER_PROMPT, content="你是谁"),
                    ChatMessage(role=ASSISTANT_PROMPT, content="我是一个智能机器人"),
                ],
                "expected": """问：你是谁
答：我是一个智能机器人
""",
            },
        ]

        for tc in test_cases:
            got = ChatGLMChat.prompt(tc["messages"])
            self.assertEqual(
                got,
                tc["expected"],
                "testcase '{case}' not passed".format(case=tc["name"]),
            )
