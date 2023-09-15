import unittest

from llmlite.llms.llama2 import LlamaChat, format_llama_prompt
from llmlite.llms.chat import ASSISTANT_PROMPT, SYSTEM_PROMPT, USER_PROMPT
from llmlite.apis.messages import ChatMessage


class TestLlama(unittest.TestCase):
    def test_format_llama_prompt_with_system_prompt(self):
        # format system prompt
        system_prompt = "Please complete the code in python"
        system_prompt_generated = """<s>[INST] <<SYS>>
Please complete the code in python
<</SYS>>

"""

        content = format_llama_prompt(
            role=SYSTEM_PROMPT, content=system_prompt, history=None
        )
        self.assertEqual(content, system_prompt_generated, "system prompt not right")

        # format user prompt
        user_prompt = "Please output `hello world`"
        user_prompt_generated = """<s>[INST] <<SYS>>
Please complete the code in python
<</SYS>>

Please output `hello world` [/INST] """

        content = format_llama_prompt(
            role=USER_PROMPT, content=user_prompt, history=content
        )
        self.assertEqual(content, user_prompt_generated, "user prompt not right")

        # format the answer
        answer = 'print("hello world")'
        user_prompt_generated = """<s>[INST] <<SYS>>
Please complete the code in python
<</SYS>>

Please output `hello world` [/INST] print("hello world") </s>"""

        content = format_llama_prompt(
            role=ASSISTANT_PROMPT,
            content=answer,
            history=content,
        )
        self.assertEqual(content, user_prompt_generated, "assistant prompt not right")

        # format user prompt again
        user_prompt = "Thanks for the answer"
        user_prompt_generated = """<s>[INST] <<SYS>>
Please complete the code in python
<</SYS>>

Please output `hello world` [/INST] print("hello world") </s><s>[INST] Thanks for the answer [/INST] """
        content = format_llama_prompt(
            role=USER_PROMPT, content=user_prompt, history=content
        )
        self.assertEqual(content, user_prompt_generated, "assistant prompt not right")

    def test_format_llama_prompt_with_no_system_prompt(self):
        # format user prompt
        user_prompt = "Please output `hello world`"
        user_prompt_generated = "<s>[INST] Please output `hello world` [/INST] "
        content = format_llama_prompt(
            role=USER_PROMPT, content=user_prompt, history=None
        )
        self.assertEqual(
            content,
            user_prompt_generated,
            "user prompt not right",
        )

        # format the answer
        answer = 'print("hello world")'
        user_prompt_generated = (
            '<s>[INST] Please output `hello world` [/INST] print("hello world") </s>'
        )
        content = format_llama_prompt(
            role=ASSISTANT_PROMPT, content=answer, history=content
        )
        self.assertEqual(
            content,
            user_prompt_generated,
            "assistant prompt not right",
        )

        # format user prompt again
        user_prompt = "Thanks for the answer"
        user_prompt_generated = '<s>[INST] Please output `hello world` [/INST] print("hello world") </s><s>[INST] Thanks for the answer [/INST] '
        content = format_llama_prompt(
            role=USER_PROMPT, content=user_prompt, history=content
        )
        self.assertEqual(
            content,
            user_prompt_generated,
            "user prompt not right",
        )

    def test_prompt(self):
        test_cases = [
            {
                "name": "prompt with both system_prompt and user_prompt",
                "messages": [
                    ChatMessage(
                        role=SYSTEM_PROMPT, content="You are a intelligent agent"
                    ),
                    ChatMessage(role=USER_PROMPT, content="Who you are"),
                ],
                "expected": """<s>[INST] <<SYS>>
You are a intelligent agent
<</SYS>>

Who you are [/INST] """,
            },
            {
                "name": "prompt with both system_prompt user_prompt and assistant_prompt",
                "messages": [
                    ChatMessage(
                        role=SYSTEM_PROMPT, content="You are a intelligent agent"
                    ),
                    ChatMessage(role=USER_PROMPT, content="Who you are"),
                    ChatMessage(role=ASSISTANT_PROMPT, content="I'm an agent"),
                ],
                "expected": """<s>[INST] <<SYS>>
You are a intelligent agent
<</SYS>>

Who you are [/INST] I'm an agent </s>""",
            },
            {
                "name": "continuous conversation with system_prompt exists",
                "messages": [
                    ChatMessage(
                        role=SYSTEM_PROMPT, content="You are a intelligent agent"
                    ),
                    ChatMessage(role=USER_PROMPT, content="Who you are"),
                    ChatMessage(role=ASSISTANT_PROMPT, content="I'm an agent"),
                    ChatMessage(role=USER_PROMPT, content="You're so clever"),
                    ChatMessage(role=ASSISTANT_PROMPT, content="Thanks"),
                ],
                "expected": """<s>[INST] <<SYS>>
You are a intelligent agent
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
            got = LlamaChat.prompt(tc["messages"])
            self.assertEqual(
                got,
                tc["expected"],
                "testcase '{case}' not passed".format(case=tc["name"]),
            )
