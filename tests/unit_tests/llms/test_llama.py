from llmlite.llms.llama import format_llama_prompt
from llmlite.apis import ChatLLM
from llmlite import consts
from llmlite.llms.messages import ChatMessage


class TestLlama:
    def test_format_llama_prompt_with_system_prompt(self):
        # format system prompt
        system_prompt = "Please complete the code in python"
        system_prompt_generated = """<s>[INST] <<SYS>>
Please complete the code in python
<</SYS>>

"""

        content = format_llama_prompt(
            role=consts.SYSTEM_PROMPT, content=system_prompt, history=None
        )
        assert content == system_prompt_generated

        # format user prompt
        user_prompt = "Please output `hello world`"
        user_prompt_generated = """<s>[INST] <<SYS>>
Please complete the code in python
<</SYS>>

Please output `hello world` [/INST] """

        content = format_llama_prompt(
            role=consts.USER_PROMPT, content=user_prompt, history=content
        )
        assert content == user_prompt_generated

        # format the answer
        answer = 'print("hello world")'
        user_prompt_generated = """<s>[INST] <<SYS>>
Please complete the code in python
<</SYS>>

Please output `hello world` [/INST] print("hello world") </s>"""

        content = format_llama_prompt(
            role=consts.ASSISTANT_PROMPT,
            content=answer,
            history=content,
        )
        assert content == user_prompt_generated

        # format user prompt again
        user_prompt = "Thanks for the answer"
        user_prompt_generated = """<s>[INST] <<SYS>>
Please complete the code in python
<</SYS>>

Please output `hello world` [/INST] print("hello world") </s><s>[INST] Thanks for the answer [/INST] """
        content = format_llama_prompt(
            role=consts.USER_PROMPT, content=user_prompt, history=content
        )
        assert content == user_prompt_generated

    def test_format_llama_prompt_with_no_system_prompt(self):
        # format user prompt
        user_prompt = "Please output `hello world`"
        user_prompt_generated = "<s>[INST] Please output `hello world` [/INST] "
        content = format_llama_prompt(
            role=consts.USER_PROMPT, content=user_prompt, history=None
        )
        assert content == user_prompt_generated

        # format the answer
        answer = 'print("hello world")'
        user_prompt_generated = (
            '<s>[INST] Please output `hello world` [/INST] print("hello world") </s>'
        )
        content = format_llama_prompt(
            role=consts.ASSISTANT_PROMPT, content=answer, history=content
        )
        assert content == user_prompt_generated

        # format user prompt again
        user_prompt = "Thanks for the answer"
        user_prompt_generated = '<s>[INST] Please output `hello world` [/INST] print("hello world") </s><s>[INST] Thanks for the answer [/INST] '
        content = format_llama_prompt(
            role=consts.USER_PROMPT, content=user_prompt, history=content
        )
        assert content == user_prompt_generated

    def test_prompt(self):
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
                "expected": """<s>[INST] <<SYS>>
You are an intelligent agent
<</SYS>>

Who you are [/INST] """,
            },
            {
                "name": "prompt with both system_prompt user_prompt and assistant_prompt",
                "messages": [
                    ChatMessage(
                        role=consts.SYSTEM_PROMPT,
                        content="You are an intelligent agent",
                    ),
                    ChatMessage(role=consts.USER_PROMPT, content="Who you are"),
                    ChatMessage(role=consts.ASSISTANT_PROMPT, content="I'm an agent"),
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
                        role=consts.SYSTEM_PROMPT,
                        content="You are an intelligent agent",
                    ),
                    ChatMessage(role=consts.USER_PROMPT, content="Who you are"),
                    ChatMessage(role=consts.ASSISTANT_PROMPT, content="I'm an agent"),
                    ChatMessage(role=consts.USER_PROMPT, content="You're so clever"),
                    ChatMessage(role=consts.ASSISTANT_PROMPT, content="Thanks"),
                ],
                "expected": """<s>[INST] <<SYS>>
You are an intelligent agent
<</SYS>>

Who you are [/INST] I'm an agent </s><s>[INST] You're so clever [/INST] Thanks </s>""",
            },
            {
                "name": "prompt with only user_prompt",
                "messages": [
                    ChatMessage(role=consts.USER_PROMPT, content="Who you are"),
                ],
                "expected": """<s>[INST] Who you are [/INST] """,
            },
            {
                "name": "prompt with user_prompt and assistant_prompt",
                "messages": [
                    ChatMessage(role=consts.USER_PROMPT, content="Who you are"),
                    ChatMessage(role=consts.ASSISTANT_PROMPT, content="I'm an agent"),
                ],
                "expected": """<s>[INST] Who you are [/INST] I'm an agent </s>""",
            },
            {
                "name": "continuous conversation with no system_prompt",
                "messages": [
                    ChatMessage(role=consts.USER_PROMPT, content="Who you are"),
                    ChatMessage(role=consts.ASSISTANT_PROMPT, content="I'm an agent"),
                    ChatMessage(role=consts.USER_PROMPT, content="You're so clever"),
                    ChatMessage(role=consts.ASSISTANT_PROMPT, content="Thanks"),
                ],
                "expected": """<s>[INST] Who you are [/INST] I'm an agent </s><s>[INST] You're so clever [/INST] Thanks </s>""",
            },
        ]

        for tc in test_cases:
            got = ChatLLM.prompt("llama-2", tc["messages"])
            assert got == tc["expected"]
