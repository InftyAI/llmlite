import unittest

from llms.llama import format_llama_prompt
from llms.chat import SYSTEM_PROMPT, USER_PROMPT


class TestLlama(unittest.TestCase):
    def test_format_llama_prompt(self):
        system_prompt = "Please complete the code in python"
        system_prompt_generated = """<s>[INST] <<SYS>>
Please complete the code in python
<</SYS>>

"""

        content = format_llama_prompt(role=SYSTEM_PROMPT, content=system_prompt)
        self.assertEqual(content, system_prompt_generated, "system prompt not right")

        user_prompt = "Please output `hello world`"
        user_prompt_generated = """<s>[INST] <<SYS>>
Please complete the code in python
<</SYS>>

Please output `hello world` [/INST] """
        content = format_llama_prompt(
            role=USER_PROMPT, content=user_prompt, history=content
        )
        self.assertEqual(content, user_prompt_generated, "user prompt not right")

        no_system_prompt_content = format_llama_prompt(
            role=USER_PROMPT, content=user_prompt
        )
        no_system_prompt_content_generated = "Please output `hello world`"
        self.assertEqual(
            no_system_prompt_content,
            no_system_prompt_content_generated,
            "user prompt not right",
        )

        answer = 'print("hello world")'
        user_prompt_2 = "Thanks for the answer"
        user_prompt_2_generated = """<s>[INST] <<SYS>>
Please complete the code in python
<</SYS>>

Please output `hello world` [/INST] print("hello world") </s><s>[INST] Thanks for the answer [/INST] """
        content = format_llama_prompt(
            content=user_prompt_2, history=content, answer=answer
        )
        self.assertEqual(
            content, user_prompt_2_generated, "continuous user prompt not right"
        )
