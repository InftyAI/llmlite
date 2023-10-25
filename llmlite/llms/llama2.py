from typing import List
import logging

import torch
import transformers  # type: ignore
from transformers import AutoTokenizer, LlamaForCausalLM

from llmlite.llms.chat import (
    ASSISTANT_PROMPT,
    SYSTEM_PROMPT,
    USER_PROMPT,
    LocalChat,
)
from llmlite.llms.messages import ChatMessage
from llmlite.utils.validation import general_validations


class LlamaChat(LocalChat):
    """

    Llama2 required prompt template: https://huggingface.co/blog/llama2#how-to-prompt-llama-2
    The prompt template for the first turn looks like:

        <s>[INST] <<SYS>>
        {{ system_prompt }}
        <</SYS>>

        {{ user_message }} [/INST]

    As the conversation progresses, all the interactions will be appended to the previous prompt like:

        <s>[INST] <<SYS>>
        {{ system_prompt }}
        <</SYS>>

        {{ user_msg_1 }} [/INST] {{ model_answer_1 }} </s><s>[INST] {{ user_msg_2 }} [/INST]

    """

    def __init__(
        self,
        model_name_or_path: str,
        task: str | None = "text-generation",
        torch_dtype: torch.dtype = torch.float16,
        **kwargs,
    ) -> None:
        super().__init__(model_name_or_path, task, torch_dtype, **kwargs)

        self.pipeline = build_pipeline(
            model_name_or_path=model_name_or_path,
            task=task,
            torch_dtype=torch_dtype,
            **kwargs,
        )
        self.logger = logging.getLogger("llmlite.LlamaChat")

    @classmethod
    def validate(cls) -> bool:
        return True

    @classmethod
    def support_system_prompt(cls) -> bool:
        return True

    @classmethod
    def prompt(cls, messages: List[ChatMessage], **kwargs) -> str | None:
        if not general_validations(messages, cls.support_system_prompt()):
            return None

        return get_full_prompts(messages)

    def completion(
        self,
        messages: List[ChatMessage],
        **kwargs,
    ) -> str | None:
        prompt = self.prompt(messages)
        if prompt is None:
            return None

        self.logger.debug(f"Llama prompt: {prompt}")

        sequences = self.pipeline(
            prompt,
            return_full_text=False,
            **kwargs,
        )

        return sequences[0]["generated_text"]


def get_full_prompts(messages: List[ChatMessage]):
    has_system_prompt = False
    prompt = None

    for message in messages:
        role = message.role
        content = message.content

        if role == SYSTEM_PROMPT:
            # We only can accept one system prompt.
            if has_system_prompt:
                continue

            prompt = format_llama_prompt(
                role=SYSTEM_PROMPT, content=content, history=None
            )
            has_system_prompt = True

        elif role == USER_PROMPT:
            prompt = format_llama_prompt(
                role=USER_PROMPT, content=content, history=prompt
            )

        elif role == ASSISTANT_PROMPT:
            prompt = format_llama_prompt(
                role=ASSISTANT_PROMPT, content=content, history=prompt
            )

        else:
            logger = logging.getLogger("llmlite.LlamaChat")
            logger.error("unavailable instruction role: %s", role)

    return prompt


# TODO: trim the tokens when exceeded.
def format_llama_prompt(
    role: str,
    content: str,
    history: str | None,
) -> str:
    if content is None:
        return ""

    result = ""

    # SYSTEM_PROMPT will only be added once in the very beginning of the conversation.
    if role == SYSTEM_PROMPT:
        result = "<s>[INST] <<SYS>>\n"
        result += content + "\n"
        result += "<</SYS>>\n\n"
        return result

    if role == USER_PROMPT:
        # No system prompt
        if history is None:
            return "<s>[INST] " + content + " [/INST] "
        else:
            # end with system prompt
            if history.endswith("<</SYS>>\n\n"):
                return history + content + " [/INST] "
            else:
                # end with assistant prompt
                return history + "<s>[INST] " + content + " [/INST] "

    if role == ASSISTANT_PROMPT:
        # history or content should not be nil here.
        if history is None or content is None:
            return result
        return history + content + " </s>"

    return result


def build_pipeline(
    model_name_or_path: str,
    task: str | None,
    torch_dtype: torch.dtype,
    **kwargs,
) -> transformers.pipeline:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        **kwargs,
    )
    model = (
        LlamaForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        .half()
        .cuda()
        .eval()
    )

    return transformers.pipeline(
        task=task,
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch_dtype,
        device=0,
        **kwargs,
    )
