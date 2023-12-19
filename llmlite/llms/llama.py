import logging
from typing import List, Optional, Dict, Any

from llmlite import consts
from llmlite.llms.messages import ChatMessage
from llmlite.llms.model import Model


class Llama(Model):
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
        backend: str,
        **kwargs,
    ) -> None:
        super().__init__(model_name_or_path, backend, **kwargs)

    __config__ = {
        "support_system_prompt": True,
        "backends": [consts.BACKEND_HF, consts.BACKEND_VLLM],
        "default_backend": consts.BACKEND_HF,
        "architecture": "LlamaForCausalLM",
    }

    @classmethod
    def load_with_hf(
        cls,
        model_name_or_path: str,
        **kwargs: Dict[str, Any],
    ):
        # Llama requires task, default to "text-generation".
        if kwargs.get("task", None) is None:
            kwargs.update({"task": "text-generation"})
        return super().load_with_hf(model_name_or_path, **kwargs)

    @classmethod
    def prompt(
        cls,
        model_name_or_path: str,
        messages: List[ChatMessage],
        **kwargs,
    ) -> Optional[str]:
        has_system_prompt = False
        prompt = None

        for message in messages:
            role = message.role
            content = message.content

            if role == consts.SYSTEM_PROMPT:
                # We only can accept one system prompt.
                if has_system_prompt:
                    continue

                prompt = format_llama_prompt(
                    role=consts.SYSTEM_PROMPT, content=content, history=None
                )
                has_system_prompt = True

            elif role == consts.USER_PROMPT:
                prompt = format_llama_prompt(
                    role=consts.USER_PROMPT, content=content, history=prompt
                )

            elif role == consts.ASSISTANT_PROMPT:
                prompt = format_llama_prompt(
                    role=consts.ASSISTANT_PROMPT, content=content, history=prompt
                )

            else:
                logger = logging.getLogger("llmlite.LlamaChat")
                logger.error("unavailable instruction role: %s", role)

        return prompt


def format_llama_prompt(
    role: str,
    content: str,
    history: str | None,
) -> str:
    if content is None:
        return ""

    result = ""

    # SYSTEM_PROMPT will only be added once in the very beginning of the conversation.
    if role == consts.SYSTEM_PROMPT:
        result = "<s>[INST] <<SYS>>\n"
        result += content + "\n"
        result += "<</SYS>>\n\n"
        return result

    if role == consts.USER_PROMPT:
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

    if role == consts.ASSISTANT_PROMPT:
        # history or content should not be nil here.
        if history is None or content is None:
            return result
        return history + content + " </s>"

    return result
