import logging

import torch
import transformers
from transformers import AutoTokenizer, LlamaForCausalLM

from llms.chat import LocalChat, SYSTEM_PROMPT, USER_PROMPT


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
        task: str,
        torch_dtype: torch.dtype = torch.float16,
    ) -> None:
        self.pipeline = build_pipeline(
            model_name_or_path=model_name_or_path, task=task, torch_dtype=torch_dtype
        )

    @classmethod
    def support_system_prompt() -> bool:
        return True

    @classmethod
    def prompt(
        cls,
        system_prompt: str = None,
        user_prompt: str = None,
    ) -> str:
        if system_prompt is not None and user_prompt is not None:
            system_content = format_llama_prompt(
                role=SYSTEM_PROMPT, content=system_prompt
            )
            return format_llama_prompt(content=user_prompt, history=system_content)
        elif system_prompt is not None:
            # This is meaningless as we only have system prompt.
            return format_llama_prompt(role=SYSTEM_PROMPT, content=system_prompt)
        else:
            return format_llama_prompt(content=user_prompt)

    def completion(
        self,
        system_prompt: str = None,
        user_prompt: str = None,
    ) -> str:
        prompt = LlamaChat.prompt(system_prompt=system_prompt, user_prompt=user_prompt)

        logging.debug(
            f"system_prompt: {system_prompt}\n, user_prompt: {user_prompt}\n, final_prompt: {prompt}\n"
        )

        sequences = self.pipeline(
            prompt,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
            max_length=4090,
            return_full_text=False,
        )

        return sequences[0]["generated_text"]


# TODO: trim the tokens when exceeded.
def format_llama_prompt(
    role: str = USER_PROMPT,
    content: str = None,
    history: str = None,
    answer: str = None,
):
    result = ""

    # SYSTEM_PROMPT will only be added once in the very beginning of the conversation.
    if role == SYSTEM_PROMPT:
        result = "<s>[INST] <<SYS>>\n"
        result += content + "\n"
        result += "<</SYS>>\n\n"
        return result

    # This is for the first run conversation of USER_PROMPT.
    # No system prompt, return the content directly.
    if answer is None and history is None:
        return content

    # This is for the first run conversation of USER_PROMPT.
    # History is the system prompt.
    if answer is None:
        result = history + content + " [/INST] "
        return result

    # For continuous conversations here.
    result = history + answer + " </s><s>[INST] " + content + " [/INST] "
    return result


def build_pipeline(
    model_name_or_path: str,
    task: str,
    torch_dtype: torch.dtype,
) -> transformers.pipeline:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )
    model = LlamaForCausalLM.from_pretrained(model_name_or_path).half().cuda().eval()

    logging.info(
        "Transformer pipeline with model: %s, task: %s, torch_dtype: %s, model dtype: %s",
        model_name_or_path,
        task,
        torch_dtype,
        model.dtype,
    )

    return transformers.pipeline(
        task=task,
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch_dtype,
        device=0,
    )
