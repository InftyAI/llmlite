from typing import List, Tuple
import logging

from transformers import AutoTokenizer, AutoModel  # type: ignore
import torch

from llmlite.llms.chat import Chat
from llmlite.llms.messages import ChatMessage
from llmlite.utils.validation import general_validations


class ChatGLMChat(Chat):
    """
    ChatGLM is mainly used for chinese questions and answers. Currently don't support system prompt yet.
    """

    def __init__(
        self,
        model_name_or_path: str,
        task: str | None = "text-generation",
        torch_dtype: torch.dtype = torch.float16,
        **kwargs,
    ) -> None:
        super().__init__(model_name_or_path, task, torch_dtype, **kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            **kwargs,
        )
        self.model = (
            AutoModel.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                **kwargs,
            )
            .half()
            .cuda()
            .eval()
        )
        self.logger = logging.getLogger("llmlite.ChatGLMChat")

    @classmethod
    def validate(cls) -> bool:
        return True

    @classmethod
    def support_system_prompt(cls) -> bool:
        return False

    @classmethod
    def prompt(cls, messages: List[ChatMessage], **kwargs) -> str | None:
        if not general_validations(messages, cls.support_system_prompt()):
            return None

        prompt = ""
        query, history = build_history(messages)

        # This is inspired by /root/.cache/huggingface/modules/transformers_modules/chatglm2-6b/tokenization_chatglm.py::build_prompt().
        for i, (old_query, response) in enumerate(history):
            prompt += "[Round {}]\n\n问：{}\n\n答：{}\n\n".format(
                i + 1, old_query, response
            )
        prompt += "[Round {}]\n\n问：{}\n\n答：".format(len(history) + 1, query)
        return prompt

    def completion(
        self,
        messages: List[ChatMessage],
        **kwargs,
    ) -> str | None:
        """
        This is how ChatGLM chat() looks like:

        def chat(
            self,
            tokenizer,
            query: str,
            history: List[Tuple[str, str]] = None,
            max_length: int = 8192,
            num_beams=1,
            do_sample=True,
            top_p=0.8,
            temperature=0.8,
            logits_processor=None,
            **kwargs,
            )

        """

        query, history = build_history(messages)
        kwargs["history"] = history

        response, _ = self.model.chat(
            self.tokenizer,
            query,
            **kwargs,
        )
        return response


def build_history(messages: List[ChatMessage]) -> Tuple[str, List[Tuple[str, str]]]:
    query = messages[-1].content
    history = []

    pair_number = int(len(messages) / 2)
    for i in range(0, pair_number):
        history.append((messages[2 * i].content, messages[2 * i + 1].content))

    return query, history
