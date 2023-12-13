from typing import List, Tuple, Optional
import logging

from transformers import AutoTokenizer  # type: ignore
import torch

from llmlite.llms.model import Model
from llmlite.utils.util import get_class
from llmlite.llms.messages import ChatMessage
from llmlite import consts


class ChatGLM(Model):
    """
    ChatGLM is mainly used for chinese questions and answers. Currently don't support system prompt yet.
    """

    def __init__(
        self,
        model_name_or_path: str,
        task: Optional[str] = "text-generation",
        torch_dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__(model_name_or_path, task, torch_dtype)

    __config__ = {
        "support_system_prompt": False,
        "backends": [consts.BACKEND_HF, consts.BACKEND_VLLM],
        "default_backend": consts.BACKEND_HF,
        "architecture": "AutoModel",
    }

    def load_with_hf(self, architecture: str):
        modelClass = get_class("transformers", architecture)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name_or_path,
            trust_remote_code=True,
        )
        self._model = (
            modelClass.from_pretrained(self._model_name_or_path).half().cuda().eval()
        )

        self.logger = logging.getLogger("llmlite.ChatGLMChat")

    @classmethod
    def prompt(cls, messages: List[ChatMessage], **kwargs) -> str | None:
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

        if self.hf_backend_flag == False:
            prompt = self.prompt(messages)
            response = self._backend.completion(prompt)
            return response
        
        query, history = build_history(messages)
        response, _ = self._model.chat(
            self._tokenizer,
            query,
            history=history,
        )
        return response


def build_history(messages: List[ChatMessage]) -> Tuple[str, List[Tuple[str, str]]]:
    query = messages[-1].content
    history = []

    pair_number = int(len(messages) / 2)
    for i in range(0, pair_number):
        history.append((messages[2 * i].content, messages[2 * i + 1].content))

    return query, history
