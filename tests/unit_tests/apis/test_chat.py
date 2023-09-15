import pytest

from llmlite.llms.chatglm2 import ChatGLMChat
from llmlite.llms.llama2 import LlamaChat
from llmlite.apis.chat import (
    fetch_llm,
    UnavailableModelException,
)


class TestChat:
    def test_fetch_llm(self):
        test_cases = [
            {
                "name": "local llama-2 model",
                "model": "/models/Llama-2-7b-chat-hf",
                "chat": LlamaChat,
            },
            {
                "name": "huggingface llama-2 model",
                "model": "meta-llama/Llama-2-7b-chat-hf",
                "chat": LlamaChat,
            },
            {
                "name": "local chatglm2 model",
                "model": "THUDM/chatglm2-6b",
                "chat": ChatGLMChat,
            },
            {
                "name": "model with non-exist model",
                "model": "non-exists-model",
                "wantException": UnavailableModelException,
            },
        ]

        for tc in test_cases:
            model = tc["model"]

            if "wantException" not in tc:
                llm = fetch_llm(model)
                assert llm == tc["chat"]
            else:
                with pytest.raises(UnavailableModelException):
                    fetch_llm(model)
