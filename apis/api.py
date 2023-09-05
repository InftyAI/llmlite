from llms.chat import Chat
from llms.llama import LlamaChat
from llms.llama_hf import LlamaHFChat


class ChatLLM:
    """

    How To Use:
        from chatllm import ChatLLM
        llm = ChatLLM(
            model_name_or_path="<model_name_or_path>",
            task="<task name>",
            host="<local or api>",
            )
        result = llm.chat(
            prompt="<user prompt>",
            system_prompt="<optional system prompt>",
        )

    """

    def __init__(
        self,
        model_name_or_path: str,
        task: str = "text-generation",
        host: str = "local",
    ):
        if model_name_or_path is None:
            raise Exception("model_name_or_path must exist")

        llm = fetch_llm(model_name_or_path, host)
        self.chat = llm(model_name_or_path=model_name_or_path, task=task)

    def completion(
        self,
        prompt: str,
        system_prompt: str = None,
    ) -> str:
        if prompt is None:
            raise Exception("user prompt must exist")

        return self.chat.completion(system_prompt=system_prompt, user_prompt=prompt)


def fetch_llm(model_name: str, host: str) -> Chat:
    model_name = model_name.lower()

    if "llama_2" in model_name or "llama-2" in model_name:
        return LlamaChat if host == "local" else LlamaHFChat

    raise UnavailableModelException(
        'model unavailable, supporting model family: "llama_2".'
    )


class UnavailableModelException(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message
