from llms.chat import Chat
from llms.llama import LlamaChat
from llms.llama_hf import LlamaHFChat


class ChatLLM:
    """

    How To Use:
        chat = ChatLLM(
            model_name_or_path="<model_name_or_path>",
            task="<task name>", # optional, default to `text-generation`.
            host="<local or api>", # optional, default to `local`.
            )
        result = chat.completion(
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
        """
        Args:
            model_name_or_path (str): The model name or the model path.
            task (str): The task defining which pipeline will be returned, default to `text-generation`.
            host (str): The place to load the model, default to `local`.
                        `local` means loading the pre-downloaded models, so `model_name_or_path` should be the model path.
                        `api` means requesting the hosted inference apis, e.g. HuggingFace Hosted Inference API,
                        so `model_name_or_path` should be the exactly model name.
        """

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
