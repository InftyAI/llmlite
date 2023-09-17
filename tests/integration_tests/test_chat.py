from llmlite.apis import ChatLLM, ChatMessage
from llmlite.utils.envs import MODEL_PATH


# This is help to test more efficiently with models pre-downloaded.
def build_model(model_name: str) -> str:
    if MODEL_PATH is not None:
        return MODEL_PATH + "/" + model_name.lower()
    return model_name


class TestChat:
    def test_chat_with_llama(self):
        test_cases = [
            {
                "name": "local model with llama 2",
                "model": build_model("meta-llama/Llama-2-7b-chat-hf"),
                "task": "text-generation",
                "messages": [
                    ChatMessage(
                        role="system", content="You're an agent based on fact."
                    ),
                    ChatMessage(
                        role="user", content="How many people are there in China?"
                    ),
                ],
            },
        ]

        for tc in test_cases:
            chat = ChatLLM(
                model_name_or_path=tc["model"],
                task=tc["task"],
            )
            result = chat.completion(messages=tc["messages"])
            assert len(result) > 0

    def test_chat_with_chatglm(self):
        test_cases = [
            {
                "name": "local model with chatglm2",
                "model": build_model("THUDM/chatglm2-6b"),
                "task": "text-generation",
                "messages": [
                    ChatMessage(role="user", content="中国共有多少人口？"),
                ],
            },
        ]

        for tc in test_cases:
            chat = ChatLLM(
                model_name_or_path=tc["model"],
                task=tc["task"],
            )
            result = chat.completion(messages=tc["messages"])
            assert len(result) > 0
