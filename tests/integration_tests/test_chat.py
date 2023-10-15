import os

from llmlite.apis import ChatLLM, ChatMessage


# This is help to test more efficiently with models pre-downloaded.
def build_model(model_name: str) -> str:
    path = os.getenv("MODEL_PATH")
    if path is not None:
        return path + "/" + model_name.lower()
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
                temperature=0.2,  # optional, default to '0.2'
                max_length=2048,  # optional, default to '2048'
                do_sample=True,  # optional, default to False
                top_p=0.7,  # optional, default to '0.7'
                top_k=3,  # optional, default to '3'
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
                temperature=0.2,  # optional, default to '0.2'
                max_length=2048,  # optional, default to '2048'
                do_sample=True,  # optional, default to False
                top_p=0.7,  # optional, default to '0.7'
                top_k=3,  # optional, default to '3'
            )
            result = chat.completion(messages=tc["messages"])
            assert len(result) > 0

    def test_chat_with_chatgpt(self):
        test_cases = [
            {
                "name": "test with gpt-3.5-turbo",
                "model": "gpt-3.5-turbo",
                "messages": [
                    ChatMessage(
                        role="system", content="You're an agent based on fact."
                    ),
                    ChatMessage(
                        role="user", content="How many people are there in China?"
                    ),
                ],
            },
            {
                "name": "test with gpt-4",
                "model": "gpt-4",
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
            )
            result = chat.completion(messages=tc["messages"])
            assert len(result) > 0
