from llmlite.apis import ChatLLM, ChatMessage

chat = ChatLLM(
    model_name_or_path="meta-llama/Llama-2-7b-chat-hf",
    task="text-generation",
    backend="vllm",
    # temperature=0.2,
    # max_length=2048,
    # do_sample=True,
    # top_p=0.7,
    # top_k=3,
)

result = chat.completion(
    messages=[
        [
            ChatMessage(role="system", content="You're a honest assistant."),
            ChatMessage(
                role="user", content="There's a llama in my garden, what should I do?"
            ),
        ],
        [
            ChatMessage(role="system", content="You're a honest assistant."),
            ChatMessage(role="user", content="How many people are their in China?"),
        ],
    ],
    temperature=0.7,  # You can also overwrite the configurations in each conservation.
)

print(result)
