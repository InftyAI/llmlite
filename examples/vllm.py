from llmlite import ChatLLM, ChatMessage

chat = ChatLLM(
    model_name_or_path="meta-llama/Llama-2-7b-chat-hf",
    backend="vllm",
)

results = chat.completion(
    messages=[
        [
            ChatMessage(role="system", content="You're a honest assistant."),
            ChatMessage(
                role="user", content="There's a llama in my garden, what should I do?"
            ),
        ],
        [
            ChatMessage(role="user", content="How many people are there in China?"),
        ],
    ],
    max_tokens=2048,
    # temperature=0.7,
    # top_p=0.8,
    # top_k=3,
)

for result in results:
    print(f"RESULT: \n{result}\n\n")
