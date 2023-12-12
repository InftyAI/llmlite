from llmlite.apis import ChatLLM, ChatMessage

chat = ChatLLM(
    model_name_or_path="/workspace/models/meta-llama/llama-2-7b-chat-hf",
    task="text-generation",
    backend="vllm",
    # temperature=0.2,
    # do_sample=True,
    # top_p=0.7,
    # top_k=3,
)

result = chat.completion(
    messages=[
        ChatMessage(role="system", content="You're a honest assistant."),
        ChatMessage(
            role="user", content="There's a llama in my garden, what should I do?"
        ),
    ],
)

print(result)
