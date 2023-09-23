from llmlite.apis import ChatLLM, ChatMessage

chat = ChatLLM(
    model_name_or_path="meta-llama/Llama-2-7b-chat-hf",
    task="text-generation",
    temperature=0.2,  # optional, default to '0.2'
    max_length=2048,  # optional, default to '2048'
    do_sample=True,  # optional, default to False
    top_p=0.7,  # optional, default to '0.7'
    top_k=3,  # optional, default to '3'
)

result = chat.completion(
    messages=[
        ChatMessage(role="system", content="You're a honest assistant."),
        ChatMessage(
            role="user", content="There's a llama in my garden, what should I do?"
        ),
    ],
    temperature=0.7,  # You can also overwrite the configurations in each conservation.
)

print(result)
