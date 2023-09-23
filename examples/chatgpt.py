from llmlite.apis import ChatLLM, ChatMessage

chat = ChatLLM(
    model_name_or_path="gpt-3.5-turbo",
    temperature=0.2,  # optional, default to '0.2'
    max_length=2048,  # optional, default to '2048'
    top_p=0.7,  # optional, default to '0.7'
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
