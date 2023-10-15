from llmlite.apis import ChatLLM, ChatMessage

# You should set the OPENAI_API_KEY first.

chat = ChatLLM(
    model_name_or_path="gpt-3.5-turbo",
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
