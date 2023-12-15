from llmlite.apis import ChatLLM, ChatMessage

chat = ChatLLM(
    model_name_or_path="codellama/CodeLlama-13b-instruct-hf",
    task="text-generation",
    max_length=2048,
    # temperature=0.2,
    # do_sample=True,
    # top_p=0.7,
    # top_k=3,
)

result = chat.completion(
    messages=[
        ChatMessage(
            role="user",
            content="import socket\n\ndef ping_exponential_backoff(host: str):",
        ),
    ]
)

print(result)
