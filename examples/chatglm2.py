from llmlite.apis import ChatLLM, ChatMessage

chat = ChatLLM(
    model_name_or_path="THUDM/chatglm2-6b",
    task="text-generation",
    temperature=0.2,
    max_length=2048,
    do_sample=True,
    top_p=0.7,
    top_k=3,
)

result = chat.completion(
    messages=[
        ChatMessage(role="user", content="中国共有多少人口？"),
    ],
    temperature=0.7,  # You can also overwrite the configurations in each conservation.
)

print(result)
