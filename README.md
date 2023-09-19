# llmlite

A library helps to communicate with all kinds of LLMs consistently.

## How to install

```cmd
pip install llmlite==0.0.7
```

## How to use

### Chatting

```python
from llmlite.apis import ChatLLM, ChatMessage

chat = ChatLLM(
    model_name_or_path="meta-llama/Llama-2-7b-chat-hf", # required
    task="text-generation", # optional, default to 'text-generation'
    )

result = chat.completion(
  messages=[
    ChatMessage(role="system", content="You're a honest assistant."),
    ChatMessage(role="user", content="There's a llama in my garden, what should I do?"),
  ]
)

# Output: Oh my goodness, a llama in your garden?! 😱 That's quite a surprise! 😅 As an honest assistant, I must inform you that llamas are not typically known for their gardening skills, so it's possible that the llama in your garden may have wandered there accidentally or is seeking shelter. 🐮 ...

```

`chat.completion()` also support other parameters like `temperature`, `max_length`, `do_sample`, `top_k`, `top_p` to control the length, randomness and diversity of the generated text.

### Prompting

You can use `llmlite` to help you generate full prompts, for instance:

```python
from llmlite.apis import ChatMessage, LlamaChat

messages = [
    ChatMessage(role="system", content="You're a honest assistant."),
    ChatMessage(role="user", content="There's a llama in my garden, what should I do?"),
]

LlamaChat.prompt(messages)

# Output:
# <s>[INST] <<SYS>>
# You're a honest assistant.
# <</SYS>>

# There's a llama in my garden, what should I do? [/INST]
```

### Logging

Set the env variable `LOG_LEVEL` for log configuration, default to `INFO`, others like DEBUG, INFO, WARNING etc..

## Integrations

| Model | State | Note |
| ---- | ---- | ---- |
| Llama-2 | Done ✅ | |
| ChatGLM2 | Done ✅ | |
| ChatGPT | WIP ⏳ | [issue#6](https://github.com/InftyAI/ChatLLM/issues/6) |
| Claude-2 | RoadMap 📋 | [issue#7](https://github.com/InftyAI/ChatLLM/issues/7)
| Falcon | RoadMap 📋 | [issue#8](https://github.com/InftyAI/ChatLLM/issues/8)
| StableLM | RoadMap 📋 | [issue#11](https://github.com/InftyAI/ChatLLM/issues/11) |
| ... | ... | ... |

## Contributions

🚀 All kinds of contributions are welcomed ! Please follow [Contributing](/CONTRIBUTING.md).

## Contributors

🎉 Thanks to all these contributors.

<a href="https://github.com/InftyAI/ChatLLM/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=InftyAI/ChatLLM" />
</a>
