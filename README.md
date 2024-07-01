# llmlite

 [![Latest Release](https://img.shields.io/github/v/release/inftyai/llmlite?include_prereleases)](https://github.com/inftyai/llmlite/releases/latest)

**🌵** llmlite is a library helps to communicate with all kinds of LLMs consistently.

## Features

- State-of-the-art LLMs support
- Continuous Batching via [vLLM](https://github.com/vllm-project/vllm)
- Quantization([issue#37] (<https://github.com/InftyAI/llmlite/issues/37>))
- Loading specific adapters ([issue#51](https://github.com/InftyAI/llmlite/issues/51))
- Streaming ([issue#52](https://github.com/InftyAI/llmlite/issues/52))

### Model Support

| Model | State | System Prompt | Note |
| ---- | ---- | ---- | ---- |
| ChatGPT | Done ✅ | Yes | |
| Llama-2 | Done ✅ | Yes | |
| CodeLlama | Done ✅ | Yes | |
| ChatGLM2 | Done ✅ | No | |
| Baichuan2 | Done ✅ | Yes | |
| ChatGLM3 | WIP ⏳ | Yes | |
| Claude-2 | RoadMap 📋 | | [issue#7](https://github.com/InftyAI/ChatLLM/issues/7)
| Falcon | RoadMap 📋 | | [issue#8](https://github.com/InftyAI/ChatLLM/issues/8)
| StableLM | RoadMap 📋 | | [issue#11](https://github.com/InftyAI/ChatLLM/issues/11) |

### Backend Support

| backend | State |
| ---- | ---- |
| [huggingface](https://github.com/huggingface) | Done ✅ |
| [vLLM](https://github.com/vllm-project/vllm) | Done ✅ |

## How to install

```cmd
pip install llmlite==0.0.15
```

## How to use

### Chat

```python
from llmlite import ChatLLM, ChatMessage

chat = ChatLLM(
    model_name_or_path="meta-llama/Llama-2-7b-chat-hf", # required
    task="text-generation",
    )

result = chat.completion(
  messages=[
    ChatMessage(role="system", content="You're a honest assistant."),
    ChatMessage(role="user", content="There's a llama in my garden, what should I do?"),
  ]
)

# Output: Oh my goodness, a llama in your garden?! 😱 That's quite a surprise! 😅 As an honest assistant, I must inform you that llamas are not typically known for their gardening skills, so it's possible that the llama in your garden may have wandered there accidentally or is seeking shelter. 🐮 ...

```

### Continuous Batching

_This is mostly supported by vLLM, you can enable this by configuring the **backend**._

```python
from llmlite import ChatLLM, ChatMessage

chat = ChatLLM(
    model_name_or_path="meta-llama/Llama-2-7b-chat-hf",
    backend="vllm",
)

results = chat.completion(
    messages=[
        [
            ChatMessage(role="system", content="You're a honest assistant."),
            ChatMessage( role="user", content="There's a llama in my garden, what should I do?"),
        ],
        [
            ChatMessage(role="user", content="What's the population of the world?"),
        ],
    ],
    max_tokens=2048,
)

for result in results:
    print(f"RESULT: \n{result}\n\n")
```

`llmlite` also supports other parameters like `temperature`, `max_length`, `do_sample`, `top_k`, `top_p` to help control the length, randomness and diversity of the generated text.

See **[examples](./examples/)** for reference.

### Prompting

You can use `llmlite` to help you generate full prompts, for instance:

```python
from llmlite import ChatLLM

messages = [
    ChatMessage(role="system", content="You're a honest assistant."),
    ChatMessage(role="user", content="There's a llama in my garden, what should I do?"),
]

ChatLLM.prompt("meta-llama/Llama-2-7b-chat-hf", messages)

# Output:
# <s>[INST] <<SYS>>
# You're a honest assistant.
# <</SYS>>

# There's a llama in my garden, what should I do? [/INST]
```

### Logging

Set the env variable `LOG_LEVEL` for log configuration, default to `INFO`, others like DEBUG, INFO, WARNING etc..

## Contributions

🚀 All kinds of contributions are welcomed ! Please follow [Contributing](/CONTRIBUTING.md).

## Contributors

🎉 Thanks to all these contributors.

<a href="https://github.com/InftyAI/ChatLLM/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=InftyAI/ChatLLM" />
</a>
