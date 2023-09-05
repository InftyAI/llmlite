# ChatLLM

A library helps to communicate with all kinds of LLMs consistently.

## How to use

```python
from chatllm import ChatLLM
llm = ChatLLM(
    model_name_or_path="<model_name_or_path>",
    task="<task name>", # optional, default to `text-generation`.
    host="<local or api>", # optional, default to `local`.
    )
result = llm.chat(
    prompt="<user prompt>",
    system_prompt="<optional system prompt>",
)
```

## Integrations

| Name | Host | State |
| ---- | ----- | ------- |
| meta-llama/Llama-2~ | Local | Done ✅ |
| meta-llama/Llama-2~ | API | WIP ⏳ |
| THUDM/chatglm2~ | Local | WIP ⏳ |
| ChatGPT | API | RoadMap 📋 |
| ... | ... | ... |

## Contributions

🚀 All kinds of contributions are welcomed ! Please follow [Contributing](/CONTRIBUTING.md).
