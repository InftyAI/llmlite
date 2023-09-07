import sys
sys.path.append('/home/jerry_kon/my_chatglm/ChatLLM')
from apis.api import ChatLLM
chat = ChatLLM(
            model_name_or_path="/models/llama-2-7b-chat-hf",
            task="text-generation",
            host="local",
            )
result = chat.completion(
            prompt="How many people are there in China"
        )

print(result)
