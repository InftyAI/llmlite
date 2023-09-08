from transformers import AutoTokenizer, AutoModel
from llms.chat import LocalChat


class ChatglmChat(LocalChat):
    def __init__(self, model_name_or_path, task=None):
        self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, trust_remote_code=True
                )
        self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True).half().cuda()
        self.model = self.model.eval()

    def completion(self, system_prompt, user_prompt):
        history = []
        response, history = self.model.chat(
                self.tokenizer, user_prompt, history=history
                )
        return response

    @classmethod
    def support_system_prompt(self) -> bool:
        return False

    @classmethod
    def prompt(cls, system_prompt: str = None, user_prompt: str = None):
        pass
