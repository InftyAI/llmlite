from transformers import AutoTokenizer, AutoModel


class ChatglmChat:
    def __init__(self, model_name_or_path, task=None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        self.model = (
            AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
            .half()
            .cuda()
        )
        self.model = self.model.eval()

    def completion(self, system_prompt, user_prompt):
        history = []
        response, history = self.model.chat(
            self.tokenizer, user_prompt, history=history
        )
        return response
