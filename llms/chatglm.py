from transformers import AutoTokenizer, AutoModel

#if not history:
#    prompt = query
#else:
#    prompt = ""
#    for i, (old_query, response) in enumerate(history):
#        prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
#    prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
#response, history = model.chat(tokenizer, query, history=history)
#print(response)
#print(history)

class ChatglmChat:
    def __init__(self,model_name_or_path, task = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True).half().cuda()
        self.model = self.model.eval()
    def completion(self,system_prompt, user_prompt):
        history = []
        response, history = self.model.chat(self.tokenizer, user_prompt, history=history)
        return response
