from llmlite.utils import util
from llmlite import consts


class TestValidation:
    def test_parse_model_name(self):
        test_cases = [
            {
                "name": "parse llama2",
                "model_name": "Llama-2-13b-chat-hf",
                "want_model_type": consts.MODEL_TYPE_LLAMA,
                "want_version": 2,
            },
            {
                "name": "parse codellama",
                "model_name": "CodeLlama-7b-hf",
                "want_model_type": consts.MODEL_TYPE_LLAMA,
                "want_version": 2,
            },
            {
                "name": "parse chatglm2",
                "model_name": "chatglm2-6b",
                "want_model_type": consts.MODEL_TYPE_CHATGLM,
                "want_version": 2,
            },
            {
                "name": "parse chatglm3",
                "model_name": "chatglm3-6b",
                "want_model_type": consts.MODEL_TYPE_CHATGLM,
                "want_version": 3,
            },
            {
                "name": "parse baichuan2",
                "model_name": "Baichuan2-13B-Chat",
                "want_model_type": consts.MODEL_TYPE_BAICHUAN,
                "want_version": 2,
            },
            {
                "name": "parse gpt-3.5",
                "model_name": "gpt-3.5turbo",
                "want_model_type": consts.MODEL_TYPE_GPT,
                "want_version": 3.5,
            },
            {
                "name": "parse gpt-4",
                "model_name": "gpt-4",
                "want_model_type": consts.MODEL_TYPE_GPT,
                "want_version": 4,
            },
        ]

        for tc in test_cases:
            model_type, version = util.parse_model_name(tc["model_name"])
            assert model_type == tc["want_model_type"]
            assert version == tc["want_version"]
