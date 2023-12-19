import os


# This is help to test more efficiently with models pre-downloaded.
def build_model(model_name: str) -> str:
    path = os.getenv("MODEL_PATH")
    if path is not None:
        return path + "/" + model_name
    return model_name
