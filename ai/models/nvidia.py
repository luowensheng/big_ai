# https://build.nvidia.com/mistralai/mixtral-8x7b-instruct
from .openai import OpenAIModel

class NvidiaModel(OpenAIModel):
    def __init__(self, *, model: str, url="https://integrate.api.nvidia.com/v1", api_key: str):
        super().__init__(model=model, url=url, api_key=api_key)
