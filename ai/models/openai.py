# https://github.com/openai/openai-python
from openai import OpenAI


class OpenAIModel:
    
    def __init__(self, *, model: str, url=None, api_key: str):
        self.url = url
        self.model = model
        self.client = OpenAI(base_url=url or None, api_key=api_key)

    def run(self, messages, temperature, stream=True, **kwargs):


        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            stream=stream,
        )

        if not stream:
            yield {"text": response.choices[0].message.content}
            return
        
        for text in response:
            yield {"text": text.choices[0].delta.content}
