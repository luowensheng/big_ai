# https://github.com/groq/groq-python?tab=readme-ov-file
from ..utils import str_to_json
import json
import requests
from groq import Groq



class GroqModel:
    def __init__(self, *, model: str, url="https://api.groq.com/openai/v1/chat/completions", api_key: str, **kwargs):

        self.url = url
        self.model = model
        self.other_kwargs = kwargs
        self.api_key = api_key

        self.client = Groq(
            api_key=self.api_key  # This is the default and can be omitted
        )

    def run(self, messages, temperature, top_p: float, top_k: int, stream=True, **kwargs):

        # request_data = {
        #     "model": self.model,
        #     "messages": messages,
        #     "stream": stream,
        #     "format": kwargs.get("format"),
        #     "keep_alive": self.other_kwargs.get("keep_alive"),
        #     "options": {
        #         "temperature": temperature,
        #         "top_p": top_p,
        #         "top_k": top_k
        #     }
        # }

        chat_completion = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            stream=stream,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k
        )

        if not stream:
            text = chat_completion.choices[0].message.content
            yield {"text": text}
            return
        
        for chunk in chat_completion:
            yield {"text": chunk.choices[0].delta.content}
            