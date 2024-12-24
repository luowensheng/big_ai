# https://github.com/anthropics/anthropic-sdk-python

from ..utils import str_to_json
import json
import requests
from anthropic import Anthropic


class ClaudeModel:
    def __init__(self, *, model: str, api_key: str, anthropic_version: str,  **kwargs):
        self.url = "https://api.anthropic.com/v1/messages"
        self.model = model
        # self.api_key = api_key
        self.anthropic_version = anthropic_version
        self.client = Anthropic(api_key=api_key)

    def run(self, messages, temperature, stream=True, **kwargs):

        request_data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream
        }

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": self.anthropic_version,
        }

        response = requests.post(self.url, json=request_data, headers=headers, stream=stream)
        

        message = self.client.messages.create(
            max_tokens=kwargs.get("max_tokens"),
            messages=message,
            model=self.model,
            stream=stream
        )

        if not stream:
            yield {"text": message.content}
            return 
        
        for chunk in message:
            yield {"text": chunk.content}

        