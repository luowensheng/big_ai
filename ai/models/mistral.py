# https://github.com/mistralai/client-python

from ..utils import str_to_json
from mistralai import Mistral


class MistralModel:
    def __init__(self, *, model: str, api_key: str, server_url: str=None, **kwargs):
        self.model = model
        self.api_key = api_key
        self.safe_prompt = kwargs.get("safe_prompt") or False
        self.server_url = server_url


    def run(self, messages, temperature, top_p: float, stream=True, **kwargs):


        with Mistral(
            api_key=self.api_key,
        ) as s:
            res = s.chat.complete(
                model=self.model, 
                messages=messages,
                temperature=temperature,
                stream=stream,
                top_p=top_p,
                server_url=self.server_url,
                safe_prompt=self.safe_prompt
            )

            if res is  None:
                return
            
            if not stream:
                yield {"text": res.choices[0].message.content}
                return
            
            for chunk in res:
                yield {"text": chunk[0]}




        # request_data = {
        #     "model": self.model,
        #     "messages": messages,
        #     "temperature": temperature,
        #     "stream": stream,
        #     "safe_prompt": self.safe_prompt,
        #     "random_seed": kwargs.get("seed", 1337),
        #     "top_p": top_p,
        #     "response_format": {
        #         "type": kwargs.get("format")
        #     },
        #     "max_tokens": kwargs.get("max_tokens")
        # }

        # headers = {
        #     "Content-Type": "application/json",
        #     "Authorization": self.api_key,
        # }

        # response = requests.post(self.url, json=request_data, headers=headers, stream=stream)
        