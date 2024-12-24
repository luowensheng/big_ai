# https://github.com/ollama/ollama-python
from ollama import chat
from ollama import ChatResponse
from ollama import Client


class OllamaModel:
    def __init__(self, *, model: str, ollama_host="http://localhost:11434", **kwargs):
        self.model = model
        self.other_kwargs = kwargs
        self.client = Client(
            host=ollama_host,
            headers=kwargs.get("headers") or None
        )

    def run(self, messages, temperature, top_p: float, top_k: int, stream=True, **kwargs):

        response: ChatResponse = self.client.chat(model=self.model, messages=messages, keep_alive="-1h", options={
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
        }, stream=stream)

        if not stream:
            yield {"text": response.message.content}
            return 
        
        for chunk in response:
            yield {"text": chunk['message']['content']}

