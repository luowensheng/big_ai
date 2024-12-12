from ..utils import str_to_json
import json
import requests

class ClaudeModel:
    def __init__(self, *, model: str, api_key: str, anthropic_version: str,  **kwargs):
        self.url = "https://api.anthropic.com/v1/messages"
        self.model = model
        self.api_key = api_key
        self.anthropic_version = anthropic_version

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
        
        if response.status_code == 200:
            try:
                if stream:
                    # Handle streaming response
                    for chunk in response.iter_lines():
                        if chunk:
                            # Decode the chunk and remove "data: " prefix
                            chunk_data = chunk.decode('utf-8')
                            if chunk_data.startswith("data: "):
                            
                                # Skip [DONE] message
                                if chunk_data == "[DONE]":
                                    break
                            try:
                                # Parse the JSON data
                                chunk_json = str_to_json(chunk_data)
                                # Extract the content if it exists
                                if (choices := chunk_json.get('choices')) and len(choices) > 0:
                                    if content := choices[0].get('delta', {}).get('content'):
                                        yield {"text": content}
                            except ValueError:
                                pass
                            except json.JSONDecodeError:
                                print(f"Error decoding chunk: {chunk_data}")
                else:
                    # Handle regular response
                    response_data = response.json()
                    if (choices := response_data.get('choices')) and len(choices) > 0:
                        if content := choices[0].get('message', {}).get('content'):
                            yield {"text": content}
                            return
                    return
                        
            except Exception as e:
                print(f"Error processing response: {str(e)}")
                raise
        else:
            error_msg = f"API request failed with status code {response.status_code}"
            try:
                error_data = response.json()
                if 'error' in error_data:
                    error_msg += f": {error_data['error'].get('message', '')}"
            except:
                error_msg += f": {response.text}"
            raise Exception(error_msg)