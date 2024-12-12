import json
import requests
import re


class GeminiModel:
    def __init__(self, *, model: str, api_key: str):
        self.model = model
        self.api_key = api_key

    def run(self, messages, temperature,top_p, top_k, stream=True, **kwargs):

        if not stream:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        else:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:streamGenerateContent?key={self.api_key}"

        messages_final = []
        instructions = []
        for message in messages:

            if not message:
                continue

            role = "user" if message["role"] != "assistant" else "model"
            
            match message["role"]:
                case "system":
                    instructions.append(message["content"])
                    continue

                case "user":
                    role = "user"
                
                case "assistant":
                    role = "model"

            messages_final.append({
            "role": role,
                "parts": [
                    {
                        "text": message["content"]
                    }
                ]   
            })
        request_data = {
            "contents": messages_final,
            "generationConfig": {
                "temperature": temperature,
                "topP": top_p,
                "topK": top_k
            }
        }

        if instructions:
            request_data["system_instruction"] = {"parts": { "text": "\n".join(instructions)}}

        headers = {
            "Content-Type": "application/json",
        }

        response = requests.post(url, json=request_data, headers=headers, stream=stream)
        if response.status_code == 200:
            try:
                if not stream:
                    response_data = response.json()
                    content = response_data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text") or ""
                    yield {"text": content}
                else:
                    # Handle streaming response
                    for chunk in response.iter_lines(1024,  delimiter=b'"candidates":'):
                        if chunk:
                            text_value = extract_text_value(chunk.decode('utf-8'))
                            if not text_value:
                                continue
                            yield {"text": text_value}

            except json.JSONDecodeError:
                raise Exception("Failed to parse JSON response")
        else:
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")


def replace_escaped_chars(input_string):
    return re.sub(r'\\([ntr])', lambda match: {'n': '\n', 't': '\t', 'r': '\r'}[match.group(1)], input_string)

def unquote(text: str):
    return replace_escaped_chars(text.strip("\""))

def extract_text_value(input_str):
    # Regex to match the value of "text"
    match = re.search(r'"text":\s*("[^"]+")', input_str)
    # Return the captured group or None if no match
    return unquote(match.group(1)) if match else None
