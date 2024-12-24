# https://github.com/google-gemini/cookbook/blob/main/quickstarts/Streaming.ipynb
import json
import requests
import re
import google.generativeai as genai
from google.generativeai.types import generation_types



class GeminiModel:
    def __init__(self, *, model: str, api_key: str):
        self.model = model
        self.api_key = api_key


    def run(self, messages: list[dict], temperature: float, top_p: float, top_k: int, stream=True, **kwargs):

        contents = []
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
            parts = []

            if isinstance(message["content"], str):
                parts.append({"text": message["content"]})
            else:
                for item in message["content"]:

                    if item["type"] == "text":
                        parts.append({"text": item["text"]})

                    elif item["type"] == "image_url":
                        base64_image = item["image_url"]["url"]
                        ext = base64_image.split(";")[0].split("/")[-1].lower()

                        parts.append({
                            "inline_data":{
                                "mime_type": f"image/{ext}",
                                "data": base64_image.split(",")[-1]
                        }})
                    else:
                        raise Exception(f"Invalid type: {item['type']}")

            contents.append({
                "role": role,
                "parts": parts  
            })
        instruction = ""

        if instructions:
            instruction = "\n".join(instructions)

        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(f"models/{self.model}", system_instruction=instruction or None)
        response = model.generate_content(contents, stream=stream, generation_config={
            "temperature": temperature,
        })

        if not stream:
            yield {"text": response.text}
            return
        
        for chunk in response:
            yield {"text": chunk.text}



def replace_escaped_chars(input_string):
    return re.sub(r'\\([ntr])', lambda match: {'n': '\n', 't': '\t', 'r': '\r'}[match.group(1)], input_string)

def unquote(text: str):
    return replace_escaped_chars(text.strip("\""))

def extract_text_value(input_str):
    # Regex to match the value of "text"
    match = re.search(r'"text":\s*("[^"]+")', input_str)
    # Return the captured group or None if no match
    return unquote(match.group(1)) if match else None
