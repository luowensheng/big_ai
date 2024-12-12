import os
import subprocess
import sys
import inspect
import json
from threading import Lock
from typing import Iterable, Optional, Protocol
from .types import ChatCompletion
from .config import APIConfig
from .utils import load_module_from_path, str_to_json, render_template
from typing import TypedDict

class Response(TypedDict):
    text: str

class Model(Protocol):
    def run(
        messages:list[dict],
        temperature: float,
        seed: int,
        top_p: float,
        top_k: int,
    )->Iterable[Response]:...

def install_requirements(requirement_path: str):
    """Installs requirements from the provided requirement_path."""
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirement_path])
        print(f"Successfully installed requirements from {requirement_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements from {requirement_path}: {e}")
        
class ModelManager:

    def __init__(self, config_path: str):
        
        self.lock = Lock()
        self.config = json.load(open(config_path))
        self.model: Optional[Model] = None
        self.current_model_name = None
        self.available_models = list(self.config["models"])

        for model_id, config in self.config["models"].items():

            config: dict
            requirement_path = config.get("requirement_path")
            if requirement_path:
                install_requirements(requirement_path)
            
            env = config.get("env")
            if not env:
                continue

            env: dict
            self.config["models"][model_id]["kwargs"] = self.config["models"][model_id].get("kwargs") or {}
            for key, env_key in env.items():
                value = os.getenv(env_key)
                assert value, f'Could not load ENV variable "{env_key}" for {model_id}'
                self.config["models"][model_id]["kwargs"][key] = value
            

    def load_model(self, name: str):

        self.current_model_name = name
        
        kwargs = self.config["models"][name].get("kwargs") or {}
        id = self.config["models"][name].get("id")
        if id:
            from .models import API_MODELS
            load_model = API_MODELS[id]
        
        else:
            filepath = self.config["models"][name]["path"]

            dirpath, _ = os.path.split(filepath)
            dirpath = os.path.abspath(dirpath)
            
            if not dirpath in sys.path:
                module_name = os.path.basename(dirpath)
                sys.path.append(dirpath)
            

            module = load_module_from_path(module_name, filepath)
            load_model = getattr(module, "load_model")

        model = load_model(**kwargs)

        assert hasattr(model, "run") and inspect.isgeneratorfunction(getattr(model, "run")), f"invalid model. Must contain generator method 'run'"

        self.current_model_name = name
        self.model = model

    def get_models(self):
        return self.available_models
    
    def execute_api(self, api: APIConfig, data: dict):
        from pprint import pprint
        pprint({"api": api, "data": data})

        instruction = api.get("instruction", "")
        messages = []

        if instruction:
            instruction = render_template(instruction, data)
            messages.append({
                "role": "system",
                "content": instruction
            })
        
        prompt = render_template(api.get("template", ""), data)
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        chat: ChatCompletion = {
            "model": api.get("model", next(iter(self.get_models()))),
            "messages": messages,
            "temperature": api.get("temperature"),
            "stream": api.get("stream", False)
        }

        is_json = api.get("response_format", "") == "json"
        for item in self.run(**chat):
            if is_json:
                yield {"content": str_to_json(item.get("text", ""))}
            else:
                yield {"content": item.get("text", "")}
        
    def run(
            self,
            model:str,
            messages:list[dict],
            temperature: float,
            seed: int,
            top_p: float,
            top_k: int,
    )->Iterable[Response]:
        
        with self.lock:
            if self.current_model_name != model or self.model is None:
                self.load_model(model)
            
            yield from self.model.run(
                messages=messages, 
                temperature=temperature, 
                seed=seed, 
                top_p=top_p, 
                top_k=top_k
            )

