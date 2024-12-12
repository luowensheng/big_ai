from fastapi import FastAPI, Path
from fastapi.responses import StreamingResponse
import gradio as gr
import uvicorn
# import llama_cpp
from .model import ModelManager
import os
import yaml
from .default import API_DIR
from .config import APIConfig


class FastAPIGradio(FastAPI):
    def mount_gradio_page(self, path: str, blocks: gr.Blocks):
        gr.mount_gradio_app(self, blocks, path=path)
        



# app.mount_gradio_page("/chat", chat.create_interface(model_manager))
# app.mount_gradio_page("/create", create.create_interface(model_manager))


def create_api_server(config_path: str,host: str, port: int):

    app = FastAPIGradio()

    model_manager = ModelManager(config_path)

    @app.get("/api/models")
    def models():
        return {
            "success": True,
            "data": model_manager.get_models()
        }

    DEFAULT_MODEL = model_manager.get_models()[0]


    @app.post("/api/{id}")
    def handle( id: str = Path(..., regex="^[^/\\\]+$"), data: dict = {}):
        
        config_path = os.path.join(API_DIR, f"{id}.yaml")
        if not os.path.join(config_path):
            return {
                "success": False,
                "error": f"not found: {id}"
            }

        config: APIConfig = yaml.safe_load(open(config_path))
        try:
            output = model_manager.execute_api(config, data)
            output =  next(output)
            
            return output
        
        except Exception as e:
            return {
                "success": False,
                "error": f"Error {e}"
            }



    @app.post("/api/generate")
    def generate(data: dict):

        try:
            output = model_manager.run(data)
            return StreamingResponse(({"success": True, "data": result} for result in output), media_type="application/json")
        except Exception as e:
            return {
                "success": False,
                "error": f"Error {e}"
            }

    uvicorn.run(app, host=host, port=port)