from pprint import pprint
import base64
import gradio as gr
from gradio.components.chatbot import FileMessage
from gradio.data_classes import FileData
from .model import ModelManager
from .default import API_DIR, DEFAULT_MODELS_CONFIG_PATH
import os


CHATBOT_HEIGHT = 555


class ModelAdapter:
    def __init__(self, config_path: str):
        self.manager = ModelManager(config_path)

    def send_message(self, 
                     model: str, 
                     ai_history: list[gr.MessageDict], 
                     new_message: str, 
                     instruction, 
                     temp: float, 
                     seed_value: int, 
                     top_p_value: float, 
                     top_k_value: int, 
                     image_input: str):
        
        if not new_message:
            yield ai_history , "", image_input
            return

        new_message = new_message.strip()
        
        yield ai_history, new_message, None

        if image_input:
            assert isinstance(image_input, str), f"TYPE: {type(image_input)}"
            content = new_message
            ai_history.append({"role": "user",  "content": content})
            filename = os.path.split(image_input)[-1]
            ext = os.path.splitext(image_input)[-1].lower().strip(".").strip()
            ai_history.append({"role": "user",  "content": FileMessage(file=FileData(path=image_input, mime_type=f"image/{ext}"), alt_text=filename,)})

        else:
            content = new_message
            ai_history.append({"role": "user",  "content": content})

        yield ai_history, new_message, None


        messages = []
        instruction and messages.append({"role": "system", "content": instruction, })

        for _, history_message in enumerate(ai_history):

            if isinstance(history_message["content"], str):
                messages.append({"role": history_message["role"], "content": history_message["content"]})
                continue
            
            assert isinstance(history_message["content"], FileMessage)
            image_path = history_message["content"].file.path

                            
            assert image_path, f"No Path for image: '{image_path}'"
            
            imageb64 = base64.b64encode(open(image_path, 'rb').read()).decode("utf8")
            ext = os.path.splitext(image_path)[-1].lower().strip(".").strip()
            imageb64_url = f'data:image/{ext};base62,{imageb64}'
            i = len(messages)

            if i - 1 >= 0 and messages[i-1]["role"] == "user":
                last_message = messages[i-1]["content"]
                if isinstance(last_message, str):
                    last_message = [{"type": "text", "text": last_message}]

                last_message.append({"type": "image_url", "image_url": {"url": imageb64_url}})

                messages[i-1]["content"] = last_message
                continue
            

        ai_history.append({"role": "assistant", "content": ""})
        for reply in self.manager.run(
            model=model,
            messages=messages,
            temperature=temp,
            seed=seed_value,
            top_p=top_p_value,
            top_k=top_k_value,
        ):
            ai_history[-1]["content"] += reply["text"]
            yield ai_history, new_message, None

    def resend_message(self, model: str, ai_history: list[list[str]], instruction, temp: float, seed_value: int, top_p_value: float, top_k_value: int, custom_params):
        
        if not ai_history:
            yield ai_history, message
            return
        message = ai_history[-1][0]
        ai_history = ai_history[:-1]
        
        yield from self.send_message(model, ai_history, message, instruction, temp, seed_value, top_p_value, top_k_value, custom_params)


def create_interface(config_path: str, host: str, port: int):
    
    adapter = ModelAdapter(config_path)
    MODELS = adapter.manager.get_models()
    # Create the Gradio interface
    with gr.Blocks(fill_height=True) as app:

        with gr.Tab("Chat"):
        
            with gr.Row():
                with gr.Column(scale=1):
                    # Model selection dropdown
                    model_select = gr.Dropdown(
                        label="Select Model", 
                        choices=[""]+MODELS,  # Example models
                        value="" if not MODELS else MODELS[0]
                    )

                    with gr.Accordion("Config", open=False):
                        instructions = gr.Textbox(label="Instructions", lines=5)

                        # Temperature slider
                        temperature = gr.Slider(
                            label="Temperature", 
                            minimum=0, 
                            maximum=2, 
                            step=0.01, 
                            value=1
                        )

                        # Seed input field
                        seed = gr.Number(
                            label="Seed", 
                            value=42
                        )

                        # Top_p slider
                        top_p = gr.Slider(
                            label="Top P", 
                            minimum=0, 
                            maximum=1, 
                            step=0.00001, 
                            value=0.9
                        )

                        # Top_k slider
                        top_k = gr.Slider(
                            label="Top K", 
                            minimum=1, 
                            maximum=100, 
                            step=1, 
                            value=50
                        )


                with gr.Column(scale=3):
                    ai_chatbot = gr.Chatbot(label="Chat", height=CHATBOT_HEIGHT, show_copy_button=True, type="messages") 
                    with gr.Row():
                        with gr.Column(scale=6):
                            with gr.Group():
                                with gr.Row(equal_height=True):
                                    ai_textbox = gr.Textbox(interactive=True, label="Message", scale=4, lines=4)
                                    with gr.Column(scale=1, min_width=0):
                                        image_input = gr.Image( min_width=0,type="filepath", height=170) 
                                        btn = gr.Button("X", min_width=0)
                                        btn.click(lambda: None, outputs=image_input)

                        with gr.Column(scale=1, min_width=5):
                            ai_send_btn = gr.Button("Send")

                            clear_btn = gr.Button("Clear")
                            retry_btn = gr.Button("Retry")

        with gr.Tab("Models"):

                
                with gr.Column(scale=3):
                    save_btn = gr.Button("Save")
                    models_text_area = gr.Textbox(value=open(DEFAULT_MODELS_CONFIG_PATH).read(), label="Edit MODELS", lines=25)

                def update_models(content: str):
                    with open(DEFAULT_MODELS_CONFIG_PATH, 'w') as f:
                        f.write(content)

                save_btn.click(update_models, models_text_area)
            
        with gr.Tab("APIs"):

            with gr.Row(min_height=100):
                with gr.Column(scale=1):
                    current_filename = gr.State("")
                    FILES = get_files()
                    api = gr.Dropdown(label="Select Existing API", choices=[""]+FILES)
                    with gr.Group():
                        with gr.Column():
                            gr.Markdown("Create New API")
                            with gr.Row():
                                new_api = gr.Textbox(label="API Name", placeholder="Snake case name")
                                create_new = gr.Button("Create")
                    
                
                with gr.Column(scale=3):
                    text_area = gr.Textbox(label="Edit YAML", lines=25)

                create_new.click(lambda x: write_file(x, "") or "", new_api, new_api).then(
                    lambda: gr.Dropdown.update(choices=[""]+get_files())
                )
                text_area.change(write_file, [current_filename, text_area])
                api.change(lambda x: (open(os.path.join(API_DIR, f"{x}.yaml")).read(), x), api, [text_area, current_filename])
            

        # Define the function to run when the button is clicked
        clear_btn.click(lambda : [], outputs=[ai_chatbot])
        ai_send_btn.click(adapter.send_message, inputs=[model_select, ai_chatbot, ai_textbox, instructions, temperature, seed, top_p, top_k, image_input], outputs=[ai_chatbot, ai_textbox, image_input]).then(lambda: "", outputs=ai_textbox)
        
        ai_textbox.submit(adapter.send_message, inputs=[model_select, ai_chatbot, ai_textbox, instructions, temperature, seed, top_p, top_k, image_input], outputs=[ai_chatbot, ai_textbox, image_input]).then(lambda: "", outputs=ai_textbox)

        retry_btn.click(adapter.resend_message, inputs=[model_select, ai_chatbot, instructions, temperature, seed, top_p, top_k, image_input], outputs=[ai_chatbot, ai_textbox, image_input])


    app.launch(server_name=host, server_port=port)


def get_files():
    return [x.strip(".yaml") for x in os.listdir(API_DIR) if x.endswith(".yaml")]

def write_file(api_id: str, content: str):
    import yaml
    yaml.safe_load(content)
    with open(os.path.join(API_DIR, f"{api_id}.yaml"), 'w') as f:
        f.write(content)