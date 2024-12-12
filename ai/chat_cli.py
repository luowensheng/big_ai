import argparse
from .model import ModelManager
import random
from .default import DEFAULT_MODELS_CONFIG_PATH

def start_chat(config_path, model_id, instruction, message):
    history = []
    manager = ModelManager(config_path=config_path)

    available_models = manager.get_models()
    if not available_models:
        print("No models")
        return
    print("\nAvailable models: ", available_models, "\n")
    if not model_id:
        model_id = random.choice(available_models)
        print("\nNo model provided. Randomly selected: ", model_id, "\n")

    if instruction:
        history.append({
            "role": "system",
            "content": instruction
        })
        
    while True:
        
        while not message:
            message = input("User:\t") or ""
            message =  message.strip()
        
        if message == "/bye":
            break

        history.append({
            "role": "user",
            "content": message
        })

        reply = []
        
        print(f"\n{model_id}:\t", end="", flush=True)

        for response in manager.run(**{"messages": history, "model": model_id, "stream": True}):
            text = response["text"]
            reply.append(text)
            print(text, end="", flush=True)
        print("\n")

        history.append({
            "role": "assistant",
            "content": "".join(reply)
        })

        message = ""


def main():
    parser = argparse.ArgumentParser(description="CLI to load a model.")
    parser.add_argument("--model_id", "-m", default="", type=str, help="ID of the model to load")
    parser.add_argument("--config_path", type=str, default=DEFAULT_MODELS_CONFIG_PATH, help="Config Path")
    parser.add_argument("--instruction", type=str, default="", help="Instruction")
    parser.add_argument("--message", type=str, default="", help="Init Message")

    args = parser.parse_args()

    # Load the model

    start_chat(args.config_path, args.model_id, args.instruction, args.message)
    # Add further processing if necessary
    print("Model loaded successfully", args.model_id)

if __name__ == "__main__":


    main()
