# import argparse
# from .default import CACHE_DIR

# def main():
#     parser = argparse.ArgumentParser(description="CLI to load a model.")
#     parser.add_argument("model_id", type=str, help="ID of the model to load")
#     parser.add_argument("--filename", type=str, default="*Q8_0.gguf", help="Filename to use")
#     parser.add_argument("--cache_dir", type=str, default=CACHE_DIR, help="Cache directory")
#     args = parser.parse_args()

#     # Load the model
#     load_model(path=args.model_id, loader="llamacpp" if not "mlx" in args.model_id.lower() else "mlx", filename=args.filename, cache_dir=args.cache_dir)
#     # Add further processing if necessary
#     print("Model loaded successfully", args.model_id)

# if __name__ == "__main__":


#     main()
