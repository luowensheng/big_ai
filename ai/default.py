import os

# CACHE_DIR = "/Volumes/STORAGE_2/models" #"./cache"
# if not os.path.exists(CACHE_DIR):
#     os.makedirs(CACHE_DIR)

APP_DIR = os.path.expanduser("~/.cache/ai")
if os.path.exists("./ai/assets"):
    ASSETS_DIR = os.path.abspath("./ai/assets")
else:
    ASSETS_DIR = os.path.join(APP_DIR, "assets")

os.makedirs(ASSETS_DIR, exist_ok=True)

DEFAULT_MODELS_CONFIG_PATH = os.path.join(ASSETS_DIR, "models.json")
if not os.path.exists(DEFAULT_MODELS_CONFIG_PATH):
    with open(DEFAULT_MODELS_CONFIG_PATH, "w") as f:
        f.write('{"models": {}}')

API_DIR = os.path.join(ASSETS_DIR, "apis")
os.makedirs(API_DIR, exist_ok=True)

# MODELS_DIR = os.path.join(ASSETS_DIR, "models")
# os.makedirs(MODELS_DIR, exist_ok=True)

TEMPERATURE: float = 0.6
TOP_P: float = 0.95 
TOP_K: int = 40 
MAX_NEW_TOKENS = 255
# min_p: float = 0.05 
# typical_p: float = 1, 
# stream: bool = False,
# stop: str | List[str] | None = [], 
# seed: int | None = None, 
# response_format: ChatCompletionRequestResponseFormat | None = None, 
# max_tokens: int | None = None, 
# presence_penalty: float = 0, 
# frequency_penalty: float = 0, 
# repeat_penalty: float = 1, 
# tfs_z: float = 1, 
# mirostat_mode: int = 0, 
# mirostat_tau: float = 5, 
# mirostat_eta: float = 0.1, 
# model: str | None = None, 