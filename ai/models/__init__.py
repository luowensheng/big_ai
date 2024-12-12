from .claude import ClaudeModel
from .gemini import GeminiModel
from .groq import GroqModel
from .mistral import MistralModel
from .nvidia import NvidiaModel
from .ollama import OllamaModel
from .openai import OpenAIModel


def load_hf(*args, **kwargs):
    from .hf import LLMModel
    return LLMModel(*args, **kwargs)


def load_hfmm(*args, **kwargs):
    from .hf import MLLMModel
    return MLLMModel(*args, **kwargs)

def load_mlx(*args, **kwargs):
    from .mlx_model import MLXModel
    return MLXModel(*args, **kwargs)


def load_llamacpp(*args, **kwargs):
    from .llama_cpp_model import LLamaCPPModel
    return LLamaCPPModel(*args, **kwargs)


API_MODELS = {
 "claude": ClaudeModel,
 "gemini": GeminiModel,
 "groq": GroqModel,
 "mistral": MistralModel,
 "nvidia": NvidiaModel,
 "ollama": OllamaModel,
 "openai": OpenAIModel,
 "llamacpp": load_llamacpp,
 "mlx": load_mlx,
 "hf": load_hf,
 "hf-mm": load_hfmm
}
