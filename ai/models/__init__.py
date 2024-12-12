from .claude import ClaudeModel
from .gemini import GeminiModel
from .groq import GroqModel
from .mistral import MistralModel
from .nvidia import NvidiaModel
from .ollama import OllamaModel
from .openai import OpenAIModel

API_MODELS = {
 "claude": ClaudeModel,
 "gemini": GeminiModel,
 "groq": GroqModel,
 "mistral": MistralModel,
 "nvidia": NvidiaModel,
 "ollama": OllamaModel,
 "openai": OpenAIModel,
}
