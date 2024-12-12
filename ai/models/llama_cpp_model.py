from ..utils import assert_library_exists
from .. import default 

assert_library_exists(
"llama_cpp", 
"""
Missing 'llama_cpp' library. 
To install, please visit to 'https://github.com/abetlen/llama-cpp-python'

Here is some useful installation for your system:
                      
Environment Variables
# Linux and Mac
CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" \
  pip install llama-cpp-python
# mac (Metal)
To install with Metal (MPS), set the GGML_METAL=on environment variable before installing:

CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python
# Windows
$env:CMAKE_ARGS = "-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"
pip install llama-cpp-python
CLI / requirements.txt
Supported Backends
Below are some common backends, their build commands and any additional environment variables required.

OpenBLAS (CPU)
To install with OpenBLAS, set the GGML_BLAS and GGML_BLAS_VENDOR environment variables before installing:

CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" pip install llama-cpp-python
CUDA
To install with CUDA support, set the GGML_CUDA=on environment variable before installing:

CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python  

                              
""")

from llama_cpp import Llama
import os
from . import default

import llama_cpp

LLAMA_CPP = llama_cpp

class LLamaCPPModel:
    def __init__(self, path: str, **kwargs) -> None:
        if os.path.exists(path):
            self.model = Llama(
                model_path=path,
                **kwargs
            )
        else:
            
            kwargs["cache_dir"] = kwargs.get("cache_dir", default.CACHE_DIR)
            kwargs["verbose"] = kwargs.get("verbose", False)
            
            self.model = Llama.from_pretrained(
                repo_id=path,
                **kwargs
            )
    
    def run(self, messages: list, **kwds):

        stream = kwds.get("stream") or False

        completions = self.model.create_chat_completion(
            messages=messages,
            temperature = kwds.get("temperature", default.TEMPERATURE), #: float = 0.2, 
            top_p = kwds.get("top_p", default.TOP_P), #: float = 0.95, 
            top_k = kwds.get("top_k", default.TOP_K), #: int = 40, 
            # min_p = kwds.get("min_p"), #: float = 0.05, 
            # typical_p = kwds.get("typical_p"), #: float = 1, 
            stream = stream, #: bool = False, 
            stop = kwds.get("stop"), #: str | List[str] | None = [], 
            seed = kwds.get("seed"), #: int | None = None, 
            response_format = kwds.get("response_format"), #: ChatCompletionRequestResponseFormat | None = None, 
            max_tokens = kwds.get("max_new_tokens", default.MAX_NEW_TOKENS), #: int | None = None, 
            # presence_penalty = kwds.get("presence_penalty"), #: float = 0, 
            # frequency_penalty = kwds.get("frequency_penalty"), #: float = 0, 
            # repeat_penalty = kwds.get("repeat_penalty"), #: float = 1, 
        )

        if not stream:
            yield post_process_reply(completions)
        
        else:
            for completion in completions:
                yield post_process_stream_reply(completion)


def post_process_reply(completion: llama_cpp.CreateChatCompletionResponse):
    content = completion["choices"][-1]
    return {
        "text": content["message"]["content"],
        "finish_reason": content["finish_reason"],
    }

def post_process_stream_reply(completion: llama_cpp.CreateChatCompletionStreamResponse):
    content = completion["choices"][-1]
    return {
        "text": content["delta"].get("content", ""),
        "finish_reason": content["finish_reason"],
    }