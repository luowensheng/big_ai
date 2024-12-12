from ..utils import assert_library_exists

assert_library_exists(
"mlx_lm", "Missing 'mlx_lm' package. Please install with this commad: pip install mlx-lm"
)

assert_library_exists(
    "huggingface_hub",
    "Missing 'huggingface_hub' package. Please install with this commad: pip install huggingface-hub"
)

import huggingface_hub.constants as constants
from .default import CACHE_DIR

from time import perf_counter
from typing import Any
import os




class MLXModel:
    def __init__(self, path: str, **kwargs) -> None:
        constants.HF_HUB_CACHE = CACHE_DIR
        from mlx_lm import load

        self.model, self.tokenizer = load(path, **kwargs)
        self.path = path
        assert hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template is not None, f"model does not have 'apply_chat_template' method"
        
    def run(self, messages, **kwds: Any) -> Any:
        from mlx_lm import generate
        from mlx_lm.utils import stream_generate

        stream = kwds.get("stream")
 

        model_id, _ = os.path.splitext(self.path.split("/")[-1])
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        start = perf_counter()

        if not stream:

            outputs = generate(
                self.model,
                    tokenizer=self.tokenizer,
                    prompt=prompt,
                    max_tokens=kwds.get("max_completion_tokens", 256),
                    top_p=kwds.get("top_p", 50),
                    repetition_penalty=kwds.get("presence_penalty", 1.1), #1.1,
                )
            
            yield {"text": outputs, "elapsed_time": perf_counter() - start, "model_id": model_id}
        
        else:
            outputs = stream_generate(
                self.model,
                    tokenizer=self.tokenizer,
                    prompt=prompt,
                    max_tokens=kwds.get("max_completion_tokens", 256),
                    top_p=kwds.get("top_p", 50),
                    repetition_penalty=kwds.get("presence_penalty", 1.1), #1.1,
                )
            
            for chunk in outputs:
                end = perf_counter()
                yield {"text": chunk, "elapsed_time": end - start, "model_id": model_id}
                start = end


def post_process_reply(completion: list[dict]):
    content = completion["choices"][-1]
    return {
        "text": content["message"],
        "finish_reason": content["finish_reason"],
        "usage": completion["usage"]
    }
        