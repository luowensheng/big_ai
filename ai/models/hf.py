from ..utils import assert_library_exists

assert_library_exists("torch", "Missing 'torch' library. To install, please visit to 'https://pytorch.org/get-started/locally/'")
assert_library_exists("transformers", "Missing 'transformers' library. Install using 'pip install --upgrade transformers'")

import torch
import gc
from transformers import pipeline, TextStreamer


def clear_memory():
  gc.collect()
  torch.cuda.empty_cache()

class MLLMModel:
    def __init__(self, **kwargs):
        from transformers import MllamaForConditionalGeneration, AutoProcessor
        clear_memory()

        model_id = kwargs["model_id"]
        token=kwargs.get("hf_token")
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=getattr(torch, kwargs.get("torch_dtype", "bfloat16")),
            device_map=kwargs.get("device_map", "auto"),
            token=token,
        )
        tie_weights = kwargs.get("tie_weights")
        if tie_weights is None or tie_weights:
            self.model.tie_weights()

        self.processor = AutoProcessor.from_pretrained(model_id, token=token)
        clear_memory()

    def run(self, messages: list[dict], *args, **kwargs):
        stream = kwargs.get("stream")
        streamer = None
        if stream:
            streamer = TextStreamer(self.model.tokenizer)

        try:

            processed_messages = []
            for message in messages:
                processed_messages.append(message)
            
            input_text = self.processor.apply_chat_template(processed_messages, add_generation_prompt=True)

            inputs = self.processor(
                text=input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(self.model.device)

            outputs = self.model.generate(
                **inputs, 
                streamer=streamer, 
                max_new_tokens=kwargs.get("max_new_tokens", 200)
            )

            if not stream:
                text = self.processor.decode(outputs[0])
                yield {"text": text}
                return

            for output in outputs:
                chunk = output["generated_text"]
                yield {"text": chunk}

        finally:
            clear_memory()


class LLMModel:
    def __init__(self, **kwargs):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        clear_memory()
        token=kwargs.get("hf_token")
        model_id = kwargs["model_id"]
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=kwargs.get("torch_dtype", "auto"),
            device_map=kwargs.get("device_map", "auto"),
            token=token
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)

        clear_memory()

    def run(self, messages: list[dict], *args, **kwargs):

        try:

            processed_messages = []
            for message in messages:
                if not isinstance(message['content'], str):
                    continue
                processed_messages.append(message)
            
            text = self.tokenizer.apply_chat_template(
                processed_messages, 
                tokenize=False,
                add_generation_prompt=True
            )

            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)


            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=kwargs.get("max_new_tokens", 200)
            )
            
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            yield {"text": text}

        finally:
            clear_memory()

