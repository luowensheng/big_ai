from typing import Any, Dict, List, Literal, Optional, TypedDict


RoleType = Literal["system", "user", "assistant"]

class Message(TypedDict):
    role: RoleType
    content: str
    name: Optional[str]  # Optional: name of the sender, e.g., "assistant"


class ChatCompletion(TypedDict, total=False):  # total=False makes all fields optional
    model: str  # The model to use (e.g., "gpt-4", "gpt-3.5-turbo", etc.)
    messages: List[Message]  # The list of messages in the conversation
    temperature: float  # Sampling temperature between 0 and 2
    top_p: float  # Nucleus sampling parameter
    frequency_penalty: float  # Penalty for frequent tokens (between -2.0 and 2.0)
    presence_penalty: float  # Penalty for tokens already mentioned (between -2.0 and 2.0)
    logit_bias: Dict[str, int]  # Modify the likelihood of specific tokens
    max_completion_tokens: int  # Maximum number of tokens for the completion
    n: int  # Number of completions to generate
    stream: bool  # If set to True, stream the response
    stop: Any  # Sequences where the API will stop generation
    logprobs: bool  # Whether to return log probabilities of output tokens
    top_logprobs: int  # Number of top logprobs to return for each token
    response_format: Dict[str, Any]  # Specify output format, such as JSON
    seed: int  # For deterministic results
