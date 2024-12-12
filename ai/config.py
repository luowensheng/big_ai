from typing import Literal, TypedDict


class InputType(TypedDict):
    type: str

class APIConfig(TypedDict):
    id: str
    temperature: int
    inputs: dict[str, InputType]
    response_format: Literal["json"]
    instruction: str
    template: str


class APIRequestParams(TypedDict):
    model: str
    params: dict