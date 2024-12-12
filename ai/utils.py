import json
import importlib.util
from jinja2 import  Template
import platform

def get_os_name():
    system = platform.system().lower()
    if "linux" in system:
        return "linux"
    elif "darwin" in system:
        return "mac"
    elif "windows" in system:
        return "windows"
    else:
        return "other"
    
def assert_library_exists(library_name: str, message: str):
    assert importlib.util.find_spec(library_name) is not None, message


def render_template(template:str, data: dict):
    return Template(template).render(data)

def load_module_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_id_from_filename(filename: str):
    return "/".join(filename.split("--")[1:]) #.lower()

def str_to_json(json_string: str, is_list: bool=False)-> dict:

    json_string = json_string.replace("\n", "").strip("```")
    json_string = json_string.strip("```json")
    start_char = "{" if not is_list else "["
    end_char = "}" if not is_list else "]"

    start_index = json_string.index(start_char)

    end_index = len(json_string) -1

    while json_string[end_index] != end_char:
        end_index -= 1

        assert end_index > 0

    valid_json = json_string[start_index:end_index+1]

    return json.loads(valid_json)