import argparse
from . import default 


def main():

    parser = argparse.ArgumentParser(description='CLI App Manager')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create new app command
    create_app_parser = subparsers.add_parser('api', help='Create a new application')
    create_app_parser.add_argument('--config_path', default=default.DEFAULT_MODELS_CONFIG_PATH, help='Name of the application')
    create_app_parser.add_argument('--host', default="localhost", type=str,  help='Host Name of the application')
    create_app_parser.add_argument('--port', '-p', default=4455, type=int, help='Port')


    # Create new page command
    create_page_parser = subparsers.add_parser('ui', help='Create a new page in an application')
    create_page_parser.add_argument('--config_path', default=default.DEFAULT_MODELS_CONFIG_PATH, help='Name of the application')
    create_page_parser.add_argument('--host', default="localhost", type=str,  help='Host Name of the application')
    create_page_parser.add_argument('--port', '-p', default=4455, type=int, help='Port')


    server_parser = subparsers.add_parser("chat", help='CLI App Manager')
    server_parser.add_argument('--model_id', type=str,  help='Model ID')
    server_parser.add_argument('--instruction', '-i', default="", type=str, help='Instruction')
    server_parser.add_argument('--config_path', default=default.DEFAULT_MODELS_CONFIG_PATH, help='Name of the application')

    server_parser.add_argument('--message', '-m', type=str, default="", help='Message')

    
    args = parser.parse_args()

    match args.command:

        case 'ui':
            from .gradio_ui import create_interface
            create_interface(args.config_path, args.host, args.port)

        case 'api':
            from .api import create_api_server
            create_api_server(args.config_path, args.host, args.path)

        case "chat":
            from .chat_cli import start_chat
            start_chat(args.config_path)

        case _:
            print(f"Command not found: '{args.command}'")
            parser.print_help()

if __name__ == '__main__':
    main()