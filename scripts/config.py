import json
import os


def load_config():
    # Get the root directory of the project (assuming this script is located in 'scripts' folder)
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Construct the path to the 'config/config.json' file
    config_path = os.path.join(root_dir, 'config', 'config.json')

    # Check if the config file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    # Load the JSON config
    with open(config_path, 'r') as file:
        config = json.load(file)

    return config