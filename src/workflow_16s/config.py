# ===================================== IMPORTS ====================================== #

import os
import yaml
from pathlib import Path
from typing import Dict, Union

# ================================= GLOBAL VARIABLES ================================= #

DEFAULT_CONFIG_PATH = (
    Path(
        os.path.abspath(
            os.path.join(
                os.path.dirname(
                    os.path.abspath(__file__)
                ), "..", ".."
            )
        )
    ) 
    / "references" 
    / "config.yaml"
)

# ==================================== FUNCTIONS ===================================== #

def resolve_relative_paths(config: Dict, config_dir: Path) -> Dict:
    """Converts any relative paths in the configuration to absolute paths based on 
    the directory of the config file."""
    for key, value in config.items():
        if isinstance(value, str):
            # Check if the value is a relative path
            if value.startswith("./") or value.startswith("../"):
                # Convert relative path to absolute path
                config[key] = (config_dir / value).resolve()
        elif isinstance(value, dict):
            # Recursively handle nested dictionaries
            config[key] = resolve_relative_paths(value, config_dir)
    return config


def get_config(
    file_path: Union[str, Path] = DEFAULT_CONFIG_PATH
) -> Dict:
    # Load the YAML configuration file
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    
    # Resolve any relative paths in the config
    config_dir = Path(os.path.dirname(os.path.abspath(file_path)))
    config = resolve_relative_paths(config, config_dir)
    
    return config
