# ===================================== IMPORTS ====================================== #

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Union

# ================================= GLOBAL VARIABLES ================================= #

DEFAULT_CONFIG_PATH = (
    Path(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))) 
    / "references" 
    / "config.yaml"
)

# ==================================== FUNCTIONS ===================================== #

def get_config(
    file_path: Union[str, Path] = DEFAULT_CONFIG_PATH
) -> Dict:
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config



    
