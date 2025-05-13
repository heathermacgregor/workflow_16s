# ===================================== IMPORTS ====================================== #

from pathlib import Path
from typing import Union

from datetime import datetime

import logging
from logging.handlers import RotatingFileHandler

from workflow_16s.utils.dir_utils import SubDirs

# ==================================== FUNCTIONS ===================================== #

def setup_logging(
    log_dir_path : Union[str, Path],
    log_filename : Union[str, None] = None,
    max_file_size: int = 5 * 1024 * 1024,  # 5 MB
    backup_count : int = 3
) -> logging.Logger:
    """
    Sets up logging for the entire package.
    
    Args:
        project_dir:   Root directory of the project.
        log_dir:       Subdirectory for storing logs.
        log_filename:  Log file name. Defaults to current datetime.
        max_file_size: Maximum size of log files before rotating. Defaults to 5 MB.
        backup_count:  Number of backup log files to keep.
        
    Returns:
        logger:        Configured logger for the package.
    """
    # Set log filename
    if log_filename is None:
        log_filename = datetime.now().strftime("%Y-%m-%d_%H:%M:%S.log")
    log_file_path = log_dir_path / log_filename

    # Configure root logger
    logger = logging.getLogger('workflow_16s')
    logger.setLevel(logging.DEBUG)

    # File handler (rotating)
    file_handler = RotatingFileHandler(
        filename    = log_file_path,
        maxBytes    = max_file_size,
        backupCount = backup_count
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_formatter = logging.Formatter(
        '%(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)

    # Attach handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    print(f"Logging initialized. Logs will be stored in {log_file_path}.")
    return logger
