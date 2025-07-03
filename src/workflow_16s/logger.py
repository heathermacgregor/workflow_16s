# ===================================== IMPORTS ====================================== #

# Standard Library
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Union

# 3rd‑party (Rich)
from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# Local
from workflow_16s.utils.dir_utils import SubDirs  # (if you still need this)

# ==================================== FUNCTIONS ===================================== #

def setup_logging(
    log_dir_path: Union[str, Path],
    log_filename: Union[str, None] = None,
    max_file_size: int = 5 * 1024 * 1024,  # 5 MB
    backup_count: int = 3,
    console_level: int = logging.INFO,     # console shows INFO+
    file_level: int = logging.DEBUG        # file keeps DEBUG+
) -> logging.Logger:
    """
    Configure workflow_16s logging with:
      • colourful Rich console output (custom theme)
      • rotating file handler for full DEBUG logs
    """
    # ───────────────────── log‑file path ──────────────────────
    log_dir_path = Path(log_dir_path)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    if log_filename is None:
        log_filename = datetime.now().strftime("%Y-%m-%d_%H%M%S.log")
    log_file_path = log_dir_path / log_filename

    # ─────────────────── root / package logger ─────────────────
    logger = logging.getLogger("workflow_16s")
    logger.setLevel(logging.DEBUG)                     # keep everything

    # ───────────────────────── FILE …──────────────────────────
    file_handler = RotatingFileHandler(
        filename=log_file_path,
        maxBytes=max_file_size,
        backupCount=backup_count,
        encoding="utf‑8",
    )
    file_handler.setLevel(file_level)
    file_fmt = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s:%(filename)s:%(funcName)s(): %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_fmt)

     # 1. Use EXACT theme keys required by Rich
    custom_theme = Theme({
        "logging.time": "bold white",
        "logging.level.debug": "dim cyan",
        "logging.level.info": "bold white",  # Now white instead of blue
        "logging.level.warning": "bold yellow",
        "logging.level.error": "bold red",
        "logging.level.critical": "reverse bold bright_white on red",
        "logging.name": "bold white",
        "logging.function": "bold white",  # Note: must be "function" not "func"
        "logging.message": "white",
    })
    console = Console(theme=custom_theme)

    # 2. Configure RichHandler with correct parameters
    rich_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        level=console_level,
        show_time=True,           # Must be True to show timestamp
        show_path=False,
        markup=False,             # Disable markup interpretation
        log_time_format="[%X]"    # Time format matching your theme
    )

    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(rich_handler)  # Uses built-in formatting

    logger.info("Logging initialised → %s", log_file_path)
    return logger
