# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import warnings
from typing import Any

# Third-Party Imports
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    track, 
)
from rich.text import Text

# ========================== INITIALIZATION & CONFIGURATION ========================== #

warnings.filterwarnings("ignore") # Suppress warnings

# ================================= GLOBAL VARIABLES ================================= #

DEFAULT_PROGRESS_TEXT_N = 65

# ==================================== FUNCTIONS ===================================== #

def create_progress() -> Progress:
    return Progress(
        # Green spinner (turns bold green when finished)
        SpinnerColumn(style="green"),
        
        # White task description text
        TextColumn(
            "[progress.description]{task.description}".ljust(DEFAULT_PROGRESS_TEXT_N), 
            style="white"
        ),
        
        # Bar: yellow filling, green when complete, black background
        BarColumn(
            bar_width=40,
            complete_style="yellow",
            finished_style="green",
            style="black"  # Background color
        ),
        
        # Cyan "M/N" counter
        MofNCompleteColumn(),#style="cyan"),
        
        # Magenta elapsed time
        TimeElapsedColumn(),#style="magenta"),
        
        # Red remaining time estimate
        TimeRemainingColumn(),#style="red"),
        expand=False
    )

# ============================== CUSTOM PROGRESS COLUMN ============================== #

class MofNCompleteColumn(ProgressColumn):
    """Renders completed count/total (e.g., '3/10') with bold styling"""
    
    def render(self, task: Task) -> Text:
        """Render the progress count as 'completed/total'"""
        return Text(
            f"{task.completed}/{task.total}".rjust(10),
            style="cyan",
            justify="right"
        )
        

def get_progress_bar(transient: bool = False) -> Progress:
    """Return a customized progress bar with consistent styling"""
    return Progress(
        SpinnerColumn(
            "dots", 
            style="bright_green", 
            speed=0.75
        ),
        TextColumn(
            "[white]{task.description}".ljust(DEFAULT_PROGRESS_TEXT_N), 
            justify="left"
        ),
        MofNCompleteColumn(),
        BarColumn(
            bar_width=40,
            style="black", # Background color
            complete_style="bright_yellow",
            finished_style="green",
            #pulse_style="yellow"
        ),
        TextColumn(
            "[progress.percentage]{task.percentage:>3.0f}%".rjust(5), 
            style="bright_green"
        ),
        TextColumn(
            "E".rjust(2), 
            style="yellow"
        ),
        TimeElapsedColumn(),
        TextColumn(
            "R".rjust(2), 
            style="cyan"
        ),
        TimeRemainingColumn(),
        transient=transient,
        expand=False
    )
