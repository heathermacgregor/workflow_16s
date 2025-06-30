# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import warnings
from typing import Any, Optional

# Third-Party Imports
from rich.progress import (
    BarColumn,
    Column,
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

DEFAULT_DESCRIPTION_STYLE = "white"
DEFAULT_BAR_COLUMN_COMPLETE_STYLE = "honeydew2"
DEFAULT_FINISHED_STYLE = "dark_cyan" 
DEFAULT_PROGRESS_PERCENTAGE_STYLE = "honeydew2"
DEFAULT_M_OF_N_COMPLETE_STYLE = "honeydew2"
DEFAULT_TIME_ELAPSED_STYLE = "light_sky_blue1"
DEFAULT_TIME_REMAINING_STYLE = "thistle1"

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
            style=DEFAULT_M_OF_N_COMPLETE_STYLE,
            justify="right"
        )

class TimeElapsedColumn(ProgressColumn):
    """Renders time elapsed."""
    
    def render(self, task: "Task") -> Text:
        """Show time elapsed."""
        elapsed = task.finished_time if task.finished else task.elapsed
        if elapsed is None:
            return Text("-:--:--", style=DEFAULT_TIME_ELAPSED_STYLE)
        delta = timedelta(seconds=max(0, int(elapsed)))
        return Text(str(delta), style=DEFAULT_TIME_ELAPSED_STYLE)

class TimeRemainingColumn(ProgressColumn):
    """Renders estimated time remaining."""

    # Only refresh twice a second to prevent jitter
    max_refresh = 0.5

    def __init__(
        self,
        compact: bool = False,
        elapsed_when_finished: bool = False,
        table_column: Optional[Column] = None,
    ):
        self.compact = compact
        self.elapsed_when_finished = elapsed_when_finished
        super().__init__(table_column=table_column)


    def render(self, task: "Task") -> Text:
        """Show time remaining."""
        if self.elapsed_when_finished and task.finished:
            task_time = task.finished_time
            style = DEFAULT_TIME_ELAPSED_STYLE
        else:
            task_time = task.time_remaining
            style = DEFAULT_TIME_REMAINING_STYLE

        if task.total is None:
            return Text("", style=style)

        if task_time is None:
            return Text("--:--" if self.compact else "-:--:--", style=style)

        # Based on https://github.com/tqdm/tqdm/blob/master/tqdm/std.py
        minutes, seconds = divmod(int(task_time), 60)
        hours, minutes = divmod(minutes, 60)

        if self.compact and not hours:
            formatted = f"{minutes:02d}:{seconds:02d}"
        else:
            formatted = f"{hours:d}:{minutes:02d}:{seconds:02d}"

        return Text(formatted, style=style)
        

def get_progress_bar(transient: bool = False) -> Progress:
    """Return a customized progress bar with consistent styling"""
    return Progress(
        SpinnerColumn(
            "dots", 
            style=DEFAULT_BAR_COLUMN_COMPLETE_STYLE, 
            speed=0.75
        ),
        TextColumn(
            task.description.ljust(DEFAULT_PROGRESS_TEXT_N), 
            style=DEFAULT_DESCRIPTION_STYLE,
            justify="left"
        ),
        MofNCompleteColumn(),
        BarColumn(
            bar_width=40,
            style="black", # Background color
            complete_style=DEFAULT_BAR_COLUMN_COMPLETE_STYLE,
            finished_style=DEFAULT_FINISHED_STYLE,
            #pulse_style="yellow"
        ),
        TextColumn(
            "[progress.percentage]{task.percentage:>3.0f}%".rjust(5), 
            style=DEFAULT_PROGRESS_PERCENTAGE_STYLE
        ),
        TextColumn(
            "E".rjust(2), 
            style=DEFAULT_TIME_ELAPSED_STYLE
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
