# ===================================== IMPORTS ====================================== #

# Third-Party Imports
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    track,
    TaskID
)

# ==================================== FUNCTIONS ===================================== #

def create_progress() -> Progress:
    return Progress(
        SpinnerColumn(style="green", finished_style="green bold"),
        TextColumn("[blue]{task.description}"),
        BarColumn(
            bar_width=40,
            complete_style="yellow",
            finished_style="green",
            pulse_style="yellow",
            style="black"
        ),
        TextColumn("[cyan]{task.completed}/{task.total}"),
        TextColumn("[magenta]Elapsed: {task.elapsed:.0f}s"),
        TextColumn("[red]Remaining: {task.time_remaining:.0f}s"),
        expand=False
    )
