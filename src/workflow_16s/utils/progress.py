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
        # Spinner with default style (we'll customize through markup)
        SpinnerColumn(),
        
        # Blue task description using markup instead of style parameter
        TextColumn("[blue][progress.description]{task.description}"),
        
        # Bar with color customization through markup
        BarColumn(bar_width=40),
        
        # M/N complete column using markup
        TextColumn("[cyan]{task.completed}/{task.total}"),
        
        # Time columns using markup
        TextColumn("[magenta]⏱ {task.elapsed:.0f}s"),
        TextColumn("[red]⏳ {task.time_remaining:.0f}s"),
        expand=False,
        
        # Customize bar colors through the Progress styles parameter
        styles={
            "bar.complete": "yellow",
            "bar.finished": "green",
            "bar.pulse": "yellow",
            "bar.back": "black"
        }
    )
