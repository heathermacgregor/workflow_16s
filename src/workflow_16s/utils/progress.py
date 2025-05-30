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
        # Green spinner (turns bold green when finished)
        SpinnerColumn(style="green"),
        
        # White task description text
        TextColumn("[progress.description]{task.description}", style="white"),
        
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
