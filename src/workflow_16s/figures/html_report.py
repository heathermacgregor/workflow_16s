from pathlib import Path
from typing import Union
import pandas as pd

def generate_html_report(amplicon_data: "AmpliconData",
                         output_path: Union[str, Path],
                         width: int = 900,
                         height: int = 600) -> None:
    """
    Write an HTML page displaying only the first sample‑map Plotly figure,
    resized to `width` × `height` pixels.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ts = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    fig_html = "<p><strong>No sample‑map figure found.</strong></p>"
    if isinstance(amplicon_data.figures, dict) and "map" in amplicon_data.figures:
        for _col, f in amplicon_data.figures["map"].items():
            if f is not None and hasattr(f, "to_html"):
                # ── resize ──────────────────────────────────────────────────────────
                f.update_layout(width=width, height=height)
                fig_html = f.to_html(full_html=False, include_plotlyjs="cdn")
                break

    html_doc = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>16S Sample‑Map Debug</title>
  <style>body{{font-family:Arial,sans-serif;margin:40px;}}</style>
</head>
<body>
  <h1>16S Amplicon Analysis – Sample‑Map Debug</h1>
  <p>Generated: {ts}</p>
  {fig_html}
</body>
</html>"""

    output_path.write_text(html_doc, encoding="utf-8")
