# ──────────────────────────────────────────────────────────────────────────────
# Standard library
# ──────────────────────────────────────────────────────────────────────────────
import base64
import json
import logging
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Third‑party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from plotly.offline import get_plotlyjs_version

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Custom JSON encoder for NumPy scalars / arrays
# ──────────────────────────────────────────────────────────────────────────────
class NumpySafeJSONEncoder(json.JSONEncoder):
    """Serialize NumPy data types so that json.dumps succeeds."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────
def generate_html_report(
    amplicon_data: "AmpliconData",
    output_path: Union[str, Path],
    include_sections: List[str] | None = None,
) -> None:
    """
    Write an HTML debug page with interactive Plotly / Matplotlib figures.

    Parameters
    ----------
    amplicon_data
        Object containing `.figures`, a nested mapping of section → figure(s).
    output_path
        Destination *.html* file.
    include_sections
        Which of amplicon_data.figures' top‑level keys to include.
    """
    include_sections = include_sections or ["map", "ordination"]

    ts = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── structure figures into sections / tabs → plot metadata ───────────────
    sections_data = _prepare_sections(amplicon_data.figures, include_sections)

    # ── Plotly CDN tag (fallback to an old version if discovery fails) ───────
    try:
        plotly_js_version = get_plotlyjs_version()
    except Exception:
        plotly_js_version = "3.0.1"
    plotly_js_tag = (
        f'<script src="https://cdn.plot.ly/plotly-{plotly_js_version}.min.js"></script>'
    )

    # ── JSON serialisation of all figure payloads ────────────────────────────
    plot_data_json = json.dumps(
        sections_data["plot_data"], cls=NumpySafeJSONEncoder, ensure_ascii=False
    )
    # Do *not* let a stray "</script>" close the tag early
    plot_data_json = plot_data_json.replace("</", "<\\/")

    # ── build the section / subsection / tab HTML ────────────────────────────
    sections_html: list[str] = []
    for section in sections_data["sections"]:
        section_id    = section["id"]
        section_title = section["title"]

        subsections_html: list[str] = []
        for i, subsection in enumerate(section["subsections"]):
            tabs_html    = subsection["tabs_html"]
            buttons_html = subsection["buttons_html"]

            subsections_html.append(
                f"""
                <div class="subsection">
                  <h3>{subsection['title']}</h3>
                  <div class="tabs">{buttons_html}</div>
                  <div class="tab-content">{tabs_html}</div>
                </div>
                """
            )

        sections_html.append(
            f"""
            <div class='section' id='{section_id}'>
              <h2>{section_title}</h2>
              {''.join(subsections_html)}
            </div>
            """
        )

    # ── final HTML – header, styles, JSON payload, behaviour script ──────────
    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset='utf-8' />
  <title>16S Analysis Debug Report</title>
  {plotly_js_tag}
  <style>
    body          {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
    .section      {{ margin-bottom: 30px; padding-bottom: 20px; border-bottom: 1px solid #eee; }}
    .subsection   {{ margin-left: 20px; margin-bottom: 20px; }}
    .tabs         {{ display: flex; margin-bottom: -1px; flex-wrap: wrap; }}
    .tab-button   {{ padding: 8px 12px; background: #eee; border: 1px solid #ccc;
                     cursor: pointer; border-radius: 4px 4px 0 0; margin-right: 5px; margin-bottom: 5px; font-size: 0.9em; }}
    .tab-button.active {{ background: #fff; border-bottom: 1px solid #fff; }}
    .tab-content  {{ border: 1px solid #ccc; padding: 15px; border-radius: 0 4px 4px 4px; }}
    .plot-container {{ width: 900px; height: 600px; }}
    .error        {{ color: #d32f2f; padding: 10px; border: 1px solid #ffcdd2; background: #ffebee; }}
    .section-controls {{ margin: 10px 0; }}
    .section-button  {{ background: #f0f0f0; border: 1px solid #ddd; padding: 5px 10px;
                        cursor: pointer; border-radius: 4px; margin-right: 5px; }}
  </style>
</head>
<body>
  <h1>16S Amplicon Analysis – Debug Report</h1>
  <p>Generated: {ts}</p>
  <p>Sections shown: {', '.join(include_sections)}</p>

  <div class="section-controls">
    <button class="section-button" onclick="toggleAllSections(true)">Expand All</button>
    <button class="section-button" onclick="toggleAllSections(false)">Collapse All</button>
  </div>

  {''.join(sections_html)}

  <!-- ── raw plot metadata – safe from </script> because of type="application/json" ── -->
  <script id="plot-data" type="application/json">{plot_data_json}</script>

  <!-- ── behaviour ──────────────────────────────────────────────────────────────── -->
  <script>
    /* ---------- data ---------- */
    const plotData = JSON.parse(
        document.getElementById('plot-data').textContent);

    /* ---------- state ---------- */
    const initializedPlots = new Set();

    /* ---------- helpers ---------- */
    function renderPlot(containerId, plotId) {{
        const container = document.getElementById(containerId);
        if (!container) {{
            console.error('Container not found:', containerId);
            return;
        }}

        container.innerHTML = '';

        const plotDiv = document.createElement('div');
        plotDiv.id        = plotId;
        plotDiv.className = 'plot-container';
        container.appendChild(plotDiv);

        const data = plotData[plotId];
        if (!data) {{
            plotDiv.innerHTML = '<div class="error">Plot data not available</div>';
            return;
        }}

        if (data.type === 'plotly') {{
            if (data.layout) data.layout.showlegend = false;
            if (typeof Plotly === 'undefined') {{
                plotDiv.innerHTML = '<div class="error">Plotly library missing</div>';
                console.error('Plotly not loaded');
                return;
            }}
            Plotly.newPlot(plotId, data.data, data.layout)
                  .catch(err => {{
                      plotDiv.innerHTML = `<div class="error">Plotly error: ${{err}}</div>`;
                      console.error('Plotly error:', err);
                  }});
        }} else if (data.type === 'image') {{
            const img = document.createElement('img');
            img.src         = 'data:image/png;base64,' + data.data;
            img.style.maxWidth = '100%';
            plotDiv.appendChild(img);
        }} else if (data.type === 'error') {{
            plotDiv.innerHTML = `<div class="error">${{data.error}}</div>`;
        }} else {{
            plotDiv.innerHTML = '<div class="error">Unknown plot type</div>';
        }}
    }}

    /* ---------- tab switching ---------- */
    function showTab(tabId, plotId) {{
        const tabEl = document.getElementById(tabId);
        if (!tabEl) {{
            console.error('Tab element not found:', tabId);
            return;
        }}

        const subsection = tabEl.closest('.subsection');
        if (!subsection) {{
            console.error('Tab not inside a .subsection container:', tabId);
            return;
        }}

        subsection.querySelectorAll('.tab-pane')
                  .forEach(pane => pane.style.display = 'none');
        subsection.querySelectorAll('.tab-button')
                  .forEach(btn  => btn.classList.remove('active'));

        tabEl.style.display = 'block';
        const btn = subsection.querySelector(`.tab-button[data-tab="{{tabId}}"]`
                                              .replace('{{tabId}}', tabId));
        if (btn) btn.classList.add('active');

        if (!initializedPlots.has(plotId)) {{
            renderPlot(`container-${{plotId}}`, plotId);
            initializedPlots.add(plotId);
        }}
    }}

    /* ---------- section hide/show ---------- */
    function toggleSection(sectionId) {{
        const s = document.getElementById(sectionId);
        if (s) s.style.display = (s.style.display === 'none' ? 'block' : 'none');
    }}
    function toggleAllSections(show) {{
        document.querySelectorAll('.section')
                .forEach(s => (s.style.display = show ? 'block' : 'none'));
    }}

    /* ---------- first‑tab initialisation ---------- */
    document.addEventListener('DOMContentLoaded', () => {{
        document.querySelectorAll('.subsection').forEach(subsection => {{
            const firstTab = subsection.querySelector('.tab-pane');
            if (firstTab) {{
                const plotId = firstTab.dataset.plotId;
                showTab(firstTab.id, plotId);
            }}
        }});
    }});
  </script>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────
def _prepare_sections(figures_dict: Dict, include_sections: List[str]) -> Dict:
    """Flatten nested figure structure → UI sections + serialisable plot data."""
    sections: list[Dict[str, Any]] = []
    plot_data: Dict[str, Any] = {}

    ordination_methods = ["pca", "pcoa", "tsne", "umap"]

    for section_title in include_sections:
        if section_title not in figures_dict:
            continue

        section_data = {
            "id":   section_title.replace(" ", "_"),
            "title": section_title.title(),
            "subsections": [],
        }

        # ── ordination gets a subsection per method ───────────────────────────
        if section_title == "ordination":
            for method in ordination_methods:
                subsection_figures: Dict[str, Any] = {}
                # drill down: table‑type → level → method → colour → figure
                for table_type, levels in figures_dict[section_title].items():
                    for level, methods in levels.items():
                        if method in methods:
                            for colour, fig in methods[method].items():
                                key = f"{table_type} – {level} – {colour}"
                                subsection_figures[key] = fig
                if not subsection_figures:
                    continue

                tabs_html, buttons_html, subsection_plot_data = (
                    _prepare_figures_from_dict(subsection_figures)
                )
                plot_data.update(subsection_plot_data)
                section_data["subsections"].append(
                    {"title": method.upper(),
                     "tabs_html": tabs_html,
                     "buttons_html": buttons_html},
                )

        # ── any other section collapses everything into one subsection ───────
        else:
            subsection_figures: Dict[str, Any] = {}
            _flatten_figures(figures_dict[section_title], [], subsection_figures)

            if subsection_figures:
                tabs_html, buttons_html, subsection_plot_data = (
                    _prepare_figures_from_dict(subsection_figures)
                )
                plot_data.update(subsection_plot_data)
                section_data["subsections"].append(
                    {"title": "All",
                     "tabs_html": tabs_html,
                     "buttons_html": buttons_html},
                )

        if section_data["subsections"]:
            sections.append(section_data)

    return {"sections": sections, "plot_data": plot_data}


def _flatten_figures(
    figures_dict: Dict,
    parent_keys: List[str],
    out: Dict[str, Any],
) -> None:
    """Recursive walk that flattens arbitrarily nested dicts of figures."""
    for key, value in figures_dict.items():
        keys = parent_keys + [key]
        if isinstance(value, dict):
            _flatten_figures(value, keys, out)
        else:
            out[" – ".join(keys)] = value


def _prepare_figures_from_dict(
    figures: Dict[str, Any]
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Convert a flat {title: figure} mapping → HTML tabs/buttons + serialisable data
    """
    tabs: list[str]    = []
    buttons: list[str] = []
    plot_data: Dict[str, Any] = {}

    for i, (title, fig) in enumerate(figures.items()):
        tab_id  = f"tab-{i}"
        plot_id = f"plot-{i}"

        active = "active" if i == 0 else ""
        buttons.append(
            f'<button class="tab-button {active}" '
            f'data-tab="{tab_id}" '
            f'onclick="showTab(\'{tab_id}\', \'{plot_id}\')">{title}</button>'
        )
        tabs.append(
            f'<div id="{tab_id}" class="tab-pane" '
            f'style="display:{"block" if i==0 else "none"}" '
            f'data-plot-id="{plot_id}">'
            f'<div id="container-{plot_id}" class="plot-container"></div>'
            f'</div>'
        )

        if hasattr(fig, "to_plotly_json"):
            try:
                pj = fig.to_plotly_json()
                layout = pj.get("layout", {})
                layout["showlegend"] = False
                plot_data[plot_id] = {
                    "type":   "plotly",
                    "data":   pj["data"],
                    "layout": layout,
                }
            except Exception as exc:
                logger.exception("Serialising Plotly figure failed")
                plot_data[plot_id] = {"type": "error", "error": str(exc)}

        elif isinstance(fig, Figure):
            try:
                buf = BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight")
                buf.seek(0)
                plot_data[plot_id] = {
                    "type": "image",
                    "data": base64.b64encode(buf.read()).decode(),
                }
            except Exception as exc:
                logger.exception("Serialising Matplotlib figure failed")
                plot_data[plot_id] = {"type": "error", "error": str(exc)}

        else:
            plot_data[plot_id] = {
                "type": "error",
                "error": f"Unsupported figure type: {type(fig)}",
            }

    return "\n".join(tabs), "\n".join(buttons), plot_data
