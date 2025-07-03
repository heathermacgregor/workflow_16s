# ──────────────────────────────────────────────────────────────────────────────
# Standard library
# ──────────────────────────────────────────────────────────────────────────────
import base64
import itertools
import json
import logging
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

# Third‑party
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from plotly.offline import get_plotlyjs_version

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# NumPy‑aware JSON encoder
# ──────────────────────────────────────────────────────────────────────────────
class NumpySafeJSONEncoder(json.JSONEncoder):
    def default(self, obj):  # noqa: D401
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
# Public API
# ──────────────────────────────────────────────────────────────────────────────
def generate_html_report(
    amplicon_data: "AmpliconData",
    output_path: Union[str, Path],
    include_sections: List[str] | None = None,
) -> None:
    """Write an HTML file with interactive Plotly/Matplotlib figures."""
    include_sections = include_sections or ["map", "ordination"]
    ts = pd.Timestamp.now().strftime("%Y‑%m‑%d %H:%M:%S")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── flatten the figure tree → sections + serialisable plot data ───────────
    id_counter = itertools.count()
    sections, plot_data = _prepare_sections(
        amplicon_data.figures, include_sections, id_counter
    )

    # ── CDN tag for Plotly ────────────────────────────────────────────────────
    try:
        plotly_ver = get_plotlyjs_version()
    except Exception:
        plotly_ver = "3.0.1"
    plotly_js_tag = (
        f'<script src="https://cdn.plot.ly/plotly-{plotly_ver}.min.js"></script>'
    )

    # ── JSON payload (escape "</" so it can never close the <script>) ────────
    payload = json.dumps(plot_data, cls=NumpySafeJSONEncoder, ensure_ascii=False)
    payload = payload.replace("</", "<\\/")  # safety

    # ── build the nested HTML bits ────────────────────────────────────────────
    sections_html = "\n".join(_section_html(s) for s in sections)

    # ── the full document ─────────────────────────────────────────────────────
    html = _HTML_TEMPLATE.format(
        plotly_js_tag=plotly_js_tag,
        generated_ts=ts,
        section_list=", ".join(include_sections),
        sections_html=sections_html,
        plot_data_json=payload,
    )
    output_path.write_text(html, encoding="utf‑8")


# ──────────────────────────────────────────────────────────────────────────────
# Template (in one place so it’s easy to scan)
# ──────────────────────────────────────────────────────────────────────────────
_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>16S Analysis Debug Report</title>
  {plotly_js_tag}
  <style>
    body              {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
    .section          {{ margin-bottom: 30px; padding-bottom: 20px; border-bottom: 1px solid #eee; }}
    .subsection       {{ margin-left: 20px; margin-bottom: 20px; }}
    .tabs             {{ display: flex; margin-bottom: -1px; flex-wrap: wrap; }}
    .tab-button       {{ padding: 8px 12px; background: #eee; border: 1px solid #ccc; cursor: pointer;
                         border-radius: 4px 4px 0 0; margin-right: 5px; margin-bottom: 5px; font-size: 0.9em; }}
    .tab-button.active{{ background: #fff; border-bottom: 1px solid #fff; }}
    .tab-content      {{ border: 1px solid #ccc; padding: 15px; border-radius: 0 4px 4px 4px; }}
    .plot-container   {{ width: 900px; height: 600px; }}
    .error            {{ color: #d32f2f; padding: 10px; border: 1px solid #ffcdd2; background: #ffebee; }}
    .section-controls {{ margin: 10px 0; }}
    .section-button   {{ background: #f0f0f0; border: 1px solid #ddd; padding: 5px 10px; cursor: pointer;
                         border-radius: 4px; margin-right: 5px; }}
  </style>
</head>
<body>
  <h1>16S Amplicon Analysis – Debug Report</h1>
  <p>Generated: {generated_ts}</p>
  <p>Sections: {section_list}</p>

  <div class="section-controls">
    <button class="section-button" onclick="toggleAllSections(true)">Expand All</button>
    <button class="section-button" onclick="toggleAllSections(false)">Collapse All</button>
  </div>

  {sections_html}

  <!-- serialised figure data -->
  <script id="plot-data" type="application/json">{plot_data_json}</script>

  <script>
    /* ---- data ---- */
    const plotData = JSON.parse(document.getElementById('plot-data').textContent);

    /* ---- state ---- */
    const rendered = new Set();

    /* ---- helpers ---- */
    function renderPlot(containerId, plotId) {{
        const container = document.getElementById(containerId);
        if (!container) return console.error('Missing container', containerId);

        container.innerHTML = '';
        const div = document.createElement('div');
        div.id        = plotId;
        div.className = 'plot-container';
        container.appendChild(div);

        const payload = plotData[plotId];
        if (!payload) {{
            div.innerHTML = '<div class="error">Plot data unavailable</div>';
            return;
        }}

        if (payload.type === 'plotly') {{
            if (payload.layout) {{
                payload.layout.showlegend = false;
                payload.layout.width  = 900;
                payload.layout.height = 600;
            }}
            Plotly.newPlot(
                plotId, payload.data, payload.layout, {{responsive: true}}
            ).catch(err => {{
                div.innerHTML = `<div class="error">Plotly error: ${{
                    err
                }}</div>`;
                console.error(err);
            }});
        }} else if (payload.type === 'image') {{
            const img = document.createElement('img');
            img.src = 'data:image/png;base64,' + payload.data;
            img.style.maxWidth = '100%';
            div.appendChild(img);
        }} else if (payload.type === 'error') {{
            div.innerHTML = `<div class="error">${{payload.error}}</div>`;
        }} else {{
            div.innerHTML = '<div class="error">Unknown plot type</div>';
        }}
    }}

    /* ---- tab logic ---- */
    function showTab(tabId, plotId) {{
        const pane = document.getElementById(tabId);
        if (!pane) return;

        const subsection = pane.closest('.subsection');
        subsection.querySelectorAll('.tab-pane')
                   .forEach(p => p.style.display = 'none');
        subsection.querySelectorAll('.tab-button')
                   .forEach(b => b.classList.remove('active'));

        pane.style.display = 'block';
        subsection.querySelector(`[data-tab="${{tabId}}"]`)
                  .classList.add('active');

        if (!rendered.has(plotId)) {{
            renderPlot(`container-${{plotId}}`, plotId);
            rendered.add(plotId);
        }}
    }}

    /* ---- section toggles ---- */
    function toggleAllSections(show) {{
        document.querySelectorAll('.section')
                .forEach(s => (s.style.display = show ? 'block' : 'none'));
    }}

    /* ---- first‑tab bootstrapping ---- */
    document.addEventListener('DOMContentLoaded', () => {{
        document.querySelectorAll('.subsection').forEach(sub => {{
            const first = sub.querySelector('.tab-pane');
            if (first) showTab(first.id, first.dataset.plotId);
        }});
    }});
  </script>
</body>
</html>"""

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _prepare_sections(
    figures: Dict,
    include_sections: List[str],
    id_counter,
) -> Tuple[List[Dict], Dict]:
    """Flatten figures → list‑of‑sections, dict‑of‑plotPayload."""
    sections = []
    plot_data: Dict[str, Any] = {}

    ordination_methods = ["pca", "pcoa", "tsne", "umap"]

    for sec in include_sections:
        if sec not in figures:
            continue

        sec_data = {"id": f"sec-{uuid.uuid4().hex}", "title": sec.title(), "subsections": []}

        if sec == "ordination":
            for meth in ordination_methods:
                figs = _collect_ord_figs(figures[sec], meth)
                if not figs:
                    continue
                tabs, btns, pd = _figs_to_html(figs, id_counter, sec_data["id"])
                plot_data.update(pd)
                sec_data["subsections"].append({"title": meth.upper(),
                                                "tabs_html": tabs,
                                                "buttons_html": btns})
        else:
            flat: Dict[str, Any] = {}
            _flatten(figures[sec], [], flat)
            if flat:
                tabs, btns, pd = _figs_to_html(flat, id_counter, sec_data["id"])
                plot_data.update(pd)
                sec_data["subsections"].append({"title": "All",
                                                "tabs_html": tabs,
                                                "buttons_html": btns})
        if sec_data["subsections"]:
            sections.append(sec_data)

    return sections, plot_data


def _collect_ord_figs(tree, meth):
    out = {}
    for table_type, levels in tree.items():
        for level, methods in levels.items():
            if meth in methods:
                for colour, fig in methods[meth].items():
                    out[f"{table_type} – {level} – {colour}"] = fig
    return out


def _flatten(tree, keys, out):
    for k, v in tree.items():
        new_keys = keys + [k]
        if isinstance(v, dict):
            _flatten(v, new_keys, out)
        else:
            out[" – ".join(new_keys)] = v


def _figs_to_html(figs: Dict[str, Any], counter, prefix) -> Tuple[str, str, Dict]:
    tabs, btns, plot_data = [], [], {}
    for title, fig in figs.items():
        idx     = next(counter)
        tab_id  = f"{prefix}-tab-{idx}"
        plot_id = f"{prefix}-plot-{idx}"

        btns.append(
            f'<button class="tab-button {"active" if idx==0 else ""}" '
            f'data-tab="{tab_id}" '
            f'onclick="showTab(\'{tab_id}\', \'{plot_id}\')">{title}</button>'
        )
        tabs.append(
            f'<div id="{tab_id}" class="tab-pane" '
            f'style="display:{"block" if idx==0 else "none"}" '
            f'data-plot-id="{plot_id}">'
            f'<div id="container-{plot_id}" class="plot-container"></div></div>'
        )

        try:
            if hasattr(fig, "to_plotly_json"):
                pj = fig.to_plotly_json()
                pj.setdefault("layout", {})["showlegend"] = False
                plot_data[plot_id] = {"type": "plotly",
                                      "data": pj["data"],
                                      "layout": pj["layout"]}
            elif isinstance(fig, Figure):
                buf = BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight")
                buf.seek(0)
                plot_data[plot_id] = {"type": "image",
                                      "data": base64.b64encode(buf.read()).decode()}
            else:
                plot_data[plot_id] = {"type": "error",
                                      "error": f"Unsupported figure type {type(fig)}"}
        except Exception as exc:  # pragma: no cover
            logger.exception("Serialising figure failed")
            plot_data[plot_id] = {"type": "error", "error": str(exc)}

    return "\n".join(tabs), "\n".join(btns), plot_data


def _section_html(sec):
    sub_html = "\n".join(
        f'<div class="subsection">\n'
        f'  <h3>{sub["title"]}</h3>\n'
        f'  <div class="tabs">{sub["buttons_html"]}</div>\n'
        f'  <div class="tab-content">{sub["tabs_html"]}</div>\n'
        f'</div>'
        for sub in sec["subsections"]
    )
    return f'<div class="section" id="{sec["id"]}">\n' \
           f'  <h2>{sec["title"]}</h2>\n{sub_html}\n</div>'
