"""The house style for interactive charts: a Plotly template, set as the default when this
module is imported, and pages saved as self-contained HTML.

    import plotly_style                     # from here on, every figure is in the house style
    import plotly.express as px

    fig = px.bar(df, x="region", y="revenue", title="Revenue by region")
    plotly_style.save(fig, out, "revenue_by_region")   # revenue_by_region.html

The static charts (PNG, for reports, slides and documents) are `seaborn_style.py`'s, beside
this file, drawn with seaborn: this module writes no image, so it needs no kaleido and no
Chrome. Both read the same palette, so a page and a PNG of the same data match.

Importing it is the whole setup: the template is registered as "house" and made
`plotly.io.templates.default`, so `px.*`, `go.Figure` and `make_subplots` all pick it
up. For what a template cannot decide on its own (a reference line, the "actual" series
next to forecasts, positive against negative), take the colour from `PALETTE` by its key,
never as hex copied into a script, so that those stay consistent too:

    MUTED = plotly_style.PALETTE["text"]["light_background"]["muted"]
    fig.add_hline(0, line_color=MUTED)

Every colour comes from `palette.json`, beside this file: the palette the slides are built
from. The categorical order, the sequential and diverging scales and the text colours are
the ones `seaborn_style.py` describes, read from the same keys.

The font is DM Sans where the viewer has it installed, a system sans otherwise: a page
names its fonts and does not embed them.

Where it lives: in `skills/builtin/resources/`, with the palette, shared by every builtin
skill; the `interactive-charts` skill copies it, with palette.json, next to the script.

Needs `plotly` only.
"""

import json
import re
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio

NAME = "house"

# The palette shared with the slides; it travels with this file (copied beside it).
PALETTE_FILE = Path(__file__).resolve().with_name("palette.json")
if not PALETTE_FILE.is_file():
    raise SystemExit(
        f"plotly_style needs palette.json beside it ({PALETTE_FILE}): copy it from "
        "skills/builtin/resources/ along with plotly_style.py."
    )
# Every colour a chart uses, by its palette key: what the template does not set itself is
# taken from here by role, e.g. a reference line in PALETTE["text"]["light_background"]
# ["muted"], the second series' colour in PALETTE["charts"]["categorical"][1].
PALETTE = json.loads(PALETTE_FILE.read_text())

# The house style's DM Sans where the viewer has it, then the system sans a page falls back to.
FONT = (
    '"DM Sans", "DM Sans 14pt", Inter, system-ui, -apple-system, "Segoe UI", '
    "Helvetica, Arial, sans-serif"
)

# The legend sits above the plot by default. With subplot titles it would cover the first
# row's, so a figure that has both puts it on the right instead:
#     fig.update_layout(legend=plotly_style.LEGEND_RIGHT)
LEGEND_RIGHT = {"orientation": "v", "x": 1.02, "xanchor": "left", "y": 1, "yanchor": "top"}


def colorscale(kind: str) -> list[list[float | str]]:
    """A Plotly colour scale from the palette: "sequential" for magnitude, light (little)
    to dark (a lot); "diverging" for a value around a meaningful zero, negative to
    positive through a neutral midpoint."""
    charts = PALETTE["charts"]
    if kind == "sequential":
        return [[stop["pos"], stop["color"]] for stop in charts["sequential"]]
    if kind == "diverging":
        return [[0.0, charts["negative"]], [0.5, charts["neutral"]], [1.0, charts["positive"]]]
    raise ValueError(f"no colour scale {kind!r}: 'sequential' or 'diverging'")


def tint(color: str, alpha: float) -> str:
    """A "#rrggbb" colour at the given opacity, for a band or a fill in a series' colour."""
    r, g, b = (int(color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


def _template() -> go.layout.Template:
    text = PALETTE["text"]["light_background"]
    axis = {
        "showline": True,
        "linecolor": PALETTE["lines"]["divider_light"],
        "ticks": "outside",
        "tickcolor": PALETTE["lines"]["divider_light"],
        "ticklen": 4,
        "gridcolor": PALETTE["charts"]["grid"],
        "zeroline": False,
        "automargin": True,
        "title": {"standoff": 10, "font": {"color": text["body"]}},
        "tickfont": {"color": text["body"]},
    }
    colorbar = {"outlinewidth": 0, "thickness": 14, "tickfont": {"color": text["body"]}}
    return go.layout.Template(
        layout={
            "colorway": PALETTE["charts"]["categorical"],
            "font": {"family": FONT, "size": 13, "color": text["body"]},
            "title": {
                "x": 0,
                "xanchor": "left",
                "xref": "container",
                "pad": {"l": 16},
                "font": {"size": 17, "color": text["title"]},
            },
            "paper_bgcolor": PALETTE["surfaces"]["card_light"],
            "plot_bgcolor": PALETTE["surfaces"]["card_light"],
            "margin": {"l": 64, "r": 32, "t": 80, "b": 56},
            "xaxis": dict(axis, showgrid=False),
            "yaxis": dict(axis, showgrid=True),
            "colorscale": {
                "sequential": colorscale("sequential"),
                "sequentialminus": colorscale("sequential"),
                "diverging": colorscale("diverging"),
            },
            "coloraxis": {"colorbar": colorbar},
            "legend": {
                "orientation": "h",
                "x": 0,
                "xanchor": "left",
                "y": 1.02,
                "yanchor": "bottom",
                "title_text": "",
            },
            "hovermode": "closest",
            "hoverlabel": {
                "bgcolor": PALETTE["surfaces"]["card_light"],
                "bordercolor": PALETTE["lines"]["divider_light"],
                "font": {"family": FONT, "color": text["heading"]},
            },
            "annotationdefaults": {"font": {"color": text["body"], "size": 12}},
            "bargap": 0.2,
            "bargroupgap": 0.05,
        },
        data={
            "bar": [go.Bar(marker_line_width=0)],
            "histogram": [go.Histogram(marker_line_width=0)],
            "scatter": [go.Scatter(line={"width": 2}, marker={"size": 8})],
            "box": [go.Box(line={"width": 1.5}, marker={"size": 4})],
            "heatmap": [go.Heatmap(xgap=1, ygap=1, colorbar=colorbar)],
        },
    )


pio.templates[NAME] = _template()
pio.templates.default = NAME


def safe_name(text: str) -> str:
    """A string usable as a file name."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_") or "chart"


def save(fig: go.Figure, out: Path, name: str) -> str:
    """Write the chart as <out>/<name>.html and return the file name.

    Self-contained, with plotly.js inline (about 4.8 MB): a page served on its own, as a
    frontend preview does, cannot load a script from beside it, and one from a CDN would
    not work offline.
    """
    base = safe_name(name)
    fig.write_html(Path(out) / f"{base}.html", include_plotlyjs=True, full_html=True)
    return f"{base}.html"
