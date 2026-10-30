"""The house style for every static chart the builtin skills draw: seaborn on matplotlib,
set up when this module is imported, and saved as PNG.

    import seaborn_style                       # from here on, every figure is in the house style
    import seaborn as sns

    fig, ax = seaborn_style.figure()
    sns.barplot(df, x="revenue", y="region", ax=ax)
    seaborn_style.title(fig, "Revenue by region")
    seaborn_style.save(fig, out, "revenue_by_region")   # revenue_by_region.png

Importing it is the whole setup: `seaborn.set_theme` with the house rcParams, so every
axes-level seaborn function and plain matplotlib pick it up. What rcParams cannot decide is
a function here: `figure()` (the size, and the layout that keeps labels inside it),
`title()` (the chart's title, left-aligned at the top of the figure), `legend()` (the
legend above the plot, in one row) and `save()`. For what a style cannot decide on its own
(a reference line, the "actual" series next to forecasts, positive against negative), take
the colour from `PALETTE` by its key, never as hex copied into a script, so that those
stay consistent too:

    MUTED = seaborn_style.PALETTE["text"]["light_background"]["muted"]
    ax.axhline(0, color=MUTED)

Every colour comes from `palette.json` and every text size from `typography.json`, both
beside this file: what the slides and documents are built from, so a chart pasted into a
deck matches it. Sizes are in points, as matplotlib draws text, so a chart placed at its
own size has the slides' body and heading sizes. Nothing here, and nothing in the palette's
`charts` section, is a colour the template does not use:

- the categorical order is the template theme's six accents (accent1-6, what PowerPoint
  itself gives a chart's series), reordered for separation. Three of them are close blues
  (primary, secondary, quaternary; the last two nearly identical to deuteranopes), so the
  order puts the four that stay apart under colour blindness first: primary blue, coral
  (quinary), sky (tertiary), deep violet (senary). Past four series, fold the rest into
  "Other" or split the chart; the fifth and sixth (quaternary, secondary) only with
  direct labels.
- the sequential colormap runs from the template's pale blue through sky and primary to
  the deep violet; the diverging one from coral through the template's off-white to primary.
  Both are registered by name ("house_sequential", "house_diverging", and their "_r").
- titles and text are the template's text colours for light backgrounds; the axis is its
  light divider, the grid its pale blue.

Sky (tertiary) reaches only 2.4:1 against white: fine for a bar or a filled area, too faint for a
thin line or a small marker on its own, where it is better left to a later position.

The font is DM Sans, registered with matplotlib from `assets/fonts/` (in the skills'
resources) or `fonts/` (beside this file, in a run folder), so every PNG is drawn in it
whether or not it is installed on the machine.

Where it lives: in `skills/builtin/resources/`, with the palette and the shared assets,
shared by every builtin skill. Scripts find it by their own position in that tree
(`Path(__file__).resolve().parents[...]`), never by the working directory or PYTHONPATH;
data_utils.record_run copies it, with palette.json, typography.json and the fonts, into
every run folder, so the folder reruns without the skill.

Needs `seaborn` (which brings matplotlib). No browser: the interactive HTML version of a
chart is `plotly_style.py`'s, beside this file.
"""

import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # files only: never a window, whatever the machine has

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colormaps, font_manager
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.ticker import FuncFormatter, LogFormatter, NullFormatter, ScalarFormatter

HERE = Path(__file__).resolve().parent

# The palette and type shared with the slides; they travel with this file (copied beside it).
PALETTE_FILE = HERE / "palette.json"
TYPOGRAPHY_FILE = HERE / "typography.json"
for _needed in (PALETTE_FILE, TYPOGRAPHY_FILE):
    if not _needed.is_file():
        raise SystemExit(
            f"seaborn_style needs {_needed.name} beside it ({_needed}): copy palette.json and "
            "typography.json from skills/builtin/resources/ along with seaborn_style.py."
        )
# Every colour a chart uses, by its palette key: what the style does not set itself is
# taken from here by role, e.g. a reference line in PALETTE["text"]["light_background"]
# ["muted"], the second series' colour in PALETTE["charts"]["categorical"][1].
PALETTE = json.loads(PALETTE_FILE.read_text())
# The text roles (title, body, caption…) and their sizes in points.
TYPOGRAPHY = json.loads(TYPOGRAPHY_FILE.read_text())

# Where the DM Sans files are: the skills' resources, or a run folder's copy.
FONT_DIRS = (HERE / "assets" / "fonts", HERE / "fonts")
FONT = "DM Sans"

# Default size of a chart, in CSS pixels (96 per inch): the figure is WIDTH / 96 inches
# wide, and the PNG is written at DPI, twice that many pixels.
WIDTH, HEIGHT = 900, 520
PX_PER_INCH = 96
DPI = 2 * PX_PER_INCH


def _size(role: str) -> float:
    return float(TYPOGRAPHY["roles"][role]["size"])


def _register_fonts() -> list[str]:
    """Add the shipped DM Sans files to matplotlib; the font list to use."""
    folder = next((d for d in FONT_DIRS if any(d.glob("DMSans-*.ttf"))), None)
    if folder is None:
        print(
            f"seaborn_style: no DM Sans files in {' or '.join(map(str, FONT_DIRS))}; "
            "drawing in a system sans."
        )
        return ["DejaVu Sans"]
    for path in sorted(folder.glob("DMSans-*.ttf")):
        font_manager.fontManager.addfont(str(path))
    return [FONT, "DejaVu Sans"]


def _register_colormaps() -> None:
    charts = PALETTE["charts"]
    maps = {
        "sequential": [(stop["pos"], stop["color"]) for stop in charts["sequential"]],
        "diverging": [
            (0.0, charts["negative"]),
            (0.5, charts["neutral"]),
            (1.0, charts["positive"]),
        ],
    }
    for kind, stops in maps.items():
        cmap = LinearSegmentedColormap.from_list(f"house_{kind}", stops)
        for each in (cmap, cmap.reversed()):
            if each.name not in colormaps:
                colormaps.register(each)


def colormap(kind: str) -> str:
    """The name of a house colormap, for `cmap=`: "sequential" for magnitude, light (little)
    to dark (a lot); "diverging" for a value around a meaningful zero, negative to
    positive through a neutral midpoint (centre it: `center=0`, or `vmin=-m, vmax=m`)."""
    if kind not in ("sequential", "diverging"):
        raise ValueError(f"no colormap {kind!r}: 'sequential' or 'diverging'")
    return f"house_{kind}"


def tint(color: str, alpha: float) -> tuple[float, float, float, float]:
    """A "#rrggbb" colour at the given opacity, for a band or a fill in a series' colour."""
    return to_rgba(color, alpha)


def _rc(fonts: list[str]) -> dict[str, Any]:
    text = PALETTE["text"]["light_background"]
    divider = PALETTE["lines"]["divider_light"]
    surface = PALETTE["surfaces"]["card_light"]
    body = _size("body")
    return {
        "font.family": "sans-serif",
        "font.sans-serif": fonts,
        "font.size": body,
        "text.color": text["body"],
        # "$1.2M to $3.4M" is text, not a formula.
        "text.parse_math": False,
        "figure.facecolor": surface,
        "figure.dpi": PX_PER_INCH,
        "figure.constrained_layout.use": True,
        "figure.titlesize": _size("heading"),
        "figure.titleweight": "bold",
        "savefig.facecolor": surface,
        "savefig.dpi": DPI,
        "axes.facecolor": surface,
        "axes.edgecolor": divider,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": text["body"],
        "axes.labelsize": body,
        "axes.labelpad": 8.0,
        # An axes title is a panel's, in small multiples: the chart's own is title().
        "axes.titlesize": _size("emphasis"),
        "axes.titleweight": "bold",
        "axes.titlecolor": text["heading"],
        "axes.titlelocation": "left",
        "axes.titlepad": 8.0,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "axes.prop_cycle": matplotlib.cycler(color=PALETTE["charts"]["categorical"]),
        "grid.color": PALETTE["charts"]["grid"],
        "grid.linewidth": 1.0,
        "xtick.color": divider,
        "ytick.color": divider,
        "xtick.labelcolor": text["body"],
        "ytick.labelcolor": text["body"],
        "xtick.labelsize": body,
        "ytick.labelsize": body,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "legend.frameon": False,
        "legend.fontsize": body,
        "legend.title_fontsize": body,
        # Dates as "2024", "Feb", "Mar"…: no repeated year on every tick, even in a narrow panel.
        "date.converter": "concise",
        "lines.linewidth": 2.0,
        "lines.markersize": 6.0,
        "patch.linewidth": 0.0,
        "image.cmap": colormap("sequential"),
    }


_register_colormaps()
sns.set_theme(
    context="notebook",
    style="ticks",
    palette=PALETTE["charts"]["categorical"],
    rc=_rc(_register_fonts()),
)


def figure(
    width: int = WIDTH, height: int = HEIGHT, nrows: int = 1, ncols: int = 1, **kwargs: Any
) -> tuple[Any, Any]:
    """A new figure and its axes (an array of them for nrows / ncols > 1), `width` x
    `height` CSS pixels; other arguments go to `plt.subplots` (`sharey=True`…)."""
    return plt.subplots(
        nrows,
        ncols,
        figsize=(width / PX_PER_INCH, height / PX_PER_INCH),
        layout="constrained",
        **kwargs,
    )


def title(fig: Any, text: str) -> None:
    """The chart's title: left-aligned at the top of the figure, above the legend."""
    # 12 points in from the left edge, wherever the plot itself starts.
    fig.suptitle(
        text,
        x=12 / 72 / fig.get_figwidth(),
        ha="left",
        color=PALETTE["text"]["light_background"]["title"],
        wrap=True,  # a title longer than the figure is wide breaks, rather than being cut
    )


def legend(ax: Any, ncols: int | None = None, title: str | None = None) -> None:
    """Move the legend of `ax` (seaborn's, or the labelled artists') above the plot, under the
    chart's title, in one row (or `ncols` columns), without a frame or a title unless given.

    A panel with a title of its own (small multiples) gets it on the right instead, top
    aligned, where it covers neither that title nor the next panel's: call it on the last
    panel of the row."""
    old = ax.get_legend()
    if old is not None:
        handles = list(old.legend_handles)
        labels = [t.get_text() for t in old.get_texts()]
        old.remove()
    else:
        handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    if ax.get_title(loc="left") or ax.get_title():
        place = {"loc": "upper left", "bbox_to_anchor": (1.02, 1.0), "ncols": ncols or 1}
    else:
        # Attached to the axes, so the layout makes room for it under the figure's title.
        place = {"loc": "lower left", "bbox_to_anchor": (0.0, 1.01), "ncols": ncols or len(labels)}
    ax.legend(handles, labels, title=title, alignment="left", borderaxespad=0.0, **place)


def safe_name(text: str) -> str:
    """A string usable as a file name."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_") or "chart"


def _compact_ticks(fig: Any) -> None:
    """Tick labels in thousands / millions ("60k", "1.2M") on every numeric axis that reaches
    10,000, as Plotly writes them: long numbers would run into each other in a narrow panel.
    A log axis gets plain numbers ("1", "10", "1,000"). Category names, dates and a formatter
    set by the script keep their own labels."""
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            if isinstance(axis.get_major_formatter(), LogFormatter):
                # Its labels are mathtext ("10^{2}"), printed as written with parse_math off.
                axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.6g}"))
                axis.set_minor_formatter(NullFormatter())
                continue
            if type(axis.get_major_formatter()) is not ScalarFormatter:
                continue
            reach = max(abs(v) for v in axis.get_view_interval())
            scale, suffix = next(
                ((s, x) for s, x in ((1e9, "B"), (1e6, "M"), (1e3, "k")) if reach >= 10 * s),
                (None, ""),
            )
            if scale is not None:
                axis.set_major_formatter(
                    FuncFormatter(lambda v, _, s=scale, x=suffix: f"{v / s:g}{x}" if v else "0")
                )


def save(fig: Any, out: Path, name: str) -> str:
    """Write the chart as <out>/<name>.png (for reports), close it, and return the file name.
    Large numbers on an axis are written as "60k", "1.2M" (see _compact_ticks)."""
    base = safe_name(name)
    _compact_ticks(fig)
    fig.savefig(Path(out) / f"{base}.png")
    plt.close(fig)
    return f"{base}.png"
