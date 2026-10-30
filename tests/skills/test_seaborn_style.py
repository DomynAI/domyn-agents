"""The chart styles every builtin skill shares, in `skills/builtin/resources/`:
`seaborn_style.py` (seaborn, PNG), which reads its colours from `palette.json` and its text
sizes from `typography.json` beside it, and `plotly_style.py` (Plotly, HTML only), which
reads the same palette.

They sit in `builtin/resources/`, and every script that draws a chart finds them by its own
place in that tree, never by the working directory or PYTHONPATH: the skills' absolute
path differs per install, and the agent runs the scripts from wherever it happens to be.
So what is checked here is the finding, as the agent runs them — a fresh interpreter,
started somewhere else, with a decoy `seaborn_style` on PYTHONPATH — and that a run
folder keeps its own copy, palette, typography and fonts included, so it reruns without
the skill.

The test environment has neither seaborn nor plotly, so the lookup is checked with
`find_spec`, which locates a module without running it. What needs either is skipped
without it.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import domyn.agents
from domyn.agents.interfaces.cli.config import builtin_skill_dirs
from domyn.agents.skills import LocalDirSkillSource, SkillQuery

BUILTIN = Path(domyn.agents.__file__).parent / "skills" / "builtin"
RESOURCES = BUILTIN / "resources"
STYLE = RESOURCES / "seaborn_style.py"
PLOTLY_STYLE = RESOURCES / "plotly_style.py"
PALETTE = RESOURCES / "palette.json"
TYPOGRAPHY = RESOURCES / "typography.json"
FONTS = RESOURCES / "assets" / "fonts"
DATA_SKILLS = BUILTIN / "data"

# Where a script that imported `common` would get seaborn_style from, without importing it.
FIND = "import common, importlib.util; print(importlib.util.find_spec('seaborn_style').origin)"


def locate(cwd: Path, script_dir: Path, decoy: Path) -> subprocess.CompletedProcess[str]:
    """`FIND` as a script in `script_dir` would run it: that directory first on the path,
    started from `cwd`, with a `seaborn_style` of someone else's on PYTHONPATH."""
    code = f"import sys; sys.path.insert(0, {str(script_dir)!r}); {FIND}"
    env = {**os.environ, "PYTHONPATH": str(decoy)}
    return subprocess.run(
        [sys.executable, "-c", code], cwd=cwd, env=env, capture_output=True, text=True, check=False
    )


@pytest.fixture
def decoy(tmp_path: Path) -> Path:
    folder = tmp_path / "decoy"
    folder.mkdir()
    (folder / "seaborn_style.py").write_text("raise SystemExit('the wrong seaborn_style')\n")
    return folder


class TestItIsFoundByTheTreeAlone:
    @pytest.mark.parametrize("skill", ["data-analysis", "time-series-forecasting"])
    def test_from_anywhere_and_ahead_of_pythonpath(
        self, skill: str, tmp_path: Path, decoy: Path
    ) -> None:
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        result = locate(elsewhere, DATA_SKILLS / skill / "scripts", decoy)
        assert result.returncode == 0, result.stderr
        assert Path(result.stdout.strip()) == STYLE

    def test_every_skill_puts_the_resources_on_the_path(self) -> None:
        """tabular-classification-regression's common.py needs scikit-learn to import, which
        the test environment lacks, so its path setup is checked by reading it."""
        for common in DATA_SKILLS.glob("*/scripts/common.py"):
            text = common.read_text()
            assert (
                'sys.path.insert(1, str(Path(__file__).resolve().parents[3] / "resources"))' in text
            ), common

    def test_every_chart_script_imports_it_after_common(self) -> None:
        """common is what puts builtin/resources/ on the path, so it has to come first. isort keeps
        that order only because `seaborn_style` sorts after `common`: a rename that broke
        it would leave every chart script failing on its first import."""
        scripts = [
            s for s in BUILTIN.glob("*/*/scripts/*.py") if "import seaborn_style" in s.read_text()
        ]
        assert scripts
        for script in scripts:
            if script.name == "common.py":
                continue  # imports it inside the functions that draw
            lines = script.read_text().splitlines()
            first = {
                name: next(i for i, ln in enumerate(lines) if re.match(rf"import {name}\b", ln))
                for name in ("common", "seaborn_style")
            }
            assert first["common"] < first["seaborn_style"], script

    def test_only_the_interactive_charts_use_plotly(self) -> None:
        """A PNG is seaborn's: Plotly would need kaleido, and kaleido a Chrome."""
        allowed = {PLOTLY_STYLE, BUILTIN / "visualization" / "interactive-charts" / "SKILL.md"}
        for path in BUILTIN.rglob("*"):
            if path.suffix in (".py", ".md") and path not in allowed:
                text = path.read_text().lower()
                assert "import plotly" not in text and "kaleido" not in text, path


class TestARunFolderKeepsItsOwnCopy:
    def test_record_run_copies_it(self, tmp_path: Path) -> None:
        sys.path.insert(0, str(DATA_SKILLS))
        try:
            import data_utils
        finally:
            sys.path.remove(str(DATA_SKILLS))
        out = data_utils.output_dir(tmp_path / "run")
        script = out.parent / "step.py"
        script.write_text("")
        data_utils.record_run(out, script, ("step",))
        assert (out / "seaborn_style.py").read_bytes() == STYLE.read_bytes()
        assert (out / "palette.json").read_bytes() == PALETTE.read_bytes()
        assert (out / "typography.json").read_bytes() == TYPOGRAPHY.read_bytes()
        shipped = sorted(p.name for p in FONTS.iterdir())
        assert sorted(p.name for p in (out / "fonts").iterdir()) == shipped
        assert not (out / "plotly_style.py").exists()

    def test_a_rerun_copies_them_from_the_run_folder(self, tmp_path: Path) -> None:
        """A rerun of the folder's own data_utils finds the style next to it."""
        first = tmp_path / "first"
        (first / "fonts").mkdir(parents=True)
        for src in (DATA_SKILLS / "data_utils.py", STYLE, PALETTE, TYPOGRAPHY):
            (first / src.name).write_bytes(src.read_bytes())
        (first / "fonts" / "DMSans-Regular.ttf").write_bytes(
            (FONTS / "DMSans-Regular.ttf").read_bytes()
        )
        code = (
            f"import sys; sys.path.insert(0, {str(first)!r}); import data_utils; "
            f"from pathlib import Path; out = data_utils.output_dir(Path({str(tmp_path / 'run')!r})); "
            "s = out.parent / 'step.py'; s.write_text(''); data_utils.record_run(out, s, ('step',))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr
        files = tmp_path / "run" / "files"
        for name in (
            "seaborn_style.py",
            "palette.json",
            "typography.json",
            "fonts/DMSans-Regular.ttf",
        ):
            assert (files / name).is_file(), name

    def test_the_copy_is_the_one_a_rerun_uses(self, tmp_path: Path, decoy: Path) -> None:
        files = tmp_path / "run" / "files"
        files.mkdir(parents=True)
        for src in (
            DATA_SKILLS / "data-analysis" / "scripts" / "common.py",
            DATA_SKILLS / "data_utils.py",
            STYLE,
            PALETTE,
            TYPOGRAPHY,
        ):
            (files / src.name).write_bytes(src.read_bytes())
        result = locate(tmp_path, files, decoy)
        assert result.returncode == 0, result.stderr
        assert Path(result.stdout.strip()) == files / "seaborn_style.py"


class TestTheChartsSkills:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("skill", "style"), [("charts", STYLE), ("interactive-charts", PLOTLY_STYLE)]
    )
    async def test_it_is_indexed_and_the_style_is_where_it_says(
        self, skill: str, style: Path
    ) -> None:
        """Each skill tells the model to copy its style from `<its directory>/../../resources/`."""
        index = await LocalDirSkillSource(paths=builtin_skill_dirs()).index(SkillQuery(task=""))
        found = next(s for s in index.skills if s.name == skill)
        assert found.path is not None
        assert Path(found.path).resolve().parents[2] / "resources" / style.name == style
        text = Path(found.path).read_text()
        assert "../../resources/" in text and style.name in text

    @pytest.mark.asyncio
    async def test_the_resources_are_not_a_skill(self) -> None:
        """resources/ is a directory under builtin/, so it is scanned like a category: nothing
        in it may read as a skill (a flat `.md` would)."""
        index = await LocalDirSkillSource(paths=[RESOURCES]).index(SkillQuery(task=""))
        assert not index.skills


def import_from_resources(name: str):  # type: ignore[no-untyped-def]
    sys.path.insert(0, str(RESOURCES))
    try:
        return __import__(name)
    finally:
        sys.path.remove(str(RESOURCES))


def palette_colours(node: object) -> set[str]:
    if isinstance(node, dict):
        return {c for v in node.values() for c in palette_colours(v)}
    if isinstance(node, list):
        return {c for item in node for c in palette_colours(item)}
    return {node} if isinstance(node, str) and node.startswith("#") else set()


class TestThePalette:
    def test_every_chart_colour_is_one_of_the_templates(self) -> None:
        """The `charts` section arranges colours extracted from the master deck; it adds none."""
        palette = json.loads(PALETTE.read_text())
        charts = palette.pop("charts")
        assert palette_colours(charts) <= palette_colours(palette)
        accents = ("primary", "secondary", "tertiary", "quaternary", "quinary", "senary")
        assert sorted(charts["categorical"]) == sorted(palette[a] for a in accents)

    @pytest.mark.parametrize("module", [STYLE, PLOTLY_STYLE])
    def test_no_colour_is_written_into_the_modules(self, module: Path) -> None:
        """Every hex code lives in palette.json; the modules have none of their own."""
        assert not re.findall(r"#[0-9A-Fa-f]{6}\b", module.read_text())

    @pytest.mark.parametrize("module", [STYLE, PLOTLY_STYLE])
    def test_they_keep_no_copy_of_a_palette_entry(self, module: Path) -> None:
        """The styles read the palette themselves: no module constant holds a read-out of it."""
        import ast

        names = {
            target.id
            for node in ast.parse(module.read_text()).body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        assert "PALETTE" in names
        for gone in ("BRAND", "CATEGORICAL", "SEQUENTIAL", "DIVERGING", "MUTED", "INK", "TEXT"):
            assert gone not in names


class TestTheStyle:
    """seaborn_style.py: the seaborn theme and the PNGs."""

    @pytest.fixture
    def style(self):  # type: ignore[no-untyped-def]
        pytest.importorskip("seaborn")
        return import_from_resources("seaborn_style")

    def test_importing_it_sets_the_theme_from_the_palette(self, style) -> None:  # type: ignore[no-untyped-def]
        """The slides and the charts read one file: a colour changed there changes both."""
        import matplotlib as mpl
        from matplotlib.colors import to_hex

        palette = json.loads(PALETTE.read_text())
        charts, text = palette["charts"], palette["text"]["light_background"]
        rc = mpl.rcParams
        assert [to_hex(c["color"]).upper() for c in rc["axes.prop_cycle"]] == [
            c.upper() for c in charts["categorical"]
        ]
        assert rc["axes.prop_cycle"].by_key()["color"][0].upper() == palette["primary"].upper()
        assert to_hex(rc["text.color"]).upper() == text["body"].upper()
        assert to_hex(rc["grid.color"]).upper() == charts["grid"].upper()
        assert to_hex(rc["axes.edgecolor"]).upper() == palette["lines"]["divider_light"].upper()
        assert to_hex(rc["figure.facecolor"]).upper() == palette["surfaces"]["card_light"].upper()
        assert rc["image.cmap"] == "house_sequential"
        assert rc["text.parse_math"] is False

    def test_text_sizes_are_the_typographys(self, style) -> None:  # type: ignore[no-untyped-def]
        import matplotlib as mpl

        roles = json.loads(TYPOGRAPHY.read_text())["roles"]
        assert mpl.rcParams["font.size"] == roles["body"]["size"]
        assert mpl.rcParams["figure.titlesize"] == roles["heading"]["size"]

    def test_it_draws_in_the_shipped_dm_sans(self, style) -> None:  # type: ignore[no-untyped-def]
        """Registered from the files, so a PNG is in DM Sans whether or not it is installed."""
        from matplotlib import font_manager

        found = font_manager.findfont(font_manager.FontProperties(family=style.FONT))
        assert Path(found).parent == FONTS

    def test_the_colormaps_are_the_palettes(self, style) -> None:  # type: ignore[no-untyped-def]
        from matplotlib import colormaps
        from matplotlib.colors import to_hex

        charts = json.loads(PALETTE.read_text())["charts"]
        diverging = colormaps[style.colormap("diverging")]
        assert to_hex(diverging(0.0)).upper() == charts["negative"].upper()
        assert to_hex(diverging(1.0)).upper() == charts["positive"].upper()
        sequential = colormaps[style.colormap("sequential")]
        assert to_hex(sequential(1.0)).upper() == charts["sequential"][-1]["color"].upper()
        assert f"{style.colormap('sequential')}_r" in colormaps

    def test_an_unknown_colormap_is_refused(self, style) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ValueError, match="sequential"):
            style.colormap("rainbow")

    def test_save_writes_the_png_only(self, style, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        fig, ax = style.figure()
        ax.bar(["a", "b"], [1, 2])
        style.title(fig, "A chart")
        assert style.save(fig, tmp_path, "a chart") == "a_chart.png"
        assert sorted(p.name for p in tmp_path.iterdir()) == ["a_chart.png"]

    def test_the_png_is_twice_the_size_asked_for(self, style, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        from PIL import Image

        fig, ax = style.figure(width=400, height=300)
        ax.plot([0, 1], [0, 1])
        with Image.open(tmp_path / style.save(fig, tmp_path, "sized")) as im:
            assert im.size == (800, 600)

    def test_large_numbers_and_log_axes_read_as_plain_numbers(  # type: ignore[no-untyped-def]
        self, style, tmp_path: Path
    ) -> None:
        """ "60k" rather than "60000"; "1,000" on a log axis rather than its mathtext, which
        prints as written with parse_math off."""
        fig, (big, log) = style.figure(ncols=2)
        big.plot([0, 60_000], [0, 1])
        log.plot([1, 1_000], [1, 2])
        log.set_xscale("log")
        style.save(fig, tmp_path, "ticks")
        big_labels = [t.get_text() for t in big.get_xticklabels()]
        log_labels = [t.get_text() for t in log.get_xticklabels()]
        assert "60k" in big_labels and "0" in big_labels
        assert "1,000" in log_labels and not any("$" in t for t in log_labels)

    def test_the_legend_sits_above_the_plot_under_the_title(  # type: ignore[no-untyped-def]
        self, style, tmp_path: Path
    ) -> None:
        fig, ax = style.figure()
        for name in ("one", "two"):
            ax.plot([0, 1], [0, 1], label=name)
        style.title(fig, "Two series")
        style.legend(ax)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        legend = ax.get_legend().get_window_extent(renderer)
        plot = ax.get_window_extent(renderer)
        title = fig.get_suptitle() and fig._suptitle.get_window_extent(renderer)
        assert legend.y0 >= plot.y1  # above the plot
        assert title.y0 >= legend.y1  # under the title
        assert ax.get_legend()._ncols == 2  # one row

    def test_a_panels_legend_goes_to_its_right(self, style) -> None:  # type: ignore[no-untyped-def]
        fig, axes = style.figure(ncols=2)
        for ax in axes:
            ax.plot([0, 1], [0, 1], label="one")
            ax.set_title("panel")
        style.legend(axes[-1])
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        legend = axes[-1].get_legend().get_window_extent(renderer)
        assert legend.x0 >= axes[-1].get_window_extent(renderer).x1

    @pytest.mark.parametrize("missing", ["palette.json", "typography.json"])
    def test_without_its_files_it_says_what_is_missing(self, missing: str, tmp_path: Path) -> None:
        pytest.importorskip("seaborn")
        for src in (STYLE, PALETTE, TYPOGRAPHY):
            if src.name != missing:
                (tmp_path / src.name).write_bytes(src.read_bytes())
        result = subprocess.run(
            [sys.executable, "-c", "import seaborn_style"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode != 0
        assert missing in result.stderr


class TestThePlotlyStyle:
    """plotly_style.py: the template for interactive pages, and nothing that needs a browser."""

    @pytest.fixture
    def style(self):  # type: ignore[no-untyped-def]
        pytest.importorskip("plotly")
        return import_from_resources("plotly_style")

    def test_it_writes_no_image(self) -> None:
        text = PLOTLY_STYLE.read_text()
        assert "write_image" not in text and "import kaleido" not in text

    def test_importing_it_makes_it_the_default(self, style) -> None:  # type: ignore[no-untyped-def]
        import plotly.graph_objects as go
        import plotly.io as pio

        assert pio.templates.default == style.NAME
        fig = go.Figure(go.Bar(x=["a"], y=[1]))
        assert fig.layout.template.layout.colorway[0] == style.PALETTE["primary"]

    def test_the_template_picks_its_colours_from_the_palette(self, style) -> None:  # type: ignore[no-untyped-def]
        import plotly.io as pio

        palette = json.loads(PALETTE.read_text())
        charts, text = palette["charts"], palette["text"]["light_background"]
        layout = pio.templates[style.NAME].layout
        assert list(layout.colorway) == charts["categorical"]
        assert [list(stop) for stop in layout.colorscale.sequential] == [
            [s["pos"], s["color"]] for s in charts["sequential"]
        ]
        assert [stop[1] for stop in layout.colorscale.diverging] == [
            charts["negative"],
            charts["neutral"],
            charts["positive"],
        ]
        assert layout.title.font.color == text["title"]
        assert layout.font.color == text["body"]
        assert layout.hoverlabel.font.color == text["heading"]
        assert layout.yaxis.gridcolor == charts["grid"]
        assert layout.xaxis.linecolor == palette["lines"]["divider_light"]
        assert layout.paper_bgcolor == layout.plot_bgcolor == palette["surfaces"]["card_light"]

    def test_an_unknown_colour_scale_is_refused(self, style) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ValueError, match="sequential"):
            style.colorscale("rainbow")

    def test_the_page_is_self_contained(self, style, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        import plotly.graph_objects as go

        name = style.save(go.Figure(go.Bar(x=["a"], y=[1])), tmp_path, "a chart")
        assert name == "a_chart.html"
        assert sorted(p.name for p in tmp_path.iterdir()) == ["a_chart.html"]
        page = (tmp_path / name).read_text()
        # A preview serving the page alone has nothing beside it to load.
        assert 'src="plotly.min.js"' not in page
        assert "Plotly" in page and len(page) > 1_000_000

    def test_without_the_palette_it_says_what_is_missing(self, tmp_path: Path) -> None:
        pytest.importorskip("plotly")
        (tmp_path / "plotly_style.py").write_bytes(PLOTLY_STYLE.read_bytes())
        result = subprocess.run(
            [sys.executable, "-c", "import plotly_style"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode != 0
        assert "palette.json" in result.stderr
