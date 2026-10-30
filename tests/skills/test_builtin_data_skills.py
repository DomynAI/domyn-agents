"""The built-in data skills and the code they share in `builtin/data/`.

The shared scripts (`data_utils.py`, `prepare_table.py`, `prepare_series.py`,
`series_format.py`) sit one level above the skills, and each skill's scripts put that
directory on `sys.path` themselves. Moving a skill or one of those files leaves every
script failing on its first import, so it is checked here, as the agent runs them: a
fresh interpreter, started in the skill's `scripts/` directory.

The prepared table is how the three skills share cleaned data. What matters is that it
round-trips (a code "001" stays text, a category "NA" stays a value), and that every
row the cleaning removes has its decision recorded.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import ClassVar

import pandas as pd
import pytest

import domyn.agents
from domyn.agents.skills import LocalDirSkillSource, SkillQuery

DATA_SKILLS = Path(domyn.agents.__file__).parent / "skills" / "builtin" / "data"
SHARED_SCRIPTS = ("data_utils.py", "prepare_table.py", "prepare_series.py", "series_format.py")

sys.path.insert(0, str(DATA_SKILLS))
import data_utils  # noqa: E402


def run(script: Path, *args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=cwd or script.parent,
        capture_output=True,
        text=True,
        check=False,
    )


class TestTheSharedScripts:
    def test_they_sit_next_to_the_skills_not_inside_one(self) -> None:
        for name in SHARED_SCRIPTS:
            assert (DATA_SKILLS / name).is_file(), name
            assert not list(DATA_SKILLS.glob(f"*/scripts/{name}")), f"a stale copy of {name}"

    @pytest.mark.parametrize(
        ("skill", "module"),
        [
            ("data-analysis", "common"),
            ("time-series-forecasting", "common"),
            ("time-series-forecasting", "series_format"),
        ],
    )
    def test_the_skills_scripts_find_them(self, skill: str, module: str) -> None:
        scripts = DATA_SKILLS / skill / "scripts"
        code = f"import common, {module}, data_utils, series_format; print(data_utils.__file__)"
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=scripts, capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr
        assert Path(result.stdout.strip()) == DATA_SKILLS / "data_utils.py"

    def test_every_skill_puts_the_shared_directory_on_the_path(self) -> None:
        """tabular-classification-regression's common.py needs scikit-learn to import, which the test
        environment lacks, so its path setup is checked by reading it."""
        for common in DATA_SKILLS.glob("*/scripts/common.py"):
            text = common.read_text()
            assert "sys.path.append(str(Path(__file__).resolve().parents[2]))" in text, common
            assert "import data_utils" in text, common

    @pytest.mark.asyncio
    async def test_none_of_them_is_indexed_as_a_skill(self) -> None:
        index = await LocalDirSkillSource(paths=[DATA_SKILLS]).index(SkillQuery(task=""))
        assert {skill.name for skill in index.skills} == {
            "data-analysis",
            "tabular-classification-regression",
            "time-series-forecasting",
        }


class TestTheRunFolder:
    """report.md at the top of a run folder, everything the scripts write in its files/."""

    def test_scripts_write_only_into_files(self, tmp_path: Path) -> None:
        source = tmp_path / "data.csv"
        source.write_text("a,b\n1,x\n2,y\n")
        run_folder = tmp_path / "analysis"
        result = run(
            DATA_SKILLS / "prepare_table.py", "--data", str(source), "--out", str(run_folder)
        )
        assert result.returncode == 0, result.stderr
        assert [p.name for p in run_folder.iterdir()] == ["files"]
        assert {"table.csv", "table.json", "table.md", "run.sh"} <= {
            p.name for p in (run_folder / "files").iterdir()
        }

    def test_the_files_folder_itself_is_not_nested(self, tmp_path: Path) -> None:
        assert (
            data_utils.output_dir(tmp_path / "run" / "files")
            == (tmp_path / "run" / "files").resolve()
        )
        assert data_utils.output_dir(tmp_path / "run") == (tmp_path / "run" / "files").resolve()

    def test_inputs_are_found_from_the_run_folder(self, tmp_path: Path) -> None:
        data_utils.write_table(
            data_utils.output_dir(tmp_path / "run"), pd.DataFrame({"a": [1.0]}), {}
        )
        table, _ = data_utils.load_table(tmp_path / "run")
        assert list(table["a"]) == [1.0]


class TestRunSh:
    """run.sh reruns with the interpreter that ran the scripts, not whichever `python` is
    first on PATH, and survives being edited by hand."""

    ORDER = ("prepare_table", "profile")

    def record(self, out: Path, script: str, monkeypatch: pytest.MonkeyPatch) -> list[str]:
        monkeypatch.setattr(sys, "argv", [script, "--out", str(out)])
        path = out / script
        path.write_text("")
        data_utils.record_run(out, path, self.ORDER)
        return (out / "run.sh").read_text().splitlines()

    def test_it_runs_the_scripts_with_the_interpreter_that_ran_them(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        lines = self.record(tmp_path, "profile_data.py", monkeypatch)
        assert f'PY="${{PY:-{sys.executable}}}"' in lines
        assert lines[-1].startswith('"$PY" profile_data.py --out ')

    def test_a_rerun_replaces_the_line_and_keeps_the_order(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.record(tmp_path, "profile_data.py", monkeypatch)
        self.record(tmp_path, "prepare_table.py", monkeypatch)
        lines = self.record(tmp_path, "profile_data.py", monkeypatch)
        commands = [line.split()[1] for line in lines if line.startswith('"$PY" ')]
        assert commands == ["prepare_table.py", "profile_data.py"]
        assert sum(line.startswith('PY="${PY:-') for line in lines) == 1

    def test_lines_edited_by_hand_are_kept_rather_than_crashing_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The older format, plus what an agent added: a variable and its own script.
        (tmp_path / "run.sh").write_text(
            '#!/bin/sh\ncd "$(dirname "$0")"\nPY=/project/.venv/bin/python\n'
            "python prepare_table.py --data /x.csv\n\npython analysis.py\n"
        )
        lines = self.record(tmp_path, "profile_data.py", monkeypatch)
        assert lines.count("#!/bin/sh") == 1
        assert "PY=/project/.venv/bin/python" in lines
        assert '"$PY" prepare_table.py --data /x.csv' in lines
        assert lines[-1] == '"$PY" analysis.py'

    def test_it_reruns_with_another_interpreter_or_falls_back(self, tmp_path: Path) -> None:
        source = tmp_path / "data.csv"
        source.write_text("a,b\n1,x\n2,y\n")
        out = tmp_path / "run"
        assert (
            run(
                DATA_SKILLS / "prepare_table.py", "--data", str(source), "--out", str(out)
            ).returncode
            == 0
        )
        script = out / "files" / "run.sh"
        for py in (sys.executable, "/no/such/python"):
            result = subprocess.run(
                ["sh", "-x", str(script)],
                env={"PATH": "/usr/bin:/bin", "PY": py},
                capture_output=True,
                text=True,
            )
            assert f"+ command -v {py}" in result.stderr
        assert "+ PY=python" in result.stderr


class TestThePreparedTable:
    def test_it_round_trips_with_its_types(self, tmp_path: Path) -> None:
        df = pd.DataFrame(
            {
                "code": ["001", "NA", None, "None"],
                "amount": [1.5, None, 3.0, -2.25],
                "units": pd.array([1, None, 3, 4], dtype="Int64"),
                "flag": pd.array([True, False, None, True], dtype="boolean"),
                "day": pd.to_datetime(["2024-01-31", None, "2024-02-01", "2024-03-01"]),
                "at": pd.to_datetime(
                    ["2024-01-31 13:05:00", "2024-01-31 00:00:00", None, "2024-02-01 00:00:00"]
                ),
                "note": ["a, b", 'quote "x"', "two\nlines", "plain"],
            }
        )
        data_utils.write_table(tmp_path, df, {"source": "test"})
        back, meta = data_utils.load_table(tmp_path)

        assert meta["columns"] == {
            "code": "string",
            "amount": "number",
            "units": "integer",
            "flag": "boolean",
            "day": "date",
            "at": "date",
            "note": "string",
        }
        assert list(back["code"].fillna("<missing>")) == ["001", "NA", "<missing>", "None"]
        for col in df.columns:
            pd.testing.assert_series_equal(
                back[col].astype(object).where(back[col].notna(), None),
                df[col].astype(object).where(df[col].notna(), None),
                check_names=False,
            )

    def test_a_table_that_no_longer_matches_its_types_is_refused(self, tmp_path: Path) -> None:
        data_utils.write_table(tmp_path, pd.DataFrame({"a": [1.0]}), {"source": "test"})
        (tmp_path / data_utils.TABLE_FILE).write_text("b\n1\n")
        with pytest.raises(SystemExit):
            data_utils.load_table(tmp_path)

    @pytest.mark.parametrize(
        ("text", "options", "expected"),
        [
            (["$1,234.50", "1.234,5", "(12)", "abc"], {}, [1234.5, 1234.5, -12.0, None]),
            (["1.234", "2.000,5"], {"thousands": ".", "decimal": ","}, [1234.0, 2000.5]),
            (["12%", "3"], {"percent": True}, [0.12, 3.0]),
        ],
    )
    def test_numbers_stored_as_text_are_parsed(
        self, text: list[str], options: dict[str, object], expected: list[float | None]
    ) -> None:
        parsed = data_utils.parse_number(pd.Series(text), **options)  # type: ignore[arg-type]
        assert [None if pd.isna(v) else v for v in parsed] == expected


class TestCommentsBeforeTheHeader:
    """Exports often open with `# source: ...` lines. Read as they were, the first comment
    became the header and pandas stopped on the first data row, so every script failed."""

    @pytest.mark.parametrize(
        ("text", "separator", "columns"),
        [
            # A comment with a comma in it used to decide both the header and the separator.
            (
                "# Source: bike export\n# Units: count, temp in C\n\ndate,count\n2024-01-01,3\n",
                ",",
                ["date", "count"],
            ),
            ("﻿# exported 2024-01-01\nid;price\n1;8,60\n", ";", ["id", "price"]),
            # A comment that splits like the header is still a comment: the header follows.
            ("# a, b\nid,name\n1,x\n", ",", ["id", "name"]),
        ],
    )
    def test_they_are_skipped(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        text: str,
        separator: str,
        columns: list[str],
    ) -> None:
        path = tmp_path / "data.csv"
        path.write_text(text, encoding="utf-8")
        assert data_utils.detect_sep(path) == separator
        df = data_utils.read_table(path, decimal="," if separator == ";" else ".")
        assert list(df.columns) == columns
        assert len(df) == 1
        assert "Skipped the first" in capsys.readouterr().out

    def test_a_header_that_starts_with_a_hash_is_kept(self, tmp_path: Path) -> None:
        path = tmp_path / "data.csv"
        path.write_text("#,name\n1,a\n2,b\n")
        df = data_utils.read_table(path)
        assert list(df.columns) == ["#", "name"]
        assert len(df) == 2

    def test_a_hash_inside_the_data_is_not_a_comment(self, tmp_path: Path) -> None:
        path = tmp_path / "data.csv"
        path.write_text("id,colour\n1,#ff0000\n2,#00ff00\n")
        assert data_utils.read_table(path)["colour"].tolist() == ["#ff0000", "#00ff00"]

    def test_the_profile_runs_and_says_so(self, tmp_path: Path) -> None:
        source = tmp_path / "data.csv"
        source.write_text("# Source: export, v2\nid,price\n1,3.5\n2,4.0\n3,5.5\n")
        script = DATA_SKILLS / "data-analysis" / "scripts" / "profile_data.py"
        result = run(script, "--data", str(source), "--out", str(tmp_path / "run"))
        assert result.returncode == 0, result.stderr
        assert "# Source: export, v2" in result.stdout
        assert (
            "comments before the header, were skipped"
            in (tmp_path / "run" / "files" / "profile.md").read_text()
        )


class TestTheCleaning:
    @pytest.fixture
    def source(self, tmp_path: Path) -> Path:
        path = tmp_path / "orders.csv"
        path.write_text(
            "order_id;zip;price;age;city\n"
            "001;02134;1.234,50;34;NYC\n"
            "002;02139;99,00;-1;New York\n"
            "002;02139;99,00;-1;New York\n"
            "003;10001;5.000,00;151;Boston\n"
        )
        return path

    def prepare(self, source: Path, spec: dict[str, object]) -> subprocess.CompletedProcess[str]:
        out = source.parent / "out"
        (source.parent / "cleaning.json").write_text(json.dumps(spec))
        return run(
            DATA_SKILLS / "prepare_table.py",
            "--data", str(source),
            "--spec", str(source.parent / "cleaning.json"),
            "--out", str(out),
        )  # fmt: skip

    def test_the_spec_is_applied_and_every_removal_is_explained(self, source: Path) -> None:
        result = self.prepare(
            source,
            {
                "types": {"order_id": "string", "zip": "string"},
                "missing": {"age": [-1]},
                "numbers": {"price": {"thousands": ".", "decimal": ","}},
                "categories": {"city": {"NYC": "New York"}},
                "duplicates": {"keep": "first", "reason": "exported twice"},
                "exclude": [{"column": "age", "outside": [0, 120], "reason": "impossible age"}],
            },
        )
        assert result.returncode == 0, result.stderr

        table, meta = data_utils.load_table(source.parent / "out")
        assert list(table["order_id"]) == ["001", "002"]
        assert list(table["zip"]) == ["02134", "02139"]
        assert list(table["price"]) == [1234.5, 99.0]
        assert list(table["city"]) == ["New York", "New York"]
        assert (meta["rows_source"], meta["rows"]) == (4, 2)

        removed = {s["step"]: s for s in meta["steps"] if s["step"] in ("duplicates", "exclude")}
        assert removed["duplicates"]["affected"] == 1
        assert "exported twice" in removed["duplicates"]["what"]
        assert removed["exclude"]["affected"] == 1
        assert "impossible age" in removed["exclude"]["what"]
        summary = (source.parent / "out" / "files" / "table.md").read_text()
        assert "exported twice" in summary and "impossible age" in summary

    @pytest.mark.parametrize(
        "spec",
        [
            {"duplicates": {"keep": "first"}},
            {"exclude": [{"column": "age", "outside": [0, 120]}]},
            {"typo": {}},
            {"numbers": {"no_such_column": {}}},
        ],
    )
    def test_a_spec_that_would_remove_rows_without_a_reason_or_is_wrong_is_refused(
        self, source: Path, spec: dict[str, object]
    ) -> None:
        result = self.prepare(source, spec)
        assert result.returncode != 0
        assert "cleaning spec" in result.stderr
        assert not (source.parent / "out" / "files" / data_utils.TABLE_FILE).exists()


class TestTheBacktestSize:
    """forecast_split.py sizes the backtest from the history, not a fixed count: a fixed
    3 windows left a one-day-ahead forecast scored on 3 points."""

    @pytest.fixture
    def series(self, tmp_path: Path):  # type: ignore[no-untyped-def]
        def make(periods: int, freq: str, n_series: int = 1) -> Path:
            dates = pd.date_range("2020-01-01", periods=periods, freq=freq)
            rows = pd.DataFrame(
                [
                    {"date": d, "id": f"s{k}", "y": float(i % 7 + k)}
                    for k in range(n_series)
                    for i, d in enumerate(dates)
                ]
            )
            source = tmp_path / f"source_{periods}_{freq}_{n_series}.csv"
            rows.to_csv(source, index=False)
            out = tmp_path / f"series_{periods}_{freq}_{n_series}"
            args = ["--data", str(source), "--time-col", "date", "--target", "y"]
            result = run(
                DATA_SKILLS / "prepare_series.py", *args, "--id-cols", "id", "--out", str(out)
            )
            assert result.returncode == 0, result.stderr
            return out / "files" / "series.csv"

        return make

    def windows(self, series: Path, *args: str) -> dict[str, object]:
        out = series.parent.parent / f"split_{series.parent.name}_{'_'.join(args)}"
        script = DATA_SKILLS / "time-series-forecasting" / "scripts" / "forecast_split.py"
        result = run(script, "--series", str(series), *args, "--out", str(out))
        assert result.returncode == 0, result.stderr
        meta = json.loads((out / "files" / "split.json").read_text())
        return {"n": len(meta["windows"]), "available": meta["windows_available"]}

    def test_a_share_of_the_available_windows_is_used(self, series) -> None:  # type: ignore[no-untyped-def]
        # 731 days, 1 day ahead: 717 fit after two weekly cycles; 10% of them.
        assert self.windows(series(731, "D"), "--horizon", "1") == {"n": 72, "available": 717}

    def test_short_histories_keep_at_least_three_windows(self, series) -> None:  # type: ignore[no-untyped-def]
        # 84 months, 6 ahead: 10 fit after two yearly cycles; 10% would be 1.
        assert self.windows(series(84, "MS"), "--horizon", "6")["n"] == 3

    def test_many_series_are_capped_by_the_fit_budget(self, series) -> None:  # type: ignore[no-untyped-def]
        assert (
            self.windows(series(400, "D", n_series=50), "--horizon", "1", "--max-fits", "200")["n"]
            == 4
        )

    def test_an_exact_number_still_wins(self, series) -> None:  # type: ignore[no-untyped-def]
        assert self.windows(series(731, "D"), "--horizon", "7", "--windows", "10")["n"] == 10


class TestTheMaseCheck:
    """MASE divides by the seasonal-naive error averaged over the whole history. On a
    series whose level or noise grows, that unit is too small for the recent periods, and
    a forecast comparison decided on MASE was read against 1 when it could not be."""

    CASES = """
import json, numpy as np, common
rng = np.random.default_rng(0)
t = np.arange(720)
weekly = 10 * np.sin(2 * np.pi * t / 7)
cases = {
    "stable": 100 + weekly + rng.normal(0, 5, 720),
    "growing": 100 + t + weekly + rng.normal(0, 5, 720) * (1 + t / 240),
    "level_shift": np.where(t < 360, 100, 250) + weekly + rng.normal(0, 5, 720),
    "short": np.arange(6.0),
}
out = {}
for name, y in cases.items():
    d = common.drift(y, 7)
    out[name] = None if d is None else common.is_drifting(d)
print(json.dumps(out))
"""

    def test_it_flags_only_the_non_stationary_series(self) -> None:
        scripts = DATA_SKILLS / "time-series-forecasting" / "scripts"
        result = subprocess.run(
            [sys.executable, "-c", self.CASES],
            cwd=scripts,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout) == {
            "stable": False,
            "growing": True,
            "level_shift": True,
            "short": None,
        }


class TestTheFeatures:
    """Other columns reach ARIMA and CatBoost as features, each declared known (its value
    for a period is usable when forecasting that period) or lagged (only past values).
    A wrong call either way makes the backtest see the future, so the rules that stop
    it are checked here: a known feature needs a reason and future values, a lagged one
    can't use a lag shorter than the horizon, and each model sees the value at the right
    period."""

    SCRIPTS = DATA_SKILLS / "time-series-forecasting" / "scripts"

    @pytest.fixture
    def source(self, tmp_path: Path) -> Path:
        # Hourly rows over 60 days, prepared as daily totals: the holiday is a flag in
        # every hour of the day, the temperature varies within the day.
        hours = pd.date_range("2024-01-01", periods=60 * 24, freq="h")
        day = (hours - hours[0]).days
        rows = pd.DataFrame(
            {
                "time": hours,
                "cnt": 10 + day % 7,
                "holiday": (day % 10 == 0).astype(int),
                "temp": day + hours.hour / 100,
            }
        )
        path = tmp_path / "hourly.csv"
        rows.to_csv(path, index=False)
        return path

    def prepare(
        self, source: Path, tmp_path: Path, spec: dict, *, future: bool = True
    ) -> subprocess.CompletedProcess[str]:
        spec_path = tmp_path / "features.json"
        spec_path.write_text(json.dumps(spec))
        args = ["--data", str(source), "--time-col", "time", "--target", "cnt"]
        args += ["--freq", "D", "--aggregate", "sum", "--features", str(spec_path)]
        if future:
            calendar = tmp_path / "calendar.csv"
            pd.DataFrame(
                {"time": pd.date_range("2024-03-01", periods=7, freq="D"), "holiday": 1}
            ).to_csv(calendar, index=False)
            args += ["--future", str(calendar)]
        return run(DATA_SKILLS / "prepare_series.py", *args, "--out", str(tmp_path / "series"))

    def split(self, tmp_path: Path, horizon: int) -> subprocess.CompletedProcess[str]:
        series = tmp_path / "series" / "files" / "series.csv"
        return run(
            self.SCRIPTS / "forecast_split.py",
            "--series", str(series), "--horizon", str(horizon), "--out", str(tmp_path / "run"),
        )  # fmt: skip

    SPEC: ClassVar[dict[str, dict[str, str]]] = {
        "holiday": {"use": "known", "aggregate": "max", "reason": "set in advance"},
        "temp": {"use": "lagged", "aggregate": "mean"},
    }

    def test_they_are_aggregated_per_period_with_the_future_values_apart(
        self, source: Path, tmp_path: Path
    ) -> None:
        assert self.prepare(source, tmp_path, self.SPEC).returncode == 0
        files = tmp_path / "series" / "files"
        series = pd.read_csv(files / "series.csv")
        assert list(series.columns) == ["id", "period", "y", "holiday", "temp"]
        assert series["holiday"].tolist()[:11] == [1] + [0] * 9 + [1]  # max: a day flag
        assert series["temp"].iloc[1] == pytest.approx(1.115)  # mean of 1.00 … 1.23
        future = pd.read_csv(files / "future_features.csv")
        assert list(future.columns) == ["id", "period", "holiday"]  # known features only
        assert future["period"].iloc[0] == "2024-03-01"
        assert json.loads((files / "series.json").read_text())["features"]["temp"] == {
            "use": "lagged",
            "aggregate": "mean",
        }
        assert (files / "features.json").exists()  # the spec is kept with the run

    @pytest.mark.parametrize(
        ("rule", "message"),
        [
            ({"use": "known"}, "needs a `reason`"),
            ({"use": "sometimes"}, "`use` must be one of"),
            ({"use": "known", "reason": "r", "lags": [1]}, "applies only to lagged"),
            ({"use": "lagged", "lags": [0]}, "positive integers"),
        ],
    )
    def test_the_spec_is_checked(
        self, source: Path, tmp_path: Path, rule: dict, message: str
    ) -> None:
        result = self.prepare(source, tmp_path, {"holiday": rule})
        assert result.returncode != 0
        assert message in result.stderr

    def test_a_known_feature_without_future_values_stops_the_split(
        self, source: Path, tmp_path: Path
    ) -> None:
        assert self.prepare(source, tmp_path, self.SPEC, future=False).returncode == 0
        result = self.split(tmp_path, 1)
        assert result.returncode != 0
        assert "--future" in result.stderr

    def test_a_lag_shorter_than_the_horizon_is_refused(self, source: Path, tmp_path: Path) -> None:
        spec = {"temp": {"use": "lagged", "lags": [1, 7]}}
        assert self.prepare(source, tmp_path, spec).returncode == 0
        result = self.split(tmp_path, 3)
        assert result.returncode != 0
        assert "lags [1]" in result.stderr

    def test_each_model_sees_the_value_it_may_know(self, source: Path, tmp_path: Path) -> None:
        """Known: the value of the period forecast, also past the history. Lagged: the
        value one horizon back, never the period's own."""
        assert self.prepare(source, tmp_path, self.SPEC).returncode == 0
        assert self.split(tmp_path, 2).returncode == 0
        code = f"""
import json, numpy as np, common
args = common.base_parser("x").parse_args(["--out", {str(tmp_path / "run")!r}])
panel = common.load_panel(args)
rows = np.array([0])
T = len(panel.periods)
cols = panel.feature_columns(rows, 20)
ahead = panel.feature_columns(rows, T + 1)
hist = panel.feature_history(0, 0, 3)
print(json.dumps({{
    "spec": panel.feature_spec["temp"]["lags"],
    "names": sorted(cols),
    "holiday_20": float(cols["holiday"][0]),
    "temp_lag_at_20": float(cols["temp_lag2"][0]),
    "holiday_ahead": float(ahead["holiday"][0]),
    "temp_lag_ahead": float(ahead["temp_lag2"][0]),
    "hist_lag": [None if np.isnan(v) else float(v) for v in hist["temp_lag2"]],
}}))
"""
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=self.SCRIPTS, capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
        seen = json.loads(result.stdout.strip().splitlines()[-1])
        assert seen["spec"] == [2]  # default lag: the horizon
        assert seen["names"] == ["holiday", "temp_lag2"]
        assert seen["holiday_20"] == 1.0  # day 20 is a holiday (every 10th day)
        assert seen["temp_lag_at_20"] == pytest.approx(18.115)  # day 18's mean
        assert seen["holiday_ahead"] == 1.0  # from future_features.csv
        assert seen["temp_lag_ahead"] == pytest.approx(59.115)  # the last observed day
        assert seen["hist_lag"] == [None, None, pytest.approx(0.115)]


class TestFeaturesForPreparedSeries:
    """A data-analysis run prepares the series without features, and the forecast reuses
    them. Without a trace of the columns left out, the forecast would silently run
    without a holiday flag the source had: so they are recorded, the split warns, and
    --from-series adds them without re-deciding the target's preparation."""

    SCRIPTS = DATA_SKILLS / "time-series-forecasting" / "scripts"
    SPEC: ClassVar[dict[str, dict[str, str]]] = {
        "holiday": {"use": "known", "aggregate": "max", "reason": "set in advance"},
    }

    @pytest.fixture
    def analysis(self, tmp_path: Path) -> Path:
        """Daily series prepared as an analysis would: no --features."""
        days = pd.date_range("2024-01-01", periods=60, freq="D")
        index = pd.RangeIndex(len(days))
        source = tmp_path / "daily.csv"
        pd.DataFrame(
            {"time": days, "cnt": 10 + index % 7, "holiday": (index % 10 == 0).astype(int)}
        ).to_csv(source, index=False)
        result = run(
            DATA_SKILLS / "prepare_series.py",
            "--data", str(source), "--time-col", "time", "--target", "cnt",
            "--out", str(tmp_path / "analysis"),
        )  # fmt: skip
        assert result.returncode == 0, result.stderr
        assert "NOTE: 1 other numeric columns are not carried" in result.stdout
        return tmp_path / "analysis" / "files" / "series.csv"

    def add_features(
        self, series: Path, tmp_path: Path, *extra: str
    ) -> subprocess.CompletedProcess[str]:
        spec = tmp_path / "features.json"
        spec.write_text(json.dumps(self.SPEC))
        calendar = tmp_path / "calendar.csv"
        pd.DataFrame(
            {"time": pd.date_range("2024-03-01", periods=7, freq="D"), "holiday": 0}
        ).to_csv(calendar, index=False)
        return run(
            DATA_SKILLS / "prepare_series.py",
            "--from-series", str(series), "--features", str(spec), "--future", str(calendar),
            *extra,
        )  # fmt: skip

    def split(self, series: Path, out: Path) -> subprocess.CompletedProcess[str]:
        return run(
            self.SCRIPTS / "forecast_split.py",
            "--series", str(series), "--horizon", "2", "--out", str(out),
        )  # fmt: skip

    def test_the_columns_left_out_are_recorded_and_the_split_warns(
        self, analysis: Path, tmp_path: Path
    ) -> None:
        meta = json.loads(analysis.with_name("series.json").read_text())
        assert meta["unused_columns"] == ["holiday"]
        result = self.split(analysis, tmp_path / "forecast")
        assert result.returncode == 0, result.stderr
        assert "WARNING: the series carry no features" in result.stdout
        assert "--from-series" in result.stdout

    def test_from_series_adds_the_features_to_the_same_target(
        self, analysis: Path, tmp_path: Path
    ) -> None:
        result = self.add_features(analysis, tmp_path, "--out", str(tmp_path / "forecast"))
        assert result.returncode == 0, result.stderr
        files = tmp_path / "forecast" / "files"
        before, after = pd.read_csv(analysis), pd.read_csv(files / "series.csv")
        assert list(after.columns) == ["id", "period", "y", "holiday"]
        pd.testing.assert_frame_equal(after[["id", "period", "y"]], before)
        meta = json.loads((files / "series.json").read_text())
        assert meta["extends"] == str(analysis.resolve())
        assert meta["unused_columns"] == []
        assert (files / "future_features.csv").exists()
        split = self.split(files / "series.csv", tmp_path / "forecast")
        assert split.returncode == 0, split.stderr
        assert "WARNING: the series carry no features" not in split.stdout

    @pytest.mark.parametrize(
        ("extra", "message"),
        [
            (("--target", "cnt"), "come from its series.json"),
            (("--freq", "W"), "come from its series.json"),
        ],
    )
    def test_the_target_settings_cannot_be_changed(
        self, analysis: Path, tmp_path: Path, extra: tuple[str, ...], message: str
    ) -> None:
        result = self.add_features(analysis, tmp_path, *extra, "--out", str(tmp_path / "f"))
        assert result.returncode != 0
        assert message in result.stderr

    def test_it_needs_features_and_a_folder_of_its_own(
        self, analysis: Path, tmp_path: Path
    ) -> None:
        bare = run(
            DATA_SKILLS / "prepare_series.py",
            "--from-series", str(analysis), "--out", str(tmp_path / "f"),
        )  # fmt: skip
        assert bare.returncode != 0
        assert "pass --features" in bare.stderr
        same = self.add_features(analysis, tmp_path, "--out", str(tmp_path / "analysis"))
        assert same.returncode != 0
        assert "would replace the series" in same.stderr

    def test_a_source_changed_since_is_refused(self, analysis: Path, tmp_path: Path) -> None:
        source = tmp_path / "daily.csv"
        df = pd.read_csv(source)
        df.loc[5, "cnt"] += 100
        df.to_csv(source, index=False)
        result = self.add_features(analysis, tmp_path, "--out", str(tmp_path / "forecast"))
        assert result.returncode != 0
        assert "no longer matches" in result.stderr
