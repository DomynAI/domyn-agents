"""Shared helpers for the time-series-forecasting scripts.

Not run directly. Every entry-point script (`forecast_*.py`) imports this module,
which is why they all live in one directory.

The scripts read only prepared series (series.csv / series.json, written by
prepare_series.py; see series_format.py). What lives here is what has to be
*identical* across models for the comparison to be fair: which series take part,
which periods are held out (the backtest windows), which features each model may see
at each period, and how errors are scaled. The
`forecast_split.py` settles them once and writes them to `split.json` in the output
folder; the model scripts take only `--out` and read everything from there.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# data_utils.py is shared by all the data skills and lives one level above them, in
# builtin/data/. Appended, not prepended: in an output folder, the copy that
# record_run put next to this file comes first.
sys.path.append(str(Path(__file__).resolve().parents[2]))
# seaborn_style.py, the chart style shared by every builtin skill, lives with the palette
# in builtin/resources/; scripts that draw charts import it after this module.
# Found from this file's place in the tree, never from the working directory or
# PYTHONPATH, and inserted right after the script's own directory: in an output folder
# the copy record_run put there still comes first, and nothing else can stand in for it.
sys.path.insert(1, str(Path(__file__).resolve().parents[3] / "resources"))
import data_utils
import series_format as sf

META_FILE = "split.json"

# A change between the first and last third of a history this large makes it non-stationary
# for MASE (see drift).
DRIFT = 1.5

# Order of the commands in run.sh, whatever order the scripts were run in.
RUN_ORDER = ("prepare_table", "prepare_series", "split", "baseline", "catboost", "compare")


def split_parser(description: str) -> argparse.ArgumentParser:
    """The arguments of forecast_split.py: which series, and which periods are held out."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--series",
        required=True,
        type=Path,
        help="series.csv written by prepare_series.py (or the folder holding it)",
    )
    p.add_argument("--horizon", required=True, type=int, help="how many periods ahead to forecast")
    windows = p.add_mutually_exclusive_group()
    windows.add_argument(
        "--windows-share",
        type=float,
        default=0.10,
        help="share of the available backtest windows to use, 0-1 (default 0.10; see "
        "Panel._make_windows). The usual way to size the backtest",
    )
    windows.add_argument(
        "--windows",
        type=int,
        default=None,
        help="an exact number of backtest windows instead of a share",
    )
    p.add_argument(
        "--max-fits",
        type=int,
        default=2000,
        help="cap on windows × series, the number of ARIMA fits per run (default 2000)",
    )
    p.add_argument(
        "--season-length",
        type=int,
        default=None,
        help="periods per seasonal cycle (default: the one in series.json)",
    )
    p.add_argument("--out", required=True, type=Path, help="output folder (created if missing)")
    return p


def base_parser(description: str) -> argparse.ArgumentParser:
    """The arguments every model script accepts: the folder forecast_split.py wrote to.

    The series, horizon, windows and season length are read from its split.json, so a
    model script cannot be run on other series or another backtest.
    """
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--out", required=True, type=Path, help="output folder prepared by forecast_split.py"
    )
    p.add_argument("--seed", type=int, default=42)
    return p


def load_panel(args: argparse.Namespace) -> Panel:
    """The panel and windows forecast_split.py settled in `args.out`."""
    meta_path = data_utils.output_dir(args.out) / META_FILE
    if not meta_path.exists():
        sys.exit(f"{meta_path} not found: run forecast_split.py with --out {args.out} first.")
    meta = json.loads(meta_path.read_text())
    settings = meta["settings"]
    args.windows_reason = meta.get("windows_reason")
    args.series = Path(settings["series"])
    args.horizon, args.windows = settings["horizon"], settings["windows"]
    args.season_length = settings["season_length"]
    return Panel(args)


class Panel:
    """The series on one calendar: `values[i, t]` is series i at period t.

    Periods before a series starts are NaN. Series that stop before the last period
    are left out, so the backtest windows are the same dates for every series.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.out: Path = data_utils.output_dir(args.out)

        frame, self.meta = sf.load_series(args.series)
        self.series_path = data_utils.find_input(args.series, sf.SERIES_FILE)
        self.target: str = self.meta["target"]
        self.freq: str = self.meta["freq"]
        self.season: int = args.season_length or int(self.meta["season_length"])

        last = frame["period"].max()
        ends = frame.groupby("id")["period"].max()
        stopped = ends[ends < last]
        if len(stopped):
            print(
                f"WARNING: {len(stopped)} series end before {last} and are left out "
                f"(e.g. {stopped.index[:5].tolist()})"
            )
            frame = frame[~frame["id"].isin(stopped.index)]
        self.stopped = stopped.index.tolist()

        self.periods = pd.period_range(frame["period"].min(), last, freq=self.freq)
        self.ids = np.array(sorted(frame["id"].unique()))
        wide = frame.pivot(index="id", columns="period", values="y")
        self.values = wide.reindex(index=self.ids, columns=self.periods).to_numpy(dtype=float)
        self.start = np.array([int(np.flatnonzero(~np.isnan(r))[0]) for r in self.values])
        self._load_features(frame)
        self._make_windows()
        self._write_meta()

    def _load_features(self, frame: pd.DataFrame) -> None:
        """`X[name][i, t]`: feature `name` of series i at period t, for the history and
        the --horizon periods after it (NaN where unknown).

        What a model may see at period t (`feature_columns`, `feature_history`): a known
        feature's value at t, and a lagged feature's values at t - k for its lags, all at
        least --horizon, so from any forecast origin every value used was observed before
        it. A known feature's future values come from future_features.csv, and all of
        them are required: without them the forecast itself cannot be made.
        """
        h, T, n = self.args.horizon, len(self.periods), len(self.ids)
        spec = self.meta.get("features") or {}
        future = sf.load_future(self.series_path, self.meta)
        ahead = pd.period_range(self.periods[-1] + 1, periods=h, freq=self.freq)
        self.X: dict[str, np.ndarray] = {}
        self.feature_spec: dict[str, dict] = {}
        for name, rule in spec.items():
            x = np.full((n, T + h), np.nan)
            wide = frame.pivot(index="id", columns="period", values=name)
            x[:, :T] = wide.reindex(index=self.ids, columns=self.periods).to_numpy(dtype=float)
            resolved = dict(rule)
            if rule["use"] == "known":
                if future is not None and len(future):
                    wide = future.pivot(index="id", columns="period", values=name)
                    x[:, T:] = wide.reindex(index=self.ids, columns=ahead).to_numpy(dtype=float)
                if np.isnan(x[:, T:]).any():
                    sys.exit(
                        f"Known feature {name!r} has no values for some of the {h} periods to "
                        f"forecast ({ahead[0]} to {ahead[-1]}). Pass them to prepare_series.py "
                        "with --future (a file with the time and id columns and this one), or "
                        "declare the feature lagged if its future values are not known."
                    )
            else:
                resolved["lags"] = rule.get("lags") or [h]
                short = [k for k in resolved["lags"] if k < h]
                if short:
                    sys.exit(
                        f"Lagged feature {name!r}: lags {short} are shorter than the horizon "
                        f"({h}), so forecasting {h} periods ahead would need values from inside "
                        "the forecast window. Use lags of at least the horizon, or declare it "
                        "known if its values really are known in advance."
                    )
            self.X[name] = x
            self.feature_spec[name] = resolved
        self.max_feature_lag = max(
            (max(f["lags"]) for f in self.feature_spec.values() if f["use"] == "lagged"),
            default=0,
        )
        if self.feature_spec:
            print(
                "features: "
                + ", ".join(
                    name if f["use"] == "known" else f"{name} (lags {f['lags']})"
                    for name, f in self.feature_spec.items()
                )
            )

    def feature_columns(self, rows: np.ndarray, t: int) -> dict[str, np.ndarray]:
        """The feature values the models see for `rows` at period t."""
        cols: dict[str, np.ndarray] = {}
        for name, rule in self.feature_spec.items():
            x = self.X[name]
            if rule["use"] == "known":
                cols[name] = x[rows, t]
            else:
                for k in rule["lags"]:
                    cols[f"{name}_lag{k}"] = (
                        x[rows, t - k] if t - k >= 0 else np.full(len(rows), np.nan)
                    )
        return cols

    def feature_history(self, i: int, t0: int, t1: int) -> dict[str, np.ndarray]:
        """The same columns for series i over periods t0 <= t < t1."""
        cols: dict[str, np.ndarray] = {}
        t = np.arange(t0, t1)
        for name, rule in self.feature_spec.items():
            x = self.X[name][i]
            if rule["use"] == "known":
                cols[name] = x[t0:t1]
            else:
                for k in rule["lags"]:
                    cols[f"{name}_lag{k}"] = np.where(
                        t - k >= 0, x[np.clip(t - k, 0, None)], np.nan
                    )
        return cols

    def _make_windows(self) -> None:
        """Choose the backtest windows: consecutive, --horizon periods each, the last one
        ending at the last period.

        Available windows are the ones that fit after a minimum training span: two
        seasonal cycles, and never less than `min_history`. Without --windows, the run
        uses --windows-share of them (at least 3 when there are 3), so the backtest grows
        with the history: 731 days forecast one day ahead get 72 one-day windows, 7 years
        of monthly data forecast 6 months ahead get 3 (the floor: 10% of 10 is 1). --max-fits then caps
        windows × series, since every window refits ARIMA for every series.
        """
        a, h, T = self.args, self.args.horizon, len(self.periods)
        if h < 1:
            sys.exit("--horizon must be at least 1")
        min_train = max(self.min_history, 2 * self.season)
        if T - min_train < h:  # too short for two cycles: settle for the per-series minimum
            min_train = self.min_history
        self.windows_available = max(0, (T - min_train) // h)
        if self.windows_available < 1:
            sys.exit(
                f"{T} periods are too few to backtest a {h}-period forecast: at least "
                f"{min_train + h} are needed. Lower --horizon."
            )
        series = len(self.ids)
        if a.windows is not None:
            w, how = a.windows, getattr(a, "windows_reason", None) or "set with --windows"
            if not 1 <= w <= self.windows_available:
                sys.exit(
                    f"--windows {w}: between 1 and {self.windows_available} fit in {T} periods "
                    f"with a {h}-period horizon."
                )
        else:
            share = getattr(a, "windows_share", 0.10)
            if not 0 < share <= 1:
                sys.exit("--windows-share must be above 0 and at most 1")
            w = max(min(3, self.windows_available), math.ceil(share * self.windows_available))
            how = f"{share:.0%} of the {self.windows_available} available"
            cap = max(1, getattr(a, "max_fits", 2000) // max(series, 1))
            if w > cap:
                w, how = cap, f"{how}, capped by --max-fits at {cap} for {series} series"
        a.windows, self.windows_reason = w, how
        self.origins = [T - k * h for k in range(w, 0, -1)]  # oldest first
        print(
            f"Backtest: {w} windows of {h} ({how}), about {w * h * series} forecast points "
            f"per model, training on at least {self.origins[0]} periods"
        )
        if 2 * self.season > self.origins[0]:
            print(
                f"WARNING: less than two seasonal cycles ({self.season} periods each) before the "
                "first backtest window: seasonality can barely be learned or tested."
            )

    @property
    def min_history(self) -> int:
        """Periods a series needs before an origin to take part in that window."""
        return max(2 * self.args.horizon, 8)

    def _write_meta(self) -> None:
        h = self.args.horizon
        meta = {
            "series_file": str(self.series_path),
            "target": self.target,
            "freq": self.freq,
            "season_length": self.season,
            "horizon": h,
            "windows_available": self.windows_available,
            "windows_reason": self.windows_reason,
            "windows": [
                {
                    "window": k + 1,
                    "train_until": str(self.periods[o - 1]),
                    "first_forecast_period": str(self.periods[o]),
                    "last_forecast_period": str(self.periods[o + h - 1]),
                    "series_included": int(sum(self.usable(i, o) for i in range(len(self.ids)))),
                }
                for k, o in enumerate(self.origins)
            ],
            "series": len(self.ids),
            "series_left_out_ended_early": self.stopped,
            "periods": len(self.periods),
            "first_period": str(self.periods[0]),
            "last_period": str(self.periods[-1]),
            "future_periods": self.future_periods(),
            "features": self.feature_spec or None,  # None: as in a split made before features
            "settings": {
                "series": str(self.series_path),
                "horizon": h,
                "windows": self.args.windows,
                "season_length": self.season,
            },
        }
        path = self.out / META_FILE
        if path.exists():
            before = json.loads(path.read_text())
            keys = (
                "series_file", "season_length", "horizon", "windows", "series", "periods",
                "features",
            )  # fmt: skip
            if {k: before.get(k) for k in keys} != {k: meta[k] for k in keys}:
                sys.exit(
                    f"{self.out} was already split with other series or settings (was "
                    f"{before.get('settings')}). Models trained there would not be comparable: "
                    "use a new --out folder."
                )
        path.write_text(json.dumps(meta, indent=2))

    def history(self, i: int, until: int) -> np.ndarray:
        """Series i before period index `until` (exclusive), from its first period."""
        return self.values[i, self.start[i] : until]

    def usable(self, i: int, origin: int) -> bool:
        return origin - self.start[i] >= self.min_history

    def mase_scale(self, i: int, origin: int) -> float:
        """Mean absolute seasonal-naive error on the history: the unit MASE divides by."""
        y = self.history(i, origin)
        lag = self.season if len(y) > self.season else 1
        scale = float(np.mean(np.abs(y[lag:] - y[:-lag]))) if len(y) > lag else float("nan")
        return scale if scale > 0 else float("nan")

    def future_periods(self) -> list[str]:
        last = self.periods[-1]
        return [str(last + k) for k in range(1, self.args.horizon + 1)]


def backtest_frame(
    panel: Panel,
    window: int,
    i: int,
    pred: np.ndarray,
    intervals: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """One series' forecast for one backtest window, with the actuals and the MASE scale."""
    origin, h = panel.origins[window], panel.args.horizon
    frame = pd.DataFrame(
        {
            "id": panel.ids[i],
            "window": window + 1,
            "step": np.arange(1, h + 1),
            "period": [str(p) for p in panel.periods[origin : origin + h]],
            "actual": panel.values[i, origin : origin + h],
            "pred": pred,
            "mase_scale": panel.mase_scale(i, origin),
        }
    )
    for name, values in (intervals or {}).items():
        frame[name] = values
    return frame


def future_frame(
    panel: Panel, i: int, pred: np.ndarray, intervals: dict[str, np.ndarray] | None = None
) -> pd.DataFrame:
    """One series' forecast past the last period."""
    frame = pd.DataFrame(
        {
            "id": panel.ids[i],
            "step": np.arange(1, panel.args.horizon + 1),
            "period": panel.future_periods(),
            "pred": pred,
        }
    )
    for name, values in (intervals or {}).items():
        frame[name] = values
    return frame


def save_forecasts(
    out: Path, model: str, backtest: list[pd.DataFrame], future: list[pd.DataFrame]
) -> None:
    """Backtest and future forecasts, in the one format the compare script reads."""
    pd.concat(backtest, ignore_index=True).to_csv(out / f"{model}_backtest.csv", index=False)
    pd.concat(future, ignore_index=True).to_csv(out / f"{model}_forecast.csv", index=False)


def forecast_metrics(frame: pd.DataFrame) -> dict[str, float]:
    """The metrics every model is scored on, over the backtest points in `frame`.

    MASE: average error in units of the seasonal-naive error on each series' own history
    (on a stable series, below 1 beats "same as one season ago" in-sample; comparable
    across series of any size). On a non-stationary one (see drift) the unit doesn't fit
    the recent periods, and the values can't be read against 1. WAPE: total absolute error as a share of the total actual value. Bias: total
    over- (positive) or under-forecast as a share of the total. Coverage: share of actuals
    inside the 80% / 95% prediction intervals, when a model gives them.
    """
    err = frame["pred"] - frame["actual"]
    abs_err = err.abs()
    total = frame["actual"].abs().sum()
    scaled = (abs_err / frame["mase_scale"]).dropna()
    out = {
        "MASE": float(scaled.mean()) if len(scaled) else float("nan"),
        "WAPE_%": float(100 * abs_err.sum() / total) if total else float("nan"),
        "bias_%": float(100 * err.sum() / total) if total else float("nan"),
        "MAE": float(abs_err.mean()),
        "RMSE": float(np.sqrt((err**2).mean())),
    }
    for level in (80, 95):
        lo, hi = f"lo{level}", f"hi{level}"
        if lo in frame and frame[lo].notna().any():
            has = frame[lo].notna()
            inside = (frame["actual"] >= frame[lo]) & (frame["actual"] <= frame[hi])
            out[f"coverage{level}_%"] = float(100 * inside[has].mean())
    return out


def drift(y: np.ndarray, season: int) -> dict[str, float] | None:
    """How much the level and the seasonal-naive error change from the first to the last
    third of a history. None when a third is too short to tell."""
    y = y[~np.isnan(y)]
    lag = season if len(y) > 3 * (season + 2) else 1
    k = len(y) // 3
    if k < lag + 2:
        return None
    early, late = y[:k], y[-k:]
    scale = [np.mean(np.abs(part[lag:] - part[:-lag])) for part in (early, late)]
    typical = np.mean(np.abs(y))
    level = abs(late.mean() - early.mean()) / typical if typical else 0.0
    error = max(scale) / min(scale) if min(scale) > 0 else float("inf")
    return {"level_change": float(level), "error_ratio": float(error)}


def is_drifting(d: dict[str, float]) -> bool:
    """A level change of half the typical value counts the same as a 1.5x error ratio."""
    return d["level_change"] >= DRIFT - 1 or d["error_ratio"] >= DRIFT


markdown_table = data_utils.markdown_table
write_json = data_utils.write_json


def record_run(out: Path, script: Path, step: str) -> None:
    """Copy the script (with this module, series_format and data_utils) into the output
    folder, and log its command in run.sh."""
    here = Path(__file__).resolve()
    data_utils.record_run(out, script, RUN_ORDER, (here, Path(sf.__file__)))
