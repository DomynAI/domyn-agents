"""The prepared-series format shared by the data-analysis and time-series-forecasting skills.

It lives in `builtin/data/`, next to data_utils.py, and both skills' scripts import it
from there, so a series prepared once by `prepare_series.py` is read by both without
being prepared again.

The format is two files in one folder:

- `series.csv` — long format, one row per series per period, sorted by id then period:

      id,period,y
      north,2024-01,1520.0
      north,2024-02,1611.0

  `id` names the series (`series` when there is only one); `period` is a pandas Period
  label (`2024-01-15` daily, `2024-01-01/2024-01-07` weekly, `2024-01` monthly,
  `2024Q1` quarterly, `2024` yearly, `2024-01-15 13:00` hourly); `y` is the value.
  Every series is on a regular calendar from its first period to its last, with no
  missing periods and no missing values.
- `series.json` — what the file is: `target` (the name of what `y` measures), `freq`,
  `season_length`, the source file and the preparation settings.

Optionally, **features**: other columns measured per series and period (a holiday flag,
the temperature), declared in a features spec (see `check_feature_spec`) and written
as extra columns of `series.csv`, after `y`. Each is declared `known` (its value for a
period is known before that period, like a holiday: models may use the value of the
period they forecast) or `lagged` (only known afterwards, like the weather: models may
only use past values). `series.json` holds the spec under `features`. Known features
need their values for the periods to forecast too: those go in `future_features.csv`
(`id`, `period`, one column per known feature), next to `series.csv`.

Scripts consume it with `load_series(path)`, which checks those guarantees again
rather than trusting them.
"""

import json
import sys
from pathlib import Path
from typing import Any

import data_utils
import numpy as np
import pandas as pd

FORMAT_VERSION = 2  # 2 added features; a version 1 file is read as one without them
SERIES_FILE = "series.csv"
SERIES_META = "series.json"
FUTURE_FILE = "future_features.csv"
FEATURE_USES = ("known", "lagged")
FEATURE_AGGREGATES = ("mean", "sum", "max", "min", "last")
SINGLE_ID = "series"
ID_JOIN = " | "

# Periods in one seasonal cycle, by frequency.
SEASON_LENGTH = {"min": 60, "h": 24, "D": 7, "W": 52, "M": 12, "Q": 4, "Y": 1}


def period_freq(freq: str) -> str:
    """A pandas offset alias (MS, ME, QS-OCT, W-SUN, ...) as a Period frequency."""
    if freq.startswith(("B", "C")):
        sys.exit(
            "Business-day data is not supported as a frequency: aggregate it to weeks "
            "(--freq W --aggregate ...), or use --freq D with --fill-gaps interpolate."
        )
    if freq.startswith("W"):
        return freq if "-" in freq else "W-SUN"
    for prefix in ("min", "M", "Q", "Y", "A"):
        if freq.startswith(prefix):
            return {"A": "Y"}.get(prefix, prefix)
    if freq.upper() in ("H", "h"):
        return "h"
    return freq


def season_length_for(freq: str) -> int:
    for prefix, length in SEASON_LENGTH.items():
        if freq.startswith(prefix):
            return length
    return 1


def infer_freq(times: pd.Series, ids: pd.Series) -> str:
    """The frequency of the longest series' dates."""
    longest = ids.value_counts().index[0]
    dates = pd.DatetimeIndex(sorted(times[ids == longest].unique()))
    freq = pd.infer_freq(dates) if len(dates) >= 3 else None
    if freq is None:
        gaps = pd.Series(dates).diff().dropna()
        common_gap = gaps.mode().iloc[0] if len(gaps) else "unknown"
        sys.exit(
            f"Could not infer the period from the dates (the most common gap is {common_gap}). "
            "Pass --freq (D, W, M, Q, Y, h), and --aggregate if several rows share a period."
        )
    print(f"Inferred frequency: {freq}")
    return freq


def check_feature_spec(spec: Any) -> dict[str, dict[str, Any]]:
    """The features spec, validated: `{column: {"use": ..., ...}}`.

        {
          "holiday": {"use": "known", "aggregate": "max",
                      "reason": "public holidays are set years ahead"},
          "temp":    {"use": "lagged", "aggregate": "mean", "lags": [1, 7]}
        }

    - `use` (required): `known` or `lagged` (see the module docstring). `known` needs a
      `reason`, because it is a claim that the value is available before the period,
      and a wrong claim leaks the future into every backtest.
    - `aggregate`: how rows sharing a period combine (default `mean`), as `--aggregate`
      does for the target.
    - `lags` (`lagged` only): which past periods to use (default: the forecast
      horizon). Every lag must be at least the horizon, checked when the backtest is
      set up: a shorter one would need a value from inside the forecast window.
    """
    if not isinstance(spec, dict) or not spec:
        sys.exit("features spec: expected a non-empty object {column: {...}}")
    out: dict[str, dict[str, Any]] = {}
    for name, rule in spec.items():
        where = f"features spec, {name!r}"
        if name in ("id", "period", "y"):
            sys.exit(f"{where}: the name is reserved by the series format; rename the column")
        if not isinstance(rule, dict):
            sys.exit(f'{where}: expected an object like {{"use": "known", "reason": "..."}}')
        unknown = set(rule) - {"use", "aggregate", "lags", "reason"}
        if unknown:
            sys.exit(f"{where}: unknown keys {sorted(unknown)}")
        use = rule.get("use")
        if use not in FEATURE_USES:
            sys.exit(f"{where}: `use` must be one of {list(FEATURE_USES)}, not {use!r}")
        if use == "known" and not str(rule.get("reason") or "").strip():
            sys.exit(f"{where}: a known feature needs a `reason` (why it is known in advance)")
        aggregate = rule.get("aggregate", "mean")
        if aggregate not in FEATURE_AGGREGATES:
            sys.exit(f"{where}: `aggregate` must be one of {list(FEATURE_AGGREGATES)}")
        lags = rule.get("lags")
        if lags is not None and use != "lagged":
            sys.exit(f"{where}: `lags` applies only to lagged features")
        if lags is not None and (
            not isinstance(lags, list)
            or not lags
            or not all(isinstance(k, int) and k >= 1 for k in lags)
        ):
            sys.exit(f"{where}: `lags` must be a list of positive integers")
        out[str(name)] = {
            "use": use,
            "aggregate": aggregate,
            **({"lags": sorted(set(lags))} if lags else {}),
            **({"reason": rule["reason"]} if rule.get("reason") else {}),
        }
    return out


def to_regular(
    frame: pd.DataFrame,
    freq: str,
    aggregate: str | None,
    fill_gaps: str | None,
    features: dict[str, dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, int]:
    """(id, time, y, *features) rows as the canonical (id, period, y, *features) frame,
    and how many values were filled.

    Several rows in one period are refused unless `aggregate` says how to combine them;
    missing periods or values inside a series are refused unless `fill_gaps` says how
    to fill them. A series starts at its first row: the periods before it are not gaps.
    Features combine by their own `aggregate`. Their gaps are refused too, and with
    `fill_gaps` either way they are interpolated: zero is a count's "nothing happened",
    never a holiday flag's or a temperature's.
    """
    features = features or {}
    frame = frame.assign(period=frame["time"].dt.to_period(freq))
    dup = frame.duplicated(["id", "period"], keep=False)
    if dup.any() and not aggregate:
        example = frame.loc[dup, ["id", "period"]].head(3).astype(str).values.tolist()
        sys.exit(
            f"{int(dup.sum())} rows share a series and a {freq} period (e.g. {example}). "
            "Pass --aggregate sum|mean|last to combine them, or a finer --freq."
        )
    if dup.any():
        print(f"Combining {int(dup.sum())} rows that share a period with --aggregate {aggregate}")
    rules = {"y": aggregate or "last", **{n: f["aggregate"] for n, f in features.items()}}
    frame = frame.sort_values("time").groupby(["id", "period"], as_index=False).agg(rules)

    parts, gaps, feature_gaps = [], [], []
    for sid, group in frame.groupby("id", sort=True):
        periods = pd.period_range(group["period"].min(), group["period"].max(), freq=freq)
        g = group.set_index("period").reindex(periods)
        y = g["y"]
        missing = y.index[y.isna()]
        gaps += [(sid, str(p)) for p in missing]
        if len(missing) and fill_gaps == "zero":
            y = y.fillna(0.0)
        elif len(missing) and fill_gaps == "interpolate":
            y = y.interpolate(limit_direction="both")
        part = pd.DataFrame({"id": sid, "period": y.index.astype(str), "y": y.to_numpy()})
        for name in features:
            x = g[name]
            feature_gaps += [(sid, name, str(p)) for p in x.index[x.isna()]]
            part[name] = (x.interpolate(limit_direction="both") if fill_gaps else x).to_numpy()
        parts.append(part)
    if (gaps or feature_gaps) and not fill_gaps:
        detail = f"e.g. {gaps[:5]}" if gaps else f"in features, e.g. {feature_gaps[:5]}"
        sys.exit(
            f"{len(gaps) + len(feature_gaps)} missing periods or values inside the series "
            f"({detail}). Pass --fill-gaps zero when a missing row means nothing happened "
            "(no sales that day), or --fill-gaps interpolate for a measurement that was not "
            "recorded. Feature gaps are interpolated either way."
        )
    if gaps:
        print(f"Filled {len(gaps)} missing periods / values with --fill-gaps {fill_gaps}")
    if feature_gaps:
        print(f"Interpolated {len(feature_gaps)} missing feature values")
    return pd.concat(parts, ignore_index=True), len(gaps) + len(feature_gaps)


def write_series(
    out: Path, frame: pd.DataFrame, meta: dict[str, Any], future: pd.DataFrame | None = None
) -> Path:
    """Write series.csv and series.json (and future_features.csv, when there are known
    features) to `out`; refuse to replace a different preparation."""
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / SERIES_META
    if meta_path.exists():
        before = json.loads(meta_path.read_text())
        keys = ("source", "target", "freq")
        if {k: before.get(k) for k in keys} != {k: meta.get(k) for k in keys}:
            sys.exit(
                f"{out} already holds series prepared from {before.get('source')} "
                f"({before.get('target')}, {before.get('freq')}). Use another --out folder."
            )
    frame.to_csv(out / SERIES_FILE, index=False)
    if future is not None:
        future.to_csv(out / FUTURE_FILE, index=False)
    else:
        (out / FUTURE_FILE).unlink(missing_ok=True)
    meta_path.write_text(json.dumps({"format_version": FORMAT_VERSION, **meta}, indent=2))
    return out / SERIES_FILE


def load_series(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read a prepared series.csv (and the series.json next to it), checking the format.

    Returns the frame with `period` as pandas Periods, and the metadata.
    """
    path = data_utils.find_input(path, SERIES_FILE)
    meta_path = path.parent / SERIES_META
    if not path.exists() or not meta_path.exists():
        sys.exit(
            f"{path} (with {SERIES_META} next to it) not found. Prepare the data first with "
            "prepare_series.py, and pass its series.csv."
        )
    meta = json.loads(meta_path.read_text())
    features = list(meta.get("features") or {})
    frame = pd.read_csv(path, dtype={"id": str, "period": str})
    expected_cols = ["id", "period", "y", *features]
    if list(frame.columns) != expected_cols:
        sys.exit(
            f"{path} has columns {list(frame.columns)}, not {', '.join(expected_cols)}: "
            "rerun prepare_series.py."
        )
    freq = meta["freq"]
    frame["period"] = pd.PeriodIndex(frame["period"], freq=freq)
    for col in ["y", *features]:
        frame[col] = frame[col].astype(float)
    if frame[["y", *features]].isna().any().any():
        sys.exit(f"{path} has missing values: rerun prepare_series.py with --fill-gaps.")
    if frame.duplicated(["id", "period"]).any():
        sys.exit(
            f"{path} has repeated (id, period) rows: rerun prepare_series.py with --aggregate."
        )
    for sid, group in frame.groupby("id"):
        expected = pd.period_range(group["period"].min(), group["period"].max(), freq=freq)
        if len(expected) != len(group):
            sys.exit(f"Series {sid!r} in {path} has missing periods: rerun prepare_series.py.")
    return frame.sort_values(["id", "period"]).reset_index(drop=True), meta


def load_future(path: Path, meta: dict[str, Any]) -> pd.DataFrame | None:
    """The known features' values for the periods after the history, from
    future_features.csv next to series.csv; None when there are no known features."""
    known = [n for n, f in (meta.get("features") or {}).items() if f["use"] == "known"]
    if not known:
        return None
    file = data_utils.find_input(path, SERIES_FILE).parent / FUTURE_FILE
    if not file.exists():
        return pd.DataFrame(columns=["id", "period", *known])
    frame = pd.read_csv(file, dtype={"id": str, "period": str})
    if list(frame.columns) != ["id", "period", *known]:
        sys.exit(f"{file} has columns {list(frame.columns)}: rerun prepare_series.py.")
    frame["period"] = pd.PeriodIndex(frame["period"], freq=meta["freq"])
    return frame


def describe(frame: pd.DataFrame, meta: dict[str, Any]) -> str:
    lengths = frame.groupby("id").size()
    features = meta.get("features") or {}
    listed = ", ".join(f"{n} ({f['use']})" for n, f in features.items())
    return (
        f"{frame['id'].nunique()} series of {meta['target']} at frequency {meta['freq']} "
        f"(season length {meta['season_length']}), {frame['period'].min()} to "
        f"{frame['period'].max()}, {int(lengths.min())}–{int(lengths.max())} periods each "
        f"(median {int(np.median(lengths))})" + (f"; features: {listed}" if listed else "")
    )
