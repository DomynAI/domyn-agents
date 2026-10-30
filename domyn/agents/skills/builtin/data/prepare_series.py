r"""Prepare time series for analysis and forecasting: reshape, put on a regular calendar,
and write the shared series.csv / series.json format (see series_format.py).

Shared by the data skills (it lives in builtin/data/). Run it once per dataset: its
output is read by the data-analysis skill's timeseries_analysis.py and by every
time-series-forecasting script, so an analysis followed by a forecast works from the
same prepared file.

Three input layouts (--layout):

  long               one row per series per period   date, store, sales
                                                       2024-01, north, 1520
  dates-as-columns   one row per series, one          store, 2024-01, 2024-02, ...
                     column per period                north, 1520, 1611, ...
  series-as-columns  one row per period, one          date, north, south, ...
                     column per series                2024-01, 1520, 980, ...

Several id columns (store and product, say) are joined into one id, "north | shoes".
Event-level data (one row per order) is long data with several rows per period: pass
--freq and --aggregate sum to turn it into totals per period.

Start from the prepared table (--table, from prepare_table.py) so the cleaning applies to
the series too; --data reads a raw file that needs no cleaning.

Other columns can go along as features (long layout only): --features names a JSON spec
declaring each one `known` (its value for a period is known in advance, like a holiday)
or `lagged` (only past values are usable, like the weather), with how it aggregates;
see series_format.check_feature_spec. Known features also need their values for the
periods to forecast: --future reads them from a file with the same time and id columns
(the rows after each series' last period are used; earlier ones are ignored, so the
data file itself works when it already holds them). The spec is copied to
files/features.json.

    python prepare_series.py --table <run folder>/files/table.csv --time-col date \
        --target sales --id-cols store --out <run folder>
    python prepare_series.py --table ... --features <run folder>/files/features.json \
        --future /abs/calendar_2025.csv --out <run folder>

Features for series already prepared (by a data-analysis run, say): --from-series reads
that series.json and prepares again from the same source with the same settings, adding
only the features, into a new --out. The target is checked to come out identical, so
the analysis and the forecast describe the same series.

    python prepare_series.py --from-series <analysis run folder>/files/series.csv \
        --features <run folder>/files/features.json --future ... --out <run folder>

Without --features, the numeric columns of a long table that were not carried are
listed, and recorded in series.json (`unused_columns`), so that forecast_split.py can
point them out: a forecast run on series prepared without them would not say so.
"""

import argparse
import json
import shutil
from pathlib import Path

import data_utils
import numpy as np
import pandas as pd
import series_format as sf


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--table", type=Path, help="a prepared table.csv (prepare_table.py): the cleaned data"
    )
    source.add_argument(
        "--data", type=Path, help="a raw CSV, TSV, Parquet, Excel, JSON or JSONL file"
    )
    source.add_argument(
        "--from-series",
        type=Path,
        help="a prepared series.csv to add --features to: its source and settings are reused",
    )
    p.add_argument(
        "--layout",
        choices=["long", "dates-as-columns", "series-as-columns"],
        default=None,
        help="default: long",
    )
    p.add_argument(
        "--target",
        help="long: the value column. Other layouts: a name for what the values measure",
    )
    p.add_argument("--time-col", help="long / series-as-columns: the date column")
    p.add_argument(
        "--id-cols",
        nargs="*",
        default=None,
        help="long / dates-as-columns: the column(s) naming each series (none for a single series)",
    )
    p.add_argument(
        "--value-cols",
        nargs="*",
        default=None,
        help="wide layouts: the period / series columns to use (default: all the others)",
    )
    p.add_argument("--freq", default=None, help="D, W (W-SUN), M, Q, Y, h (default: inferred)")
    p.add_argument(
        "--season-length", type=int, default=None, help="default: 7 D, 52 W, 12 M, 4 Q, 24 h"
    )
    p.add_argument(
        "--aggregate",
        choices=["sum", "mean", "last"],
        default=None,
        help="how to combine several rows per series and period",
    )
    p.add_argument(
        "--fill-gaps",
        choices=["zero", "interpolate"],
        default=None,
        help="missing periods / values: zero (no row means nothing happened) or interpolate "
        "(default: refuse, listing them)",
    )
    p.add_argument(
        "--features",
        type=Path,
        default=None,
        help="JSON spec of the columns to carry as features, each known or lagged "
        "(long layout only)",
    )
    p.add_argument(
        "--future",
        type=Path,
        default=None,
        help="a table or data file with the known features' values for the periods to "
        "forecast (same time and id columns)",
    )
    p.add_argument("--sep", default=None, help="CSV / TXT separator (default: detected)")
    p.add_argument("--decimal", default=".", help="decimal mark in CSV numbers (default '.')")
    p.add_argument("--encoding", default="utf-8")
    p.add_argument("--sheet", default=None, help="Excel sheet (default: the first)")
    p.add_argument("--out", required=True, type=Path, help="folder for series.csv and series.json")
    return p


def require(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise SystemExit(f"Columns not in the data: {missing}. Available: {list(df.columns)}")


def series_id(df: pd.DataFrame, id_cols: list[str]) -> pd.Series:
    if not id_cols:
        return pd.Series(sf.SINGLE_ID, index=df.index)
    return df[id_cols].astype(str).agg(sf.ID_JOIN.join, axis=1)


def numeric_features(df: pd.DataFrame, names: list[str]) -> dict[str, pd.Series]:
    """The feature columns as numbers (booleans as 0 / 1); text is refused."""
    out = {}
    for name in names:
        col = df[name]
        if col.dtype == bool or str(col.dtype) == "boolean":
            col = col.astype(float)
        values = pd.to_numeric(col, errors="coerce")
        bad = int((values.isna() & col.notna()).sum())
        if bad:
            raise SystemExit(
                f"Feature {name!r} has {bad} non-numeric values (e.g. "
                f"{col[values.isna() & col.notna()].iloc[0]!r}). Features must be numbers or "
                "booleans: map categories to numbers in the cleaning spec, or one column per "
                "category (0 / 1)."
            )
        out[name] = values
    return out


def to_long(
    df: pd.DataFrame, a: argparse.Namespace, features: list[str] | None = None
) -> pd.DataFrame:
    """The input as (id, time, y, *features) rows, whatever its layout."""
    features = features or []
    if a.layout == "long":
        require(df, [a.time_col, a.target, *a.id_cols, *features])
        return pd.DataFrame(
            {
                "id": series_id(df, a.id_cols),
                "time": df[a.time_col],
                "y": df[a.target],
                **numeric_features(df, features),
            }
        )
    if a.layout == "dates-as-columns":
        require(df, a.id_cols)
        cols = a.value_cols or [c for c in df.columns if c not in a.id_cols]
        require(df, cols)
        undated = [c for c in cols if pd.isna(pd.to_datetime(str(c), errors="coerce"))]
        if undated:
            raise SystemExit(
                f"Columns {undated[:10]} are not dates: add them to --id-cols, or list the "
                "period columns with --value-cols."
            )
        frame = df.assign(id=series_id(df, a.id_cols)).melt(
            id_vars="id", value_vars=cols, var_name="time", value_name="y"
        )
        frame["time"] = frame["time"].astype(str)
        return frame
    require(df, [a.time_col])
    cols = a.value_cols or [c for c in df.columns if c != a.time_col]
    require(df, cols)
    frame = df.melt(id_vars=a.time_col, value_vars=cols, var_name="id", value_name="y")
    return frame.rename(columns={a.time_col: "time"}).astype({"id": str})


# What --from-series takes from the series it extends: every setting that decides the target.
REUSED = ("layout", "target", "time_col", "id_cols", "freq", "season_length", "aggregate")
READING = ("sep", "decimal", "encoding", "sheet")


def unused_columns(df: pd.DataFrame, a: argparse.Namespace, features: list[str]) -> list[str]:
    """The numeric (or boolean) columns of a long table left out of the series: candidate
    features, which a forecast would otherwise never hear about."""
    if a.layout != "long":
        return []
    used = {a.time_col, a.target, *a.id_cols, *features}
    return [str(c) for c in df.columns if c not in used and pd.api.types.is_numeric_dtype(df[c])]


def from_series(a: argparse.Namespace, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """For --from-series: fill `a` with the settings of the series it extends, and return
    that series and its source, read as it was then."""
    if not a.features:
        raise SystemExit("--from-series adds features to prepared series: pass --features")
    given = [
        f"--{k.replace('_', '-')}" for k in (*REUSED, "value_cols") if getattr(a, k) is not None
    ]
    if given:
        raise SystemExit(
            f"With --from-series, {' '.join(given)} come from its series.json: leave them "
            "out, or prepare from scratch with --table."
        )
    original, meta = sf.load_series(a.from_series)
    path = data_utils.find_input(a.from_series, sf.SERIES_FILE)
    if path.parent.resolve() == out.resolve():
        raise SystemExit(
            f"--out would replace the series in {path.parent}: write the series with features "
            "into this run's folder, and leave the prepared one as it is."
        )
    if meta.get("layout") != "long":
        raise SystemExit(
            f"{path} was prepared from --layout {meta.get('layout')}; features need a long "
            "table, one row per series and period. Prepare from scratch with --table."
        )
    for k in REUSED:
        setattr(a, k, meta.get(k))
    a.fill_gaps = a.fill_gaps or meta.get("fill_gaps")  # feature gaps may need one
    source = Path(meta["source"])
    if not source.exists():
        raise SystemExit(f"{path} was prepared from {source}, which no longer exists.")
    reading = argparse.Namespace(**{**vars(a), **(meta.get("read") or {})})
    a.data = source
    return original, read_input(source, reading)


def check_same_target(original: pd.DataFrame, regular: pd.DataFrame, path: Path) -> None:
    """Refuse series whose target differs from the one --from-series extends."""
    before = original.assign(period=original["period"].astype(str))[["id", "period", "y"]]
    after = regular.sort_values(["id", "period"])[["id", "period", "y"]]
    same = (
        len(before) == len(after)
        and (before["id"].to_numpy() == after["id"].to_numpy()).all()
        and (before["period"].to_numpy() == after["period"].to_numpy()).all()
        and np.allclose(before["y"].to_numpy(), after["y"].to_numpy())
    )
    if not same:
        raise SystemExit(
            f"Prepared again from its source, the target no longer matches {path}: the "
            "source has changed since. Prepare from scratch with --table and --features, "
            "and say in the report that the series differ from the analysis."
        )


def read_input(path: Path, a: argparse.Namespace) -> pd.DataFrame:
    """A prepared table (a folder, or a table.csv with table.json next to it) or a raw file."""
    path = path.resolve()
    if path.is_dir() or path.with_name(data_utils.TABLE_META).exists():
        df, _ = data_utils.load_table(path)
        return df
    return data_utils.read_table(path, a.sep, a.decimal, a.encoding, a.sheet)


def future_rows(
    a: argparse.Namespace,
    features: dict[str, dict],
    regular: pd.DataFrame,
    freq: str,
) -> pd.DataFrame | None:
    """The known features' values after each series' last period, from --future."""
    known = [n for n, f in features.items() if f["use"] == "known"]
    if not known:
        if a.future:
            print("WARNING: --future is ignored: no feature is declared known")
        return None
    if not a.future:
        print(
            f"WARNING: known features {known} have no values for the periods to forecast "
            "(--future): the backtest can run, but not the forecast itself."
        )
        return None
    df = read_input(a.future, a)
    require(df, [a.time_col, *a.id_cols, *known])
    frame = pd.DataFrame(
        {
            "id": series_id(df, a.id_cols),
            "period": pd.to_datetime(df[a.time_col], errors="coerce").dt.to_period(freq),
            **numeric_features(df, known),
        }
    ).dropna(subset=["period"])
    last = regular.groupby("id")["period"].max().map(lambda p: pd.Period(p, freq))
    frame = frame[frame["id"].map(last).notna()]
    frame = frame[frame["period"] > frame["id"].map(last)]
    rules = {n: features[n]["aggregate"] for n in known}
    frame = frame.groupby(["id", "period"], as_index=False).agg(rules)
    if frame[known].isna().any().any():
        raise SystemExit(f"--future has missing values for {known} after the history.")
    print(
        f"Future values of {known}: {len(frame)} rows, up to {frame['period'].max()}"
        if len(frame)
        else f"WARNING: --future has no rows after the history for {known}"
    )
    return frame.astype({"period": str}).sort_values(["id", "period"])


def main() -> None:
    a = parser().parse_args()
    out = data_utils.output_dir(a.out)
    original = None
    if a.from_series:
        original, df = from_series(a, out)
    a.layout = a.layout or "long"
    a.id_cols = a.id_cols or []
    if not a.target:
        raise SystemExit("--target is required")
    if a.layout != "dates-as-columns" and not a.time_col:
        raise SystemExit(f"--layout {a.layout} needs --time-col")
    features: dict[str, dict] = {}
    if a.features:
        if a.layout != "long":
            raise SystemExit("--features needs --layout long: wide layouts hold one value per cell")
        features = sf.check_feature_spec(json.loads(a.features.read_text()))
    if a.table:
        df, _ = data_utils.load_table(a.table)
        a.data = data_utils.find_input(a.table, data_utils.TABLE_FILE)
    elif a.data and not a.from_series:
        df = data_utils.read_table(a.data.resolve(), a.sep, a.decimal, a.encoding, a.sheet)
    unused = unused_columns(df, a, list(features))
    frame = to_long(df, a, list(features))

    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    frame["y"] = pd.to_numeric(frame["y"], errors="coerce")
    bad_dates = int(frame["time"].isna().sum())
    if bad_dates:
        print(f"WARNING: {bad_dates} rows have an unreadable date and are dropped")
        frame = frame.dropna(subset=["time"])
    blank = int(frame["y"].isna().sum())
    if blank and a.layout != "long":
        # A wide table has a cell for every series and period, so an empty one is a gap,
        # except before a series starts or after it ends, which to_regular leaves out.
        frame = frame.dropna(subset=["y"])
        print(f"{blank} empty cells left out (outside a series' span, or gaps reported below)")
    elif blank:
        if not a.fill_gaps:
            raise SystemExit(
                f"{blank} rows have a non-numeric or empty {a.target}. Decide what they mean, "
                "then pass --fill-gaps zero or --fill-gaps interpolate."
            )
        print(
            f"{blank} rows with a non-numeric or empty {a.target} are filled like missing periods"
        )
        frame = frame.dropna(subset=["y"])

    freq = sf.period_freq(a.freq or sf.infer_freq(frame["time"], frame["id"]))
    regular, filled = sf.to_regular(frame, freq, a.aggregate, a.fill_gaps, features)
    if original is not None:
        check_same_target(original, regular, a.from_series)
    future = future_rows(a, features, regular, freq) if features else None
    if a.features and a.features.resolve() != (out / "features.json").resolve():
        shutil.copyfile(a.features, out / "features.json")
    meta = {
        "target": a.target,
        "freq": freq,
        "season_length": a.season_length or sf.season_length_for(freq),
        "source": str(a.data.resolve()),
        "layout": a.layout,
        "time_col": a.time_col,
        "id_cols": a.id_cols,
        "aggregate": a.aggregate,
        "fill_gaps": a.fill_gaps,
        "filled_values": filled,
        "features": features,
        "future_features_until": (
            str(future["period"].max()) if future is not None and len(future) else None
        ),
        "unused_columns": unused,
        "read": {k: getattr(a, k) for k in READING},
        "extends": (
            str(data_utils.find_input(a.from_series, sf.SERIES_FILE).resolve())
            if a.from_series
            else None
        ),
        "series": int(regular["id"].nunique()),
        "rows": len(regular),
    }
    path = sf.write_series(out, regular, meta, future)
    here = Path(__file__).resolve()
    # Preparation comes first in run.sh whichever skill's folder this is.
    order = ("prepare_table", "prepare_series")  # first in run.sh, in whichever skill's folder
    data_utils.record_run(out, here, order, (here.with_name("series_format.py"),))

    loaded, meta = sf.load_series(path)
    print(sf.describe(loaded, meta))
    if unused and not features:
        print(
            f"NOTE: {len(unused)} other numeric columns are not carried as features: "
            f"{unused[:10]}. A forecast can use them (a holiday flag, a price, the weather): "
            "see the time-series-forecasting skill, Features."
        )
    print(f"Prepared series: {path}")


if __name__ == "__main__":
    main()
