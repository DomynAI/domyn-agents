"""Shared helpers for the tabular-classification-regression scripts.

Not run directly. Every entry-point script (`classification_*.py`, `regression_*.py`)
imports this module, which is why they all live in one directory.

What lives here is what has to be *identical* across models for the comparison to
be fair: how the data is read, which columns are features, how they are typed, and
which rows are train / validation / test. `make_split.py` decides all of it once and
writes it to the output folder (`split.csv`, and the settings in `split.json`); the
training scripts take only `--out` and rebuild the same dataset from there, so the
models are always scored on the same rows.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

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

MISSING = "__missing__"
SPLIT_FILE = "split.csv"
META_FILE = "split.json"

# Order of the commands in run.sh, whatever order the scripts were run in.
RUN_ORDER = ("prepare_table", "split", "baseline", "catboost", "compare")

# What make_split.py records, and every training script reads back.
SETTINGS = (
    "data", "table", "target", "sep", "decimal", "encoding", "sheet", "drop", "date_cols",
    "split", "time_col", "group_col", "test_size", "valid_size", "seed",
)  # fmt: skip


def split_parser(description: str) -> argparse.ArgumentParser:
    """The arguments of make_split.py: the data, the features, and the split."""
    p = argparse.ArgumentParser(description=description)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--table",
        type=Path,
        help="a prepared table.csv (prepare_table.py): the cleaned data, e.g. from an analysis",
    )
    source.add_argument(
        "--data", type=Path, help="a raw CSV, TSV, Parquet, Excel, JSON or JSONL file"
    )
    p.add_argument("--target", required=True, help="column to predict")
    p.add_argument(
        "--task",
        required=True,
        choices=["classification", "regression"],
        help="decides the checks on the target, and stratification",
    )
    p.add_argument(
        "--sep",
        default=None,
        help="CSV / TXT column separator, e.g. ';' or '|' (default: detected from the file)",
    )
    p.add_argument(
        "--decimal",
        default=".",
        help="decimal mark in CSV / TSV numbers: ',' for European files like 1,5 (default '.')",
    )
    p.add_argument(
        "--encoding",
        default="utf-8",
        help="text encoding of CSV / TSV / JSON files, e.g. latin-1, cp1252 (default utf-8)",
    )
    p.add_argument("--sheet", default=None, help="Excel sheet name (default: the first)")
    p.add_argument("--out", required=True, type=Path, help="output folder (created if missing)")
    p.add_argument("--drop", nargs="*", default=[], help="columns to exclude (ids, leaky columns)")
    p.add_argument("--date-cols", nargs="*", default=[], help="columns to parse as dates")
    p.add_argument(
        "--split",
        choices=["auto", "random", "time", "group"],
        default="auto",
        help="auto (default): time if there is one date column, else group if there is one "
        "repeated entity id, else random (see choose_split)",
    )
    p.add_argument("--time-col", help="column to order by for a time split (implies it with auto)")
    p.add_argument(
        "--group-col",
        help="entity column for a group split, kept out of the features (implies it with auto)",
    )
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--valid-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p


def base_parser(description: str) -> argparse.ArgumentParser:
    """The arguments every training script accepts: the folder make_split.py wrote to.

    Everything about the data and the split is read from its split.json, so a training
    script cannot be run on other data, other features or another split.
    """
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--out", required=True, type=Path, help="output folder prepared by make_split.py"
    )
    return p


def load_dataset(args: argparse.Namespace, task: str) -> Dataset:
    """The dataset make_split.py prepared in `args.out`, with its settings copied onto `args`."""
    meta_path = data_utils.output_dir(args.out) / META_FILE
    if not meta_path.exists():
        sys.exit(f"{meta_path} not found: run make_split.py with --out {args.out} first.")
    meta = json.loads(meta_path.read_text())
    if meta["task"] != task:
        sys.exit(f"{args.out} was split for {meta['task']}: use the {meta['task']}_* scripts.")
    for key, value in meta["settings"].items():
        setattr(args, key, Path(value) if key in ("data", "table") and value else value)
    args.split_reason = meta.get("split_reason")
    return Dataset(args, task)


# Column names that usually identify an entity (a customer, a patient, a device).
ENTITY_NAME = re.compile(
    r"(^|_)(id|key|code|uuid|no|number)$|^id_|(customer|client|user|account|member|patient|"
    r"device|household|employee|session|subscriber|policy|contract|vehicle|store|shop)",
    re.IGNORECASE,
)


def date_candidates(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    """Columns holding dates (datetime-typed, or text of which 90% parses as one) with at
    least 10 distinct values."""
    out = []
    for col in df.columns:
        if col in exclude:
            continue
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            parsed = s
        elif s.dtype == object or pd.api.types.is_string_dtype(s):
            sample = s.dropna().astype(str).head(500)
            # Bare numbers ("12", "2024") parse as dates too; they are not date columns.
            if sample.empty or sample.str.fullmatch(r"-?\d+(\.\d+)?").mean() > 0.5:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(sample, errors="coerce")
            if parsed.notna().mean() < 0.9:
                continue
        else:
            continue
        if parsed.nunique() >= 10:
            out.append(col)
    return out


def entity_candidates(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    """Columns that look like an entity id repeated over several rows: named like an id,
    text or integer, at least 20 distinct values, each on 1.5 rows or more on average.

    A unique id (one row per value) is not a group: it splits like a random split.
    """
    out = []
    for col in df.columns:
        if col in exclude or not ENTITY_NAME.search(str(col)):
            continue
        s = df[col].dropna()
        if pd.api.types.is_float_dtype(s) or pd.api.types.is_bool_dtype(s):
            continue
        distinct = s.nunique()
        if distinct >= 20 and len(s) / distinct >= 1.5:
            out.append(col)
    return out


def choose_split(df: pd.DataFrame, args: argparse.Namespace) -> tuple[str, str]:
    """Resolve --split auto into random / time / group, and say why.

    The order is the conservative one. A time or group split where random would have
    been fine only makes the scores a little pessimistic; a random split where time or
    group was needed tests the model on rows it has effectively seen already, and makes
    them optimistic. So: time if there is a date column, then group if there is a
    repeated entity id, then random. Two candidates of one kind stop the script rather
    than guess.
    """
    if args.split != "auto":
        return args.split, "set with --split"
    if args.time_col:
        return "time", f"--time-col {args.time_col} given"
    if args.group_col:
        return "group", f"--group-col {args.group_col} given"
    dates = date_candidates(df, {args.target, *args.drop})
    if len(dates) > 1:
        sys.exit(
            f"--split auto found several date columns {dates}: pass --time-col with the one "
            "recording when each row happened, or --split random / group."
        )
    if dates:
        args.time_col = dates[0]
        return "time", (
            f"'{dates[0]}' is a date column: the test rows are the most recent ones, as when "
            "the model is used on new data"
        )
    entities = entity_candidates(df, {args.target})
    if len(entities) > 1:
        sys.exit(
            f"--split auto found several repeated entity ids {entities}: pass --group-col with "
            "the one the model will meet new values of, or --split random."
        )
    if entities:
        col = args.group_col = entities[0]
        return "group", (
            f"'{col}' repeats ({len(df) / df[col].nunique():.1f} rows per value): each {col} "
            "is entirely in train, validation or test, so the test has only unseen ones"
        )
    return "random", "no date column and no repeated entity id"


def date_features(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Replace each date column with numeric parts a model can use."""
    for col in columns:
        d = pd.to_datetime(df[col], errors="coerce")
        df[f"{col}_year"] = d.dt.year
        df[f"{col}_month"] = d.dt.month
        df[f"{col}_dayofweek"] = d.dt.dayofweek
        df[f"{col}_days"] = (d - d.min()).dt.days
        df = df.drop(columns=col)
    return df


class Dataset:
    """Features, target and the fixed split, loaded the same way by every script."""

    def __init__(self, args: argparse.Namespace, task: str) -> None:
        self.args = args
        self.task = task
        self.out: Path = data_utils.output_dir(args.out)

        if args.table:
            df, _ = data_utils.load_table(args.table)
            args.data = data_utils.find_input(args.table, data_utils.TABLE_FILE)
        else:
            df = data_utils.read_table(
                args.data.resolve(), args.sep, args.decimal, args.encoding, args.sheet
            )
        # A prepared table's dates are typed already: they become date features, like
        # --date-cols, instead of being read as categories.
        typed_dates = [
            c
            for c in df.columns
            if pd.api.types.is_datetime64_any_dtype(df[c]) and c not in (*args.drop, args.target)
        ]
        args.date_cols = [*args.date_cols, *(c for c in typed_dates if c not in args.date_cols)]
        missing = [c for c in [args.target, *args.drop, *args.date_cols] if c not in df.columns]
        if missing:
            sys.exit(f"Columns not in the data: {missing}. Available: {list(df.columns)}")
        self._resolve_split(df)

        n_before = len(df)
        df = df.dropna(subset=[args.target]).reset_index(drop=True)
        self.dropped_missing_target = n_before - len(df)

        self._check_target(df[args.target])
        self.order = df[args.time_col] if args.split == "time" else None
        self.groups = df[args.group_col] if args.split == "group" else None

        excluded = {args.target, *args.drop}
        if args.group_col:
            excluded.add(args.group_col)
        X = df.drop(columns=[c for c in excluded if c in df.columns])
        X = date_features(X, [c for c in args.date_cols if c in X.columns])
        self.cat_cols: list[str] = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()
        self.cat_cols += X.select_dtypes(include="bool").columns.tolist()
        self.num_cols: list[str] = [c for c in X.columns if c not in self.cat_cols]
        for col in self.cat_cols:
            X[col] = X[col].astype("string").fillna(MISSING).astype(str)
        self.X = X
        self.y = (
            df[args.target].astype(str)
            if task == "classification"
            else df[args.target].astype(float)
        )

        self.part = self._load_or_make_split()
        self._write_meta()

    def _resolve_split(self, df: pd.DataFrame) -> None:
        a = self.args
        if getattr(a, "split_reason", None):  # reloaded from split.json
            self.split_reason = a.split_reason
        else:
            a.split, self.split_reason = choose_split(df, a)
        for col in (a.time_col, a.group_col):
            if col and col not in df.columns:
                sys.exit(f"Column '{col}' not in the data. Available: {list(df.columns)}")
        if a.split == "time" and a.time_col not in (*a.date_cols, *a.drop):
            # The model gets its parts (year, month, ...) rather than the raw text.
            a.date_cols = [*a.date_cols, a.time_col]
        print(f"Split: {a.split} — {self.split_reason}")

    def settings(self) -> dict[str, Any]:
        s = {k: getattr(self.args, k) for k in SETTINGS}
        s["data"] = str(self.args.data.resolve())
        s["table"] = str(self.args.table.resolve()) if self.args.table else None
        return s

    def _check_target(self, y: pd.Series) -> None:
        if self.task == "regression":
            if not pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y):
                sys.exit(
                    f"Target '{self.args.target}' is not numeric: use the classification scripts."
                )
            if y.nunique() <= 2:
                sys.exit(
                    f"Target '{self.args.target}' has {y.nunique()} values: use the classification scripts."
                )
        else:
            n = y.nunique()
            if n < 2:
                sys.exit(f"Target '{self.args.target}' has a single value: nothing to predict.")
            if pd.api.types.is_float_dtype(y) and n > 20:
                sys.exit(
                    f"Target '{self.args.target}' is a continuous number ({n} distinct values): "
                    "use the regression scripts."
                )
            smallest = y.value_counts().min()
            if smallest < 10:
                print(
                    f"WARNING: the rarest class has only {smallest} rows; its metrics will be unreliable."
                )

    def _load_or_make_split(self) -> pd.Series:
        path = self.out / SPLIT_FILE
        meta_path = self.out / META_FILE
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            now = {
                "task": self.task,
                "data": str(self.args.data.resolve()),
                "target": self.args.target,
            }
            before = {k: meta.get(k) for k in now}
            if before != now:
                sys.exit(
                    f"{self.out} already holds a run for {before}, not {now}: "
                    "every model in one folder must share the data, target and split. Use a new --out folder."
                )
        if path.exists():
            part = pd.read_csv(path)["part"]
            if len(part) != len(self.X):
                sys.exit(
                    f"{path} has {len(part)} rows but the data now has {len(self.X)}: "
                    "the scripts were run with different data or --drop settings. Use a new --out folder."
                )
            return part
        part = self._make_split()
        pd.DataFrame({"row": range(len(part)), "part": part}).to_csv(path, index=False)
        return part

    def _make_split(self) -> pd.Series:
        a, n = self.args, len(self.X)
        idx = np.arange(n)
        part = pd.Series("train", index=idx)
        if a.split == "time":
            if not a.time_col:
                sys.exit("--split time needs --time-col")
            ranked = pd.to_datetime(self.order, errors="coerce").rank(method="first").to_numpy()
            n_test, n_valid = round(n * a.test_size), round(n * a.valid_size)
            order = np.argsort(ranked)
            part.iloc[order[n - n_test :]] = "test"
            part.iloc[order[n - n_test - n_valid : n - n_test]] = "valid"
            return part
        if a.split == "group":
            if not a.group_col:
                sys.exit("--split group needs --group-col")
            outer = GroupShuffleSplit(n_splits=1, test_size=a.test_size, random_state=a.seed)
            rest, test = next(outer.split(idx, groups=self.groups))
            inner = GroupShuffleSplit(
                n_splits=1, test_size=a.valid_size / (1 - a.test_size), random_state=a.seed
            )
            _, va = next(inner.split(rest, groups=self.groups.iloc[rest]))
            part.iloc[test], part.iloc[rest[va]] = "test", "valid"
            return part
        strat = self.y if self.task == "classification" else None
        rest, test = train_test_split(
            idx, test_size=a.test_size, random_state=a.seed, stratify=strat
        )
        strat = self.y.iloc[rest] if self.task == "classification" else None
        _tr, va = train_test_split(
            rest, test_size=a.valid_size / (1 - a.test_size), random_state=a.seed, stratify=strat
        )
        part.iloc[test], part.iloc[va] = "test", "valid"
        return part

    def _write_meta(self) -> None:
        meta = {
            "task": self.task,
            "data": str(self.args.data.resolve()),
            "target": self.args.target,
            "split": self.args.split,
            "split_reason": self.split_reason,
            "settings": self.settings(),
            "rows": {p: int((self.part == p).sum()) for p in ("train", "valid", "test")},
            "dropped_missing_target": self.dropped_missing_target,
            "dropped_columns": self.args.drop,
            "numeric_features": self.num_cols,
            "categorical_features": self.cat_cols,
        }
        if self.task == "classification":
            meta["classes"] = sorted(self.y.unique().tolist())
            meta["class_counts"] = {k: int(v) for k, v in self.y.value_counts().items()}
        (self.out / META_FILE).write_text(json.dumps(meta, indent=2))

    def subset(self, name: str) -> tuple[pd.DataFrame, pd.Series]:
        mask = (self.part == name).to_numpy()
        return self.X[mask], self.y[mask]

    def rows(self, name: str) -> np.ndarray:
        return np.flatnonzero((self.part == name).to_numpy())


def baseline_preprocessing(
    num_cols: list[str], cat_cols: list[str], min_frequency: int = 10
) -> Any:
    """Preparation for the linear / logistic baselines, fitted on training rows only.

    Numeric: median imputation, then standardization (mean 0, std 1), so coefficients are
    comparable across features. Categorical (already strings, missing = MISSING): one-hot,
    with categories under `min_frequency` rows — and any unseen at training — in one
    "infrequent" column.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    return ColumnTransformer(
        [
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num_cols),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=min_frequency,
                    sparse_output=False,
                ),
                cat_cols,
            ),
        ]
    )


def coefficient_chart(
    out: Path, name: str, features: pd.Series, values: pd.Series, title: str, xlabel: str
) -> None:
    """Horizontal bars, positive in blue and negative in coral, largest at the top."""
    import seaborn_style as style

    fig, ax = style.figure(height=max(360, 26 * len(features) + 140))
    ax.barh(
        features.iloc[::-1].astype(str),
        values.iloc[::-1],
        color=[
            style.PALETTE["charts"]["negative"] if v < 0 else style.PALETTE["charts"]["positive"]
            for v in values.iloc[::-1]
        ],
    )
    ax.axvline(0, color=style.PALETTE["text"]["light_background"]["muted"], linewidth=1)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    style.title(fig, title)
    style.save(fig, out, name)


def importance_chart(out: Path, importance: pd.DataFrame) -> None:
    """CatBoost's feature importance (`feature`, `importance`), the top 15, largest on top."""
    import seaborn_style as style

    top = importance.head(15).iloc[::-1]
    fig, ax = style.figure(height=max(360, 26 * len(top) + 140))
    ax.barh(
        top["feature"].astype(str),
        top["importance"],
        color=style.PALETTE["charts"]["categorical"][0],
    )
    ax.set_xlabel("share of the model's decisions (sums to 100)")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    style.title(fig, "CatBoost feature importance (top 15)")
    style.save(fig, out, "catboost_feature_importance")


def learning_curve_chart(out: Path, curve: list[float], best: int, title: str, ylabel: str) -> None:
    """The validation loss by training round, with the round early stopping kept."""
    import seaborn_style as style

    fig, ax = style.figure()
    ax.plot(curve, color=style.PALETTE["charts"]["categorical"][0])
    muted = style.PALETTE["text"]["light_background"]["muted"]
    ax.axvline(best, color=muted, linestyle="--", linewidth=1)
    # The note beside the line, on the side with room for it.
    right = best < 0.7 * max(len(curve) - 1, 1)
    ax.annotate(
        "best round (kept)",
        xy=(best, 1),
        xycoords=ax.get_xaxis_transform(),
        xytext=(5 if right else -5, -2),
        textcoords="offset points",
        ha="left" if right else "right",
        va="top",
        color=muted,
    )
    ax.set_xlabel("training round")
    ax.set_ylabel(ylabel)
    style.title(fig, title)
    style.save(fig, out, "catboost_learning_curve")


def shap_effect_chart(
    out: Path,
    feature: str,
    values: pd.Series,
    categorical: bool,
    panels: list[tuple[str, str, np.ndarray]],
) -> None:
    """How one feature pushes the prediction: its SHAP values against its values.

    `panels` is (axis label, panel title, SHAP values) per panel: one for regression and
    binary classification, one per class for multiclass. Numeric features get a scatter,
    categorical ones a box per category (the 15 with the largest median push).
    """
    import seaborn as sns
    import seaborn_style as style

    color = style.PALETTE["primary"]
    muted = style.PALETTE["text"]["light_background"]["muted"]
    fig, axes = style.figure(width=max(900, 480 * len(panels)), ncols=len(panels), squeeze=False)
    for ax, (label, title, shap) in zip(axes[0], panels, strict=True):
        if categorical:
            frame = pd.DataFrame({"v": values.astype(str).to_numpy(), "s": shap})
            order = frame.groupby("v")["s"].median().sort_values().index[-15:]
            # Largest median push at the top.
            sns.boxplot(
                frame[frame.v.isin(order)],
                x="s",
                y="v",
                order=list(order[::-1]),
                orient="h",
                color=color,
                fill=False,
                width=0.6,
                linewidth=1.5,
                flierprops={"markersize": 3, "alpha": 0.5},
                ax=ax,
            )
            ax.axvline(0, color=muted, linewidth=1)
            ax.set_xlabel(label)
            ax.set_ylabel("")
            ax.grid(axis="x")
            ax.grid(axis="y", visible=False)
        else:
            ax.scatter(values, shap, s=14, alpha=0.5, color=color, linewidths=0)
            ax.axhline(0, color=muted, linewidth=1)
            ax.set_xlabel(feature)
            ax.set_ylabel(label)
        if title:
            ax.set_title(title)
    style.title(fig, f"Effect of {feature} on the prediction (SHAP values, test rows)")
    style.save(fig, out, "catboost_effect_" + style.safe_name(feature))


def classification_metrics(
    actual: np.ndarray, pred: np.ndarray, proba: np.ndarray, classes: list[str]
) -> dict[str, float]:
    """The classification metrics every script reports. `proba` columns follow `classes`."""
    from sklearn import metrics as m

    actual, pred = np.asarray(actual).astype(str), np.asarray(pred).astype(str)
    out = {
        "accuracy": m.accuracy_score(actual, pred),
        "balanced_accuracy": m.balanced_accuracy_score(actual, pred),
        "F1_macro": m.f1_score(actual, pred, average="macro"),
        "log_loss": m.log_loss(actual, np.clip(proba, 1e-15, 1), labels=classes),
    }
    if len(classes) == 2:
        out["ROC_AUC"] = m.roc_auc_score(actual == classes[1], proba[:, 1])
        out["PR_AUC"] = m.average_precision_score(actual == classes[1], proba[:, 1])
    else:
        out["ROC_AUC"] = m.roc_auc_score(actual, proba, multi_class="ovr", labels=classes)
    return {k: float(v) for k, v in out.items()}


def regression_metrics(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """The regression metrics every script reports."""
    from sklearn import metrics as m

    out = {
        "MAE": m.mean_absolute_error(actual, pred),
        "RMSE": float(np.sqrt(m.mean_squared_error(actual, pred))),
        "R2": m.r2_score(actual, pred),
        "median_abs_error": m.median_absolute_error(actual, pred),
    }
    if (np.asarray(actual) > 0).all():  # only meaningful for strictly positive quantities
        out["MAPE_%"] = 100 * m.mean_absolute_percentage_error(actual, pred)
    return {k: float(v) for k, v in out.items()}


def save_predictions(
    out: Path,
    model_name: str,
    rows: np.ndarray,
    actual: pd.Series,
    pred: np.ndarray,
    proba: np.ndarray | None = None,
    classes: list[str] | None = None,
) -> None:
    """Test-set predictions in the one format compare scripts read."""
    frame = pd.DataFrame({"row": rows, "actual": actual.to_numpy(), "pred": pred})
    if proba is not None and classes is not None:
        for i, cls in enumerate(classes):
            frame[f"proba_{cls}"] = proba[:, i]
    frame.to_csv(out / f"{model_name}_test_predictions.csv", index=False)


def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    """A DataFrame as a Markdown table; metrics keep four decimals."""
    return data_utils.markdown_table(frame, digits)


write_json = data_utils.write_json


def record_run(out: Path, script: Path, step: str) -> None:
    """Copy the script (with this module and data_utils) into the output folder, and log its
    command in run.sh."""
    data_utils.record_run(out, script, RUN_ORDER, (Path(__file__),))
