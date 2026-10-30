r"""Prepare a data file for every data skill: apply the cleaning decisions in a spec,
and write the cleaned, typed table (table.csv / table.json, see data_utils.write_table).

Shared by data-analysis, tabular-classification-regression and time-series-forecasting, so data
cleaned once is analysed, modelled and forecast as the same table. The cleaning is
decided in the spec, a JSON file you write (profile_data.py drafts one); this script
only applies it, always in the same order, and records what every step changed.

The spec (every key optional; column names are the names after `rename`):

    {
      "types":      {"zip": "string", "units": "integer"},
      "rename":     {"Unnamed: 0": "row_id"},
      "drop":       {"internal_note": "free text, not analysed"},
      "trim":       ["*"],
      "missing":    {"age": [-1, 999], "*": ["N/A", "unknown", ""]},
      "numbers":    {"price": {"thousands": ".", "decimal": ","}, "discount": {"percent": true}},
      "dates":      {"order_date": {"dayfirst": true}},
      "booleans":   {"active": {"true": ["yes", "Y"], "false": ["no", "N"]}},
      "categories": {"city": {"N. York": "New York", "NYC": "New York"}},
      "duplicates": {"subset": ["order_id"], "keep": "first", "reason": "order exported twice"},
      "exclude":    [{"column": "age", "outside": [0, 120], "reason": "impossible ages"}]
    }

Order: read (with `types: string` columns kept as text, so "02134" keeps its zero) →
rename → drop → trim → missing → numbers → dates → booleans → types (integer / number
casts) → categories → duplicates → exclude. Rows are removed only by `duplicates` and
`exclude`, and both require a `reason`. `exclude` rules take `outside: [low, high]`
(null for no bound), `equals: [values]` or `missing: true`; each counts the rows it
removed from what the rules before it left.

Writes to --out: table.csv, table.json (types, row counts, and every step with how
many values or rows it affected), table.md (the same, for the report's Method
section), and cleaning.json (a copy of the spec, when it is elsewhere).

    python prepare_table.py --data /abs/sales.csv --spec <run folder>/files/cleaning.json \
        --out <run folder>
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import data_utils
import pandas as pd

SPEC_FILE = "cleaning.json"
SPEC_KEYS = (
    "types", "rename", "drop", "trim", "missing", "numbers", "dates", "booleans",
    "categories", "duplicates", "exclude",
)  # fmt: skip


class Log:
    """What each step changed: machine-readable for table.json, readable for table.md."""

    def __init__(self) -> None:
        self.steps: list[dict[str, Any]] = []

    def add(
        self, step: str, what: str, affected: int, column: str | None = None, **extra: Any
    ) -> None:
        self.steps.append(
            {"step": step, "column": column, "what": what, "affected": int(affected), **extra}
        )
        print(f"{step:<10} {('`' + column + '`: ') if column else ''}{what} ({affected})")


def fail(message: str) -> None:
    sys.exit(f"cleaning spec: {message}")


def require_columns(df: pd.DataFrame, columns: list[str], step: str) -> None:
    missing = [c for c in columns if c != "*" and c not in df.columns]
    if missing:
        fail(f"`{step}` names columns not in the data: {missing}. Available: {list(df.columns)}")


def targets(df: pd.DataFrame, names: list[str], text_only: bool = False) -> list[str]:
    """The columns a key applies to; "*" means every (text) column."""
    if "*" in names:
        return [
            c
            for c in df.columns
            if not text_only
            or pd.api.types.is_object_dtype(df[c])
            or pd.api.types.is_string_dtype(df[c])
        ]
    return names


def is_text(s: pd.Series) -> bool:
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)


def apply(df: pd.DataFrame, spec: dict[str, Any], log: Log) -> pd.DataFrame:
    if spec.get("rename"):
        require_columns(df, list(spec["rename"]), "rename")
        df = df.rename(columns=spec["rename"])
        for old, new in spec["rename"].items():
            log.add("rename", f"renamed to `{new}`", 0, old)

    drop = spec.get("drop") or {}
    drop = drop if isinstance(drop, dict) else dict.fromkeys(drop, "")
    if drop:
        require_columns(df, list(drop), "drop")
        df = df.drop(columns=list(drop))
        for col, reason in drop.items():
            log.add("drop", f"column removed{': ' + reason if reason else ''}", 1, col)

    for col in targets(df, spec.get("trim") or [], text_only=True):
        require_columns(df, [col], "trim")
        s = df[col]
        if not is_text(s):
            continue
        trimmed = s.where(s.isna(), s.astype(str).str.strip().str.replace(r"\s+", " ", regex=True))
        changed = int((trimmed != s).fillna(False).sum())
        if changed:
            df[col] = trimmed
            log.add("trim", "surrounding / repeated spaces removed", changed, col)

    missing = spec.get("missing") or {}
    require_columns(df, list(missing), "missing")
    for key, markers in missing.items():
        # "*" reaches text columns only: a marker like -1 is a real value in most numbers.
        for col in targets(df, [key], text_only=True):
            s = df[col]
            text = s.astype(str).str.strip()
            hit = s.isin(markers) | (s.notna() & text.isin([str(m) for m in markers]))
            if hit.any():
                df[col] = s.mask(hit)
                log.add("missing", f"values {markers} treated as missing", int(hit.sum()), col)

    numbers = spec.get("numbers") or {}
    require_columns(df, list(numbers), "numbers")
    for col, opts in numbers.items():
        before = df[col].notna()
        df[col] = data_utils.parse_number(
            df[col], opts.get("thousands"), opts.get("decimal"), bool(opts.get("percent"))
        )
        lost = int((before & df[col].isna()).sum())
        log.add(
            "numbers",
            f"parsed as numbers ({opts or 'separators detected'}); "
            f"{lost} values could not be parsed and are now missing",
            lost,
            col,
        )

    dates = spec.get("dates") or {}
    require_columns(df, list(dates), "dates")
    for col, opts in dates.items():
        before = df[col].notna()
        kwargs = {"format": opts["format"]} if opts.get("format") else {"format": "mixed"}
        df[col] = pd.to_datetime(
            df[col], errors="coerce", dayfirst=bool(opts.get("dayfirst")), **kwargs
        )
        lost = int((before & df[col].isna()).sum())
        log.add(
            "dates",
            f"parsed as dates ({opts or 'format detected'}); "
            f"{lost} values could not be parsed and are now missing",
            lost,
            col,
        )

    booleans = spec.get("booleans") or {}
    require_columns(df, list(booleans), "booleans")
    for col, opts in booleans.items():
        true = {str(v).strip().lower() for v in opts.get("true", [])}
        false = {str(v).strip().lower() for v in opts.get("false", [])}
        if not true or not false:
            fail(f"`booleans.{col}` needs both `true` and `false` value lists")
        text = df[col].astype("string").str.strip().str.lower()
        parsed = pd.Series(pd.NA, index=df.index, dtype="boolean")
        parsed[text.isin(true).fillna(False)] = True
        parsed[text.isin(false).fillna(False)] = False
        lost = int((df[col].notna() & parsed.isna()).sum())
        df[col] = parsed
        log.add(
            "booleans",
            f"true = {sorted(true)}, false = {sorted(false)}; {lost} other values are now missing",
            lost,
            col,
        )

    types = spec.get("types") or {}
    require_columns(df, list(types), "types")
    for col, kind in types.items():
        if kind == "string":  # applied when reading
            log.add("types", "read as text (codes keep their leading zeros)", 0, col)
            continue
        before = df[col].notna()
        values = pd.to_numeric(df[col], errors="coerce")
        if kind == "integer":
            fractional = values.notna() & (values % 1 != 0)
            values = values.mask(fractional).astype("Int64")
        df[col] = values if kind == "integer" else values.astype(float)
        lost = int((before & df[col].isna()).sum())
        log.add(
            "types", f"cast to {kind}; {lost} values did not fit and are now missing", lost, col
        )

    categories = spec.get("categories") or {}
    require_columns(df, list(categories), "categories")
    for col, mapping in categories.items():
        hit = df[col].isin(list(mapping))
        df[col] = df[col].replace(mapping)
        log.add("categories", f"spellings merged: {mapping}", int(hit.sum()), col)

    dup = spec.get("duplicates")
    if dup:
        if not dup.get("reason"):
            fail("`duplicates` needs a `reason`: it removes rows")
        subset = dup.get("subset")
        if subset:
            require_columns(df, subset, "duplicates")
        keep = {"none": False}.get(str(dup.get("keep", "first")), dup.get("keep", "first"))
        drop_rows = df.duplicated(subset=subset, keep=keep)
        df = df[~drop_rows]
        on = f"same {subset}" if subset else "identical rows"
        log.add(
            "duplicates",
            f"rows removed ({on}, keep {dup.get('keep', 'first')}): {dup['reason']}",
            int(drop_rows.sum()),
            reason=dup["reason"],
        )

    for rule in spec.get("exclude") or []:
        col, reason = rule.get("column"), rule.get("reason")
        if not col or not reason:
            fail(f"every `exclude` rule needs a `column` and a `reason`: {rule}")
        require_columns(df, [col], "exclude")
        s = df[col]
        if "outside" in rule:
            low, high = rule["outside"]
            hit = pd.Series(False, index=df.index)
            if low is not None:
                hit |= (s < low).fillna(False)
            if high is not None:
                hit |= (s > high).fillna(False)
            what = f"outside [{'−∞' if low is None else low}, {'+∞' if high is None else high}]"
        elif "equals" in rule:
            hit = s.isin(rule["equals"])
            what = f"equal to {rule['equals']}"
        elif rule.get("missing"):
            hit = s.isna()
            what = "missing"
        else:
            fail(f"`exclude` rule for `{col}` needs `outside`, `equals` or `missing`: {rule}")
        df = df[~hit]
        log.add(
            "exclude",
            f"rows removed where `{col}` is {what}: {reason}",
            int(hit.sum()),
            col,
            rule=rule,
        )
    return df.reset_index(drop=True)


def markdown(meta: dict[str, Any]) -> str:
    removed = [s for s in meta["steps"] if s["step"] in ("duplicates", "exclude")]
    lines = [
        f"# Prepared table: {Path(meta['source']).name}",
        "",
        f"Source `{meta['source']}`: {meta['rows_source']:,} rows, {meta['columns_source']} columns. "
        f"Prepared: {meta['rows']:,} rows, {len(meta['columns'])} columns.",
        "",
        "## Rows removed, and why" if removed else "## Rows removed",
        "",
        *([f"- **{s['affected']:,} rows** — {s['what']}" for s in removed] or ["- none"]),
        "",
        "## Every step",
        "",
        "| step | column | what | affected |",
        "|---|---|---|---|",
        *(
            f"| {s['step']} | {s['column'] or ''} | {s['what'].replace('|', '/')} | {s['affected']:,} |"
            for s in meta["steps"]
        ),
        "",
        "## Column types",
        "",
        ", ".join(f"`{c}` {t}" for c, t in meta["columns"].items()),
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", required=True, type=Path, help="the source file, never modified")
    p.add_argument(
        "--spec", type=Path, help="the cleaning spec (JSON); omit to only read and type the data"
    )
    p.add_argument("--sep", default=None, help="CSV / TXT separator (default: detected)")
    p.add_argument("--decimal", default=".", help="decimal mark in CSV numbers (default '.')")
    p.add_argument("--encoding", default="utf-8")
    p.add_argument("--sheet", default=None, help="Excel sheet (default: the first)")
    p.add_argument("--out", required=True, type=Path, help="folder for table.csv and table.json")
    a = p.parse_args()

    out = data_utils.output_dir(a.out)
    spec: dict[str, Any] = json.loads(a.spec.read_text()) if a.spec else {}
    unknown = [k for k in spec if k not in SPEC_KEYS and not k.startswith("_")]
    if unknown:
        fail(f"unknown keys {unknown}; the keys are {list(SPEC_KEYS)} (prefix a note with _)")
    bad = {
        c: t
        for c, t in (spec.get("types") or {}).items()
        if t not in ("string", "integer", "number")
    }
    if bad:
        fail(
            f"`types` takes string, integer or number: {bad} (dates and booleans have their own keys)"
        )

    meta_path = out / data_utils.TABLE_META
    source = str(a.data.resolve())
    if meta_path.exists() and json.loads(meta_path.read_text()).get("source") != source:
        sys.exit(f"{out} holds a table prepared from another file: use another --out folder.")

    # Columns kept as text are named in the source, i.e. before any rename.
    rename_back = {new: old for old, new in (spec.get("rename") or {}).items()}
    as_text = [rename_back.get(c, c) for c, t in (spec.get("types") or {}).items() if t == "string"]
    df = data_utils.read_table(
        a.data.resolve(),
        a.sep,
        a.decimal,
        a.encoding,
        a.sheet,
        dtype=dict.fromkeys(as_text, str) or None,
    )
    rows_source, columns_source = df.shape
    log = Log()
    df = apply(df, spec, log)

    meta = {
        "source": source,
        "reading": {"sep": a.sep, "decimal": a.decimal, "encoding": a.encoding, "sheet": a.sheet},
        "spec": str((out / SPEC_FILE) if a.spec else ""),
        "rows_source": rows_source,
        "columns_source": columns_source,
        "rows_removed": rows_source - len(df),
        "steps": log.steps,
    }
    path = data_utils.write_table(out, df, meta)
    if a.spec and a.spec.resolve() != (out / SPEC_FILE):
        shutil.copy2(a.spec, out / SPEC_FILE)
    table_meta = json.loads(meta_path.read_text())
    (out / "table.md").write_text(markdown(table_meta))
    data_utils.record_run(out, Path(__file__), ("prepare_table", "prepare_series"))
    print(f"\n{rows_source:,} → {len(df):,} rows, {df.shape[1]} columns. Prepared table: {path}")


if __name__ == "__main__":
    main()
