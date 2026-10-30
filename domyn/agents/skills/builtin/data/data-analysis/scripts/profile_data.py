"""Profile a dataset and check its quality: Steps 2 and 3 of the skill.

Reports shape, column types, a sample of rows, missing values (including sentinel values
such as -1, "N/A", "unknown"), duplicates, numbers and dates stored as text, categories
spelled several ways, suspicious values, and which columns the default analysis would
skip (ids, constants, free text).

Changes nothing: it reports, and drafts the cleaning spec for prepare_table.py in
cleaning_suggested.json. Its `missing`, `numbers`, `dates`, `categories` and `trim`
entries are what the checks found; `_to_decide` holds what only you can decide
(numeric placeholders like -1, duplicate rows). Review it, save it as cleaning.json,
and run prepare_table.py. Run this again with --table on the result to check it.

Writes to --out: profile.md (read it, and use it for the report's "About the data"),
profile.json, cleaning_suggested.json.

    python profile_data.py --data /abs/path/sales.csv --out <run folder>
    python profile_data.py --table <run folder>/files/table.csv --out <run folder>
"""

import json
from pathlib import Path
from typing import Any

import common
import numpy as np
import pandas as pd


def is_text(s: pd.Series) -> bool:
    """Text column, whether pandas stores it as `object` (pandas < 3) or `str` (pandas 3)."""
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)


def number_like_text(s: pd.Series) -> float:
    """Share of non-empty text values that look like numbers ("1,234", "$5", "12%")."""
    text = s.dropna().astype(str).str.strip()
    text = text[text != ""]
    if text.empty:
        return 0.0
    return float(text.str.match(common.NUMBER_TEXT).mean())


def date_like_text(s: pd.Series) -> float:
    text = s.dropna().astype(str).head(500)
    if text.empty or text.str.fullmatch(r"[-+]?\d+(\.\d+)?").all():
        return 0.0  # plain numbers parse as dates too; they are not dates
    parsed = pd.to_datetime(text, errors="coerce", format="mixed")
    return float(parsed.notna().mean())


def spelling_variants(s: pd.Series, limit: int = 5) -> list[list[str]]:
    """Groups of values that are the same after trimming, lower-casing and removing punctuation."""
    values = s.dropna().astype(str)
    if values.nunique() > 5000:
        return []
    key = (
        values.str.strip()
        .str.lower()
        .str.replace(r"[\s._-]+", " ", regex=True)
        .str.replace(r"[^\w ]", "", regex=True)
    )
    groups = (
        pd.DataFrame({"raw": values, "key": key})
        .drop_duplicates()
        .groupby("key")["raw"]
        .apply(list)
    )
    return [sorted(g) for g in groups if len(g) > 1][:limit]


def leading_zero_codes(data: Path, args: Any, df: pd.DataFrame) -> list[str]:
    """Numeric columns whose raw text has leading zeros ("02134", "00017"): codes that
    reading as numbers has already damaged. Only a raw CSV / Excel file can tell."""
    if args.table or data.suffix.lower() not in (".csv", ".txt", ".tsv", ".xlsx", ".xls"):
        return []
    raw = common.data_utils.read_table(
        data, args.sep, args.decimal, args.encoding, args.sheet, nrows=2000,
        single_column_ok=True, dtype=str,
    )  # fmt: skip
    found = []
    for col in df.columns:
        if col in raw.columns and pd.api.types.is_numeric_dtype(df[col]):
            text = raw[col].dropna().astype(str).str.strip()
            if len(text) and text.str.fullmatch(r"0\d+").mean() > 0.05:
                found.append(col)
    return found


def main() -> None:
    p = common.base_parser(__doc__.splitlines()[0])
    p.add_argument(
        "--id-cols",
        nargs="*",
        default=[],
        help="columns that should be unique (checked for duplicates)",
    )
    args = p.parse_args()

    out = common.prepare_out(args.out)
    df, notes = common.load_args(args)
    data = common.source(args)
    # In a prepared table, text the spec kept as text (codes) is a decision, not an issue.
    kept_text = set()
    if args.table:
        _, table_meta = common.data_utils.load_table(data)
        kept_text = {s["column"] for s in table_meta["steps"] if s["step"] == "types"}
    kinds = common.column_kinds(df)
    n = len(df)

    rows: list[dict[str, Any]] = []
    suggest_numbers, suggest_dates, suggest_decimal, issues = [], [], [], []
    spec: dict[str, Any] = {
        "types": {}, "trim": [], "missing": {}, "numbers": {}, "dates": {}, "categories": {}
    }  # fmt: skip
    to_decide: dict[str, Any] = {"missing": {}}
    for col in df.columns:
        s = df[col]
        missing = int(s.isna().sum())
        sentinel_text = (
            int(s.dropna().astype(str).str.strip().str.lower().isin(common.SENTINEL_TEXT).sum())
            if is_text(s)
            else 0
        )
        sentinel_num = (
            {int(v): int((s == v).sum()) for v in common.SENTINEL_NUMBERS}
            if kinds[col] == "numeric"
            else {}
        )
        sentinel_num = {k: v for k, v in sentinel_num.items() if v >= max(3, 0.005 * n)}
        if sentinel_text:
            markers = s.dropna().astype(str).str.strip()
            spec["missing"][col] = sorted(
                set(s.dropna()[markers.str.lower().isin(common.SENTINEL_TEXT)].astype(str))
            )
        if sentinel_num:
            to_decide["missing"][col] = sorted(sentinel_num)
        if is_text(s) and (s.dropna().astype(str) != s.dropna().astype(str).str.strip()).any():
            spec["trim"].append(col)
        row = {
            "column": col,
            "type": str(s.dtype),
            "kind": kinds[col],
            "distinct": int(s.nunique()),
            "missing": missing,
            "missing_%": 100 * missing / n if n else 0.0,
            "example": str(s.dropna().iloc[0])[:40] if s.notna().any() else "",
        }
        rows.append(row)

        if missing / max(n, 1) > 0.05:
            issues.append(f"`{col}`: {row['missing_%']:.1f}% missing")
        if sentinel_text:
            issues.append(
                f'`{col}`: {sentinel_text} values look like a missing marker ("N/A", "unknown", "-", …)'
            )
        for value, count in sentinel_num.items():
            issues.append(
                f"`{col}`: {count} rows equal {value}, a common placeholder for missing — check before averaging"
            )
        if is_text(s) and kinds[col] in ("categorical", "id") and col not in kept_text:
            share = number_like_text(s)
            comma_decimal = s.dropna().astype(str).str.fullmatch(r"-?\d+,\d+").mean()
            if share >= 0.9 and comma_decimal >= 0.9:
                suggest_decimal.append(col)
                spec["numbers"][col] = {"decimal": ","}
                issues.append(
                    f"`{col}`: numbers with a comma decimal mark ({100 * comma_decimal:.0f}% of values, e.g. "
                    f"{s.dropna().iloc[0]!r}) — suggested under `numbers` with decimal ','"
                )
            elif share >= 0.9:
                suggest_numbers.append(col)
                spec["numbers"][col] = {}
                issues.append(
                    f"`{col}`: numbers stored as text ({100 * share:.0f}% of values) — suggested under "
                    "`numbers` (set `thousands` / `decimal` if the file is consistent)"
                )
            elif date_like_text(s) >= 0.9:
                suggest_dates.append(col)
                spec["dates"][col] = {}
                issues.append(
                    f"`{col}`: dates stored as text — suggested under `dates` "
                    '(add "dayfirst": true for 31/01/2024)'
                )
        if kinds[col] == "categorical" and is_text(s):
            counts = s.value_counts()
            for group in spelling_variants(s):
                issues.append(f"`{col}`: the same value spelled several ways: {group}")
                keep = max(group, key=lambda v: counts.get(v, 0))
                spec["categories"].setdefault(col, {}).update({v: keep for v in group if v != keep})
            top_share = s.value_counts(normalize=True).iloc[0] if s.notna().any() else 0
            if top_share > 0.95:
                issues.append(f"`{col}`: one value covers {100 * top_share:.1f}% of rows")
        if kinds[col] == "numeric":
            values = s.dropna()
            if len(values) and (values < 0).mean() < 0.01 and (values < 0).any():
                issues.append(
                    f"`{col}`: {int((values < 0).sum())} negative values in an otherwise non-negative column"
                )
        if kinds[col] == "date":
            future = int((s > pd.Timestamp.now()).sum())
            if future:
                issues.append(f"`{col}`: {future} dates in the future")

    for col in leading_zero_codes(data, args, df):
        spec.setdefault("types", {})[col] = "string"
        issues.append(
            f'`{col}`: a code with leading zeros ("0…") read as a number, which drops them — '
            "suggested under `types` as string"
        )
    duplicates = int(df.duplicated().sum())
    if duplicates:
        issues.insert(0, f"{duplicates} fully duplicated rows ({100 * duplicates / n:.1f}%)")
        to_decide["duplicates"] = {
            "subset": None,
            "keep": "first",
            "reason": "state why they are copies",
        }
    for col in args.id_cols:
        if col in df.columns and df[col].duplicated().any():
            to_decide[f"duplicates on {col}"] = {
                "subset": [col],
                "keep": "first",
                "reason": f"state why {col} repeats",
            }
            issues.insert(
                0,
                f"`{col}` should be unique but has {int(df[col].duplicated().sum())} repeated values",
            )

    # Columns the suggested flags will convert are analyzed after conversion, not skipped.
    converted = {*suggest_numbers, *suggest_dates, *suggest_decimal}
    skipped = {
        c: k for c, k in kinds.items() if k in ("id", "constant", "text") and c not in converted
    }
    columns = pd.DataFrame(rows).set_index("column")
    spec = {k: v for k, v in spec.items() if v}
    to_decide = {k: v for k, v in to_decide.items() if v}
    if to_decide:
        spec["_to_decide"] = to_decide
    spec["_note"] = (
        "Drafted by profile_data.py. Review every entry, move what you decide from "
        "_to_decide to its key, add types / drop / exclude, and save as cleaning.json."
    )
    (out / "cleaning_suggested.json").write_text(json.dumps(spec, indent=2, default=str))

    sample = df.head(10).map(lambda v: str(v)[:30])
    md = [
        f"# Profile: {data.name}",
        f"\n**Rows:** {n:,} · **Columns:** {df.shape[1]} · **Fully duplicated rows:** {duplicates}",
        *(f"\n**Note:** {note}" for note in notes),
        "\n## Columns\n",
        common.markdown_table(columns, 1),
        "\n## Possible issues\n",
        "\n".join(f"- {i}" for i in issues)
        or "- none found by the automatic checks (still check units and what one row is)",
        "\n## Skipped by the default analysis\n",
        "\n".join(f"- `{c}` ({k})" for c, k in skipped.items()) or "- none",
        "\n## Suggested cleaning\n",
        "Drafted in `cleaning_suggested.json` for prepare_table.py: "
        + (", ".join(f"`{k}`" for k in spec if not k.startswith("_")) or "nothing to fix found")
        + (" — plus `_to_decide`: " + ", ".join(to_decide) if to_decide else ""),
        "\n## First rows\n",
        common.markdown_table(sample.set_index(sample.columns[0])),
    ]
    (out / "profile.md").write_text("\n".join(md) + "\n")
    (out / "profile.json").write_text(
        json.dumps(
            {
                "data": str(data),
                "rows": n,
                "columns": df.shape[1],
                "duplicated_rows": duplicates,
                "notes": notes,
                "issues": issues,
                "kinds": kinds,
                "skipped": skipped,
                "suggested_cleaning": spec,
                "per_column": json.loads(columns.to_json(orient="index")),
            },
            indent=2,
            default=lambda v: v.item() if isinstance(v, np.generic) else str(v),
        )
    )
    common.record_run(out, Path(__file__))
    print((out / "profile.md").read_text())


if __name__ == "__main__":
    main()
