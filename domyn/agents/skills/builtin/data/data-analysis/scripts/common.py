"""Shared helpers for the data-analysis scripts, and for your own `analysis.py`.

Not run directly. `profile_data.py` and `default_analysis.py` import it, and copy it
into the output folder, so a question-specific `analysis.py` written there can use the
same loading and column typing, and draw charts in the house style (seaborn_style.py,
copied there too):

    import common
    import seaborn_style as style  # after common, which puts it on the path
    import seaborn as sns

    df, notes = common.load(Path("/abs/<run folder>/files/table.csv"))  # the prepared table
    fig, ax = style.figure()
    sns.boxplot(df, x="region", y="price", saturation=1, ax=ax)
    style.title(fig, "Price by region")
    style.save(fig, Path(__file__).resolve().parent, "price_by_region")  # price_by_region.png

`load` reads a prepared table (table.csv from prepare_table.py) with its column types,
or a raw file as it is: cleaning happens once, in prepare_table.py, never here.
"""

import argparse
import re
import sys
from pathlib import Path

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

# Order of the commands in run.sh, whatever order the scripts were run in.
RUN_ORDER = (
    "prepare_table",
    "prepare_series",
    "profile_data",
    "default_analysis",
    "timeseries_analysis",
    "analysis",
)

# Values that usually mean "missing" when they appear in a column.
SENTINEL_TEXT = {"", "na", "n/a", "nan", "null", "none", "-", "?", "unknown", "missing", "#n/a"}
SENTINEL_NUMBERS = (-1, -999, -9999, 999, 9999, 99999)

NUMBER_TEXT = re.compile(r"^\s*[-+(]?\s*[$€£¥]?\s*[-+]?[\d.,\s]+%?\s*[)]?\s*$")


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--table", type=Path, help="the prepared table.csv (prepare_table.py): the cleaned data"
    )
    source.add_argument(
        "--data", type=Path, help="a raw CSV, TSV, Parquet, Excel, JSON or JSONL file, as it is"
    )
    p.add_argument("--out", required=True, type=Path, help="output folder (created if missing)")
    p.add_argument("--sheet", default=None, help="Excel sheet name (default: the first)")
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
    p.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="read only the first N rows (large files: say so in the report)",
    )
    return p


def source(args: argparse.Namespace) -> Path:
    """The file a script was given: the prepared table, or the raw data."""
    return Path(args.table or args.data).resolve()


# Numbers stored as text are parsed in prepare_table.py; kept for your own analysis.py.
to_number = data_utils.parse_number


def load(
    path: Path,
    *,
    nrows: int | None = None,
    sheet: str | None = None,
    sep: str | None = None,
    decimal: str = ".",
    encoding: str = "utf-8",
) -> tuple[pd.DataFrame, list[str]]:
    """Read the prepared table, or a raw file as it is.

    A prepared table (table.csv with table.json next to it) comes back with its column
    types; the reading options don't apply to it. A raw file is read with them and
    nothing else: numbers or dates stored as text stay text until prepare_table.py
    parses them. Returns the frame and one note per thing the report should say.
    """
    path = data_utils.find_input(path, data_utils.TABLE_FILE) if path.is_dir() else path.resolve()
    if path.name == data_utils.TABLE_FILE and (path.parent / data_utils.TABLE_META).exists():
        df, meta = data_utils.load_table(path)
        notes = [
            f"prepared table from `{meta['source']}`: {meta['rows_source']:,} → {meta['rows']:,} "
            f"rows after {len(meta['steps'])} cleaning steps (see table.md)"
        ]
        if nrows:
            df = df.head(nrows)
            notes.append(f"only the first {nrows} rows were read")
        return df, notes
    df = data_utils.read_table(
        path, sep, decimal, encoding, sheet, nrows=nrows, single_column_ok=True
    )
    notes: list[str] = []
    if path.suffix.lower() in (".csv", ".txt") and sep is None:
        detected = data_utils.detect_sep(path, encoding)
        if detected != ",":
            notes.append(f"columns are separated by {detected!r} (detected from the file)")
    if path.suffix.lower() in (".csv", ".txt", ".tsv"):
        tab = "\t" if path.suffix.lower() == ".tsv" else None
        skipped = data_utils.comment_lines(path, encoding, sep or tab)
        if skipped:
            notes.append(
                f"the first {len(skipped)} line(s), comments before the header, were skipped"
            )
    if decimal != ".":
        notes.append(f"numbers were read with {decimal!r} as the decimal mark")
    if nrows:
        notes.append(f"only the first {nrows} rows were read")
    return df, notes


def load_args(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str]]:
    """`load` with a script's own options."""
    return load(
        source(args),
        nrows=args.nrows,
        sheet=args.sheet,
        sep=args.sep,
        decimal=args.decimal,
        encoding=args.encoding,
    )


def column_kinds(df: pd.DataFrame) -> dict[str, str]:
    """numeric / categorical / date / id / constant / text, one per column.

    `id`, `constant` and `text` are the columns the default analysis skips: a
    near-unique value per row, a single value, or long free text.
    """
    kinds: dict[str, str] = {}
    n = len(df)
    for col in df.columns:
        s = df[col]
        distinct = s.nunique(dropna=True)
        if distinct <= 1:
            kinds[col] = "constant"
        elif pd.api.types.is_bool_dtype(s):
            kinds[col] = "categorical"
        elif pd.api.types.is_datetime64_any_dtype(s):
            kinds[col] = "date"
        elif pd.api.types.is_numeric_dtype(s):
            integer_like = pd.api.types.is_integer_dtype(s) or bool((s.dropna() % 1 == 0).all())
            kinds[col] = (
                "id"
                if integer_like and n > 50 and distinct >= 0.95 * s.notna().sum()
                else "numeric"
            )
        else:
            text = s.dropna().astype(str)
            if text.str.len().mean() > 50 and distinct > 0.5 * len(text):
                kinds[col] = "text"
            elif n > 50 and distinct >= 0.95 * len(text):
                kinds[col] = "id"
            else:
                kinds[col] = "categorical"
    return kinds


def prepare_out(out: Path) -> Path:
    """Where the scripts write: the run folder's `files/` (report.md stays at its top)."""
    return data_utils.output_dir(out)


markdown_table = data_utils.markdown_table
safe_name = data_utils.safe_name
write_json = data_utils.write_json


def record_run(out: Path, script: Path, modules: tuple[Path, ...] = ()) -> None:
    """Copy the script (with this module and data_utils) into the output folder, and log its
    command in run.sh."""
    data_utils.record_run(out, script, RUN_ORDER, (Path(__file__), *modules))
