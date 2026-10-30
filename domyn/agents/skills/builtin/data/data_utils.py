"""Helpers shared by every data skill's scripts (data-analysis, tabular-classification-regression,
time-series-forecasting): reading a data file, parsing numbers stored as text, the
prepared table (`table.csv` / `table.json`, see `write_table`), report tables, JSON, and
recording a run in the output folder. Charts are drawn with seaborn_style.py (seaborn, PNG), in
builtin/resources/ with the palette, shared by every builtin skill.

It lives one level above the skills, in `builtin/data/`, and each skill's scripts
find it there (see the import at the top of their `common.py`). `record_run` copies
it into every output folder next to the scripts, so a folder reruns on its own with
`sh run.sh`, without the skill; it copies seaborn_style.py, palette.json, typography.json and
the DM Sans files along with it, so the charts rerun in the house style too.

Not a skill, and not run directly. Keep it free of anything specific to one skill.
"""

import csv
import json
import re
import shlex
import shutil
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SEPARATORS = (",", ";", "\t", "|")

# Lines before a CSV's header that start with this are comments (see comment_lines).
COMMENT_PREFIX = "#"
_NUMBER = re.compile(r"[-+]?\d+(?:[.,]\d+)?")

# Options whose value is a path: made absolute in run.sh (see command_line).
PATH_OPTIONS = ("--data", "--table", "--spec", "--series", "--features", "--future", "--out")

# Every run folder holds report.md at the top and everything else in this subfolder:
# the scripts write only there, and run.sh lives there too (see output_dir).
FILES_DIR = "files"

# The house chart style and what it reads, shared by every builtin skill: in
# builtin/resources/, or next to this file in an output folder (where record_run copied them,
# the fonts into FONTS_DIR).
RESOURCES_DIR = "resources"
CHART_STYLE = "seaborn_style.py"
PALETTE = "palette.json"
TYPOGRAPHY = "typography.json"
FONTS_DIR = "fonts"
SHIPPED_FONTS = Path("assets") / "fonts"

# The prepared table: written by prepare_table.py, read by every data skill.
TABLE_FILE = "table.csv"
TABLE_META = "table.json"
TABLE_FORMAT_VERSION = 1
COLUMN_TYPES = ("number", "integer", "boolean", "date", "string")


def output_dir(out: Path) -> Path:
    """Where a script writes, for `--out <run folder>`: `<run folder>/files`, created.

    The run folder itself holds only report.md, which the agent writes, so a reader
    opening it sees the report first and the evidence one level down. A path that is
    already the files folder (a copied script rerun from run.sh) is used as it is.
    """
    out = out.resolve()
    files = out if out.name == FILES_DIR else out / FILES_DIR
    files.mkdir(parents=True, exist_ok=True)
    return files


def find_input(path: Path, name: str) -> Path:
    """An input given as a file, a run folder or its files folder: the file `name` in it."""
    path = path.resolve()
    if not path.is_dir():
        return path
    for candidate in (path / name, path / FILES_DIR / name):
        if candidate.exists():
            return candidate
    return path / FILES_DIR / name


def _split_preamble(path: Path, encoding: str) -> tuple[list[str], list[str]]:
    """The lines before the header that start with `#` (and blank lines among them), and
    the first 50 non-blank lines after them.

    A file made of nothing but such lines has no preamble: there is no header to skip to.
    """
    preamble: list[str] = []
    rest: list[str] = []
    with path.open(encoding=encoding, errors="replace", newline="") as f:
        for line in f:
            text = line.lstrip("\ufeff").strip()
            if not rest and (not text or text.startswith(COMMENT_PREFIX)):
                preamble.append(line)
                continue
            if text:
                rest.append(line)
            if len(rest) >= 50:
                break
    return (preamble, rest) if rest else ([], [])


def _fields(line: str, sep: str) -> list[str]:
    return next(csv.reader([line], delimiter=sep), [])


def comment_lines(path: Path, encoding: str = "utf-8", sep: str | None = None) -> list[str]:
    """The lines to skip before a CSV's header: comments starting with `#`, and blank lines.

    Exports often open with a few `# source: ...` / `# units: ...` lines. Read as they
    are, the first comment becomes the header and pandas stops on the first data row
    ("Expected 2 fields in line 4, saw 4"), or the separator is detected on the comments.

    A header can start with `#` too (`#,name,price`, `#id;...`). The last `#` line is
    taken as the header when it splits into as many fields as the line after it and that
    line holds a number, i.e. is data rather than the real header.
    """
    preamble, rest = _split_preamble(path, encoding)
    marked = [i for i, line in enumerate(preamble) if line.strip()]
    if not marked:
        return preamble
    sep = sep or _best_sep(rest)
    last = preamble[marked[-1]].lstrip("\ufeff")
    head, following = _fields(last, sep), _fields(rest[0], sep)
    looks_like_data = any(_NUMBER.fullmatch(value.strip()) for value in following)
    if len(head) > 1 and len(head) == len(following) and looks_like_data:
        return preamble[: marked[-1]]
    return preamble


def detect_sep(path: Path, encoding: str = "utf-8") -> str:
    """The separator that splits the first lines into the same number (> 1) of fields.

    Files exported with a comma decimal mark (European locales) are usually separated
    by ';'. Read with the default ',', such a file comes back as one column, or split
    in the middle of every number. Comment lines before the header are not counted
    (see `comment_lines`).
    """
    return _best_sep(_split_preamble(path, encoding)[1] or _first_lines(path, encoding))


def _first_lines(path: Path, encoding: str) -> list[str]:
    with path.open(encoding=encoding, errors="replace", newline="") as f:
        return [line for line in (f.readline() for _ in range(50)) if line.strip()]


def _best_sep(lines: list[str]) -> str:
    best, best_score = ",", 0.0
    for sep in SEPARATORS:
        counts = [len(row) for row in csv.reader(lines, delimiter=sep)]
        if not counts or counts[0] < 2:
            continue
        score = sum(c == counts[0] for c in counts) / len(counts) + counts[0] / 1000
        if score > best_score:
            best, best_score = sep, score
    return best


def read_table(
    path: Path,
    sep: str | None = None,
    decimal: str = ".",
    encoding: str = "utf-8",
    sheet: str | None = None,
    nrows: int | None = None,
    single_column_ok: bool = False,
    dtype: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Read a CSV / TSV / TXT, Parquet, Excel, JSON or JSONL file.

    A CSV read as a single column almost always means the wrong separator: that stops
    the script, unless `single_column_ok` (profiling, which should show it rather
    than refuse). Text columns that look like comma-decimal numbers print a warning.
    `dtype` forces column types where the format allows it (CSV, Excel): `{"zip": str}`
    keeps "02134" from becoming the number 2134. Comment lines starting with `#` before a
    CSV's header are skipped and printed (see `comment_lines`).
    """
    suffix = path.suffix.lower()
    if suffix in (".csv", ".txt", ".tsv"):
        if sep is None:
            sep = "\t" if suffix == ".tsv" else detect_sep(path, encoding)
        print(f"Reading {path.name} with separator {sep!r} and decimal mark {decimal!r}")
        skipped = comment_lines(path, encoding, sep)
        comments = [line.lstrip("\ufeff").strip() for line in skipped if line.strip()]
        if comments:
            # Printed whole: such lines often say what the columns are (units, source).
            print(f"Skipped the first {len(skipped)} line(s), comments before the header:")
            for comment in comments[:20]:
                print(f"  {comment[:200]}")
            if len(comments) > 20:
                print(f"  ... and {len(comments) - 20} more")
        df = pd.read_csv(
            path,
            skiprows=len(skipped),
            sep=sep,
            decimal=decimal,
            encoding=encoding,
            nrows=nrows,
            low_memory=False,
            dtype=dtype,
        )
        if df.shape[1] == 1:
            message = f"{path.name} was read as a single column with separator {sep!r}"
            if not single_column_ok:
                sys.exit(f"{message}: pass --sep.")
            print(f"WARNING: {message}. If that is wrong, pass --sep.")
        comma_numbers = [
            c
            for c in df.select_dtypes(include="object").columns
            if df[c].dropna().astype(str).str.fullmatch(r"-?\d+,\d+").mean() > 0.9
        ]
        if comma_numbers and decimal != ",":
            print(
                f"WARNING: {comma_numbers} look like numbers with a comma decimal mark, and "
                "would be read as text. Rerun with --decimal ','."
            )
        return df
    if suffix == ".parquet":
        df = pd.read_parquet(path)
        return df.head(nrows) if nrows else df
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path, sheet_name=sheet or 0, nrows=nrows, dtype=dtype)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True, nrows=nrows, encoding=encoding)
    if suffix == ".json":
        df = pd.read_json(path, encoding=encoding)
        return df.head(nrows) if nrows else df
    sys.exit(f"Unsupported file type: {path.suffix}")


def parse_number(
    series: pd.Series,
    thousands: str | None = None,
    decimal: str | None = None,
    percent: bool = False,
) -> pd.Series:
    """Numbers stored as text: currency signs, thousands separators, %, (negative).

    With `thousands` / `decimal` unset, each value's last separator followed by one or
    two digits is taken as the decimal mark ("1.234,5" and "1,234.5" both give 1234.5).
    Pass them when the file is consistent, which avoids guessing on values like
    "1,234" (a thousand, or 1.234?). `percent` turns "12%" into 0.12; otherwise the
    sign is dropped and it gives 12. Unparseable values become missing.
    """
    text = series.astype("string").str.strip()
    negative = text.str.startswith("(") & text.str.endswith(")")
    had_percent = text.str.endswith("%")
    cleaned = text.str.replace(r"[$€£¥%()\s]", "", regex=True)
    if thousands is not None or decimal is not None:
        if thousands:
            cleaned = cleaned.str.replace(thousands, "", regex=False)
        if decimal and decimal != ".":
            cleaned = cleaned.str.replace(decimal, ".", regex=False)
    else:
        european = cleaned.str.contains(r"\d\.\d{3},\d", regex=True) | cleaned.str.fullmatch(
            r"\d+,\d{1,2}"
        )
        cleaned = cleaned.where(
            ~european, cleaned.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
        )
        cleaned = cleaned.where(european, cleaned.str.replace(",", "", regex=False))
    values = pd.to_numeric(cleaned, errors="coerce").astype(float)
    values = values.where(~negative.fillna(False), -values)
    if percent:
        values = values.where(~had_percent.fillna(False), values / 100)
    return values


def column_type(s: pd.Series) -> str:
    """The prepared-table type of a column (one of COLUMN_TYPES), from its pandas dtype."""
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "date"
    if pd.api.types.is_integer_dtype(s):
        return "integer"
    if pd.api.types.is_numeric_dtype(s):
        return "number"
    return "string"


def write_table(out: Path, df: pd.DataFrame, meta: dict[str, Any]) -> Path:
    """Write the prepared table: `table.csv` and `table.json`, in `out`.

    table.csv is plain CSV whatever the source was: `,` separator, `.` decimal mark,
    UTF-8, dates as ISO 8601 (`2024-01-31`, or `2024-01-31T13:05:00` when any time is
    set), booleans as `true` / `false`, and an empty cell for a missing value. CSV
    carries no types, so table.json records each column's (`columns`), and
    `load_table` restores them: codes like "001" stay text, dates come back as dates.
    """
    out.mkdir(parents=True, exist_ok=True)
    types = {str(c): column_type(df[c]) for c in df.columns}
    frame = df.copy()
    for col, kind in types.items():
        s = frame[col]
        if kind == "date":
            timed = bool((s.dropna() != s.dropna().dt.normalize()).any())
            frame[col] = s.dt.strftime("%Y-%m-%dT%H:%M:%S" if timed else "%Y-%m-%d")
        elif kind == "boolean":
            frame[col] = s.map({True: "true", False: "false"})
        elif kind == "integer":
            frame[col] = s.astype("Int64")
    frame.to_csv(out / TABLE_FILE, index=False, encoding="utf-8")
    payload = {"format_version": TABLE_FORMAT_VERSION, **meta, "rows": len(df), "columns": types}
    write_json(out / TABLE_META, payload)
    return out / TABLE_FILE


def load_table(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read a prepared table (table.csv, or the folder holding it) with its column types.

    Every cell is read as text first, and only an empty cell is missing: a category
    called "NA" or "None" stays a value, since missing values were decided when the
    table was prepared.
    """
    path = find_input(path, TABLE_FILE)
    meta_path = path.parent / TABLE_META
    if not path.exists() or not meta_path.exists():
        sys.exit(
            f"{path} (with {TABLE_META} next to it) not found. Prepare the data first with "
            "prepare_table.py, and pass its table.csv."
        )
    meta = json.loads(meta_path.read_text())
    types: dict[str, str] = meta["columns"]
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""], encoding="utf-8")
    if list(raw.columns) != list(types):
        sys.exit(f"{path} does not match {TABLE_META}: rerun prepare_table.py.")
    df = pd.DataFrame(index=raw.index)
    for col, kind in types.items():
        s = raw[col]
        if kind == "number":
            df[col] = pd.to_numeric(s).astype(float)
        elif kind == "integer":
            df[col] = pd.to_numeric(s).astype("Int64")
        elif kind == "boolean":
            df[col] = s.map({"true": True, "false": False}).astype("boolean")
        elif kind == "date":
            df[col] = pd.to_datetime(s, format="ISO8601")
        else:
            df[col] = s.astype(object)
    return df, meta


def markdown_table(frame: pd.DataFrame, digits: int = 3) -> str:
    """A DataFrame as a Markdown table, without needing `tabulate`.

    Numbers get thousands separators and at most `digits` decimals, without trailing
    zeros; missing values show as "–".
    """

    def cell(v: Any) -> str:
        if not isinstance(v, (float, np.floating)):
            return str(v).replace("|", "\\|")
        if np.isnan(v):
            return "–"
        if abs(v) >= 1e6:
            return f"{v:.3g}"
        text = f"{v:,.{digits}f}"
        text = text.rstrip("0").rstrip(".") if "." in text else text
        return "0" if text == "-0" else text

    header = [frame.index.name or "", *map(str, frame.columns)]
    rows = [[cell(r[0]), *map(cell, r[1:])] for r in frame.itertuples()]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows])


def safe_name(text: str) -> str:
    """A string usable as a file name."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_") or "column"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON, turning numpy scalars into plain numbers."""
    path.write_text(
        json.dumps(payload, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    )


def command_line(script: Path, argv: Sequence[str] | None = None) -> str:
    """This run's command for run.sh, with every path option made absolute.

    run.sh starts by changing into the output folder, so a relative `--out runs/x`
    replayed from there would write to runs/x/files/runs/x.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    for k, a in enumerate(args):
        if a in PATH_OPTIONS and k + 1 < len(args):
            args[k + 1] = str(Path(args[k + 1]).resolve())
        elif "=" in a and a.split("=", 1)[0] in PATH_OPTIONS:
            name, value = a.split("=", 1)
            args[k] = f"{name}={Path(value).resolve()}"
    return f'"$PY" {shlex.quote(script.name)} ' + " ".join(shlex.quote(a) for a in args)


# A command line in run.sh: an interpreter, then a script. `"$PY"` is what command_line
# writes; a bare `python` / `python3` is the older format, or a line added by hand.
_RUN_COMMAND = re.compile(r'^\s*(?:"\$PY"|\$PY|\S*python[\d.]*)\s+(\S+\.py)(?:\s|$)')
_BARE_PYTHON = re.compile(r"^(\s*)python[\d.]*(?=\s)")
_RUN_DEFAULT_PY = 'PY="${PY:-'


def _run_header(python: str) -> list[str]:
    """run.sh's first lines.

    The interpreter that ran the scripts is the default, because a bare `python` is
    whichever one comes first on PATH, which may not have the packages.
    `PY=/path/to/python sh run.sh` picks another one, and a default that no longer exists
    (another machine, a deleted environment) falls back to `python`.
    """
    quoted = re.sub(r'(["$`\\])', r"\\\1", python)
    return [
        "#!/bin/sh",
        'cd "$(dirname "$0")"',
        "# The Python that ran these scripts. To use another: PY=/path/to/python sh run.sh",
        f'{_RUN_DEFAULT_PY}{quoted}}}"',
        'command -v "$PY" >/dev/null 2>&1 || PY=python',
    ]


def _run_script(line: str) -> str | None:
    """The script a run.sh line runs, or None for anything else (a variable, a comment)."""
    match = _RUN_COMMAND.match(line)
    return Path(match.group(1).strip("'\"")).name if match else None


def record_run(out: Path, script: Path, order: Sequence[str], modules: Iterable[Path] = ()) -> None:
    """Copy the script, the modules it needs, this file and the chart style (seaborn_style.py,
    palette.json, typography.json, and the DM Sans files into `fonts/`) into the output
    folder, and log its command in run.sh, so the folder reruns without the skill.

    run.sh keeps one line per script (a rerun replaces it), sorted by `order`: each
    line goes at the first entry its script name contains, unknown scripts last. Lines
    that run no script (a variable, a comment) are kept, between the header and the
    commands, and a command with a bare `python` is switched to `"$PY"`. See `_run_header`.
    """
    out = out.resolve()
    here = Path(__file__).resolve()
    resources = next(
        (d for d in (here.parents[1] / RESOURCES_DIR, here.parent) if (d / CHART_STYLE).is_file()),
        None,
    )
    style = [resources / name for name in (CHART_STYLE, PALETTE, TYPOGRAPHY)] if resources else []
    shared = (here, *(p for p in style if p.is_file()))
    for src in (*shared, *(Path(m).resolve() for m in modules), script.resolve()):
        if src.parent != out:
            shutil.copy2(src, out / src.name)
    # The fonts the style draws in: from the skills' assets, or a rerun's own copy.
    fonts = next(
        (
            d
            for d in (here.parents[1] / RESOURCES_DIR / SHIPPED_FONTS, here.parent / FONTS_DIR)
            if any(d.glob("*.ttf"))
        ),
        None,
    )
    if fonts is not None and fonts.resolve() != (out / FONTS_DIR).resolve():
        (out / FONTS_DIR).mkdir(exist_ok=True)
        for src in (*fonts.glob("*.ttf"), *fonts.glob("*LICENSE*")):
            shutil.copy2(src, out / FONTS_DIR / src.name)
    run = out / "run.sh"
    header = _run_header(sys.executable)
    commands: dict[str, str] = {}
    others: list[str] = []
    for line in run.read_text().splitlines() if run.exists() else []:
        name = _run_script(line)
        if name is not None:
            commands[name] = _BARE_PYTHON.sub(r'\1"$PY"', line, count=1)
        elif line.strip() and line not in header and not line.startswith(_RUN_DEFAULT_PY):
            # Kept, before the commands: a variable or comment added by hand.
            others.append(line)
    commands[script.name] = command_line(script)

    def rank(name: str) -> int:
        return next((i for i, step in enumerate(order) if step in name), len(order))

    ordered = [commands[name] for name in sorted(commands, key=rank)]
    run.write_text("\n".join([*header, *others, *ordered]) + "\n")
