"""The skill's Default Analysis: what to compute when there is no specific question.

- Numeric columns: missing %, mean, median, std, variance, min / max, 1-25-75-99%
  quantiles, skewness, excess kurtosis, IQR outliers, each with plain flags.
- Categorical columns: distinct values, top values with shares, dominant values.
- Date columns: range and rows per period, with empty periods.
- Relationships: Pearson and Spearman correlations, the strongest pairs, pairs where
  the two disagree, near-duplicate features, and, with --target, what relates to it.
- Default charts: histograms of skewed / fat-tailed columns, a boxplot of columns with
  outliers, a correlation heatmap and a scatter of the strongest pair, bar charts of
  categorical columns, rows per period for dates.

Skips ids, constant columns and free text (and anything in --skip), and says so.
Run it on the prepared table (--table, from prepare_table.py), so text numbers and
dates are already parsed and the cleaning decisions apply.

Writes to --out: default_analysis.md (read it: tables, flags, and which chart shows
what), default_analysis.json, and the charts as <name>.png.

    python default_analysis.py --table <run folder>/files/table.csv --out <run folder>
"""

import json
import textwrap
from pathlib import Path
from typing import Any

import common
import numpy as np
import pandas as pd
import seaborn as sns
import seaborn_style as style
from matplotlib import ticker

# What every chart here takes from the palette by role.
MUTED = style.PALETTE["text"]["light_background"]["muted"]
CAPTION = style.TYPOGRAPHY["roles"]["caption"]["size"]


def numeric_summary(num: pd.DataFrame) -> pd.DataFrame:
    q = num.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
    iqr = q.loc[0.75] - q.loc[0.25]
    low, high = q.loc[0.25] - 1.5 * iqr, q.loc[0.75] + 1.5 * iqr
    outliers = ((num < low) | (num > high)).sum()
    table = pd.DataFrame(
        {
            "missing_%": 100 * num.isna().mean(),
            "mean": num.mean(),
            "median": q.loc[0.5],
            "std": num.std(),
            "var": num.var(),
            "min": num.min(),
            "p01": q.loc[0.01],
            "q1": q.loc[0.25],
            "q3": q.loc[0.75],
            "p99": q.loc[0.99],
            "max": num.max(),
            "skew": num.skew(),
            "excess_kurtosis": num.kurt(),
            "outliers": outliers,
            "outliers_%": 100 * outliers / num.notna().sum(),
        }
    )
    flags = []
    for _, r in table.iterrows():
        f = []
        if r["missing_%"] > 5:
            f.append(f"{r['missing_%']:.0f}% missing")
        if abs(r["skew"]) > 1:
            f.append(("right" if r["skew"] > 0 else "left") + "-skewed: describe by median and IQR")
        if r["excess_kurtosis"] > 3:
            f.append("fat tails: extreme values more common than normal")
        if r["outliers_%"] >= 1:
            f.append(f"{r['outliers_%']:.1f}% outliers (IQR rule)")
        if r["max"] > r["p99"] + 3 * (r["q3"] - r["q1"]) and r["q3"] > r["q1"]:
            f.append("max far above the 99% quantile: check the top values")
        flags.append("; ".join(f))
    table["flags"] = flags
    return table


def categorical_summary(df: pd.DataFrame, cols: list[str], top: int) -> pd.DataFrame:
    rows = []
    for col in cols:
        counts = df[col].value_counts(dropna=True)
        shares = counts / counts.sum()
        rows.append(
            {
                "column": col,
                "distinct": int(counts.size),
                "missing_%": 100 * df[col].isna().mean(),
                "top values": ", ".join(
                    f"{v} ({c:,}, {100 * s:.0f}%)"
                    for (v, c), s in zip(counts.head(top).items(), shares.head(top), strict=True)
                ),
                "flags": f"'{counts.index[0]}' covers {100 * shares.iloc[0]:.0f}% of rows"
                if shares.iloc[0] > 0.95
                else "",
            }
        )
    return pd.DataFrame(rows).set_index("column") if rows else pd.DataFrame()


def date_summary(
    df: pd.DataFrame, cols: list[str], out: Path, charts: list[tuple[str, str]]
) -> pd.DataFrame:
    rows = []
    for col in cols:
        d = df[col].dropna()
        if d.empty:
            continue
        span = (d.max() - d.min()).days
        period, label = (
            ("D", "day") if span <= 90 else ("W", "week") if span <= 730 else ("MS", "month")
        )
        per = d.dt.to_period(period[0] if period != "MS" else "M").value_counts().sort_index()
        full = pd.period_range(per.index.min(), per.index.max(), freq=per.index.freq)
        per = per.reindex(full, fill_value=0)
        empty = int((per == 0).sum())
        sparse = int(((per > 0) & (per < 0.1 * per.median())).sum())
        rows.append(
            {
                "column": col,
                "from": d.min().date(),
                "to": d.max().date(),
                "span_days": span,
                "period": label,
                "rows per period (median)": float(per.median()),
                "empty periods": empty,
                "sparse periods": sparse,
            }
        )
        per.index = per.index.to_timestamp()
        fig, ax = style.figure(height=420)
        # No margin before the first period: the date ticks would start a period early, where
        # "29" and "Mar" (the last day of one month, the next month) print on top of each other.
        ax.plot(per.index, per.values, marker="o" if len(per) < 40 else None, clip_on=False)
        ax.margins(x=0)
        ax.set_ylim(bottom=0)  # a truncated axis would exaggerate the dips
        ax.set(xlabel=col, ylabel=f"rows per {label}")
        style.title(fig, f"Rows per {label}, by {col}")
        charts.append(
            (
                style.save(fig, out, f"rows_per_{label}_{col}"),
                f"how many rows each {label} has, by `{col}`: gaps, trends and incomplete first / last periods",
            )
        )
    return pd.DataFrame(rows).set_index("column") if rows else pd.DataFrame()


def correlations(num: pd.DataFrame, target: str | None) -> dict[str, Any]:
    if num.shape[1] < 2:
        return {
            "pairs": pd.DataFrame(),
            "pearson": None,
            "spearman": None,
            "target": pd.DataFrame(),
        }
    pearson, spearman = num.corr(method="pearson"), num.corr(method="spearman")
    cols = list(num.columns)
    rows = [
        {"a": a, "b": b, "pearson": pearson.loc[a, b], "spearman": spearman.loc[a, b]}
        for i, a in enumerate(cols)
        for b in cols[i + 1 :]
    ]
    pairs = pd.DataFrame(rows)
    pairs["strongest"] = pairs[["pearson", "spearman"]].abs().max(axis=1)
    pairs["note"] = [
        "near-duplicate features"
        if s >= 0.95
        else "methods disagree: non-linear or outlier-driven"
        if abs(p - sp) >= 0.2
        else ""
        for p, sp, s in zip(pairs["pearson"], pairs["spearman"], pairs["strongest"], strict=True)
    ]
    pairs = pairs.sort_values("strongest", ascending=False)
    shown = pairs[(pairs["strongest"] >= 0.5) | (pairs["note"] != "")].drop(columns="strongest")
    target_table = pd.DataFrame()
    if target is not None and target in num.columns:
        target_table = pd.DataFrame(
            {
                "pearson": pearson[target].drop(target),
                "spearman": spearman[target].drop(target),
            }
        )
        target_table = target_table.reindex(
            target_table.abs().max(axis=1).sort_values(ascending=False).index
        )
        target_table.index.name = f"correlation with {target}"
    return {"pairs": shown, "pearson": pearson, "spearman": spearman, "target": target_table}


def charts_for(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    corr: dict[str, Any],
    cats: list[str],
    out: Path,
    charts: list[tuple[str, str]],
    max_each: int,
) -> None:
    skewed = summary[(summary["skew"].abs() > 1) | (summary["excess_kurtosis"] > 3)]
    for col in skewed.reindex(skewed["skew"].abs().sort_values(ascending=False).index).index[
        :max_each
    ]:
        values = df[col].dropna()
        log = values.min() > 0 and values.max() / values.min() > 100
        fig, ax = style.figure()
        if log:
            # Equal-width bins on the log scale, drawn on a log axis.
            ax.hist(values, bins=np.geomspace(values.min(), values.max(), 41), rwidth=0.98)
            ax.set_xscale("log")
            # Plain numbers on the ticks (1, 10, 1,000), not the default's "10^3" formula.
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.6g}"))
            ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        else:
            ax.hist(values, bins=40, rwidth=0.98)
        # Room above the bars for the two labels, so they never sit on the data.
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
        # Each label on the outer side of its line, so the two never overlap however close.
        median_side = "left" if values.median() <= values.mean() else "right"
        marks = (
            (
                "median",
                values.median(),
                style.PALETTE["charts"]["categorical"][1],
                "--",
                median_side,
            ),
            ("mean", values.mean(), MUTED, ":", "right" if median_side == "left" else "left"),
        )
        for label, x, color, dash, side in marks:
            ax.axvline(x, color=color, linestyle=dash, linewidth=2)
            ax.annotate(
                f"{label} {x:,.3g}",
                (x, 1),
                xycoords=ax.get_xaxis_transform(),
                xytext=(-4 if side == "left" else 4, -4),
                textcoords="offset points",
                ha="right" if side == "left" else "left",
                va="top",
                fontsize=CAPTION,
                color=color,
            )
        ax.set(xlabel=col + (" (log scale)" if log else ""), ylabel="number of rows")
        style.title(fig, f"Distribution of {col}" + (" (log scale)" if log else ""))
        charts.append(
            (
                style.save(fig, out, f"distribution_{col}"),
                f"the shape of `{col}`: skew {summary.loc[col, 'skew']:.2f}, excess kurtosis {summary.loc[col, 'excess_kurtosis']:.2f}; mean vs. median marked",
            )
        )

    with_outliers = (
        summary[summary["outliers_%"] >= 1].sort_values("outliers_%", ascending=False).index[:6]
    )
    if len(with_outliers):
        primary = style.PALETTE["primary"]  # one measure per panel: not series to tell apart
        width = max(600, 180 * len(with_outliers) + 120)
        # A panel's title wraps to its width (about 10 px a bold character), never into the next one.
        chars = max(10, (width - 120) // len(with_outliers) // 10)
        fig, axes = style.figure(
            width=width,
            ncols=len(with_outliers),
            squeeze=False,
        )
        for ax, col in zip(axes[0], with_outliers, strict=True):
            ax.boxplot(
                df[col].dropna(),
                widths=0.5,
                patch_artist=True,
                boxprops={"facecolor": style.tint(primary, 0.35), "edgecolor": primary},
                medianprops={"color": primary, "linewidth": 2},
                whiskerprops={"color": primary},
                capprops={"color": primary},
                flierprops={
                    "marker": "o",
                    "markersize": 4,
                    "markerfacecolor": style.tint(primary, 0.4),
                    "markeredgecolor": primary,
                    "markeredgewidth": 0.5,
                },
            )
            ax.set_xticks([])
            ax.set_title(textwrap.fill(str(col), chars))
        style.title(
            fig, "Outliers: points beyond the whiskers (1.5 × IQR); each column on its own scale"
        )
        charts.append(
            (
                style.save(fig, out, "outliers_boxplot"),
                "which columns have outliers, and how far out they are: "
                + ", ".join(f"`{c}`" for c in with_outliers),
            )
        )

    if corr["pearson"] is not None and corr["pearson"].shape[0] >= 3:
        disagree = (corr["pearson"] - corr["spearman"]).abs().max().max() >= 0.2
        method = "spearman" if disagree else "pearson"
        m = corr[method]
        size = max(520, 44 * len(m) + 260)
        fig, ax = style.figure(width=size + 120, height=size)
        sns.heatmap(
            m,
            annot=len(m) <= 12,
            fmt=".2f",
            cmap=style.colormap("diverging"),
            center=0,
            vmin=-1,
            vmax=1,
            linewidths=1,
            linecolor=style.PALETTE["surfaces"]["card_light"],
            cbar_kws={"label": f"{method.title()} correlation"},
            ax=ax,
        )
        ax.grid(False)
        ax.tick_params(axis="x", rotation=90)
        ax.tick_params(axis="y", rotation=0)
        ax.tick_params(length=0)
        ax.set(xlabel="", ylabel="")
        style.title(fig, f"Correlation between numeric columns ({method.title()})")
        charts.append(
            (
                style.save(fig, out, "correlation_heatmap"),
                f"which numeric columns move together ({method.title()}; blue = together, coral = opposite)",
            )
        )

    if not corr["pairs"].empty:
        a, b = corr["pairs"].iloc[0][["a", "b"]]
        sample = df[[a, b]].dropna()
        sample = sample.sample(min(len(sample), 5000), random_state=0)
        fig, ax = style.figure(width=760, height=560)
        ax.scatter(sample[a], sample[b], s=14, alpha=0.4, linewidths=0)
        ax.set(xlabel=a, ylabel=b)
        style.title(fig, f"{b} against {a}")
        charts.append(
            (
                style.save(fig, out, f"scatter_{a}_vs_{b}"),
                f"the strongest relationship: `{a}` and `{b}`",
            )
        )

    for col in cats[:max_each]:
        counts = df[col].value_counts(dropna=False).head(10).iloc[::-1]
        fig, ax = style.figure(height=max(320, 30 * len(counts) + 160))
        bars = ax.barh(range(len(counts)), counts.values, height=0.7)
        ax.set_yticks(range(len(counts)), [str(v) for v in counts.index])
        ax.bar_label(
            bars,
            labels=[f"{v:,} ({100 * v / len(df):.0f}%)" for v in counts.values],
            padding=4,
        )
        # Room on the right for the labels past the longest bar.
        ax.set_xlim(0, counts.max() * 1.18)
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("number of rows")
        style.title(
            fig, f"Most common values of {col}" + (" (top 10)" if df[col].nunique() > 10 else "")
        )
        charts.append(
            (
                style.save(fig, out, f"values_{col}"),
                f"how rows are spread across the values of `{col}`",
            )
        )


def main() -> None:
    p = common.base_parser(__doc__.splitlines()[0])
    p.add_argument(
        "--skip", nargs="*", default=[], help="columns to leave out, besides ids / constants / text"
    )
    p.add_argument(
        "--target", help="an outcome column: other columns are ranked by how they relate to it"
    )
    p.add_argument("--top", type=int, default=5, help="top values listed per categorical column")
    p.add_argument(
        "--max-charts", type=int, default=4, help="charts per kind (histograms, bar charts)"
    )
    args = p.parse_args()

    out = common.prepare_out(args.out)
    df, notes = common.load_args(args)
    data = common.source(args)
    kinds = common.column_kinds(df)
    for col in args.skip:
        if col not in df.columns:
            raise SystemExit(f"--skip: `{col}` is not a column")
        kinds[col] = "skipped by request"
    if args.target and args.target not in df.columns:
        raise SystemExit(f"--target: `{args.target}` is not a column")

    skipped = {
        c: k for c, k in kinds.items() if k in ("id", "constant", "text", "skipped by request")
    }
    numeric = [c for c, k in kinds.items() if k == "numeric"]
    cats = [c for c, k in kinds.items() if k == "categorical"]
    dates = [c for c, k in kinds.items() if k == "date"]
    charts: list[tuple[str, str]] = []

    summary = numeric_summary(df[numeric]) if numeric else pd.DataFrame()
    cat_table = categorical_summary(df, cats, args.top)
    date_table = date_summary(df, dates, out, charts)
    corr = correlations(df[numeric], args.target) if numeric else correlations(pd.DataFrame(), None)
    by_class = pd.DataFrame()
    if args.target and kinds.get(args.target) == "categorical" and numeric:
        grouped = df.groupby(args.target)
        by_class = grouped[numeric].median().T
        by_class.loc["(rows)"] = grouped.size()
        by_class.index.name = f"median by {args.target}"
    if numeric:
        charts_for(df, summary, corr, cats, out, charts, args.max_charts)
    else:
        charts_for(
            df,
            pd.DataFrame(columns=["skew", "excess_kurtosis", "outliers_%"]),
            corr,
            cats,
            out,
            charts,
            args.max_charts,
        )

    flags = [f"`{c}`: {f}" for c, f in summary["flags"].items() if f] if numeric else []
    flags += (
        [f"`{c}`: {f}" for c, f in cat_table["flags"].items() if f] if not cat_table.empty else []
    )
    flags += [f"`{r.a}` ~ `{r.b}`: {r.note}" for r in corr["pairs"].itertuples() if r.note]

    md = [
        f"# Default analysis: {data.name}",
        f"\n**Rows:** {len(df):,} · **Columns analyzed:** {len(numeric) + len(cats) + len(dates)} of {df.shape[1]}",
    ]
    md += [f"\n**Note:** {n}" for n in notes]
    md += [
        "\n## Skipped columns\n",
        "\n".join(f"- `{c}` ({k})" for c, k in skipped.items()) or "- none",
    ]
    md += [
        "\n## Flags\n",
        "\n".join(f"- {f}" for f in flags) or "- nothing notable flagged automatically",
    ]
    if numeric:
        md += ["\n## Numeric columns\n", common.markdown_table(summary.drop(columns="flags"))]
    if not cat_table.empty:
        md += ["\n## Categorical columns\n", common.markdown_table(cat_table, 1)]
    if not date_table.empty:
        md += ["\n## Date columns\n", common.markdown_table(date_table, 1)]
    if not corr["pairs"].empty:
        md += [
            "\n## Correlated pairs (|r| ≥ 0.5, or a note)\n",
            common.markdown_table(corr["pairs"].set_index("a"), 2),
        ]
    elif len(numeric) >= 2:
        md += ["\n## Correlated pairs\n", "No pair of numeric columns has |r| ≥ 0.5."]
    if corr["pearson"] is not None and len(numeric) <= 10:
        md += ["\n## Correlation matrix (Spearman)\n", common.markdown_table(corr["spearman"], 2)]
    if not corr["target"].empty:
        md += [f"\n## Relation to `{args.target}`\n", common.markdown_table(corr["target"], 2)]
    if not by_class.empty:
        md += [f"\n## Numeric columns by `{args.target}`\n", common.markdown_table(by_class, 3)]
    md += [
        "\n## Charts\n",
        "Place each in the finding it supports; explain how to read it.\n",
        "\n".join(f"- `{name}`: {what}" for name, what in charts) or "- none",
    ]
    (out / "default_analysis.md").write_text("\n".join(md) + "\n")

    def records(frame: pd.DataFrame) -> Any:
        return (
            json.loads(frame.to_json(orient="index", date_format="iso")) if not frame.empty else {}
        )

    (out / "default_analysis.json").write_text(
        json.dumps(
            {
                "data": str(data),
                "rows": len(df),
                "notes": notes,
                "skipped": skipped,
                "flags": flags,
                "numeric": records(summary),
                "categorical": records(cat_table),
                "dates": records(date_table),
                "correlated_pairs": json.loads(corr["pairs"].to_json(orient="records"))
                if not corr["pairs"].empty
                else [],
                "target": records(corr["target"]),
                "charts": dict(charts),
            },
            indent=2,
        )
    )
    common.record_run(out, Path(__file__))
    print((out / "default_analysis.md").read_text())


if __name__ == "__main__":
    main()
