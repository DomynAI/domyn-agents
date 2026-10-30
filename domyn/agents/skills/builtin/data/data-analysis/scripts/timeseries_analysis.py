"""Describe time series: trend, seasonality, autocorrelation, growth, outliers and data
problems, per series, with the default time-series charts.

Reads only prepared series (series.csv / series.json, written by prepare_series.py;
see series_format.py). The same prepared file is what the time-series-forecasting
skill reads, so an analysis followed by a forecast prepares the data once.

How trend and seasonality are measured (classical decomposition, per series with at
least two full seasonal cycles):
- trend: a centred moving average over one seasonal cycle;
- seasonal pattern: the average deviation from the trend at each position in the cycle
  (each month, weekday, ...), centred on zero;
- remainder: what is left after removing both.
Strength (0 to 1) is how much of the variation the component explains beyond the
remainder: 1 − var(remainder) / var(component + remainder). Above 0.6 is strong,
below 0.3 is weak. Outliers are remainders more than 3.5 robust standard deviations
(1.4826 × MAD) from the median.

Writes to --out: timeseries_stats.csv (one row per series), timeseries_analysis.md /
.json (tables and flags, ready for the report), and the charts listed in the .md.

    python timeseries_analysis.py --series <run folder>/files/series.csv --out <run folder>
"""

import argparse
from pathlib import Path
from typing import Any

import common
import numpy as np
import pandas as pd
import seaborn_style as style
import series_format as sf

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
# Reference lines (zero, the noise band) in the palette's muted text colour.
MUTED = style.PALETTE["text"]["light_background"]["muted"]


def position_label(period: pd.Period, freq: str, season: int) -> str:
    """Where a period sits in the seasonal cycle, in words."""
    if freq == "M" and season == 12:
        return MONTHS[period.month - 1]
    if freq.startswith("Q") and season == 4:
        return f"Q{period.quarter}"
    if freq == "D" and season == 7:
        return WEEKDAYS[period.dayofweek]
    if freq == "h" and season == 24:
        return f"{period.hour:02d}:00"
    if freq.startswith("W") and season == 52:
        return f"week {period.week}"
    return f"position {period.ordinal % season + 1}"


def cycle_positions(
    periods: pd.PeriodIndex, freq: str, season: int
) -> tuple[dict[int, str], list[int]]:
    """Each position in the cycle (ordinal % season): its label, and the calendar order
    (Jan..Dec, Mon..Sun) to draw them in."""
    labels: dict[int, str] = {}
    key: dict[int, int] = {}
    for q, x in zip(periods.asi8 % season, periods, strict=False):
        labels.setdefault(int(q), position_label(x, freq, season))
        key.setdefault(int(q), x.dayofweek if freq == "D" and season == 7 else int(q))
    return labels, sorted(labels, key=key.__getitem__)


def moving_average(y: np.ndarray, season: int) -> np.ndarray:
    """Centred moving average over one cycle (2×m for an even cycle), NaN at the ends."""
    s = pd.Series(y)
    if season % 2 == 0:
        ma = s.rolling(season, center=True).mean().rolling(2).mean().shift(-1)
    else:
        ma = s.rolling(season, center=True).mean()
    return ma.to_numpy()


def decompose(y: np.ndarray, periods: pd.PeriodIndex, season: int) -> dict[str, Any] | None:
    if season < 2 or len(y) < 2 * season:
        return None
    trend = moving_average(y, season)
    detrended = y - trend
    pos = periods.asi8 % season
    index = pd.Series(detrended).groupby(pos).mean()
    index = index - index.mean()
    seasonal = index.reindex(pos).to_numpy()
    remainder = y - trend - seasonal
    ok = ~np.isnan(remainder)
    r = remainder[ok]

    def strength(component: np.ndarray) -> float:
        total = np.var(component[ok] + r)
        return float(max(0.0, 1 - np.var(r) / total)) if total > 0 else 0.0

    mad = 1.4826 * np.median(np.abs(r - np.median(r)))
    outliers = (
        np.flatnonzero(ok)[np.abs(r - np.median(r)) > 3.5 * mad] if mad > 0 else np.array([], int)
    )
    return {
        "trend": trend,
        "seasonal": seasonal,
        "remainder": remainder,
        "index": index,
        "trend_strength": strength(trend),
        "seasonal_strength": strength(seasonal),
        "outliers": outliers,
    }


def autocorrelation(y: np.ndarray, lag: int) -> float:
    if lag >= len(y) - 2:
        return float("nan")
    a, b = y[:-lag] - y.mean(), y[lag:] - y.mean()
    denom = np.sum((y - y.mean()) ** 2)
    return float(np.sum(a * b) / denom) if denom else float("nan")


def series_stats(
    sid: str, g: pd.DataFrame, freq: str, season: int
) -> tuple[dict[str, Any], dict | None]:
    y, periods = g["y"].to_numpy(), pd.PeriodIndex(g["period"])
    mean = float(np.mean(y))
    row: dict[str, Any] = {
        "id": sid,
        "periods": len(y),
        "first": str(periods[0]),
        "last": str(periods[-1]),
        "mean": mean,
        "median": float(np.median(y)),
        "min": float(np.min(y)),
        "max": float(np.max(y)),
        "cv": float(np.std(y) / abs(mean)) if mean else float("nan"),
        "zero_share": float(np.mean(y == 0)),
        "acf_lag1": autocorrelation(y, 1),
        f"acf_lag{season}": autocorrelation(y, season) if season > 1 else float("nan"),
    }
    # Linear trend, per seasonal cycle, as a share of the mean level.
    t = np.arange(len(y))
    slope = float(np.polyfit(t, y, 1)[0]) if len(y) >= 3 else float("nan")
    row["trend_per_cycle_%"] = 100 * slope * max(season, 1) / abs(mean) if mean else float("nan")
    if season > 1 and len(y) >= 2 * season:
        last, prev = y[-season:].sum(), y[-2 * season : -season].sum()
        row["last_cycle_vs_previous_%"] = 100 * (last / prev - 1) if prev else float("nan")
    parts = decompose(y, periods, season)
    if parts:
        row["trend_strength"] = parts["trend_strength"]
        row["seasonal_strength"] = parts["seasonal_strength"]
        idx = parts["index"]
        level = abs(mean) or 1.0
        labels, _ = cycle_positions(periods, freq, season)
        row["peak"] = labels[int(idx.idxmax())]
        row["trough"] = labels[int(idx.idxmin())]
        row["seasonal_swing_%"] = float(100 * (idx.max() - idx.min()) / level)
        row["outliers"] = len(parts["outliers"])
        row["outlier_periods"] = ", ".join(str(periods[k]) for k in parts["outliers"][:5])
        # The last period far below what trend and season predict is often a partial one.
        expected = np.nanmean(parts["trend"][-season:]) + idx.get(int(periods.asi8[-1] % season), 0)
        row["last_period_vs_expected_%"] = (
            float(100 * (y[-1] / expected - 1)) if expected else float("nan")
        )
    return row, parts


def flags_for(row: dict[str, Any], season: int) -> list[str]:
    out = []
    if row["periods"] < 2 * season and season > 1:
        out.append(f"short: fewer than two {season}-period cycles, seasonality can't be measured")
    if row.get("seasonal_strength", 0) >= 0.6:
        out.append(f"strong seasonality (peak {row['peak']}, trough {row['trough']})")
    if row.get("trend_strength", 0) >= 0.6:
        out.append(f"strong trend ({row['trend_per_cycle_%']:+.1f}% of the mean per cycle)")
    if row["zero_share"] >= 0.3:
        out.append(f"intermittent: {100 * row['zero_share']:.0f}% of periods are zero")
    if row.get("outliers", 0):
        out.append(f"{row['outliers']} outlier periods ({row['outlier_periods']})")
    if row.get("last_period_vs_expected_%", 0) <= -40:
        out.append("last period far below expected: possibly incomplete (a partial month / week)")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--series", required=True, type=Path, help="series.csv from prepare_series.py")
    p.add_argument("--out", required=True, type=Path, help="output folder")
    p.add_argument(
        "--season-length", type=int, default=None, help="default: the one in series.json"
    )
    p.add_argument(
        "--examples", type=int, default=6, help="series drawn in the charts (the largest)"
    )
    p.add_argument(
        "--decompose",
        default=None,
        help="series id to decompose in the chart (default: the largest)",
    )
    args = p.parse_args()

    out = common.prepare_out(args.out)
    frame, meta = sf.load_series(args.series)
    freq, target = meta["freq"], meta["target"]
    season = args.season_length or int(meta["season_length"])

    rows, parts = [], {}
    for sid, g in frame.groupby("id", sort=False):
        row, part = series_stats(str(sid), g, freq, season)
        row["flags"] = "; ".join(flags_for(row, season))
        rows.append(row)
        parts[str(sid)] = part
    stats = pd.DataFrame(rows).set_index("id")
    stats.to_csv(out / "timeseries_stats.csv")
    volume = frame.groupby("id")["y"].apply(lambda v: v.abs().sum()).sort_values(ascending=False)
    examples = volume.index[: args.examples].tolist()
    charts: dict[str, str] = {}

    # Overview: the largest series, one panel each.
    k = len(examples)
    primary = style.PALETTE["primary"]
    fig, axes = style.figure(
        width=1000, height=max(420, 200 * k + 120), nrows=k, sharex=True, squeeze=False
    )
    for ax, sid in zip(axes[:, 0], examples, strict=True):
        g = frame[frame.id == sid]
        ax.plot(pd.PeriodIndex(g["period"]).to_timestamp(), g["y"], color=primary, linewidth=1.5)
        ax.set_ylim(bottom=min(0, float(g["y"].min())))
        ax.set_ylabel(target)
        ax.set_title(str(sid))
    style.title(
        fig,
        f"{target} over time ({'the largest ' + str(k) + ' series' if len(stats) > k else 'all series'})",
    )
    charts["timeseries_overview"] = style.save(fig, out, "timeseries_overview")

    focus = args.decompose or examples[0]
    if focus not in parts:
        raise SystemExit(
            f"--decompose {focus!r} is not a series id. Ids: {stats.index[:10].tolist()}"
        )
    part = parts[focus]
    if part:
        g = frame[frame.id == focus]
        t = pd.PeriodIndex(g["period"]).to_timestamp()
        components = (
            ("observed", g["y"].to_numpy()),
            ("trend", part["trend"]),
            ("seasonal pattern", part["seasonal"]),
            ("remainder", part["remainder"]),
        )
        fig, axes = style.figure(width=1000, height=820, nrows=4, sharex=True)
        for ax, (label, values) in zip(axes, components, strict=True):
            ax.plot(t, values, color=primary, linewidth=1.5)
            ax.set_ylabel(label)
        outliers = part["outliers"]
        if len(outliers):
            axes[3].scatter(
                t[outliers],
                part["remainder"][outliers],
                color=style.PALETTE["charts"]["negative"],
                s=36,
                zorder=3,
            )
        axes[3].axhline(0, color=MUTED, linewidth=1)
        style.title(
            fig,
            f"{focus}: {target} split into trend, seasonal pattern and remainder (coral: outliers)",
        )
        charts["timeseries_decomposition"] = style.save(fig, out, "timeseries_decomposition")

        # Seasonal profile: deviation from the trend at each position, as % of the mean.
        fig, ax = style.figure()
        ticks: list[str] = []
        for sid in examples:
            pr = parts.get(sid)
            if not pr:
                continue
            g = frame[frame.id == sid]
            periods = pd.PeriodIndex(g["period"])
            labels, order = cycle_positions(periods, freq, season)
            idx = pr["index"].reindex(order)
            ticks = ticks or [labels[q] for q in order]
            ax.plot(range(len(order)), 100 * idx / abs(g["y"].mean()), marker="o", label=str(sid))
        ax.axhline(0, color=MUTED, linewidth=1)
        ax.set_xticks(range(len(ticks)), ticks, rotation=90 if season > 12 else 0)
        ax.set_ylabel("% of the series' average")
        style.legend(ax)
        style.title(fig, "Seasonal pattern: how far each position sits above or below the trend")
        charts["timeseries_seasonal_profile"] = style.save(fig, out, "timeseries_seasonal_profile")

    y = frame.loc[frame.id == focus, "y"].to_numpy()
    lags = np.arange(1, min(len(y) - 2, max(2 * season, 12)) + 1)
    acf = [autocorrelation(y, int(lag)) for lag in lags]
    band = 1.96 / np.sqrt(len(y))
    fig, ax = style.figure()
    ax.bar(
        lags,
        acf,
        color=[
            style.PALETTE["charts"]["categorical"][1]
            if season > 1 and lag % season == 0
            else primary
            for lag in lags
        ],
    )
    ax.axhspan(-band, band, color=style.tint(MUTED, 0.15), linewidth=0, zorder=0)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set(xlabel="k (periods back)", ylabel="autocorrelation")
    style.title(
        fig,
        f"{focus}: how much each value resembles the one k periods earlier (coral: whole cycles)",
    )
    charts["timeseries_autocorrelation"] = style.save(fig, out, "timeseries_autocorrelation")

    if len(stats) >= 5 and "seasonal_strength" in stats:
        fig, ax = style.figure(width=700, height=620)
        ax.scatter(stats["trend_strength"], stats["seasonal_strength"], s=36, alpha=0.6)
        ax.set(xlabel="trend strength", ylabel="seasonal strength", xlim=(0, 1.02), ylim=(0, 1.02))
        style.title(fig, "Each series: how much trend and seasonality it has (0 none, 1 dominant)")
        charts["timeseries_strength"] = style.save(fig, out, "timeseries_strength")

    flagged = stats[stats["flags"] != ""]
    shown = [
        c
        for c in (
            "periods",
            "first",
            "last",
            "mean",
            "trend_per_cycle_%",
            "last_cycle_vs_previous_%",
            "trend_strength",
            "seasonal_strength",
            "peak",
            "trough",
            "seasonal_swing_%",
            "acf_lag1",
            "zero_share",
            "outliers",
        )
        if c in stats
    ]
    summary = stats[shown].describe().T[["mean", "50%", "min", "max"]] if len(stats) > 1 else None
    what = {
        "timeseries_overview": "the series over time: level, trend, seasonality and breaks at a glance",
        "timeseries_decomposition": f"{focus} split into trend, the repeating seasonal pattern and what is left",
        "timeseries_seasonal_profile": "the seasonal pattern of each large series, comparable across sizes",
        "timeseries_autocorrelation": f"{focus}: bars outside the grey band are real repetition; coral bars at "
        "whole cycles confirm seasonality",
        "timeseries_strength": "trend and seasonal strength of every series: which behave alike",
    }
    lines = [
        f"# Time series: {target}",
        "",
        sf.describe(frame, meta) + ".",
        f"Prepared from `{meta['source']}` ({meta['layout']} layout; aggregate {meta['aggregate']}, "
        f"fill gaps {meta['fill_gaps']}, {meta['filled_values']} values filled).",
        "",
        "## Per series"
        if len(stats) <= 20
        else "## Largest 20 series (all in timeseries_stats.csv)",
        "",
        common.markdown_table(stats.loc[list(volume.index[:20]), shown]),
        "",
    ]
    if summary is not None:
        lines += ["## Across series", "", common.markdown_table(summary), ""]
    lines += ["## Flags", ""]
    lines += [f"- **{sid}**: {f}" for sid, f in flagged["flags"].items()] or ["- none"]
    lines += ["", "## Charts", ""] + [f"- `{name}` — {what[name]}" for name in charts]
    (out / "timeseries_analysis.md").write_text("\n".join(lines) + "\n")
    common.write_json(
        out / "timeseries_analysis.json",
        {
            "target": target,
            "freq": freq,
            "season_length": season,
            "series": len(stats),
            "flags": flagged["flags"].to_dict(),
            "charts": charts,
            "stats": stats.reset_index().to_dict(orient="records"),
        },
    )
    here = Path(__file__).resolve()
    common.record_run(out, here, (Path(sf.__file__),))

    print((out / "timeseries_analysis.md").read_text())
    print(f"outputs in {out}")


if __name__ == "__main__":
    main()
