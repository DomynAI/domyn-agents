"""Compare forecasting models on the backtest windows: metrics table, bootstrap confidence
interval for main − baseline, and the comparison charts.

Run after forecast_baseline.py and forecast_catboost.py with the same --out. Reads
<name>_backtest.csv and <name>_forecast.csv for every model, and the prepared series
for the history in the charts.

The interval resamples whole series when there are at least 10 (a series' errors are
related to each other, so resampling them one by one would overstate the certainty),
and single forecast points otherwise, which is optimistic, and says so.

It also checks each series for non-stationarity, on the history before the first
backtest window: the level, or the size of the seasonal-naive error, changing by 1.5×
or more between the first and the last third. MASE divides by that error averaged over
the whole history, so on such a series the unit is too small (or too large) for the
recent periods, and MASE values can't be read against 1. With MASE as the deciding
metric it warns, and says which metric to rerun with.

Writes to --out: comparison.json, comparison.md (tables ready for the report),
model_comparison.png, error_by_step.png, backtest_examples.png, future_forecast.png.

    python forecast_compare.py --out <run folder>
"""

import argparse
import json
from pathlib import Path

import common
import numpy as np
import pandas as pd
import seaborn_style as style
import series_format as sf
from matplotlib.ticker import MaxNLocator

# Seasonal naive, the yardstick, in grey; the baseline and the main model in the first two
# categorical colours, the same in every chart.
COLORS = {
    "seasonal_naive": style.PALETTE["text"]["light_background"]["muted"],
    "baseline": style.PALETTE["charts"]["categorical"][0],
    "main": style.PALETTE["charts"]["categorical"][1],
}


def load(out: Path, name: str, kind: str) -> pd.DataFrame:
    path = out / f"{name}_{kind}.csv"
    if not path.exists():
        script = (
            "forecast_baseline.py" if name in ("seasonal_naive", "arima") else f"forecast_{name}.py"
        )
        raise SystemExit(f"{path} not found: run {script} first, with the same --out.")
    return pd.read_csv(path, dtype={"id": str})


def group_sums(frame: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """Per group, the sums every metric is a ratio of, so a bootstrap only re-adds them."""
    err = frame["pred"] - frame["actual"]
    scaled = (err.abs() / frame["mase_scale"]).where(frame["mase_scale"].notna())
    parts = pd.DataFrame(
        {
            "g": groups.to_numpy(),
            "abs": err.abs().to_numpy(),
            "sq": (err**2).to_numpy(),
            "err": err.to_numpy(),
            "act": frame["actual"].abs().to_numpy(),
            "scaled": scaled.fillna(0).to_numpy(),
            "scaled_n": scaled.notna().to_numpy(dtype=float),
            "n": 1.0,
        }
    )
    return parts.groupby("g").sum()


def metric_from_sums(s: pd.DataFrame | pd.Series, metric: str) -> float:
    s = s.sum() if isinstance(s, pd.DataFrame) else s
    return {
        "MASE": s["scaled"] / s["scaled_n"] if s["scaled_n"] else np.nan,
        "WAPE_%": 100 * s["abs"] / s["act"] if s["act"] else np.nan,
        "bias_%": 100 * s["err"] / s["act"] if s["act"] else np.nan,
        "MAE": s["abs"] / s["n"],
        "RMSE": np.sqrt(s["sq"] / s["n"]),
    }[metric]


def to_time(periods: pd.Series, freq: str) -> pd.DatetimeIndex:
    return pd.PeriodIndex(periods.astype(str), freq=freq).to_timestamp()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--main", default="catboost", help="name of the main model's files")
    p.add_argument("--main-label", default="CatBoost")
    p.add_argument("--baseline", default="arima")
    p.add_argument("--baseline-label", default="ARIMA")
    p.add_argument(
        "--metric",
        default="MASE",
        choices=["MASE", "WAPE_%", "MAE", "RMSE"],
        help="the metric that decides the comparison (default MASE)",
    )
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument(
        "--examples", type=int, default=4, help="series drawn in the charts (the largest)"
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out = common.data_utils.output_dir(args.out)
    meta = json.loads((out / common.META_FILE).read_text())
    target, freq, h = meta["target"], meta["freq"], meta["horizon"]
    names = {
        "seasonal_naive": "Seasonal naive",
        args.baseline: args.baseline_label,
        args.main: args.main_label,
    }
    roles = {"seasonal_naive": "seasonal_naive", args.baseline: "baseline", args.main: "main"}
    keys = ["id", "window", "step"]

    backtests = {n: load(out, n, "backtest") for n in names}
    common_rows = backtests[args.main][keys]
    for frame in backtests.values():
        common_rows = common_rows.merge(frame[keys], on=keys)
    backtests = {
        n: f.merge(common_rows, on=keys).sort_values(keys).reset_index(drop=True)
        for n, f in backtests.items()
    }
    points = len(common_rows)
    if not points:
        raise SystemExit("The models share no backtest points: were they run on the same --out?")

    table = pd.DataFrame({names[n]: common.forecast_metrics(f) for n, f in backtests.items()}).T
    by_window = pd.DataFrame(
        {
            names[n]: {
                f"window {w} ({f.loc[f.window == w, 'period'].iloc[0]} – {f.loc[f.window == w, 'period'].iloc[-1]})": common.forecast_metrics(
                    f[f.window == w]
                )[args.metric]
                for w in sorted(f["window"].unique())
            }
            for n, f in backtests.items()
        }
    )
    by_window.index.name = f"{args.metric} by backtest window"

    # Bootstrap: the same resampled groups for both models, so only the model differs.
    series_ids = backtests[args.main]["id"]
    by_series = series_ids.nunique() >= 10
    groups = series_ids if by_series else pd.Series(np.arange(points))
    sums = {
        n: group_sums(backtests[n], groups) for n in (args.main, args.baseline, "seasonal_naive")
    }
    labels = sums[args.main].index.to_numpy()
    rng = np.random.default_rng(args.seed)
    diffs, naive_diffs = [], []
    for _ in range(args.bootstrap):
        pick = labels[rng.integers(0, len(labels), len(labels))]
        m = metric_from_sums(sums[args.main].loc[pick], args.metric)
        diffs.append(m - metric_from_sums(sums[args.baseline].loc[pick], args.metric))
        naive_diffs.append(m - metric_from_sums(sums["seasonal_naive"].loc[pick], args.metric))
    lo, hi = (float(v) for v in np.nanpercentile(diffs, [2.5, 97.5]))
    nlo, nhi = (float(v) for v in np.nanpercentile(naive_diffs, [2.5, 97.5]))
    gap = float(
        table.loc[args.main_label, args.metric] - table.loc[args.baseline_label, args.metric]
    )
    if lo <= 0 <= hi:
        verdict = f"not distinguishable: the interval includes zero, so prefer the simpler {args.baseline_label}"
    elif hi < 0:
        verdict = f"{args.main_label} is reliably better"
    else:
        verdict = f"{args.baseline_label} is reliably better"
    naive_verdict = (
        f"{args.main_label} reliably beats seasonal naive"
        if nhi < 0
        else f"{args.main_label} does NOT reliably beat seasonal naive: say so prominently"
    )
    resampled = (
        f"{len(labels)} series resampled"
        if by_series
        else f"{points} forecast points resampled (fewer than 10 series: neighbouring errors are "
        "related, so this interval is optimistic)"
    )

    # Non-stationarity: the history before the first backtest window, series by series.
    history, _ = sf.load_series(Path(meta["series_file"]))
    train_until = pd.Period(meta["windows"][0]["train_until"], freq)
    checked, flagged = 0, []
    for sid, g in history[history["id"].isin(set(series_ids))].groupby("id"):
        before = pd.PeriodIndex(g["period"].astype(str), freq=freq) <= train_until
        d = common.drift(g.loc[before, "y"].to_numpy(float), meta["season_length"])
        if d is None:
            continue
        checked += 1
        if common.is_drifting(d):
            flagged.append({"id": str(sid), **{k: round(v, 2) for k, v in d.items()}})
    alt = "MAE" if series_ids.nunique() == 1 else "WAPE_%"
    mase_warning = (
        f"{len(flagged)} of {checked} series are non-stationary (level or error size changes "
        f"by {common.DRIFT}× or more over the history, e.g. {flagged[0]['id']}: level change "
        f"{flagged[0]['level_change']:.0%}, error ratio {flagged[0]['error_ratio']:.2g}×). MASE's "
        "unit, the seasonal-naive error averaged over the whole history, does not fit their "
        f"recent periods, so MASE can't be read against 1. Rerun with --metric {alt}"
        + (" (or RMSE)" if alt == "MAE" else "")
        + ", and judge 'better than seasonal naive' by the CI against it."
        if flagged and args.metric == "MASE"
        else None
    )
    if mase_warning:
        print(f"WARNING: {mase_warning}\n")

    # Charts
    order = list(names)
    values = table.loc[[names[n] for n in order], args.metric]
    fig, ax = style.figure(height=360)
    bars = ax.barh(values.index, values, color=[COLORS[roles[n]] for n in order])
    ax.bar_label(bars, labels=[f"{v:.3g}" for v in values], padding=4)
    reference = args.metric == "MASE" and not flagged
    if reference:
        ax.axvline(
            1,
            color=style.PALETTE["text"]["light_background"]["muted"],
            linestyle="--",
            linewidth=1,
        )
    # Room on the right for the value labels (and the reference line at 1).
    ax.set_xlim(0, 1.15 * max(values.max(), 1 if reference else 0))
    ax.set_xlabel(
        "MASE: error relative to 'same as one season ago' on past data"
        if args.metric == "MASE"
        else args.metric
    )
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    style.title(fig, f"Backtest {args.metric} by model (lower is better)")
    style.save(fig, out, "model_comparison")

    fig, ax = style.figure()
    for n in order:
        f = backtests[n]
        steps = sorted(f["step"].unique())
        ax.plot(
            steps,
            [common.forecast_metrics(f[f.step == s])[args.metric] for s in steps],
            marker="o",
            color=COLORS[roles[n]],
            label=names[n],
        )
    # Whole periods only, even with a single step (which MaxNLocator otherwise splits).
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
    ax.set_xlabel("periods ahead")
    ax.set_ylabel(args.metric)
    style.title(
        fig, f"How the error grows with the distance ahead ({args.metric}, lower is better)"
    )
    style.legend(ax)
    style.save(fig, out, "error_by_step")

    largest = history.groupby("id")["y"].apply(lambda v: v.abs().sum()).sort_values(ascending=False)
    examples = [s for s in largest.index if s in set(series_ids)][: args.examples]
    futures = {n: load(out, n, "forecast") for n in names}
    spans = {
        # Backtest: the windows, with as much history again before them for context.
        "backtest": max(3 * h * len(meta["windows"]), 2 * meta["season_length"]),
        # Future: enough recent history to read the forecast against, not so much that
        # a short forecast is squeezed into the right edge.
        "future": max(8 * h, 2 * meta["season_length"], 30),
    }
    # A line needs two points: a one-point segment drawn without a marker is invisible,
    # which is every window when the horizon is one period.
    marker = "o" if h <= 3 else None
    # The length of one period, to set the models' one-period forecasts side by side.
    first = pd.Period(history["period"].iloc[0], freq)
    period_length = (first + 1).to_timestamp() - first.to_timestamp()

    for kind in ("backtest", "future"):
        height = max(420, 240 * len(examples) + 120)
        fig, axes = style.figure(height=height, nrows=len(examples), sharex=True, squeeze=False)
        panels = axes[:, 0]
        for ax, sid in zip(panels, examples, strict=True):
            # The legend is built once, from the first panel's lines.
            first_panel = ax is panels[0]
            hist = history[history.id == sid].tail(spans[kind])
            ax.plot(
                to_time(hist["period"], freq),
                hist["y"],
                color=style.PALETTE["text"]["light_background"]["heading"],
                linewidth=1.5,
                label="actual" if first_panel else None,
            )
            for j, n in enumerate(order):
                src = backtests[n] if kind == "backtest" else futures[n]
                f = src[src.id == sid]
                color = COLORS[roles[n]]
                label = names[n] if first_panel else None
                has_band = roles[n] != "seasonal_naive" and "lo80" in f and f["lo80"].notna().any()
                if kind == "backtest":
                    # One-period windows are consecutive one-step forecasts: one line.
                    # Longer windows each restart from new data: one segment each.
                    parts = [f] if h == 1 else [g for _, g in f.groupby("window")]
                    for k, part in enumerate(parts):
                        ax.plot(
                            to_time(part["period"], freq), part["pred"], color=color,
                            linestyle="--", marker=marker, label=label if k == 0 else None,
                        )  # fmt: skip
                elif h == 1:
                    # A single future point: an interval bar, since a band needs a width.
                    t = to_time(f["period"], freq) + (j - 1) * 0.25 * period_length
                    pred = f["pred"].to_numpy()
                    yerr = (
                        [pred - f["lo80"].to_numpy(), f["hi80"].to_numpy() - pred]
                        if has_band
                        else None
                    )
                    ax.errorbar(
                        t, pred, yerr=yerr, fmt="o", color=color, markersize=7, capsize=4,
                        label=label,
                    )  # fmt: skip
                else:
                    t = to_time(f["period"], freq)
                    if has_band:
                        ax.fill_between(
                            t, f["lo80"], f["hi80"], color=style.tint(color, 0.15), linewidth=0
                        )
                    ax.plot(t, f["pred"], color=color, linestyle="--", marker=marker, label=label)
            ax.set_title(str(sid))
            ax.set_ylabel(target)
        style.legend(panels[0])
        if kind == "backtest":
            what = "one-step forecasts" if h == 1 else "forecasts, one dashed segment per window"
            # On two lines: in one, it runs past the figure's width.
            style.title(fig, f"Backtest: each model's {what},\nagainst what happened (black)")
            style.save(fig, out, "backtest_examples")
        else:
            band = "bars" if h == 1 else "shaded"
            style.title(
                fig,
                f"Forecast of the next {h} period{'s' if h > 1 else ''} "
                f"({band}: 80% prediction interval)",
            )
            style.save(fig, out, "future_forecast")

    coverage = {
        names[n]: {
            k: table.loc[names[n], k]
            for k in ("coverage80_%", "coverage95_%")
            if k in table and pd.notna(table.loc[names[n], k])
        }
        for n in order
    }
    common.write_json(
        out / "comparison.json",
        {
            "metric": args.metric,
            "backtest_points": points,
            "series": int(series_ids.nunique()),
            "metrics": table.to_dict(orient="index"),
            "by_window": by_window.to_dict(),
            "main_minus_baseline": gap,
            "ci95": [lo, hi],
            "main_minus_seasonal_naive_ci95": [nlo, nhi],
            "bootstrap": resampled,
            "verdict": verdict,
            "vs_seasonal_naive": naive_verdict,
            "interval_coverage": coverage,
            "nonstationary": {
                "series_checked": checked,
                "series_flagged": len(flagged),
                "threshold": common.DRIFT,
                "examples": flagged[:10],
            },
            "mase_warning": mase_warning,
        },
    )
    shown = [
        c
        for c in ("MASE", "WAPE_%", "bias_%", "MAE", "RMSE", "coverage80_%", "coverage95_%")
        if c in table
    ]
    (out / "comparison.md").write_text(
        f"Backtest: {len(meta['windows'])} windows of {h} periods, {int(series_ids.nunique())} series, "
        f"{points} forecast points per model. Target `{target}`, frequency {freq}.\n\n"
        f"{common.markdown_table(table[shown])}\n\n"
        + (f"**Warning:** {mase_warning}\n\n" if mase_warning else "")
        + f"{args.main_label} − {args.baseline_label} {args.metric}: {gap:+.3g} "
        f"(95% CI {lo:+.3g} to {hi:+.3g}; {resampled}) — {verdict}\n\n"
        f"{args.main_label} − seasonal naive {args.metric}: 95% CI {nlo:+.3g} to {nhi:+.3g} — {naive_verdict}\n\n"
        f"{common.markdown_table(by_window)}\n\n"
        "Coverage: the share of actual values inside the 80% / 95% intervals. Close to 80 / 95 "
        "means the intervals are honest; well below means they are too narrow.\n"
    )
    common.record_run(out, Path(__file__), "compare")

    print((out / "comparison.md").read_text())
    print(f"outputs in {out}")


if __name__ == "__main__":
    main()
