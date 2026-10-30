"""Compare regression models on the test rows: metrics table, bootstrap confidence
interval for main − baseline, and the comparison charts.

Run after regression_baseline.py and regression_catboost.py with the same --out.
Reads <name>_test_predictions.csv for every model.

Writes to --out: comparison.json, comparison.md (tables ready for the report),
model_comparison.png, predicted_vs_actual.png, error_distribution.png.

    python regression_compare.py --out <run folder>
"""

import argparse
import json
from pathlib import Path

import common
import numpy as np
import pandas as pd
import seaborn_style as style

HIGHER_IS_BETTER = {"R2"}


def load(out: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    path = out / f"{name}_test_predictions.csv"
    if not path.exists():
        script = (
            "regression_baseline.py" if name in ("dummy", "baseline") else f"regression_{name}.py"
        )
        raise SystemExit(f"{path} not found: run {script} first, with the same --out.")
    frame = pd.read_csv(path).sort_values("row")
    return frame["actual"].to_numpy(dtype=float), frame["pred"].to_numpy(dtype=float)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--main", default="catboost", help="name of the main model's prediction file")
    p.add_argument("--main-label", default="CatBoost")
    p.add_argument("--baseline", default="baseline")
    p.add_argument("--baseline-label", default="Linear regression")
    p.add_argument("--metric", default="RMSE", help="the metric that decides the comparison")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out = common.data_utils.output_dir(args.out)
    meta = json.loads((out / common.META_FILE).read_text())
    if meta["task"] != "regression":
        raise SystemExit("This folder holds a classification run: use classification_compare.py.")
    target = meta["target"]

    labels = {
        "dummy": "Always the training mean",
        args.baseline: args.baseline_label,
        args.main: args.main_label,
    }
    preds = {name: load(out, name) for name in labels}
    actual = preds[args.main][0]
    table = pd.DataFrame(
        {labels[n]: common.regression_metrics(a, pr) for n, (a, pr) in preds.items()}
    ).T
    if args.metric not in table.columns:
        raise SystemExit(f"--metric must be one of {list(table.columns)}")

    # Bootstrap: the same resampled rows for both models, so only the model differs.
    rng = np.random.default_rng(args.seed)
    pm, pb = preds[args.main][1], preds[args.baseline][1]
    diffs = []
    for _ in range(args.bootstrap):
        idx = rng.integers(0, len(actual), len(actual))
        diffs.append(
            common.regression_metrics(actual[idx], pm[idx])[args.metric]
            - common.regression_metrics(actual[idx], pb[idx])[args.metric]
        )
    lo, hi = (float(v) for v in np.percentile(diffs, [2.5, 97.5]))
    gap = float(
        table.loc[args.main_label, args.metric] - table.loc[args.baseline_label, args.metric]
    )
    better = (lambda d: d > 0) if args.metric in HIGHER_IS_BETTER else (lambda d: d < 0)
    if lo <= 0 <= hi:
        verdict = "not distinguishable: the interval includes zero, so prefer the simpler baseline"
    elif better(lo) and better(hi):
        verdict = f"{args.main_label} is reliably better"
    else:
        verdict = f"{args.baseline_label} is reliably better"

    # Charts: the trivial model in grey, the baseline and the main model in the first two
    # categorical colours, the same in every chart.
    colors = {
        "dummy": style.PALETTE["text"]["light_background"]["muted"],
        args.baseline: style.PALETTE["charts"]["categorical"][0],
        args.main: style.PALETTE["charts"]["categorical"][1],
    }
    direction = "higher" if args.metric in HIGHER_IS_BETTER else "lower"
    fig, ax = style.figure(height=360)
    bars = ax.barh(table.index, table[args.metric], color=[colors[n] for n in labels])
    ax.bar_label(bars, labels=[f"{float(f'{v:.3g}'):,g}" for v in table[args.metric]], padding=4)
    ax.margins(x=0.12)  # room for the values at the end of the bars
    if table[args.metric].min() >= 0:
        ax.set_xlim(left=0)
    ax.set_xlabel(
        args.metric if args.metric in HIGHER_IS_BETTER else f"{args.metric}, in units of {target}"
    )
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    style.title(fig, f"Test {args.metric} by model ({direction} is better)")
    style.save(fig, out, "model_comparison")

    names = (args.baseline, args.main)
    muted = style.PALETTE["text"]["light_background"]["muted"]
    fig, axes = style.figure(width=1000, height=500, ncols=2, sharex=True, sharey=True)
    lims = [float(actual.min()), float(actual.max())]
    for ax, name in zip(axes, names, strict=True):
        ax.scatter(actual, preds[name][1], s=14, alpha=0.5, color=colors[name], linewidths=0)
        # Labelled once, on the last panel, which carries the legend.
        ax.plot(
            lims,
            lims,
            color=muted,
            linestyle="--",
            linewidth=1,
            label="perfect prediction" if ax is axes[-1] else None,
        )
        ax.set_xlabel(f"actual {target}")
        ax.set_title(labels[name])
    axes[0].set_ylabel(f"predicted {target}")
    style.legend(axes[-1])
    style.title(fig, "Predicted vs. actual (test rows): the closer to the dashed line, the better")
    style.save(fig, out, "predicted_vs_actual")

    errors = {n: np.abs(preds[n][1] - actual) for n in names}
    top = float(np.percentile(np.concatenate(list(errors.values())), 99)) or 1.0
    fig, ax = style.figure()
    for name, err in errors.items():
        ax.hist(
            err,
            bins=np.linspace(0, top, 41),
            color=colors[name],
            alpha=0.6,
            label=f"{labels[name]} (median {float(f'{np.median(err):.3g}'):,g})",
        )
    ax.set_xlim(0, top)
    ax.set_xlabel(f"absolute error, in units of {target}")
    ax.set_ylabel("number of test rows")
    style.legend(ax)
    style.title(fig, "How far off the predictions are (test rows)")
    style.save(fig, out, "error_distribution")

    # Plain-language error bands: share of test rows predicted within a given distance.
    scale = float(np.median(np.abs(actual - np.median(actual)))) or float(np.std(actual)) or 1.0
    bands = pd.DataFrame(
        {
            label: {
                f"within {k:g}× the typical spread ({k * scale:.3g})": float(
                    (err <= k * scale).mean()
                )
                for k in (0.25, 0.5, 1.0)
            }
            for label, err in errors.items()
        }
    )
    bands.index.name = "share of test rows predicted"

    common.write_json(
        out / "comparison.json",
        {
            "metric": args.metric,
            "test_rows": len(actual),
            "metrics": table.to_dict(orient="index"),
            "main_minus_baseline": gap,
            "ci95": [lo, hi],
            "bootstrap_samples": len(diffs),
            "verdict": verdict,
            "target_summary": {
                "mean": float(actual.mean()),
                "median": float(np.median(actual)),
                "std": float(actual.std()),
                "typical_spread_MAD": scale,
            },
        },
    )
    (out / "comparison.md").write_text(
        f"Test rows: {len(actual)}. Target `{target}`: median {np.median(actual):.4g}, "
        f"typical spread around it (median absolute deviation) {scale:.4g}.\n\n"
        f"{common.markdown_table(table)}\n\n"
        f"{args.main_label} − {args.baseline_label} {args.metric}: {gap:+.4g} "
        f"(95% CI {lo:+.4g} to {hi:+.4g}, {len(diffs)} bootstrap samples) — {verdict}\n\n"
        f"{common.markdown_table(bands, 3)}\n"
    )
    common.record_run(out, Path(__file__), "compare")

    print((out / "comparison.md").read_text())
    print(f"outputs in {out}")


if __name__ == "__main__":
    main()
