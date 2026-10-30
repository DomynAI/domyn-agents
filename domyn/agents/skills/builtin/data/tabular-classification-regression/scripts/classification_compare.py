"""Compare classification models on the test rows: metrics table, bootstrap confidence
interval for main − baseline, and the comparison charts.

Run after classification_baseline.py and classification_catboost.py with the same --out.
Reads <name>_test_predictions.csv for every model.

Writes to --out: comparison.json, comparison.md (tables ready for the report),
model_comparison.png, confusion_matrices.png, roc_curves.png (binary only).

    python classification_compare.py --out <run folder>
"""

import argparse
import json
from pathlib import Path

import common
import numpy as np
import pandas as pd
import seaborn as sns
import seaborn_style as style
from sklearn import metrics as m

LOWER_IS_BETTER = {"log_loss"}


def load(out: Path, name: str, classes: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = out / f"{name}_test_predictions.csv"
    if not path.exists():
        script = (
            "classification_baseline.py"
            if name in ("dummy", "baseline")
            else f"classification_{name}.py"
        )
        raise SystemExit(f"{path} not found: run {script} first, with the same --out.")
    frame = pd.read_csv(path, dtype={"actual": str, "pred": str}).sort_values("row")
    return (
        frame["actual"].to_numpy(),
        frame["pred"].to_numpy(),
        frame[[f"proba_{c}" for c in classes]].to_numpy(),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--main", default="catboost", help="name of the main model's prediction file")
    p.add_argument("--main-label", default="CatBoost")
    p.add_argument("--baseline", default="baseline")
    p.add_argument("--baseline-label", default="Logistic regression")
    p.add_argument("--metric", default="ROC_AUC", help="the metric that decides the comparison")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out = common.data_utils.output_dir(args.out)
    meta = json.loads((out / common.META_FILE).read_text())
    if meta["task"] != "classification":
        raise SystemExit("This folder holds a regression run: use regression_compare.py.")
    classes = [str(c) for c in meta["classes"]]

    labels = {
        "dummy": "Always the most frequent class",
        args.baseline: args.baseline_label,
        args.main: args.main_label,
    }
    preds = {name: load(out, name, classes) for name in labels}
    actual = preds[args.main][0]
    table = pd.DataFrame(
        {
            labels[n]: common.classification_metrics(a, pr, pb, classes)
            for n, (a, pr, pb) in preds.items()
        }
    ).T
    if args.metric not in table.columns:
        raise SystemExit(f"--metric must be one of {list(table.columns)}")

    # Bootstrap: the same resampled rows for both models, so only the model differs.
    rng = np.random.default_rng(args.seed)
    _, pm, qm = preds[args.main]
    _, pb, qb = preds[args.baseline]
    diffs = []
    for _ in range(args.bootstrap):
        idx = rng.integers(0, len(actual), len(actual))
        if len(np.unique(actual[idx])) < len(classes):
            continue  # a resample without some class can't be scored
        sm = common.classification_metrics(actual[idx], pm[idx], qm[idx], classes)[args.metric]
        sb = common.classification_metrics(actual[idx], pb[idx], qb[idx], classes)[args.metric]
        diffs.append(sm - sb)
    lo, hi = (float(v) for v in np.percentile(diffs, [2.5, 97.5]))
    gap = float(
        table.loc[args.main_label, args.metric] - table.loc[args.baseline_label, args.metric]
    )
    better = (lambda d: d < 0) if args.metric in LOWER_IS_BETTER else (lambda d: d > 0)
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
    fig, ax = style.figure(height=360)
    bars = ax.barh(table.index, table[args.metric], color=[colors[n] for n in labels])
    ax.bar_label(bars, labels=[f"{v:.3f}" for v in table[args.metric]], padding=4)
    ax.margins(x=0.12)  # room for the values at the end of the bars
    if table[args.metric].min() >= 0:
        ax.set_xlim(left=0)
    ax.set_xlabel(args.metric)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    style.title(
        fig,
        f"Test {args.metric} by model ({'lower' if args.metric in LOWER_IS_BETTER else 'higher'} is better)",
    )
    style.save(fig, out, "model_comparison")

    names = (args.baseline, args.main)
    matrices = {n: m.confusion_matrix(actual, preds[n][1], labels=classes) for n in names}
    top = max(int(cm.max()) for cm in matrices.values())
    fig, axes = style.figure(width=1000, height=500, ncols=2)
    for ax, name in zip(axes, names, strict=True):
        # One colour scale for both, so the same shade is the same count.
        sns.heatmap(
            matrices[name],
            annot=True,
            fmt=",d",
            cmap=style.colormap("sequential"),
            vmin=0,
            vmax=top,
            cbar=False,
            square=True,
            linewidths=1,
            linecolor=style.PALETTE["surfaces"]["card_light"],
            xticklabels=classes,
            yticklabels=classes,
            ax=ax,
        )
        ax.tick_params(axis="y", labelrotation=0)
        ax.set_xlabel("predicted class")
        ax.set_ylabel("actual class")
        ax.set_title(labels[name])
        ax.grid(False)
    fig.colorbar(axes[1].collections[0], ax=axes, label="rows", shrink=0.8)
    style.title(fig, "Confusion matrices (test rows): counts of actual vs. predicted class")
    style.save(fig, out, "confusion_matrices")

    if len(classes) == 2:
        fig, ax = style.figure(width=760, height=640)
        for name in names:
            fpr, tpr, _ = m.roc_curve(actual == classes[1], preds[name][2][:, 1])
            ax.plot(
                fpr, tpr, color=colors[name], label=f"{labels[name]} (AUC = {m.auc(fpr, tpr):.2f})"
            )
        ax.plot(
            [0, 1],
            [0, 1],
            color=style.PALETTE["text"]["light_background"]["muted"],
            linestyle="--",
            linewidth=1,
            label="random guess",
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel(f"share of actual '{classes[0]}' wrongly flagged (false positive rate)")
        ax.set_ylabel(f"share of actual '{classes[1]}' caught (true positive rate)")
        style.legend(ax, ncols=1)
        style.title(fig, "ROC curve (test rows)")
        style.save(fig, out, "roc_curves")

    report = {
        "metric": args.metric,
        "test_rows": len(actual),
        "metrics": table.to_dict(orient="index"),
        "main_minus_baseline": gap,
        "ci95": [lo, hi],
        "bootstrap_samples": len(diffs),
        "verdict": verdict,
    }
    common.write_json(out / "comparison.json", report)
    per_class = {
        labels[n]: m.classification_report(
            actual, preds[n][1], labels=classes, output_dict=True, zero_division=0
        )
        for n in (args.baseline, args.main)
    }
    recall = pd.DataFrame({k: {c: v[c]["recall"] for c in classes} for k, v in per_class.items()})
    recall.index.name = "share of each actual class predicted correctly (recall)"
    counts = pd.Series(meta["class_counts"]).reindex(classes).rename("rows in data")
    (out / "comparison.md").write_text(
        f"Test rows: {len(actual)}\n\n{common.markdown_table(table)}\n\n"
        f"{args.main_label} − {args.baseline_label} {args.metric}: {gap:+.4f} "
        f"(95% CI {lo:+.4f} to {hi:+.4f}, {len(diffs)} bootstrap samples) — {verdict}\n\n"
        f"{common.markdown_table(recall.join(counts), 3)}\n"
    )
    common.record_run(out, Path(__file__), "compare")

    print((out / "comparison.md").read_text())
    print(f"outputs in {out}")


if __name__ == "__main__":
    main()
