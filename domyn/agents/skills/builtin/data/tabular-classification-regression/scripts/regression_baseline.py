"""Baseline for regression (numeric prediction): linear regression, plus the trivial
"always predict the training mean" model.

Preparation (common.baseline_preprocessing), fitted on the training rows only:
- numeric features: missing values replaced by the median, then standardized (mean 0, std 1),
  so each coefficient reads as "change in the target per one standard deviation of the feature";
- categorical features: one-hot encoded, with categories under --min-frequency rows grouped
  into one "infrequent" column, and categories unseen in training mapped to it.

With many one-hot columns, plain least squares can become unstable (huge coefficients that
cancel out): pass --alpha to use ridge regression instead, which keeps them small.

Writes to --out: baseline.joblib (preparation + model, applies to raw feature rows),
baseline_test_predictions.csv, dummy_test_predictions.csv, baseline_coefficients.csv,
baseline_coefficients.png, baseline_metrics.json.

    python regression_baseline.py --out <run folder>
"""

from pathlib import Path

import common
import joblib
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline


def main() -> None:
    parser = common.base_parser(__doc__.splitlines()[0])
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="ridge regularization strength; omit for plain linear regression",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=10,
        help="categories rarer than this are grouped as 'infrequent'",
    )
    args = parser.parse_args()

    ds = common.load_dataset(args, task="regression")
    X_train, y_train = ds.subset("train")
    X_valid, y_valid = ds.subset("valid")
    X_test, y_test = ds.subset("test")

    regressor = LinearRegression() if args.alpha is None else Ridge(alpha=args.alpha)
    model = Pipeline(
        [
            ("prep", common.baseline_preprocessing(ds.num_cols, ds.cat_cols, args.min_frequency)),
            ("model", regressor),
        ]
    )
    model.fit(X_train, y_train)
    dummy = DummyRegressor(strategy="mean").fit(X_train, y_train)

    results = {}
    for name, est in (("baseline", model), ("dummy", dummy)):
        results[name] = {
            part: common.regression_metrics(y.to_numpy(), est.predict(X))
            for part, X, y in (
                ("train", X_train, y_train),
                ("valid", X_valid, y_valid),
                ("test", X_test, y_test),
            )
        }
        common.save_predictions(ds.out, name, ds.rows("test"), y_test, est.predict(X_test))

    names = model.named_steps["prep"].get_feature_names_out()
    table = pd.DataFrame({"feature": names, "coefficient": model.named_steps["model"].coef_})
    table = table.reindex(table["coefficient"].abs().sort_values(ascending=False).index)
    table.to_csv(ds.out / "baseline_coefficients.csv", index=False)
    top = table.head(15)
    model_name = "Linear regression" if args.alpha is None else "Ridge regression"
    common.coefficient_chart(
        ds.out,
        "baseline_coefficients",
        top["feature"],
        top["coefficient"],
        f"{model_name}: strongest coefficients (top 15)",
        f"change in {args.target} per std. dev. (numeric) or vs. other categories (one-hot)",
    )

    joblib.dump(model, ds.out / "baseline.joblib")
    common.write_json(
        ds.out / "baseline_metrics.json",
        {
            "model": model_name.lower(),
            "alpha": args.alpha,
            "intercept": float(model.named_steps["model"].intercept_),
            "metrics": results,
        },
    )
    common.record_run(ds.out, Path(__file__), "baseline")

    print(f"rows: {ds.part.value_counts().to_dict()}  target mean (train): {y_train.mean():.4g}")
    print(pd.DataFrame({k: v["test"] for k, v in results.items()}).T.round(4).to_string())
    if results["baseline"]["train"]["R2"] - results["baseline"]["test"]["R2"] > 0.2:
        print("WARNING: train R2 is far above test R2 — the baseline overfits; try --alpha 1.0")
    print(f"outputs in {ds.out}")


if __name__ == "__main__":
    main()
