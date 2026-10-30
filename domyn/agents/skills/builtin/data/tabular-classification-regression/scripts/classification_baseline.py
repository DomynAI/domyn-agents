"""Baseline for classification: logistic regression, plus the trivial "always the most frequent
class" model.

Preparation (common.baseline_preprocessing), fitted on the training rows only:
- numeric features: missing values replaced by the median, then standardized (mean 0, std 1),
  so the coefficients are comparable across features;
- categorical features: one-hot encoded, with categories under --min-frequency rows grouped
  into one "infrequent" column, and categories unseen in training mapped to it.

Writes to --out: baseline.joblib (preparation + model, applies to raw feature rows),
baseline_test_predictions.csv, dummy_test_predictions.csv, baseline_coefficients.csv,
baseline_coefficients.png, baseline_metrics.json.

    python classification_baseline.py --out <run folder>
"""

from pathlib import Path

import common
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def main() -> None:
    parser = common.base_parser(__doc__.splitlines()[0])
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="inverse regularization strength (smaller = simpler model)",
    )
    parser.add_argument(
        "--class-weight",
        choices=["none", "balanced"],
        default="none",
        help="'balanced' reweights rare classes; try it when a class is under ~10%%",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=10,
        help="categories rarer than this are grouped as 'infrequent'",
    )
    args = parser.parse_args()

    ds = common.load_dataset(args, task="classification")
    X_train, y_train = ds.subset("train")
    X_valid, y_valid = ds.subset("valid")
    X_test, y_test = ds.subset("test")

    model = Pipeline(
        [
            ("prep", common.baseline_preprocessing(ds.num_cols, ds.cat_cols, args.min_frequency)),
            (
                "model",
                LogisticRegression(
                    C=args.C,
                    max_iter=5000,
                    class_weight=None if args.class_weight == "none" else "balanced",
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    dummy = DummyClassifier(strategy="prior").fit(X_train, y_train)
    classes = [str(c) for c in model.classes_]

    results = {}
    for name, est in (("baseline", model), ("dummy", dummy)):
        results[name] = {
            part: common.classification_metrics(y, est.predict(X), est.predict_proba(X), classes)
            for part, X, y in (("valid", X_valid, y_valid), ("test", X_test, y_test))
        }
        common.save_predictions(
            ds.out,
            name,
            ds.rows("test"),
            y_test,
            est.predict(X_test),
            est.predict_proba(X_test),
            classes,
        )

    # Coefficients: on the standardized scale for numeric features, per one-hot column otherwise.
    names = model.named_steps["prep"].get_feature_names_out()
    coef = model.named_steps["model"].coef_
    if coef.shape[0] == 1:
        table = pd.DataFrame({"feature": names, "coefficient": coef[0]})
        table["abs"] = table["coefficient"].abs()
        label = f"coefficient (pushes toward '{classes[1]}' when positive)"
    else:
        table = pd.DataFrame(coef.T, columns=[f"coef_{c}" for c in classes])
        table.insert(0, "feature", names)
        table["abs"] = np.abs(coef).max(axis=0)
        label = "largest |coefficient| across classes"
    table = table.sort_values("abs", ascending=False).drop(columns="abs")
    table.to_csv(ds.out / "baseline_coefficients.csv", index=False)

    top = table.head(15)
    values = (
        top["coefficient"] if "coefficient" in top else top.filter(like="coef_").abs().max(axis=1)
    )
    common.coefficient_chart(
        ds.out,
        "baseline_coefficients",
        top["feature"],
        values,
        "Logistic regression: strongest coefficients (top 15)",
        label,
    )

    joblib.dump(model, ds.out / "baseline.joblib")
    common.write_json(
        ds.out / "baseline_metrics.json",
        {
            "model": "logistic regression",
            "C": args.C,
            "class_weight": args.class_weight,
            "classes": classes,
            "metrics": results,
        },
    )
    common.record_run(ds.out, Path(__file__), "baseline")

    print(f"rows: {ds.part.value_counts().to_dict()}  classes: {classes}")
    print(pd.DataFrame({k: v["test"] for k, v in results.items()}).T.round(4).to_string())
    print(f"outputs in {ds.out}")


if __name__ == "__main__":
    main()
