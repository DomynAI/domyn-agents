"""Main model for classification: CatBoost, with early stopping on the validation rows.

No preparation beyond common.py: CatBoost takes categorical features as they are and
handles missing numeric values itself. With --tune, depth and learning rate are chosen on
the validation rows (never on test) before the final fit.

Writes to --out: catboost_model.cbm, catboost_test_predictions.csv, catboost_metrics.json,
catboost_feature_importance.csv / .png, catboost_learning_curve.png, and one
catboost_effect_<feature>.png per top feature (SHAP values).

    python classification_catboost.py --out <run folder>
"""

from pathlib import Path

import common
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool


def fit(args, ds: common.Dataset, depth: int, learning_rate: float) -> CatBoostClassifier:
    X_train, y_train = ds.subset("train")
    X_valid, y_valid = ds.subset("valid")
    model = CatBoostClassifier(
        iterations=args.iterations,
        learning_rate=learning_rate,
        depth=depth,
        loss_function="Logloss" if y_train.nunique() == 2 else "MultiClass",
        auto_class_weights=None if args.class_weight == "none" else "Balanced",
        early_stopping_rounds=args.early_stopping,
        random_seed=args.seed,
        verbose=0,
        allow_writing_files=False,  # otherwise catboost_info/ lands in the cwd
    )
    model.fit(
        Pool(X_train, y_train, cat_features=ds.cat_cols),
        eval_set=Pool(X_valid, y_valid, cat_features=ds.cat_cols),
        use_best_model=True,
    )
    return model


def effect_chart(
    ds: common.Dataset, feature: str, shap: np.ndarray, X: pd.DataFrame, classes: list[str]
) -> None:
    """Binary: one panel, for the positive class. Multiclass: one panel per class (up to 4)."""
    j = X.columns.get_loc(feature)
    if shap.ndim == 2:
        panels = [(f"push toward '{classes[1]}'", "", shap[:, j])]
    else:
        panels = [
            (f"push toward '{c}'", f"'{c}'", shap[:, k, j]) for k, c in enumerate(classes[:4])
        ]
    common.shap_effect_chart(ds.out, feature, X[feature], feature in ds.cat_cols, panels)


def main() -> None:
    parser = common.base_parser(__doc__.splitlines()[0])
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--early-stopping", type=int, default=200)
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    parser.add_argument(
        "--tune",
        action="store_true",
        help="pick depth in {4,6,8} and learning rate in {0.03,0.1} on the validation rows",
    )
    parser.add_argument(
        "--top-effects", type=int, default=3, help="features to draw an effect chart for"
    )
    args = parser.parse_args()

    ds = common.load_dataset(args, task="classification")
    X_valid, y_valid = ds.subset("valid")
    X_test, y_test = ds.subset("test")

    tuning = []
    if args.tune:
        for depth in (4, 6, 8):
            for lr in (0.03, 0.1):
                m = fit(args, ds, depth, lr)
                score = common.classification_metrics(
                    y_valid,
                    m.predict(X_valid).ravel(),
                    m.predict_proba(X_valid),
                    [str(c) for c in m.classes_],
                )
                tuning.append({"depth": depth, "learning_rate": lr, **score})
                print(f"depth={depth} lr={lr}: valid log_loss {score['log_loss']:.4f}")
        best = min(tuning, key=lambda t: t["log_loss"])
        args.depth, args.learning_rate = best["depth"], best["learning_rate"]

    model = fit(args, ds, args.depth, args.learning_rate)
    classes = [str(c) for c in model.classes_]
    results = {
        part: common.classification_metrics(
            y, model.predict(X).ravel(), model.predict_proba(X), classes
        )
        for part, X, y in (
            ("train", *ds.subset("train")),
            ("valid", X_valid, y_valid),
            ("test", X_test, y_test),
        )
    }
    common.save_predictions(
        ds.out,
        "catboost",
        ds.rows("test"),
        y_test,
        model.predict(X_test).ravel(),
        model.predict_proba(X_test),
        classes,
    )
    model.save_model(ds.out / "catboost_model.cbm")

    importance = pd.DataFrame(
        {"feature": ds.X.columns, "importance": model.get_feature_importance()}
    )
    importance = importance.sort_values("importance", ascending=False)
    importance.to_csv(ds.out / "catboost_feature_importance.csv", index=False)
    common.importance_chart(ds.out, importance)

    metric = model.get_params()["loss_function"]
    curve = model.get_evals_result()["validation"][metric]
    common.learning_curve_chart(
        ds.out,
        curve,
        model.get_best_iteration(),
        "CatBoost: validation loss by training round",
        f"{metric} (lower is better)",
    )

    shap = model.get_feature_importance(Pool(X_test, cat_features=ds.cat_cols), type="ShapValues")
    shap = shap[:, :-1] if shap.ndim == 2 else shap[:, :, :-1]  # drop the expected-value column
    for feature in importance["feature"].head(args.top_effects):
        effect_chart(ds, feature, shap, X_test, classes)

    common.write_json(
        ds.out / "catboost_metrics.json",
        {
            "model": "CatBoost",
            "classes": classes,
            "params": {
                "depth": args.depth,
                "learning_rate": args.learning_rate,
                "class_weight": args.class_weight,
                "iterations_max": args.iterations,
            },
            "best_iteration": model.get_best_iteration(),
            "tuning": tuning,
            "metrics": results,
        },
    )
    common.record_run(ds.out, Path(__file__), "catboost")

    print(
        f"best iteration {model.get_best_iteration()}  depth {args.depth}  lr {args.learning_rate}"
    )
    print(pd.DataFrame(results).T.round(4).to_string())
    print(f"outputs in {ds.out}")


if __name__ == "__main__":
    main()
