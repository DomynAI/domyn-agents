"""Main model for regression (numeric prediction): CatBoost, with early stopping on the
validation rows.

No preparation beyond common.py: CatBoost takes categorical features as they are and
handles missing numeric values itself. With --tune, depth and learning rate are chosen on
the validation rows (never on test) before the final fit.

--loss RMSE (default) fits the mean; --loss MAE fits the median, and is less pulled by a
few very large values — use it when the target has heavy outliers and MAE is the metric.

Writes to --out: catboost_model.cbm, catboost_test_predictions.csv, catboost_metrics.json,
catboost_feature_importance.csv / .png, catboost_learning_curve.png, and one
catboost_effect_<feature>.png per top feature (SHAP values).

    python regression_catboost.py --out <run folder>
"""

from pathlib import Path

import common
import pandas as pd
from catboost import CatBoostRegressor, Pool


def fit(args, ds: common.Dataset, depth: int, learning_rate: float) -> CatBoostRegressor:
    X_train, y_train = ds.subset("train")
    X_valid, y_valid = ds.subset("valid")
    model = CatBoostRegressor(
        iterations=args.iterations,
        learning_rate=learning_rate,
        depth=depth,
        loss_function=args.loss,
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


def main() -> None:
    parser = common.base_parser(__doc__.splitlines()[0])
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--early-stopping", type=int, default=200)
    parser.add_argument("--loss", choices=["RMSE", "MAE"], default="RMSE")
    parser.add_argument(
        "--tune",
        action="store_true",
        help="pick depth in {4,6,8} and learning rate in {0.03,0.1} on the validation rows",
    )
    parser.add_argument(
        "--top-effects", type=int, default=3, help="features to draw an effect chart for"
    )
    args = parser.parse_args()

    ds = common.load_dataset(args, task="regression")
    X_valid, y_valid = ds.subset("valid")
    X_test, y_test = ds.subset("test")

    tuning = []
    if args.tune:
        for depth in (4, 6, 8):
            for lr in (0.03, 0.1):
                m = fit(args, ds, depth, lr)
                score = common.regression_metrics(y_valid.to_numpy(), m.predict(X_valid))
                tuning.append({"depth": depth, "learning_rate": lr, **score})
                print(f"depth={depth} lr={lr}: valid {args.loss} {score[args.loss]:.4f}")
        best = min(tuning, key=lambda t: t[args.loss])
        args.depth, args.learning_rate = best["depth"], best["learning_rate"]

    model = fit(args, ds, args.depth, args.learning_rate)
    results = {
        part: common.regression_metrics(y.to_numpy(), model.predict(X))
        for part, X, y in (
            ("train", *ds.subset("train")),
            ("valid", X_valid, y_valid),
            ("test", X_test, y_test),
        )
    }
    common.save_predictions(ds.out, "catboost", ds.rows("test"), y_test, model.predict(X_test))
    model.save_model(ds.out / "catboost_model.cbm")

    importance = pd.DataFrame(
        {"feature": ds.X.columns, "importance": model.get_feature_importance()}
    )
    importance = importance.sort_values("importance", ascending=False)
    importance.to_csv(ds.out / "catboost_feature_importance.csv", index=False)
    common.importance_chart(ds.out, importance)

    curve = model.get_evals_result()["validation"][args.loss]
    common.learning_curve_chart(
        ds.out,
        curve,
        model.get_best_iteration(),
        "CatBoost: validation error by training round",
        f"{args.loss} in units of {args.target} (lower is better)",
    )

    shap = model.get_feature_importance(Pool(X_test, cat_features=ds.cat_cols), type="ShapValues")[
        :, :-1
    ]
    for feature in importance["feature"].head(args.top_effects):
        j = X_test.columns.get_loc(feature)
        common.shap_effect_chart(
            ds.out,
            feature,
            X_test[feature],
            feature in ds.cat_cols,
            [(f"push on predicted {args.target}", "", shap[:, j])],
        )

    common.write_json(
        ds.out / "catboost_metrics.json",
        {
            "model": "CatBoost",
            "params": {
                "depth": args.depth,
                "learning_rate": args.learning_rate,
                "loss": args.loss,
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
