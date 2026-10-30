"""Main model for forecasting: one CatBoost model across all series, on lag, rolling and
calendar features, forecasting recursively.

How it forecasts. Each training row is one series at one period; the features are what
was known just before it: the values 1, 2, 3, ... and one season ago (lags), recent
averages (rolling means), the calendar (month, weekday, ...), the features declared in
the series (a known feature's value at that period, a lagged one's past values: see
common.Panel.feature_columns) and, with several series, the series id and its size. To forecast k periods ahead, it predicts the next period,
feeds that prediction back as the newest lag, and repeats k times.

What keeps the backtest honest. For each window, everything from the window on is
hidden. The periods just before it are validation: max(horizon, one season, 8) of
them, so a one-period horizon is not validated on a single value. Training stops when
the validation error stops improving; the model is then refitted on all periods before
the window with that many rounds, and forecasts it.

Intervals. The validation span is also forecast recursively from every period in it
that leaves a full horizon (up to 30 starts), exactly as the window will be. The 80th /
95th percentile of those errors set the 80% / 95% prediction intervals, per step ahead
when there are at least 20 errors per step, otherwise pooled over steps.

Scale. Each series is divided by its average absolute value, so one model can learn
from series of very different size. Trees cannot forecast beyond the values they were
trained on: with a steady trend, pass --difference to model the change from one
period to the next instead of the level.

Writes to --out: catboost_backtest.csv / _forecast.csv, catboost_model.cbm (fitted on
the full history), catboost_feature_importance.csv / .png, catboost_metrics.json.

    python forecast_catboost.py --out <run folder>
"""

import math
from pathlib import Path

import common
import numpy as np
import pandas as pd
import seaborn_style as style
from catboost import CatBoostRegressor, Pool

LEVELS = (80, 95)
CALENDAR = {
    "h": ("hour", "dayofweek"),
    "min": ("minute", "hour"),
    "D": ("dayofweek", "month"),
    "W": ("week",),
    "M": ("month",),
    "Q": ("quarter",),
    "Y": (),
}


class Features:
    """Builds the feature rows for a set of series at one period."""

    def __init__(self, panel: common.Panel, lags: list[int], windows: list[int]) -> None:
        self.panel, self.lags, self.windows = panel, lags, windows
        self.calendar = next((v for k, v in CALENDAR.items() if panel.freq.startswith(k)), ())
        self.multi = len(panel.ids) > 1
        self.cat_cols = ["series_id"] if self.multi else []

    def at(self, Z: np.ndarray, rows: np.ndarray, t: int, scale: np.ndarray) -> pd.DataFrame:
        cols: dict[str, np.ndarray] = {}
        for k in self.lags:
            cols[f"lag_{k}"] = Z[rows, t - k] if t - k >= 0 else np.full(len(rows), np.nan)
        for w in self.windows:
            span = Z[rows, max(0, t - w) : t]
            with np.errstate(all="ignore"):
                cols[f"mean_last_{w}"] = (
                    np.nanmean(span, axis=1) if span.shape[1] else np.full(len(rows), np.nan)
                )
        period = (
            self.panel.periods[t]
            if t < len(self.panel.periods)
            else self.panel.periods[-1] + (t - len(self.panel.periods) + 1)
        )
        for name in self.calendar:
            cols[name] = np.full(len(rows), getattr(period, name))
        cols.update(self.panel.feature_columns(rows, t))
        frame = pd.DataFrame(cols)
        if self.multi:
            frame.insert(0, "series_id", self.panel.ids[rows])
            frame["log_size"] = np.log1p(np.abs(scale[rows]))
        return frame


def training_rows(
    feats: Features,
    Z: np.ndarray,
    rows: np.ndarray,
    t0: int,
    t1: int,
    scale: np.ndarray,
    difference: bool,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Features and targets for every series in `rows` at periods t0 <= t < t1 with a known lag 1."""
    Xs, ys = [], []
    for t in range(max(t0, 1), t1):
        ok = rows[(~np.isnan(Z[rows, t])) & (~np.isnan(Z[rows, t - 1]))]
        if not len(ok):
            continue
        Xs.append(feats.at(Z, ok, t, scale))
        ys.append(Z[ok, t] - Z[ok, t - 1] if difference else Z[ok, t])
    return pd.concat(Xs, ignore_index=True), np.concatenate(ys)


def recursive(
    model: CatBoostRegressor,
    feats: Features,
    Z: np.ndarray,
    rows: np.ndarray,
    origin: int,
    h: int,
    scale: np.ndarray,
    difference: bool,
) -> np.ndarray:
    """Forecast h periods from `origin`, feeding each prediction back. Returns (rows, h), scaled."""
    Z = np.concatenate([Z[:, :origin], np.full((Z.shape[0], h), np.nan)], axis=1)
    for k in range(h):
        t = origin + k
        pred = model.predict(Pool(feats.at(Z, rows, t, scale), cat_features=feats.cat_cols))
        Z[rows, t] = Z[rows, t - 1] + pred if difference else pred
    return Z[rows, origin : origin + h]


def interval_widths(errors: np.ndarray, h: int) -> dict[int, np.ndarray]:
    """Half-widths per step (scaled units) from validation errors, shape (forecasts, h)."""
    per_step = errors.shape[0] >= 20
    out = {}
    for lv in LEVELS:
        if per_step:
            out[lv] = np.nanpercentile(np.abs(errors), lv, axis=0)
        else:
            out[lv] = np.repeat(np.nanpercentile(np.abs(errors), lv), h)
    return out


def main() -> None:
    parser = common.base_parser(__doc__.splitlines()[0])
    parser.add_argument(
        "--lags", type=int, nargs="*", default=None, help="default: 1, 2, 3, season, 2×season"
    )
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--early-stopping", type=int, default=100)
    parser.add_argument("--loss", choices=["RMSE", "MAE"], default="RMSE")
    parser.add_argument(
        "--difference",
        action="store_true",
        help="model the change per period (for trending series)",
    )
    args = parser.parse_args()

    panel = common.load_panel(args)
    h, n, s = args.horizon, len(panel.ids), panel.season
    room = panel.origins[0] // 2
    lags = sorted({k for k in (args.lags or [1, 2, 3, s, 2 * s]) if 1 <= k <= max(room, 1)})
    windows = sorted({w for w in (3, s) if 1 < w <= max(room, 1)})
    feats = Features(panel, lags, windows)
    print(f"lags {lags}, rolling means over {windows}, calendar {list(feats.calendar)}")

    def params(iterations: int, early: bool) -> dict:
        return {
            "iterations": iterations,
            "learning_rate": args.learning_rate,
            "depth": args.depth,
            "loss_function": args.loss,
            "random_seed": args.seed,
            "verbose": 0,
            "allow_writing_files": False,  # otherwise catboost_info/ lands in the cwd
            **({"early_stopping_rounds": args.early_stopping} if early else {}),
        }

    def run(
        origin: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray], CatBoostRegressor, int, np.ndarray]:
        """Validate, refit and forecast from `origin`; returns rows, preds, widths, model, rounds, scale."""
        rows = np.array([i for i in range(n) if panel.usable(i, origin)])
        valid_len = max(h, panel.season, 8)
        valid_len = max(h, min(valid_len, origin - max(h, 4)))  # leave something to train on
        fit_end = origin - valid_len
        scale = np.array([np.mean(np.abs(panel.history(i, fit_end))) or 1.0 for i in range(n)])
        Z = panel.values[:, :origin] / scale[:, None]
        X_tr, y_tr = training_rows(feats, Z, rows, 0, fit_end, scale, args.difference)
        X_va, y_va = training_rows(feats, Z, rows, fit_end, origin, scale, args.difference)
        m = CatBoostRegressor(**params(args.iterations, True))
        m.fit(
            Pool(X_tr, y_tr, cat_features=feats.cat_cols),
            eval_set=Pool(X_va, y_va, cat_features=feats.cat_cols),
            use_best_model=True,
        )
        rounds = m.get_best_iteration() + 1
        # Every start in the validation span that leaves a full horizon, at most 30.
        starts = list(range(fit_end, origin - h + 1))
        starts = starts[:: max(1, math.ceil(len(starts) / 30))]
        errors = np.concatenate(
            [
                recursive(m, feats, Z, rows, t0, h, scale, args.difference) - Z[rows, t0 : t0 + h]
                for t0 in starts
            ]
        )
        widths = interval_widths(errors, h)

        X_all, y_all = training_rows(feats, Z, rows, 0, origin, scale, args.difference)
        final = CatBoostRegressor(**params(rounds, False))
        final.fit(Pool(X_all, y_all, cat_features=feats.cat_cols))
        pred = recursive(final, feats, Z, rows, origin, h, scale, args.difference)
        return rows, pred, widths, final, rounds, scale

    def intervals(
        pred: np.ndarray, widths: dict[int, np.ndarray], sc: float
    ) -> dict[str, np.ndarray]:
        return {
            f"{side}{lv}": (pred + sign * widths[lv]) * sc
            for lv in LEVELS
            for side, sign in (("lo", -1), ("hi", 1))
        }

    backtest, rounds_used = [], []
    for w, origin in enumerate(panel.origins):
        rows, pred, widths, _, rounds, scale = run(origin)
        rounds_used.append(rounds)
        for r, i in enumerate(rows):
            backtest.append(
                common.backtest_frame(
                    panel, w, i, pred[r] * scale[i], intervals(pred[r], widths, scale[i])
                )
            )
        print(f"window {w + 1}: {len(rows)} series, {rounds} rounds")

    rows, pred, widths, model, rounds, scale = run(len(panel.periods))
    future = [
        common.future_frame(panel, i, pred[r] * scale[i], intervals(pred[r], widths, scale[i]))
        for r, i in enumerate(rows)
    ]
    common.save_forecasts(panel.out, "catboost", backtest, future)
    model.save_model(panel.out / "catboost_model.cbm")

    names = model.feature_names_
    importance = pd.DataFrame({"feature": names, "importance": model.get_feature_importance()})
    importance = importance.sort_values("importance", ascending=False)
    importance.to_csv(panel.out / "catboost_feature_importance.csv", index=False)
    top = importance.head(15).iloc[::-1]
    fig, ax = style.figure(height=max(360, 26 * len(top) + 140))
    ax.barh(top["feature"].astype(str), top["importance"])
    ax.set_xlabel("share of the model's decisions (sums to 100)")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    style.title(fig, "CatBoost: what the forecasts rely on")
    style.save(fig, panel.out, "catboost_feature_importance")

    metrics = common.forecast_metrics(pd.concat(backtest))
    common.write_json(
        panel.out / "catboost_metrics.json",
        {
            "model": "CatBoost, global across series, recursive",
            "params": {
                "depth": args.depth,
                "learning_rate": args.learning_rate,
                "loss": args.loss,
                "difference": args.difference,
                "iterations_max": args.iterations,
            },
            "features": {
                "lags": lags,
                "rolling_means": windows,
                "calendar": list(feats.calendar),
                "series_id": feats.multi,
                "declared": panel.feature_spec,
            },
            "rounds_per_window": rounds_used,
            "rounds_final": rounds,
            "intervals": "from rolling validation forecasts: per step with at least 20 errors "
            "per step, otherwise pooled over steps",
            "metrics": metrics,
        },
    )
    common.record_run(panel.out, Path(__file__), "catboost")

    print(pd.Series(metrics).round(3).to_string())
    print(f"outputs in {panel.out}")


if __name__ == "__main__":
    main()
