"""Baselines for forecasting: seasonal naive (the trivial floor) and one ARIMA per series.

- Seasonal naive repeats the last observed seasonal cycle: next January = last January.
  A model that doesn't beat it has learned nothing useful. With less than one cycle of
  history it repeats the last value.
- ARIMA is the classical statistical reference: AutoARIMA (statsforecast) picks the
  orders for each series separately on its own history, including a seasonal part of
  --season-length periods when the data supports it, and gives 80% / 95% prediction
  intervals. One model per series, never shared across series. The features declared
  in the series are its exogenous regressors (regression with ARIMA errors), the same
  columns CatBoost sees: a known feature's value at each period, a lagged one's past
  values. Seasonal naive uses none.

For each backtest window, both are fitted on the periods before it and forecast it;
then both are refitted on everything and forecast --horizon periods past the end. The
ARIMA fits all run in one parallel batch, on every core by default (--jobs). The order
search uses a conditional-sum-of-squares approximation and refits only the chosen model
by maximum likelihood: on 731 days it was about 3.5x faster than the exact search
(--exact-arima), with the same backtest error.

Writes to --out: seasonal_naive_backtest.csv / _forecast.csv, arima_backtest.csv /
_forecast.csv, arima_orders.csv (the order chosen per series on the full history),
baseline_metrics.json.

    python forecast_baseline.py --out <run folder>
"""

import time
from pathlib import Path

import common
import numpy as np
import pandas as pd

LEVELS = (80, 95)


def seasonal_naive(y: np.ndarray, h: int, season: int) -> np.ndarray:
    if len(y) >= season > 1:
        cycle = y[-season:]
        return np.array([cycle[k % season] for k in range(h)])
    return np.repeat(y[-1], h)


def arima(
    panel: common.Panel, fits: list[tuple[int, int]], jobs: int
) -> tuple[dict[tuple[int, int], dict[str, np.ndarray]], pd.DataFrame]:
    """AutoARIMA for every (series, until) in `fits`, each on the periods before `until`:
    forecasts, and the orders chosen.

    All the fits go to statsforecast at once, as separate series, because its n_jobs
    parallelizes across series: one call per backtest window would leave a single series
    fitted on one core, window after window.
    """
    from statsforecast import StatsForecast
    from statsforecast.arima import arima_string
    from statsforecast.models import AutoARIMA, SeasonalNaive

    h = panel.args.horizon
    # Lagged features have no value in their first periods: those are left out of the fit.
    first = {i: panel.start[i] + panel.max_feature_lag for i, _ in fits}
    train = pd.concat(
        [
            pd.DataFrame(
                {
                    "unique_id": f"{i}@{until}",
                    "ds": np.arange(first[i], until),
                    "y": panel.values[i, first[i] : until],
                    **panel.feature_history(i, first[i], until),
                }
            )
            for i, until in fits
        ],
        ignore_index=True,
    )
    future = (
        pd.concat(
            [
                pd.DataFrame(
                    {
                        "unique_id": f"{i}@{until}",
                        "ds": np.arange(until, until + h),
                        **panel.feature_history(i, until, until + h),
                    }
                )
                for i, until in fits
            ],
            ignore_index=True,
        )
        if panel.feature_spec
        else None
    )
    sf = StatsForecast(
        models=[
            AutoARIMA(
                season_length=panel.season,
                approximation=not panel.args.exact_arima,
            )
        ],
        freq=1,
        n_jobs=jobs,
        fallback_model=SeasonalNaive(season_length=panel.season),
    )
    sf.fit(train)
    fc = sf.predict(h=h, X_df=future, level=list(LEVELS))

    def key(uid: str) -> tuple[int, int]:
        i, until = uid.split("@")
        return int(i), int(until)

    out = {}
    for uid, group in fc.groupby("unique_id"):
        g = group.sort_values("ds")
        out[key(str(uid))] = {
            "pred": g["AutoARIMA"].to_numpy(),
            **{
                f"{side}{lv}": g[f"AutoARIMA-{side}-{lv}"].to_numpy()
                for lv in LEVELS
                for side in ("lo", "hi")
            },
        }
    orders = []
    for k, uid in enumerate(sf.uids if hasattr(sf, "uids") else train["unique_id"].unique()):
        try:
            order = arima_string(sf.fitted_[k, 0].model_)
        except Exception:  # the fallback model has no ARIMA order
            order = "fallback: seasonal naive"
        i, until = key(str(uid))
        orders.append({"id": panel.ids[i], "until": until, "arima": " ".join(order.split())})
    return out, pd.DataFrame(orders)


def rename_intervals(result: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {k: v for k, v in result.items() if k != "pred"}


def main() -> None:
    parser = common.base_parser(__doc__.splitlines()[0])
    parser.add_argument(
        "--jobs",
        type=int,
        default=-1,
        help="parallel ARIMA fits (default -1: all cores; 1 to run on one core)",
    )
    parser.add_argument(
        "--exact-arima",
        action="store_true",
        help="search the ARIMA orders by full maximum likelihood; slower. By default the "
        "search uses a conditional-sum-of-squares approximation, and only the chosen "
        "model is refitted by maximum likelihood",
    )
    args = parser.parse_args()

    panel = common.load_panel(args)
    h, n, T = args.horizon, len(panel.ids), len(panel.periods)
    # Every backtest window's fits, and the final one on the full history, in one call.
    windows = [
        (w, origin, [i for i in range(n) if panel.usable(i, origin)])
        for w, origin in enumerate(panel.origins)
    ]
    fits = [(i, origin) for _, origin, rows in windows for i in rows] + [(i, T) for i in range(n)]
    started = time.time()
    print(f"fitting {len(fits)} ARIMAs ({len(panel.origins)} windows + the full history)...")
    fitted, orders = arima(panel, fits, args.jobs)
    print(f"done in {time.time() - started:.0f}s")

    naive_bt, arima_bt = [], []
    for w, origin, rows in windows:
        for i in rows:
            naive_bt.append(
                common.backtest_frame(
                    panel, w, i, seasonal_naive(panel.history(i, origin), h, panel.season)
                )
            )
            f = fitted[(i, origin)]
            arima_bt.append(common.backtest_frame(panel, w, i, f["pred"], rename_intervals(f)))

    naive_fc = [
        common.future_frame(panel, i, seasonal_naive(panel.history(i, T), h, panel.season))
        for i in range(n)
    ]
    arima_fc = [
        common.future_frame(panel, i, fitted[(i, T)]["pred"], rename_intervals(fitted[(i, T)]))
        for i in range(n)
    ]
    # The orders chosen on the full history, as before; the backtest refits are not reported.
    orders = orders[orders["until"] == T].drop(columns="until")
    common.save_forecasts(panel.out, "seasonal_naive", naive_bt, naive_fc)
    common.save_forecasts(panel.out, "arima", arima_bt, arima_fc)
    orders.to_csv(panel.out / "arima_orders.csv", index=False)

    results = {
        "seasonal_naive": common.forecast_metrics(pd.concat(naive_bt)),
        "arima": common.forecast_metrics(pd.concat(arima_bt)),
    }
    common.write_json(
        panel.out / "baseline_metrics.json",
        {
            "season_length": panel.season,
            "arima_fallbacks": int(orders["arima"].str.startswith("fallback").sum()),
            "most_common_orders": orders["arima"].value_counts().head(5).to_dict(),
            "metrics": results,
        },
    )
    common.record_run(panel.out, Path(__file__), "baseline")

    print(pd.DataFrame(results).T.round(3).to_string())
    print(f"ARIMA orders (full history): {orders['arima'].value_counts().head(3).to_dict()}")
    print(f"outputs in {panel.out}")


if __name__ == "__main__":
    main()
