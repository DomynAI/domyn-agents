"""Set up the backtest once, for every forecasting model that follows.

The first step after preparing the series. It reads series.csv, decides which series
take part and which periods are held out, and writes it all to --out/split.json. The
model scripts then take only --out, so every model forecasts exactly the same
periods of exactly the same series.

The backtest: consecutive windows of --horizon periods each, the last one ending at the
last period. For each window, a model may use only the periods before it. A series
takes part in a window when it has at least max(2 × horizon, 8) periods before it;
series that stop before the last period are left out.

How many windows. The available ones are those that fit after two seasonal cycles of
training; --windows-share (default 0.10) of them are used, at least 3 when there are 3,
and --max-fits (default 2000) caps windows × series. So the backtest grows with the
history: 731 days forecast 1 day ahead get 72 windows, 7 years of monthly data forecast
6 months ahead get 3. --windows N sets an exact number instead. The script prints the
choice and the forecast points it gives, and split.json records both.

Choose --horizon from the decision the forecast serves.

    python forecast_split.py --series <analysis run folder>/files/series.csv --horizon 6 --out <run folder>
"""

import json
from pathlib import Path

import common


def main() -> None:
    args = common.split_parser(__doc__.splitlines()[0]).parse_args()
    panel = common.Panel(args)
    common.record_run(panel.out, Path(__file__), "split")

    meta = json.loads((panel.out / common.META_FILE).read_text())
    print(
        f"{meta['series']} series of {meta['target']}, {meta['periods']} {meta['freq']} periods "
        f"({meta['first_period']} to {meta['last_period']}), season length {meta['season_length']}"
    )
    for w in meta["windows"]:
        print(
            f"window {w['window']}: forecast {w['first_forecast_period']} to "
            f"{w['last_forecast_period']} from data until {w['train_until']}, "
            f"{w['series_included']} series"
        )
    print(f"future forecast: {meta['future_periods'][0]} to {meta['future_periods'][-1]}")
    print(f"split written to {panel.out / common.META_FILE}")
    # Series prepared for an analysis carry no features, and nothing downstream would say
    # the source had some: a forecast would silently run without them.
    unused = panel.meta.get("unused_columns")
    if unused and not panel.feature_spec:
        print(
            f"WARNING: the series carry no features, but their source has other numeric "
            f"columns: {unused[:10]}. If any is worth using (a holiday, a promotion, a price), "
            f"declare it in a features spec and add it with `prepare_series.py --from-series "
            f"{panel.series_path} --features <spec> --out <new run folder>`, then split "
            "there: this folder's split is without them. Leave out columns derived from the "
            "target, and ids."
        )


if __name__ == "__main__":
    main()
