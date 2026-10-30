r"""Split the data into train / validation / test once, for every model that follows.

The first step of every run. It reads the data, settles the features (--drop,
--date-cols) and the split, and writes them to --out: split.csv (the part of each row)
and split.json (the settings, row counts and feature types). The training scripts then
take only --out, so every model is fitted and scored on exactly the same rows.

The split (--split, default auto):
- time: the most recent rows are test, the ones before them validation. For a model
  that will predict new cases from past ones. Auto picks it when there is one date
  column (or --time-col is given).
- group: every row of one entity (a customer, a patient) lands on the same side, so
  the test has only unseen entities. Auto picks it, when there is no date column, for
  one id-like column whose values repeat (or --group-col is given).
- random: stratified by class for classification. Auto's choice when neither applies.

Auto prefers time, then group, because the error is asymmetric: an unneeded time or
group split makes the scores slightly pessimistic, a missing one makes them look better
than the model will do. Pass --split random to override, and say why in the report.

    python make_split.py --data /abs/customers.csv --target churned --task classification \
        --drop customer_id --out <run folder>
"""

import json
import sys
from pathlib import Path

import common


def main() -> None:
    args = common.split_parser(__doc__.splitlines()[0]).parse_args()
    out = common.data_utils.output_dir(args.out)
    meta_path = out / common.META_FILE
    before = json.loads(meta_path.read_text()) if meta_path.exists() else None

    ds = common.Dataset(args, task=args.task)
    if before and before.get("settings") != ds.settings():
        # Dataset has already refused another task, data or target; this is the rest.
        meta_path.write_text(json.dumps(before, indent=2))
        now = ds.settings()
        changed = {k: (v, now.get(k)) for k, v in before["settings"].items() if now.get(k) != v}
        sys.exit(
            f"{out} was already split with other settings (was, now: {changed}). Models "
            "trained there would not be comparable: use a new --out folder."
        )
    ids = [
        c
        for c in ds.X.columns
        if common.ENTITY_NAME.search(str(c)) and ds.X[c].nunique() > 0.05 * len(ds.X)
    ]
    if ids:
        print(
            f"WARNING: {ids} look like identifiers and are used as features. Pass --drop for "
            "any that only name a row or an entity (Step 3)."
        )
    common.record_run(out, Path(__file__), "split")

    meta = json.loads(meta_path.read_text())
    print(f"rows: {meta['rows']}  dropped for a missing target: {meta['dropped_missing_target']}")
    print(f"features: {len(ds.num_cols)} numeric, {len(ds.cat_cols)} categorical")
    print(f"split written to {out / common.SPLIT_FILE}")


if __name__ == "__main__":
    main()
