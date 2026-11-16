"""Second label-free check of minor/massive orientation: the drainage (active residual release) rule.

The manufacturer's description of the device logic is that the drain to the bag opens if reflux stays > 15 s at the lower sensors (Z1-Z4),
or as soon as reflux reaches the upper sensors (Z5-Z6). So the "upper" code should open the drain even on
short runs, and the "lower" code only on runs > 15 s. Uses device timing and the drain-open fraction from
data/t3/reflux_runs_raw (runs > 3 s), independent of ART's labels and of the balloon.

Run from the project root:  python scripts/check_drainage_rule.py
Writes data/validation/drainage_rule_by_era.csv.
"""
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
from config import DATA_DIR  # noqa: E402
SRC = (DATA_DIR / "t3" / "reflux_runs_raw").as_posix()
OUT = DATA_DIR / "validation" / "drainage_rule_by_era.csv"

df = duckdb.sql(f"""
    SELECT era, state,
           count(*) FILTER (dur_s <= 15) AS n_3_15s,
           avg(CAST(db_open_frac > 0 AS INT)) FILTER (dur_s <= 15) AS drain_open_3_15s,
           count(*) FILTER (dur_s > 15) AS n_gt15s,
           avg(CAST(db_open_frac > 0 AS INT)) FILTER (dur_s > 15) AS drain_open_gt15s,
           avg(dur_s) AS mean_dur_s
    FROM read_parquet('{SRC}/*.parquet')
    WHERE state IN (11, 12) AND dur_s > 3 AND right(folder_id, 3) <> '002'
    GROUP BY ALL ORDER BY era, state""").df()
OUT.parent.mkdir(parents=True, exist_ok=True)
df.round(3).to_csv(OUT, index=False)
print(df.round(3).to_string(index=False))
short = df.groupby("state").apply(lambda g: (g.drain_open_3_15s * g.n_3_15s).sum() / g.n_3_15s.sum(),
                                  include_groups=False)
print(f"\nDrain open on 3-15 s runs: state 11 {short[11]:.2f}, state 12 {short[12]:.2f}. "
      "Expected if 12 = upper (massive): state 12 near 1 and state 11 well below it; runs > 15 s near 1 for both.")
print(f"Wrote {OUT}")
