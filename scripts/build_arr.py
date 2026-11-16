"""Active Residual Release (ARR): gastric volume drained into the smART+ bag, per minute, from T0 (read-only, D:).

Device behaviour (checked on raw seconds): NutrientsPlan_GRV_Volume is the running total of gastric residual drained
during the current feeding plan (it resets when a plan starts); InBetween_Plans_GRV_Volume does the same between
plans (absent in era 0). The bag fills mostly while the tube is in plain monitoring (FT state 10), not only during
reflux drainage.

Per second: ARR increment = positive change of either counter (drops are plan resets; a single change > MAX_STEP mL
is a counter glitch and set to 0). Each increment is labelled
  reflux drainage  - a reflux state (FT 11/12/14) or a reflux drain (DB_State 2) in the preceding 60 s
  other            - otherwise (passive release outside reflux)

Writes data/t3/arr_minute.parquet: folder_id, minute (PC clock), arr_ml, arr_reflux_ml, arr_other_ml.
    python scripts/build_arr.py      # full T0 pass (about 10 min)
"""
import os
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
from config import DATA_DIR, T0_DIR  # noqa: E402
T0 = T0_DIR
T3 = DATA_DIR / "t3"
MAX_STEP = 200

con = duckdb.connect()
con.sql("SET enable_progress_bar = false")
out = []
for folder in sorted(p.name for p in T0.glob("RMC1-*")):
    src = f"read_parquet('{(T0 / folder).as_posix()}/*.parquet', union_by_name=true)"
    have = set(con.sql(f"SELECT * FROM {src} LIMIT 0").columns)
    if not {"NutrientsPlan_GRV_Volume", "FT_System_State", "DB_State"} <= have:
        print(folder, "missing columns, skipped")
        continue
    ib = "InBetween_Plans_GRV_Volume" if "InBetween_Plans_GRV_Volume" in have else "NULL"
    off = con.sql(f"SELECT median(epoch(t_com) - epoch(t_ssf)) FROM {src} "
                  "WHERE t_com IS NOT NULL AND t_ssf IS NOT NULL").fetchone()[0] or 0
    d = con.sql(f"""
        WITH x AS (SELECT coalesce(t_com_f, t_ssf_f + to_seconds({off})) AS t, FT_System_State ft, DB_State db,
                          NutrientsPlan_GRV_Volume npg, {ib} AS ibg FROM {src}),
        s AS (SELECT date_trunc('second', t) AS t, last(ft ORDER BY t) ft, last(db ORDER BY t) db,
                     last(npg ORDER BY t) npg, last(ibg ORDER BY t) ibg
              FROM x WHERE t IS NOT NULL GROUP BY 1),
        c AS (SELECT t, ft, db,
                     npg - lag(npg) OVER (ORDER BY t) AS d1, ibg - lag(ibg) OVER (ORDER BY t) AS d2,
                     max(CASE WHEN ft IN (11, 12, 14) OR db = 2 THEN 1 ELSE 0 END)
                         OVER (ORDER BY t RANGE BETWEEN INTERVAL 60 SECOND PRECEDING AND CURRENT ROW) AS refl
              FROM s),
        i AS (SELECT t, refl,
                     (CASE WHEN d1 > 0 AND d1 <= {MAX_STEP} AND npg_ok THEN d1 ELSE 0 END)
                     + (CASE WHEN d2 > 0 AND d2 <= {MAX_STEP} THEN d2 ELSE 0 END) AS inc
              FROM (SELECT *, TRUE AS npg_ok FROM c))
        SELECT '{folder}' AS folder_id, date_trunc('minute', t) AS minute, sum(inc) AS arr_ml,
               sum(inc) FILTER (refl = 1) AS arr_reflux_ml, sum(inc) FILTER (refl = 0) AS arr_other_ml
        FROM i GROUP BY ALL HAVING sum(inc) > 0""").df()
    d = d.fillna({"arr_reflux_ml": 0, "arr_other_ml": 0})
    out.append(d)
    print(folder, f"ARR {d.arr_ml.sum():.0f} mL in {len(d)} minutes; during reflux drainage "
                  f"{d.arr_reflux_ml.sum() / max(d.arr_ml.sum(), 1):.0%}", flush=True)
res = pd.concat(out, ignore_index=True)
res.to_parquet(T3 / "arr_minute.parquet", index=False)
print(f"\nWrote {T3 / 'arr_minute.parquet'}: {len(res)} minutes, {res.arr_ml.sum():.0f} mL; during reflux drainage "
      f"{res.arr_reflux_ml.sum() / res.arr_ml.sum():.0%}")
