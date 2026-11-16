"""Gastric discharge per reflux episode, and pump-off periods, from T0 (read-only, D:).

Device behaviour (checked on raw seconds): when the drain opens for a reflux (DB_State 2), GRVReflux_Volume resets
to 0 and counts the mL drained in that drainage; GRVReflux_Time counts its seconds. The value holds after the drain
closes until the next reflux drainage resets it. Session_Volume (bag) moves too slowly and noisily to use per episode.

1  drain runs: contiguous DB_State = 2 (gaps <= 5 s). Volume = max GRVReflux_Volume from the opening to 10 s after
   closing (negative sensor noise clipped to 0; > 1 L set missing as a counter glitch). Reflux-triggered if FT state 11/12 occurs from 10 s before to 3 s
   after the opening; trigger = 12 if massive was present, else 11.
2  each 7-s reflux episode (T3 reflux_events_7s) gets drained_ml = sum of the volumes of the drain runs open at any
   time from 5 s before its start to 15 s after its end (a drain opened by the minor phase of a reflux is often
   already open when the massive phase starts). A long drain run spanning several episodes counts for each.
3  pump-off runs: contiguous seconds with FPF_System_State not 20 (running) or 21 (system pause), gaps <= 60 s.

Writes data/t3/drain_runs.parquet, data/t3/reflux_events_7s_drain.parquet and data/t3/pump_off_runs.parquet.
    python scripts/build_reflux_drain.py              # full T0 pass (about 10 min)
    python scripts/build_reflux_drain.py --link-only  # redo step 2 from the saved drain runs
"""
import argparse
import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
from config import DATA_DIR, T0_DIR  # noqa: E402
T0 = T0_DIR
T3 = DATA_DIR / "t3"
MAX_ML = 1000                     # a single drainage above 1 L is a counter glitch

ap = argparse.ArgumentParser()
ap.add_argument("--link-only", action="store_true")
args = ap.parse_args()
con = duckdb.connect()
con.sql("SET enable_progress_bar = false")
ev = con.sql(f"SELECT * FROM read_parquet('{(T3 / 'reflux_events_7s').as_posix()}/*.parquet')").df()
drains, pumps = [], []
for folder in ([] if args.link_only else sorted(p.name for p in T0.glob("RMC1-*"))):
    src = f"read_parquet('{(T0 / folder).as_posix()}/*.parquet', union_by_name=true)"
    have = set(con.sql(f"SELECT * FROM {src} LIMIT 0").columns)
    if not {"DB_State", "GRVReflux_Volume", "FT_System_State", "FPF_System_State"} <= have:
        print(folder, "missing columns, skipped")
        continue
    off = con.sql(f"SELECT median(epoch(t_com) - epoch(t_ssf)) FROM {src} "
                  "WHERE t_com IS NOT NULL AND t_ssf IS NOT NULL").fetchone()[0] or 0
    con.sql(f"""
        CREATE OR REPLACE TEMP TABLE s AS
        WITH x AS (SELECT coalesce(t_com_f, t_ssf_f + to_seconds({off})) AS t, FT_System_State ft, DB_State db,
                          GRVReflux_Volume gr, FPF_System_State fpf FROM {src})
        SELECT date_trunc('second', t) AS t, last(ft ORDER BY t) ft, last(db ORDER BY t) db, last(gr ORDER BY t) gr,
               last(fpf ORDER BY t) fpf
        FROM x WHERE t IS NOT NULL GROUP BY 1""")
    d = con.sql(f"""
        WITH o AS (SELECT t, epoch(t) - epoch(lag(t) OVER (ORDER BY t)) gap FROM s WHERE db = 2),
        r AS (SELECT t, sum(CASE WHEN gap IS NULL OR gap > 5 THEN 1 ELSE 0 END) OVER (ORDER BY t) rid FROM o),
        runs AS (SELECT rid, min(t) t0, max(t) t1 FROM r GROUP BY rid)
        SELECT '{folder}' AS folder_id, runs.t0 AS start_t, runs.t1 AS end_t, epoch(runs.t1) - epoch(runs.t0) + 1 AS dur_s,
               greatest(coalesce(max(s.gr) FILTER (s.t BETWEEN runs.t0 AND runs.t1 + INTERVAL 10 SECOND), 0), 0) AS vol_ml,
               max(s.ft) FILTER (s.t BETWEEN runs.t0 - INTERVAL 10 SECOND AND runs.t0 + INTERVAL 3 SECOND
                                 AND s.ft IN (11, 12)) AS trigger_state
        FROM runs JOIN s ON s.t BETWEEN runs.t0 - INTERVAL 10 SECOND AND runs.t1 + INTERVAL 10 SECOND
        GROUP BY ALL""").df()
    drains.append(d)
    p = con.sql(f"""
        WITH o AS (SELECT t, fpf, epoch(t) - epoch(lag(t) OVER (ORDER BY t)) gap FROM s
                   WHERE fpf IS NOT NULL AND fpf NOT IN (20, 21)),
        r AS (SELECT *, sum(CASE WHEN gap IS NULL OR gap > 60 THEN 1 ELSE 0 END) OVER (ORDER BY t) rid FROM o)
        SELECT '{folder}' AS folder_id, min(t) start_t, max(t) end_t, epoch(max(t)) - epoch(min(t)) + 1 AS dur_s,
               mode(fpf) AS state FROM r GROUP BY rid""").df()
    pumps.append(p)
    print(folder, f"drains {len(d)} (reflux {d.trigger_state.notna().sum()}), median vol {d.vol_ml.median():.0f} mL; "
                  f"pump-off runs {len(p)}", flush=True)

if args.link_only:
    dr = pd.read_parquet(T3 / "drain_runs.parquet")
else:
    dr = pd.concat(drains, ignore_index=True)
    dr.loc[dr.vol_ml > MAX_ML, "vol_ml"] = np.nan                 # counter glitches (one drain read 22,540 mL)
    dr.to_parquet(T3 / "drain_runs.parquet", index=False)
    pd.concat(pumps, ignore_index=True).to_parquet(T3 / "pump_off_runs.parquet", index=False)

con.register("dr", dr)
con.register("ev", ev)
out = con.sql("""
    SELECT ev.*, coalesce(sum(dr.vol_ml), 0) AS drained_ml, count(dr.start_t) AS n_drains
    FROM ev LEFT JOIN dr ON dr.folder_id = ev.folder_id
         AND dr.start_t <= CAST(ev.end_t AS TIMESTAMP) + INTERVAL 15 SECOND
         AND dr.end_t >= CAST(ev.start_t AS TIMESTAMP) - INTERVAL 5 SECOND
    GROUP BY ALL""").df()
out.to_parquet(T3 / "reflux_events_7s_drain.parquet", index=False)
print(f"\nepisodes: {len(out)}; with a drainage: {(out.n_drains > 0).mean():.0%}; drained mL per episode "
      f"(with drainage) quantiles {out[out.n_drains > 0].drained_ml.quantile([.5, .75, .9, .95, .99]).round(0).to_dict()}")
print(out.groupby("type").agg(n=("drained_ml", "size"), with_drain=("n_drains", lambda s: (s > 0).mean()),
                              median_ml=("drained_ml", "median"), p90_ml=("drained_ml", lambda s: s.quantile(.9))))
