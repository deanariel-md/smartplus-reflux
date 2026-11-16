"""Tube insertion and repositioning events from the full-rate device logs (T0).

`Confirmed_FT_Position` holds the confirmed insertion depth in cm (e.g. 60, 63, 65); 0 means not yet
confirmed (insertion or repositioning procedure). Consecutive identical values form segments; segments
shorter than MIN_SEG_S are treated as glitches and absorbed by their neighbours.

Events per folder:
  insertion    a depth-0 segment of >= MIN_PROC_S (the first one is the initial insertion, also when a depth
               was confirmed less than SETTLE_H earlier); later ones are re-insertion procedures
  reposition   a change between two confirmed (non-zero) depths, or a new depth after a depth-0 procedure

Run from the project root (reads data/t0 from T0_ROOT; writes to this project):
    python scripts/extract_tube_positions.py
Writes data/t3/tube_position_segments.parquet and data/t3/tube_events.csv.
"""
import glob
import os
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
from config import DATA_DIR, T0_DIR  # noqa: E402
T0_ROOT = T0_DIR
OUT = DATA_DIR / "t3"
MIN_SEG_S = 60
MIN_PROC_S = 120
SETTLE_H = 1          # a depth confirmed and reset within this long is still part of the initial insertion

segs, no_field = [], []
for d in sorted(glob.glob(str(T0_ROOT / "RMC1-*"))):
    files = glob.glob(os.path.join(d, "*.parquet"))
    if not files:
        continue
    fid = os.path.basename(d)[-3:]
    cols = {r[0] for r in duckdb.sql(f"DESCRIBE SELECT * FROM read_parquet('{Path(d).as_posix()}/*.parquet', "
                                     f"union_by_name=true)").fetchall()}
    if "Confirmed_FT_Position" not in cols:
        no_field.append(fid)
        continue
    s = duckdb.sql(f"""
        WITH x AS (SELECT t_com AS t, mode(Confirmed_FT_Position) AS cp      -- t_com is 1 s; ~5 rows per second
                   FROM read_parquet('{Path(d).as_posix()}/*.parquet', union_by_name=true)
                   WHERE t_com IS NOT NULL AND Confirmed_FT_Position IS NOT NULL GROUP BY 1),
        y AS (SELECT *, CASE WHEN cp IS DISTINCT FROM lag(cp) OVER (ORDER BY t) THEN 1 ELSE 0 END AS chg FROM x),
        z AS (SELECT *, sum(chg) OVER (ORDER BY t ROWS UNBOUNDED PRECEDING) AS g FROM y)
        SELECT g, min(cp) AS cp, min(t) AS ts, max(t) AS te, count(*) AS n_s FROM z GROUP BY g ORDER BY ts""").df()
    s["fid"] = fid
    segs.append(s)
seg = pd.concat(segs, ignore_index=True)
seg["dur_s"] = (seg.te - seg.ts).dt.total_seconds()

# Absorb glitches, then merge neighbours that now share a depth.
clean = []
for fid, g in seg.groupby("fid"):
    g = g[g.dur_s >= MIN_SEG_S].sort_values("ts")
    g = g.assign(k=(g.cp != g.cp.shift()).cumsum())
    clean.append(g.groupby("k").agg(fid=("fid", "first"), cp=("cp", "first"), ts=("ts", "min"), te=("te", "max"))
                 .reset_index(drop=True))
seg = pd.concat(clean, ignore_index=True)
seg["dur_h"] = (seg.te - seg.ts).dt.total_seconds() / 3600

ev = []
for fid, g in seg.groupby("fid"):
    g = g.sort_values("ts").reset_index(drop=True)
    last_depth, n_ins, first_conf = None, 0, None
    for r in g.itertuples():
        if r.cp == 0 and r.dur_h * 3600 >= MIN_PROC_S:
            n_ins += 1
            initial = n_ins == 1 and (last_depth is None                       # before any confirmed depth
                                      or (r.ts - first_conf).total_seconds() < SETTLE_H * 3600)
            ev.append({"fid": fid, "t": r.ts, "event": "insertion" if initial else "reinsertion_procedure",
                       "depth_cm": None, "proc_min": round(r.dur_h * 60, 1)})
        elif r.cp > 0:
            if first_conf is None:
                first_conf = r.ts
            if last_depth is None and n_ins == 0:
                ev.append({"fid": fid, "t": r.ts, "event": "first_confirmed_no_procedure_logged", "depth_cm": r.cp})
            elif last_depth is not None and r.cp != last_depth:
                ev.append({"fid": fid, "t": r.ts, "event": "reposition", "depth_cm": r.cp, "from_cm": last_depth})
            last_depth = r.cp
ev = pd.DataFrame(ev)

OUT.mkdir(parents=True, exist_ok=True)
seg.to_parquet(OUT / "tube_position_segments.parquet", index=False)
ev.to_csv(OUT / "tube_events.csv", index=False)
print(f"{seg.fid.nunique()} folders, {len(seg)} segments; no Confirmed_FT_Position logged in {no_field}")
print(ev.event.value_counts().to_dict())
print(ev[ev.event != "insertion"].groupby("fid").event.value_counts().unstack(fill_value=0).to_string())
