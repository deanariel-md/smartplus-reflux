"""Tube insertion, re-insertion and repositioning (from extract_tube_positions.py) and reflux around repositions.

Device data only. Answers two questions raised by the rise in reflux over device days:
  - can tube dwell time be separated from ICU day? (only if tubes were replaced)
  - is the burden positional? (episode rate in the WINDOW_H hours before vs after each reposition)

Run after extract_tube_positions.py and build_reflux_axes.py:
    python scripts/describe_tube_position.py
Writes results/descriptive/tables/13_tube_events.csv and 13_reposition_event_study.csv.
"""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
from config import DATA_DIR, OUT_DIR  # noqa: E402
T2, T3, T4 = DATA_DIR / "t2" / "minute", DATA_DIR / "t3", DATA_DIR / "t4"
TAB = OUT_DIR / "descriptive" / "tables"
WINDOW_H = 6
MIN_SIDE_H = 3        # monitored hours required on each side of a reposition

ev = pd.read_csv(T3 / "tube_events.csv", dtype={"fid": str}, parse_dates=["t"])
seg = pd.read_parquet(T3 / "tube_position_segments.parquet")
start = seg.groupby("fid").ts.min()
rp = ev[ev.event == "reposition"].copy()
rp["device_day_frac"] = (rp.t - rp.fid.map(start)).dt.total_seconds() / 86400
summary = pd.DataFrame([{
    "folders_with_position_logging": seg.fid.nunique(),
    "initial_insertion_procedures": int((ev.event == "insertion").sum()),
    "reinsertion_procedures": int((ev.event == "reinsertion_procedure").sum()),
    "reinsertion_folders": ",".join(sorted(ev[ev.event == "reinsertion_procedure"].fid)),
    "repositions": len(rp), "patients_with_reposition": rp.fid.nunique(),
    "reposition_day_median": round(rp.device_day_frac.median(), 1),
    "depth_change_cm_median_abs": round((rp.depth_cm - rp.from_cm).abs().median(), 1),
}])

con = duckdb.connect()
con.register("rp", rp[["fid", "t"]].assign(t=lambda x: x.t.astype("datetime64[us]")))
res = con.sql(f"""
    WITH w AS (SELECT fid, t, row_number() OVER () AS id FROM rp),
    art AS (SELECT fid, t FROM read_parquet('{(T4 / 'artefact_minute_list.parquet').as_posix()}')),
    m AS (SELECT right(folder_id, 3) AS fid, minute_pc AS t, monitor_s FROM read_parquet('{T2.as_posix()}/*.parquet')),
    e AS (SELECT right(folder_id, 3) AS fid, CAST(start_t AS TIMESTAMP) AS t0
          FROM read_parquet('{(T3 / 'reflux_events_7s').as_posix()}/*.parquet')),
    mon AS (SELECT w.id, sum(m.monitor_s) FILTER (m.t < w.t) / 3600 AS h_pre,
                   sum(m.monitor_s) FILTER (m.t >= w.t) / 3600 AS h_post
            FROM w JOIN m ON m.fid = w.fid AND m.t BETWEEN w.t - INTERVAL {WINDOW_H} HOUR AND w.t + INTERVAL {WINDOW_H} HOUR
            LEFT JOIN art a ON a.fid = m.fid AND a.t = m.t WHERE a.fid IS NULL GROUP BY 1),
    ec AS (SELECT w.id, count(*) FILTER (e.t0 < w.t) AS n_pre, count(*) FILTER (e.t0 >= w.t) AS n_post
           FROM w JOIN e ON e.fid = w.fid AND e.t0 BETWEEN w.t - INTERVAL {WINDOW_H} HOUR AND w.t + INTERVAL {WINDOW_H} HOUR
           GROUP BY 1)
    SELECT w.id, w.fid, w.t, h_pre, h_post, coalesce(n_pre, 0) AS n_pre, coalesce(n_post, 0) AS n_post
    FROM w JOIN mon USING (id) LEFT JOIN ec USING (id)""").df()
res = res[(res.h_pre >= MIN_SIDE_H) & (res.h_post >= MIN_SIDE_H)].copy()
res["rate_pre"], res["rate_post"] = res.n_pre / res.h_pre, res.n_post / res.h_post
res["log_ratio"] = np.log((res.n_post + 0.5) / res.h_post) - np.log((res.n_pre + 0.5) / res.h_pre)
per_pt = res.groupby("fid").log_ratio.mean()
study = pd.DataFrame([{
    "window_h": WINDOW_H, "repositions": len(res), "patients": res.fid.nunique(),
    "median_rate_pre": round(res.rate_pre.median(), 2), "median_rate_post": round(res.rate_post.median(), 2),
    "pooled_ratio_post_pre": round((res.n_post.sum() / res.h_post.sum()) / (res.n_pre.sum() / res.h_pre.sum()), 2),
    "wilcoxon_p": round(stats.wilcoxon(res.rate_pre, res.rate_post).pvalue, 3),
    "patients_with_decrease": f"{int((per_pt < 0).sum())}/{len(per_pt)}",
    "sign_test_p": round(stats.binomtest(int((per_pt < 0).sum()), len(per_pt)).pvalue, 3),
}])
summary.to_csv(TAB / "13_tube_events.csv", index=False)
study.to_csv(TAB / "13_reposition_event_study.csv", index=False)
print(summary.T.to_string(header=False))
print(study.T.to_string(header=False))
