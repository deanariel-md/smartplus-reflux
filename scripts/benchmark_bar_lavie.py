"""External benchmark: our reflux rate on the event definition of Bar-Lavie et al. 2022.

Bar-Lavie YP, Epstein D, Roimi M, Steinlauf S. Detection of gastroesophageal reflux by multichannel
intraluminal impedance technology during mechanical ventilation: the first case series.
Clin Nutr Open Sci 2022;44:118-125. doi:10.1016/j.nutos.2022.06.008
Rambam, 20 ventilated tube-fed patients, smARTrack FT (detection only, no feeding control), first
~2.9 days per patient. A GER = reflux above the LES lasting more than 3 s. Reported: median 2.7 (IQR 2.1-3.8)
GER/h per patient, range 0.6-8.6; durations 31% 3-10 s, 61% 10-30 s, 8% > 30 s.

Our matching definition: contiguous reflux (FT state 11 or 12, so a minor run that escalates to massive
is one GER) lasting > 3 s, from data/t3/reflux_runs_raw. Denominator: monitored seconds outside primary
artefact minutes, over the first 72 monitored hours and over the whole record. Analysis set (folder 002 out).
The 7-s episodes used in the main analysis are shown alongside.

Run after build_reflux_axes.py:  python scripts/benchmark_bar_lavie.py
Writes results/descriptive/tables/15_benchmark_bar_lavie.csv.
"""
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
from config import DATA_DIR, OUT_DIR  # noqa: E402
T2, T3, T4 = DATA_DIR / "t2" / "minute", DATA_DIR / "t3", DATA_DIR / "t4"
TAB = OUT_DIR / "descriptive" / "tables" / "15_benchmark_bar_lavie.csv"
EXCLUDED = ("002",)
MIN_S, FIRST_H = 3, 72
PUBLISHED = {"source": "Bar-Lavie 2022 (Rambam, n=20)", "patients": 20, "median_per_h": 2.7, "q1": 2.1, "q3": 3.8,
             "min_per_h": 0.6, "max_per_h": 8.6, "pct_3_10s": 31, "pct_10_30s": 61, "pct_gt30s": 8}

con = duckdb.connect()
con.sql(f"""
CREATE TEMP TABLE mon AS
SELECT right(m.folder_id, 3) AS fid, m.minute_pc AS t, m.monitor_s
FROM read_parquet('{T2.as_posix()}/*.parquet') m
LEFT JOIN read_parquet('{(T4 / 'artefact_minute_list.parquet').as_posix()}') a
       ON a.fid = right(m.folder_id, 3) AND a.t = m.minute_pc
WHERE a.fid IS NULL AND m.monitor_s > 0 AND right(m.folder_id, 3) NOT IN {EXCLUDED};
CREATE TEMP TABLE first AS
SELECT fid, t, sum(monitor_s) OVER (PARTITION BY fid ORDER BY t) / 3600 AS cum_h FROM mon;
-- GER: merge back-to-back 11/12 runs (gap < 1 s) into one contiguous reflux, then keep > MIN_S s.
CREATE TEMP TABLE ger AS
WITH r AS (SELECT right(folder_id, 3) AS fid, CAST(start_t AS TIMESTAMP) AS s, end_t AS e
           FROM read_parquet('{(T3 / 'reflux_runs_raw').as_posix()}/*.parquet') WHERE state IN (11, 12)),
g AS (SELECT *, CASE WHEN s <= lag(e) OVER w + INTERVAL 1 SECOND THEN 0 ELSE 1 END AS new
      FROM r WINDOW w AS (PARTITION BY fid ORDER BY s)),
k AS (SELECT *, sum(new) OVER (PARTITION BY fid ORDER BY s) AS gid FROM g)
SELECT fid, min(s) AS s, epoch(max(e) - min(s)) AS dur_s FROM k GROUP BY fid, gid;
CREATE TEMP TABLE ep AS
SELECT right(folder_id, 3) AS fid, CAST(start_t AS TIMESTAMP) AS s FROM read_parquet('{(T3 / 'reflux_events_7s').as_posix()}/*.parquet');
""")


def rates(events, window):
    lim = f"AND f.cum_h <= {FIRST_H}" if window == "first72h" else ""
    return con.sql(f"""
        WITH den AS (SELECT fid, sum(m.monitor_s) / 3600 AS h FROM mon m JOIN first f USING (fid, t)
                     WHERE true {lim} GROUP BY fid),
        num AS (SELECT x.fid, count(*) AS n FROM {events} x
                JOIN first f ON f.fid = x.fid AND f.t = date_trunc('minute', x.s)
                WHERE true {lim} GROUP BY 1)
        SELECT den.fid, den.h, coalesce(num.n, 0) AS n, coalesce(num.n, 0) / den.h AS per_h
        FROM den LEFT JOIN num USING (fid) WHERE den.h >= 12""").df()


con.sql(f"CREATE TEMP VIEW ger3 AS SELECT * FROM ger WHERE dur_s > {MIN_S}")
rows = [PUBLISHED]
for label, events in (("GER > 3 s (Bar-Lavie definition)", "ger3"), ("7-s episodes", "ep")):
    for window in ("first72h", "whole"):
        r = rates(events, window)
        q = r.per_h.quantile([0.25, 0.5, 0.75])
        row = {"source": f"this cohort: {label}, {window}", "patients": len(r), "median_per_h": q[0.5],
               "q1": q[0.25], "q3": q[0.75], "min_per_h": r.per_h.min(), "max_per_h": r.per_h.max(),
               "median_monitored_h": r.h.median()}
        if events == "ger3":
            lim = f"AND f.cum_h <= {FIRST_H}" if window == "first72h" else ""
            d = con.sql(f"""SELECT g.dur_s FROM ger3 g JOIN first f ON f.fid = g.fid AND f.t = date_trunc('minute', g.s)
                            WHERE true {lim}""").df().dur_s
            row.update(pct_3_10s=100 * (d <= 10).mean(), pct_10_30s=100 * ((d > 10) & (d <= 30)).mean(),
                       pct_gt30s=100 * (d > 30).mean())
        rows.append(row)
out = pd.DataFrame(rows)
TAB.parent.mkdir(parents=True, exist_ok=True)
out.round(2).to_csv(TAB, index=False)
print(out.round(2).T.to_string(header=False))
print(f"\nWrote {TAB}")
