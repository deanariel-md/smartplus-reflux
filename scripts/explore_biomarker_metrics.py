"""Digital-biomarker metrics for the reflux signal: how stable is it, and does it need continuous monitoring?

  A  day-to-day reliability: ICC of the daily rate (full days, random-intercept model, with and without device-day
     adjustment), days needed for a patient mean with reliability 0.8 (Spearman-Brown), lag-1 day correlation
  B  snapshot vs continuous: a k-hour window (k = 1, 2, 4, 8, 12 contiguous monitored clock hours) against the rest of
     the same full day. Agreement within a factor of 1.5 and 2, Spearman, and agreement on the tertile (cut-points 1.46
     and 4.49 episodes/h)

Writes results/explore/12_biomarker_reliability.csv and 12_snapshot_vs_continuous.csv.
"""
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
from config import DATA_DIR, OUT_DIR  # noqa: E402
OUT = OUT_DIR / "explore"
T2, T3, T4 = DATA_DIR / "t2" / "minute", DATA_DIR / "t3", DATA_DIR / "t4"
CUTS = (1.46, 4.49)

fr = pd.read_parquet(OUT / "frame_patient_day.parquet")
full = fr[fr.full_day & fr.in_analysis_set].copy()
full["day_cat"] = np.minimum(full.device_day, 8).astype(int)

# ---------------------------------------------------------------- A reliability
rows = []
for name, y in (("episodes per h", np.log(full.episodes_per_h + 0.1)), ("% time in reflux", np.log(full.pct_time_reflux + 0.1)),
                ("massive share (days with >= 1 episode)", full.massive_share)):
    d = full.assign(y=y).dropna(subset=["y"])
    for adj, f in (("none", "y ~ 1"), ("device day", "y ~ C(day_cat)")):
        m = smf.mixedlm(f, d, groups=d.patient_id).fit(reml=True)
        vb, vw = float(m.cov_re.iloc[0, 0]), float(m.scale)
        icc = vb / (vb + vw)
        rows.append({"measure": name, "adjustment": adj, "days": len(d), "patients": d.patient_id.nunique(),
                     "ICC": icc, "days_for_reliability_0.8": 0.8 * (1 - icc) / (0.2 * icc)})
rel = pd.DataFrame(rows)
s = full.sort_values(["patient_id", "device_day"])
nxt = s.groupby("patient_id").episodes_per_h.shift(-1).where(s.groupby("patient_id").device_day.diff(-1) == -1)
lag = stats.spearmanr(s.episodes_per_h[nxt.notna()], nxt.dropna())
print("A RELIABILITY (log daily rate, full days)\n", rel.round(3).to_string(index=False))
print(f"consecutive full days, episodes/h: Spearman {lag.statistic:.2f} (n = {int(nxt.notna().sum())} pairs)")
rel = pd.concat([rel, pd.DataFrame([{"measure": "episodes per h", "adjustment": "lag-1 day Spearman",
                                     "days": int(nxt.notna().sum()), "ICC": lag.statistic}])])
rel.round(4).to_csv(OUT / "12_biomarker_reliability.csv", index=False)

# ---------------------------------------------------------------- B snapshot vs continuous
con = duckdb.connect()
con.sql("SET enable_progress_bar = false")
hh = con.sql(f"""
    WITH m AS (SELECT right(m.folder_id, 3) AS fid, date_trunc('hour', m.minute_pc) AS hr, m.monitor_s
               FROM read_parquet('{T2.as_posix()}/*.parquet') m
               LEFT JOIN read_parquet('{(T4 / 'artefact_minute_list.parquet').as_posix()}') a
                      ON a.fid = right(m.folder_id, 3) AND a.t = m.minute_pc
               WHERE a.fid IS NULL AND right(m.folder_id, 3) <> '002'),
    e AS (SELECT right(folder_id, 3) AS fid, date_trunc('hour', CAST(start_t AS TIMESTAMP)) AS hr, count(*) AS n
          FROM read_parquet('{(T3 / 'reflux_events_7s_drain.parquet').as_posix()}') ev
          WHERE NOT EXISTS (SELECT 1 FROM read_parquet('{(T4 / 'artefact_minute_list.parquet').as_posix()}') a
                            WHERE a.fid = right(ev.folder_id, 3)
                              AND a.t = date_trunc('minute', CAST(ev.start_t AS TIMESTAMP)))
          GROUP BY ALL)
    SELECT m.fid, m.hr, sum(m.monitor_s) / 3600 AS mon_h, coalesce(any_value(e.n), 0) AS n
    FROM m LEFT JOIN e ON e.fid = m.fid AND e.hr = m.hr
    GROUP BY m.fid, m.hr HAVING sum(m.monitor_s) >= 2700""").df()
hh["study_date"] = (hh.hr - pd.Timedelta(hours=7)).dt.date.astype(str)
hh["slot"] = ((hh.hr - pd.Timedelta(hours=7)).dt.hour).astype(int)          # 0 = 07:00
hh = hh.merge(full[["fid", "study_date"]], on=["fid", "study_date"])
print(f"\nB SNAPSHOT VS CONTINUOUS: {len(hh)} monitored hours on {hh.groupby(['fid', 'study_date']).ngroups} full days")


def tert(r):
    return np.digitize(r, CUTS)


res = []
for k in (1, 2, 4, 8, 12):
    w = []
    for (fid, sd), g in hh.groupby(["fid", "study_date"]):
        n, h = np.zeros(24), np.zeros(24)
        n[g.slot.values], h[g.slot.values] = g.n.values, g.mon_h.values
        have = h > 0
        tot_n, tot_h = n.sum(), h.sum()
        for s0 in range(0, 24 - k + 1):
            sl = slice(s0, s0 + k)
            if not have[sl].all() or tot_h - h[sl].sum() < 6:
                continue
            wr = n[sl].sum() / h[sl].sum()
            rr = (tot_n - n[sl].sum()) / (tot_h - h[sl].sum())               # rest of the same day
            w.append((fid, sd, s0, wr, rr))
    w = pd.DataFrame(w, columns=["fid", "sd", "s0", "win", "rest"])
    lr = np.log((w.win + 0.1) / (w.rest + 0.1))
    res.append({"window_h": k, "windows": len(w), "days": w.groupby(["fid", "sd"]).ngroups,
                "spearman_vs_rest_of_day": stats.spearmanr(w.win, w.rest).statistic,
                "within_factor_1.5": float((lr.abs() <= np.log(1.5)).mean()),
                "within_factor_2": float((lr.abs() <= np.log(2)).mean()),
                "same_tertile": float((tert(w.win) == tert(w.rest)).mean()),
                "kappa_tertile": float(pd.crosstab(tert(w.win), tert(w.rest)).pipe(
                    lambda t: (np.trace(t.values) / t.values.sum() - (t.sum(0).values @ t.sum(1).values) / t.values.sum() ** 2)
                    / (1 - (t.sum(0).values @ t.sum(1).values) / t.values.sum() ** 2))),
                "zero_episode_windows": float((w.win == 0).mean())})
snap = pd.DataFrame(res)
# for reference: one whole day against the next day (the day-to-day variation a daily summary already carries)
print(snap.round(3).to_string(index=False))
snap.round(4).to_csv(OUT / "12_snapshot_vs_continuous.csv", index=False)
cv = hh.groupby(["fid", "study_date"]).apply(lambda g: (g.n / g.mon_h).std() / max((g.n / g.mon_h).mean(), 1e-9),
                                             include_groups=False)
print(f"within-day CV of hourly rates: median {cv.median():.2f} (IQR {cv.quantile(.25):.2f}-{cv.quantile(.75):.2f})")
