"""Active Residual Release (ARR, gastric volume drained into the bag) and reflux.

ARR from build_arr.py. By the device's design the drain opens on reflux, and 96% of ARR volume is released within
60 s of a reflux drainage, so "more reflux, more ARR" is partly built in. The non-circular questions are:
does each reflux release more when reflux is frequent (a fuller stomach), and does reflux now predict a large ARR
later, beyond the reflux that happens then?

  0  validation: daily ARR against the trial database's daily GRV; share released during reflux drainage
  1  description: ARR per day and per 6-h block; ARR per reflux episode
  2  same day, within patients (Mundlak mixed models, device-day adjusted): ARR vs the day's reflux rate, and ARR per
     episode vs the reflux rate; between patients (Spearman)
  3  6-h blocks within patients: reflux in block k -> large ARR in block k+1 (GEE logistic, patient clusters), without
     and with adjustment for the reflux and feed in block k+1; AUCs
  4  the reverse: ARR in block k -> reflux in block k+1, adjusted for reflux in block k and feed (Poisson GLMM)
Large ARR: >= LARGE_ML mL in a 6-h block (sensitivity: 200 mL and the top decile).

Writes results/explore/17_arr.csv.
"""
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
from config import DATA_DIR, OUT_DIR  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))
import glmm_poisson as gp  # noqa: E402

OUT, T2, T3, T4 = OUT_DIR / "explore", DATA_DIR / "t2" / "minute", DATA_DIR / "t3", DATA_DIR / "t4"
LARGE_ML, BLOCK_H = 100, 6
con = duckdb.connect()
con.sql("SET enable_progress_bar = false")

g = con.sql(f"""
    WITH m AS (SELECT right(m.folder_id, 3) AS fid, m.minute_pc AS t, m.monitor_s, m.feed_delivered_ml_last AS fd
               FROM read_parquet('{T2.as_posix()}/*.parquet') m
               LEFT JOIN read_parquet('{(T4 / 'artefact_minute_list.parquet').as_posix()}') a
                      ON a.fid = right(m.folder_id, 3) AND a.t = m.minute_pc
               WHERE a.fid IS NULL AND right(m.folder_id, 3) <> '002'),
    r AS (SELECT right(folder_id, 3) AS fid, minute AS t, arr_ml, arr_reflux_ml
          FROM read_parquet('{(T3 / 'arr_minute.parquet').as_posix()}')),
    e AS (SELECT right(folder_id, 3) AS fid, date_trunc('minute', CAST(start_t AS TIMESTAMP)) AS t, count(*) AS n
          FROM read_parquet('{(T3 / 'reflux_events_7s_drain.parquet').as_posix()}') GROUP BY ALL)
    SELECT m.*, coalesce(r.arr_ml, 0) arr, coalesce(r.arr_reflux_ml, 0) arr_refl, coalesce(e.n, 0) AS n_ep
    FROM m LEFT JOIN r USING (fid, t) LEFT JOIN e USING (fid, t) ORDER BY fid, t""").df()
g["fd_inc"] = g.groupby("fid").fd.diff().clip(lower=0).where(lambda s: s < 50, 0).fillna(0)   # feed delivered, mL
g["study_date"] = (g.t - pd.Timedelta(hours=7)).dt.date.astype(str)
g["block"] = (g.t - pd.Timedelta(hours=7)).dt.floor(f"{BLOCK_H}h")
rows = []

# ---------------------------------------------------------------- 0 validation, 1 description
fr = pd.read_parquet(OUT / "frame_patient_day.parquet")
full = fr[fr.full_day & fr.in_analysis_set][["patient_id", "fid", "study_date", "device_day", "episodes_per_h",
                                              "monitored_h", "n_all", "grv_total_ml"]]
dd = g.groupby(["fid", "study_date"]).agg(arr=("arr", "sum"), arr_refl=("arr_refl", "sum"),
                                           feed=("fd_inc", "sum")).reset_index()
day = full.merge(dd, on=["fid", "study_date"], how="left").fillna({"arr": 0, "arr_refl": 0, "feed": 0})
v = day.dropna(subset=["grv_total_ml"])
rho = stats.spearmanr(v.arr, v.grv_total_ml).statistic
share = day.arr_refl.sum() / day.arr.sum()
day["per_ep"] = day.arr / day.n_all.where(day.n_all > 0)
print(f"0 daily ARR vs trial database GRV: Spearman {rho:.2f} ({len(v)} days); released during reflux drainage {share:.0%}")
rows += [{"part": "0 validation", "item": "Spearman, daily ARR vs trial database GRV", "estimate": rho, "n": len(v)},
         {"part": "0 validation", "item": "share of ARR volume released within 60 s of a reflux drainage", "estimate": share},
         {"part": "1 description", "item": "ARR mL/day, median (IQR)", "estimate": day.arr.median(),
          "lo": day.arr.quantile(.25), "hi": day.arr.quantile(.75), "n": len(day)},
         {"part": "1 description", "item": "ARR mL per reflux episode (days with episodes), median (IQR)",
          "estimate": day.per_ep.median(), "lo": day.per_ep.quantile(.25), "hi": day.per_ep.quantile(.75),
          "n": int(day.per_ep.notna().sum())},
         {"part": "1 description", "item": "days with ARR >= 500 mL", "estimate": int((day.arr >= 500).sum()), "n": len(day)}]

# ---------------------------------------------------------------- 2 same day
day["log2_rate"] = np.log2(day.episodes_per_h + 0.1)
day["day_cat"] = np.minimum(day.device_day, 8).astype(int)
day["pm"] = day.groupby("patient_id").log2_rate.transform("mean")
day["dev"] = day.log2_rate - day.pm
day["feed_100"] = day.feed / 100
day["log_per_ep"] = np.log2(day.per_ep + 0.5)
for y, lab, scale in (("arr", "ARR, mL/day per doubling of reflux", 1), ("log_per_ep", "ARR per episode, fold per doubling of reflux", 2)):
    q = day.dropna(subset=[y])
    r = smf.mixedlm(f"{y} ~ dev + pm + C(day_cat) + feed_100", q, groups=q.patient_id).fit()
    est, lo, hi = r.params["dev"], r.conf_int().loc["dev", 0], r.conf_int().loc["dev", 1]
    if scale == 2:
        est, lo, hi = 2 ** est, 2 ** lo, 2 ** hi
    rows.append({"part": "2 same day, within patients (+ feed, device day)", "item": lab, "estimate": est, "lo": lo,
                 "hi": hi, "p": r.pvalues["dev"], "n": len(q)})
pt = day.groupby("patient_id").agg(rate=("episodes_per_h", "median"), arr=("arr", "median"))
s = stats.spearmanr(pt.rate, pt.arr)
rows.append({"part": "2 between patients", "item": "Spearman, median reflux rate vs median daily ARR",
             "estimate": s.statistic, "p": s.pvalue, "n": len(pt)})

# ---------------------------------------------------------------- 3 and 4: 6-h blocks
b = g.groupby(["fid", "block"]).agg(mon_h=("monitor_s", lambda x: x.sum() / 3600), n_ep=("n_ep", "sum"),
                                     arr=("arr", "sum"), feed=("fd_inc", "sum")).reset_index()
b = b[b.mon_h >= BLOCK_H * 0.75].sort_values(["fid", "block"])
b["rate"] = b.n_ep / b.mon_h
b["log2_rate"] = np.log2(b.rate + 0.1)
b["day_cat"] = np.minimum((b.block - b.groupby("fid").block.transform("min")).dt.days + 1, 8)
nxt = b.groupby("fid").shift(-1)
ok = (nxt.block - b.block) == pd.Timedelta(hours=BLOCK_H)
p = b[ok].copy()
for c in ("arr", "feed", "n_ep", "mon_h", "log2_rate"):
    p[f"next_{c}"] = nxt.loc[ok, c]
p["pm"] = p.groupby("fid").log2_rate.transform("mean")
p["dev"] = p.log2_rate - p.pm
p["feed_100"], p["next_feed_100"] = p.feed / 100, p.next_feed / 100
top = p.next_arr.quantile(0.9)
print(f"\n3 6-h block pairs: {len(p)} in {p.fid.nunique()} patients; ARR per block median {b.arr.median():.0f} mL, "
      f"90th percentile {top:.0f}; >= {LARGE_ML} mL in {(p.next_arr >= LARGE_ML).mean():.1%} of blocks")
for thr, lab in ((LARGE_ML, f">= {LARGE_ML} mL"), (200, ">= 200 mL"), (top, f"top decile (>= {top:.0f} mL)")):
    p["y"] = (p.next_arr >= thr).astype(int)
    if p.y.sum() < 10:
        continue
    for adj, f in (("predictive (+ device day, feed k and k+1)", " + C(day_cat) + feed_100 + next_feed_100"),
                   ("beyond reflux in block k+1", " + C(day_cat) + feed_100 + next_feed_100 + next_log2_rate")):
        m_ = smf.gee(f"y ~ dev + pm{f}", "fid", p, family=sm.families.Binomial(),
                     cov_struct=sm.cov_struct.Exchangeable()).fit()
        rows.append({"part": "3 reflux in block k -> large ARR in block k+1 (OR per doubling, within patient)",
                     "item": f"{lab}, {adj}", "estimate": np.exp(m_.params["dev"]),
                     "lo": np.exp(m_.conf_int().loc["dev", 0]), "hi": np.exp(m_.conf_int().loc["dev", 1]),
                     "p": m_.pvalues["dev"], "n": len(p), "events": int(p.y.sum())})
    for col, nm in (("rate", "reflux rate in block k"), ("arr", "ARR in block k"), ("next_feed", "feed delivered in block k+1")):
        rows.append({"part": "3 AUC for a large ARR in block k+1", "item": f"{lab}: {nm}",
                     "estimate": roc_auc_score(p.y, p[col]), "n": len(p), "events": int(p.y.sum())})

p["arr_100"] = p.arr / 100
p["arr_pm"] = p.groupby("fid").arr_100.transform("mean")
p["arr_dev"] = p.arr_100 - p.arr_pm
X, names = gp.design(p, ["arr_dev", "arr_pm", "log2_rate", "next_feed_100", "day_cat:C"])
res = gp.fit(p.next_n_ep.to_numpy(), X, np.log(p.next_mon_h.to_numpy()), p.fid.to_numpy(), names).table()
q = res[res.term == "arr_dev"].iloc[0]
rows.append({"part": "4 ARR in block k -> reflux in block k+1 (RR per 100 mL, within patient)",
             "item": "+ reflux in block k, feed in block k+1, device day", "estimate": q.RR, "lo": q.RR_lo,
             "hi": q.RR_hi, "p": q.p, "n": len(p)})
out = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print(out.round(3).to_string(index=False))
out.round(4).to_csv(OUT / "17_arr.csv", index=False)
