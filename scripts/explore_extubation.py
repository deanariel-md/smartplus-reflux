"""Reflux around device-detected extubation, and reflux vs breathing pattern (hour level).

Extubation is detected from the device itself: REE is valid only during invasive ventilation with the
flow sensor in the circuit. An extubation = the last hour with REE valid >= 50% that is followed by
>= MIN_OFF_H monitored hours with REE valid < 10% (and no return to ventilation in that window).

  A  event study   episodes per monitored hour in each hour from -48 h to +24 h around extubation
                   (artefact minutes excluded); per-patient rates pooled with a Poisson GLMM by 6-h block
  B  breathing     hour-level GLMM within patients during ventilation: spontaneous breathing (RSBI
                   recorded), respiratory rate, tidal volume, PEEP, minute ventilation

Writes results/explore/6_extubation_*.csv, 6_breathing_glmm.csv and fig_extubation.png.
"""
import sys
from pathlib import Path

import duckdb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
from config import DATA_DIR, OUT_DIR  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))
import glmm_poisson as gp  # noqa: E402

T2, T3, T4, OUT = DATA_DIR / "t2" / "minute", DATA_DIR / "t3", DATA_DIR / "t4", OUT_DIR / "explore"
MIN_OFF_H, PRE_H, POST_H = 12, 48, 24
BLUE, INK, GREY = "#2a78d6", "#2b2f36", "#8a8f98"

con = duckdb.connect()
h = con.sql(f"""
    WITH m AS (
        SELECT right(m.folder_id, 3) AS fid, date_trunc('hour', m.minute_pc) AS hr, m.monitor_s, m.ree_valid_frac,
               m.rsbi, m.rr_total, m.vte, m.peep, m.mv_exp
        FROM read_parquet('{T2.as_posix()}/*.parquet') m
        LEFT JOIN read_parquet('{(T4 / 'artefact_minute_list.parquet').as_posix()}') a
               ON a.fid = right(m.folder_id, 3) AND a.t = m.minute_pc
        WHERE a.fid IS NULL AND right(m.folder_id, 3) <> '002'),
    e AS (SELECT right(folder_id, 3) AS fid, date_trunc('hour', CAST(start_t AS TIMESTAMP)) AS hr, count(*) AS n
          FROM read_parquet('{(T3 / 'reflux_events_7s').as_posix()}/*.parquet') GROUP BY ALL)
    SELECT m.fid, m.hr, sum(m.monitor_s) / 3600 AS mon_h,
           sum(m.ree_valid_frac * m.monitor_s) / nullif(sum(m.monitor_s), 0) AS ree_valid,
           avg(CASE WHEN m.rsbi > 0 THEN 1.0 ELSE 0.0 END) AS rsbi_present,
           median(m.rsbi) FILTER (m.rsbi > 0) AS rsbi, median(m.rr_total) FILTER (m.rr_total > 0) AS rr,
           median(m.vte) FILTER (m.vte > 0) AS vte, median(m.peep) FILTER (m.peep > 0) AS peep,
           median(m.mv_exp) FILTER (m.mv_exp > 0) AS mv,
           coalesce(any_value(e.n), 0) AS n
    FROM m LEFT JOIN e ON e.fid = m.fid AND e.hr = m.hr
    GROUP BY m.fid, m.hr HAVING sum(m.monitor_s) >= 1800 ORDER BY m.fid, m.hr""").df()
h["vent"] = h.ree_valid >= 0.5
h["off"] = h.ree_valid < 0.1

# ---------------------------------------------------------------- A: detect extubations
ext = []
for fid, g in h.groupby("fid"):
    g = g.set_index("hr")
    on = g.index[g.vent]
    for t in on:
        after = g.loc[(g.index > t) & (g.index <= t + pd.Timedelta(hours=MIN_OFF_H))]
        if len(after) >= 0.75 * MIN_OFF_H and after.off.all():
            nxt = g.loc[g.index > t]
            if not nxt.vent.any() or nxt.index[nxt.vent][0] > t + pd.Timedelta(hours=MIN_OFF_H):
                ext.append({"fid": fid, "t_ext": t + pd.Timedelta(hours=1)})
                break                                            # first sustained extubation per patient
ext = pd.DataFrame(ext)
print(f"device-detected extubations: {len(ext)} of {h.fid.nunique()} patients")
w = h.merge(ext, on="fid")
w["rel_h"] = ((w.hr - w.t_ext).dt.total_seconds() // 3600).astype(int)
w = w[w.rel_h.between(-PRE_H, POST_H - 1)].copy()
w["block"] = (np.floor(w.rel_h / 6) * 6).astype(int)
prof = w.groupby("rel_h").apply(lambda g: pd.Series({"patients": g.fid.nunique(), "rate": g.n.sum() / g.mon_h.sum()}),
                                include_groups=False).reset_index()
ref = -48
w["blk"] = w.block.astype(str)
X, names = gp.design(w.assign(blk=pd.Categorical(w.block, categories=sorted(w.block.unique()))),
                     ["block:C"])
r = gp.fit(w.n, X, np.log(w.mon_h.to_numpy()), w.fid, names).table()
r = r[r.term.str.startswith("block")].assign(vs=f"block {ref}..{ref + 5} h")
print("\nEpisode rate by 6-h block relative to extubation (RR vs -48..-43 h)\n",
      r[["term", "RR", "RR_lo", "RR_hi", "p"]].round(3).to_string(index=False))
blocks = w.groupby("block").apply(lambda g: pd.Series({"patients": g.fid.nunique(), "hours": len(g),
                                                       "rate": g.n.sum() / g.mon_h.sum()}),
                                  include_groups=False).reset_index()
print(blocks.round(2).to_string(index=False))

fig, ax = plt.subplots(figsize=(6.6, 3.8), dpi=150)
bl = blocks.copy()
ax.plot(bl.block + 3, bl.rate, color=BLUE, lw=2, marker="o", ms=6, mec="white", mew=1.2, zorder=3)
ax.axvline(0, color=GREY, lw=1, ls=(0, (3, 3)))
ax.text(0.5, ax.get_ylim()[1] * 0.97 if False else bl.rate.max() * 1.02, "extubation", color=GREY, fontsize=8.5, va="bottom")
ax.set_xlabel("Hours relative to device-detected extubation (6-h blocks)", color=INK)
ax.set_ylabel("Reflux episodes per monitored hour", color=INK)
ax.set_title(f"Reflux around extubation (n = {len(ext)} patients)", loc="left", fontsize=10, color=INK)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color="#e6e8eb", lw=0.8, zorder=0)
ax.set_ylim(bottom=0)
fig.tight_layout()
fig.savefig(OUT / "fig_extubation.png")

# ---------------------------------------------------------------- B: breathing pattern during ventilation
v = h[h.vent].copy()
v["spont"] = (v.rsbi_present >= 0.5).astype(float)
v["dday"] = v.groupby("fid").hr.transform(lambda s: (s - s.min()).dt.total_seconds() / 86400)
rows = []
for f, scale in (("spont", 1), ("rr", 5), ("vte", 100), ("peep", 5), ("mv", 2), ("rsbi", 20)):
    x = v.dropna(subset=[f]).copy()
    if x.fid.nunique() < 10:
        continue
    x["xv"] = x[f] / scale
    x["pm"] = x.groupby("fid").xv.transform("mean")
    x["dev"] = x.xv - x.pm
    X, names = gp.design(x, ["dev", "pm", "dday"])
    t = gp.fit(x.n, X, np.log(x.mon_h.to_numpy()), x.fid, names).table().set_index("term")
    rows.append({"factor": f"{f} (per {scale})" if scale != 1 else "spontaneous breathing (RSBI recorded)",
                 "hours": len(x), "patients": x.fid.nunique(), "RR_within": t.loc["dev", "RR"],
                 "lo": t.loc["dev", "RR_lo"], "hi": t.loc["dev", "RR_hi"], "p": t.loc["dev", "p"]})
br = pd.DataFrame(rows)
print("\nB  Ventilated hours: within-patient RR (adjusted for time since start)\n", br.round(3).to_string(index=False))

ext.to_csv(OUT / "6_extubation_times.csv", index=False)
blocks.round(4).to_csv(OUT / "6_extubation_blocks.csv", index=False)
r.round(4).to_csv(OUT / "6_extubation_glmm.csv", index=False)
br.round(4).to_csv(OUT / "6_breathing_glmm.csv", index=False)
print(f"\nWrote {OUT}/6_*.csv and fig_extubation.png")
