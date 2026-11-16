"""Descriptive analysis of continuous reflux (primary aim).

Uses device-derived tables only; no clinical outcome is read. Run after
build_reflux_axes.py (the primary run plus the sensitivity tags):
    python scripts/describe_reflux.py
Writes results/descriptive/tables/*.csv, models.txt and figures/*.png.

Sections
  1 coverage             analysis set, monitored and artefact hours, by era
  2 events               duration, merging, drained volume, balloon, intervals, progression
  3 axes                 patient level and by device day (with at-risk counts)
  4 course over days     Poisson GLMM, device day, log(monitored h) offset, patient random intercept
  5 time of day          hourly counts; 4-h blocks and 24/12-h harmonics in the same model
  6 determinants         programmed feed rate (previous hour), invasive ventilation, PEEP
  7 device coupling      reflux-triggered pump pause, balloon time
  8 era                  axes by software era
  9 sensitivity          alternative definitions against the primary
"""
import json
import sys
from pathlib import Path

import duckdb
import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import glmm_poisson as gp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
from config import DATA_DIR, OUT_DIR  # noqa: E402
T2 = DATA_DIR / "t2" / "minute"
T3 = DATA_DIR / "t3"
T4 = DATA_DIR / "t4"
VAL = DATA_DIR / "validation"
OUT = OUT_DIR / "descriptive"
TAB, FIG = OUT / "tables", OUT / "figures"
for d in (TAB, FIG):
    d.mkdir(parents=True, exist_ok=True)

EXCLUDED = ("002",)
DAY_START_H = 7
MIN_HOUR_MON_H = 0.25     # hours with less monitoring are dropped from hourly models
SENS_TAGS = {"_raw": "raw runs, no drop/merge", "_art15": "artefact >= 15 min", "_art60": "artefact >= 60 min",
             "_noart": "no artefact rule", "_noera1": "era 1 excluded", "_covered": "all recorded time as denominator",
             "_nofault": "documented device-fault days excluded"}
AX = ["episodes_per_h", "pct_time_reflux", "massive_share"]
AX_LABEL = {"episodes_per_h": "Episodes per monitored hour", "pct_time_reflux": "% of monitored time in reflux",
            "massive_share": "Massive share of episodes"}

# Figure style (dataviz reference palette, light surface; validated pair for minor/massive).
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
C_MINOR, C_MASSIVE, C_IND = "#2a78d6", "#eb6834", "#b9b8b3"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "savefig.dpi": 200})

con = duckdb.connect()
models_txt = []


def q(sql):
    return con.sql(sql).df()


def save(df, name):
    df.to_csv(TAB / f"{name}.csv", index=False)
    return df


def miqr(s, d=2):
    s = pd.Series(s).dropna()
    return f"{s.median():.{d}f} ({s.quantile(.25):.{d}f}-{s.quantile(.75):.{d}f})" if len(s) else ""


def log_model(title, res):
    models_txt.append(f"\n### {title}\n{res.summary()}\n")
    print(f"\n{title}\n{res.summary()}")


excl = ",".join(f"'{x}'" for x in EXCLUDED)
con.sql(f"CREATE VIEW mp AS SELECT right(folder_id, 3) AS fid, right(patient_id, 3) AS patient_id "
        f"FROM read_csv_auto('{(DATA_DIR / 'folder_to_patient.csv').as_posix()}', all_varchar=true)")
con.sql(f"CREATE VIEW pd AS SELECT * FROM read_parquet('{(T4 / 'reflux_axes_patient_day.parquet').as_posix()}') "
        f"WHERE in_analysis_set")
con.sql(f"CREATE TABLE art AS SELECT fid, t FROM read_parquet('{(T4 / 'artefact_minute_list.parquet').as_posix()}')")
ERA = ("CASE WHEN fid::INT <= 20 THEN 0 WHEN fid::INT <= 38 THEN 1 WHEN fid::INT <= 53 THEN 2 "
       "WHEN fid::INT <= 82 THEN 3 ELSE 4 END")

# Events of the analysis set whose onset minute is not artefact.
con.sql(f"""
    CREATE TABLE ev AS
    SELECT e.*, {ERA.replace('fid', 'e.fid')} AS era FROM (
        SELECT right(folder_id, 3) AS fid, type AS typ, CAST(start_t AS TIMESTAMP) AS t0, span_s, in_state_s,
               n_runs, grv_reflux_ml_max AS vol
        FROM read_parquet('{(T3 / 'reflux_events_7s').as_posix()}/*.parquet')) e
    LEFT JOIN art a ON a.fid = e.fid AND a.t = date_trunc('minute', e.t0)
    WHERE a.fid IS NULL AND e.fid NOT IN ({excl})""")
con.sql(f"""CREATE TABLE bal AS SELECT right(folder_id, 3) AS fid, CAST(start_t AS TIMESTAMP) AS t
            FROM read_parquet('{(T3 / 'balloon_inflations').as_posix()}/*.parquet')""")

# ---------------------------------------------------------------- 1 coverage
cov = q(f"""SELECT {ERA} AS era, count(DISTINCT fid) AS patients, count(*) AS patient_days,
                   count(*) FILTER (full_day) AS full_days, round(sum(monitored_h)) AS monitored_h,
                   round(sum(artefact_h), 1) AS artefact_h
            FROM pd GROUP BY ROLLUP (1) ORDER BY 1 NULLS LAST""")
cov["era"] = cov["era"].astype("Int64").astype(str).replace("<NA>", "all")
save(cov, "1_coverage")
print("1 coverage\n", cov.to_string(index=False))

# ---------------------------------------------------------------- 2 events
evt = q("""
    SELECT typ, count(*) AS n, count(DISTINCT fid) AS patients,
           median(in_state_s) AS in_state_med, quantile_cont(in_state_s, .25) AS in_state_q1,
           quantile_cont(in_state_s, .75) AS in_state_q3, quantile_cont(in_state_s, .9) AS in_state_p90,
           median(span_s) AS span_med, avg((span_s >= 60)::INT) AS p_span_ge60,
           avg((n_runs > 1)::INT) AS p_merged, avg((vol > 0)::INT) AS p_drained,
           median(vol) FILTER (vol > 0) AS drained_ml_med
    FROM ev GROUP BY 1 ORDER BY 1""")
bal_cpl = q("""
    SELECT typ, avg(hit::INT) AS p_balloon_30s FROM (
        SELECT e.typ, EXISTS (SELECT 1 FROM bal b WHERE b.fid = e.fid
               AND b.t BETWEEN e.t0 - INTERVAL 2 SECOND AND e.t0 + INTERVAL 30 SECOND) AS hit
        FROM ev e WHERE e.fid NOT IN ('028', '038', '044', '092')) GROUP BY 1""")
evt = evt.merge(bal_cpl, on="typ")
# Progression: minor followed by a massive onset within 60 s; massive preceded by a minor onset within 60 s.
prog = q("""
    SELECT
      avg(EXISTS (SELECT 1 FROM ev m WHERE m.fid = e.fid AND m.typ = 'massive'
                  AND m.t0 > e.t0 AND m.t0 <= e.t0 + INTERVAL 60 SECOND)::INT) FILTER (e.typ = 'minor') AS p_minor_to_massive_60s,
      avg(EXISTS (SELECT 1 FROM ev m WHERE m.fid = e.fid AND m.typ = 'minor'
                  AND m.t0 < e.t0 AND m.t0 >= e.t0 - INTERVAL 60 SECOND)::INT) FILTER (e.typ = 'massive') AS p_massive_after_minor_60s
    FROM ev e""")
# Intervals between consecutive onsets (any type) within a study day.
iei = q(f"""
    WITH x AS (SELECT fid, t0, CAST(t0 - INTERVAL {DAY_START_H} HOUR AS DATE) AS sd,
                      epoch(t0) - lag(epoch(t0)) OVER (PARTITION BY fid ORDER BY t0) AS gap,
                      lag(CAST(t0 - INTERVAL {DAY_START_H} HOUR AS DATE)) OVER (PARTITION BY fid ORDER BY t0) AS sd_prev
               FROM ev)
    SELECT count(*) AS n_gaps, median(gap) AS gap_med_s, quantile_cont(gap, .25) AS gap_q1_s,
           quantile_cont(gap, .75) AS gap_q3_s, avg((gap <= 60)::INT) AS p_le_60s,
           avg((gap <= 300)::INT) AS p_le_5min
    FROM x WHERE gap IS NOT NULL AND sd = sd_prev""")
save(evt.round(3), "2_event_characteristics")
save(pd.concat([prog, iei], axis=1).round(3), "2_event_sequences")
print("\n2 events\n", evt.round(3).to_string(index=False), "\n", prog.round(3).to_string(index=False),
      "\n", iei.round(3).to_string(index=False))

# ---------------------------------------------------------------- 3 axes
pt = pd.read_csv(T4 / "reflux_axes_patient.csv", dtype={"patient_id": str, "fid": str})
rows = []
for c in ["full_days", "monitored_h"] + AX + ["burden_index"]:
    s = pt[c].dropna()
    rows.append({"measure": c, "n": len(s), "median": s.median(), "q1": s.quantile(.25), "q3": s.quantile(.75),
                 "min": s.min(), "max": s.max()})
save(pd.DataFrame(rows).round(3), "3_axes_patient_level")
pdf = q("SELECT * FROM pd")
full = pdf[pdf.full_day].copy()
full["day"] = np.where(full.device_day <= 10, full.device_day.astype(str), "11-14")
byday = full.groupby("day", sort=False).agg(
    at_risk=("fid", "nunique"),
    **{f"{a}": (a, lambda s, d=(3 if a == "massive_share" else 2): miqr(s, d)) for a in AX},
    burden_median=("burden_index", "median")).reset_index()
byday["order"] = byday["day"].map(lambda d: 11 if d == "11-14" else int(d))
byday = byday.sort_values("order").drop(columns="order")
# Patients still recorded (any monitoring) on each device day, full day or not.
rec = pdf[pdf.monitored_h > 0].groupby("device_day")["fid"].nunique()
byday["recorded"] = byday["day"].map(lambda d: rec.loc[11:].max() if d == "11-14" else rec.get(int(d), 0))
save(byday, "3_axes_by_device_day")
print("\n3 axes by device day (median, IQR; full days)\n", byday.to_string(index=False))

# ---------------------------------------------------------------- 4 course over days
full["day_cat"] = np.where(full.device_day >= 8, 8, full.device_day)
full["n_all"] = full.n_minor + full.n_massive
X, names = gp.design(full, ["day_cat:C"])
names = [n.replace("day_cat[8]", "day_cat[8+]") for n in names]
off = np.log(full.monitored_h.to_numpy())
day_res = {}
for lab, col in (("all episodes", "n_all"), ("minor", "n_minor"), ("massive", "n_massive")):
    r = gp.fit(full[col], X, off, full.fid, names)
    day_res[lab] = r
    log_model(f"4 Course over device days ({lab}); full days; offset log(monitored h); ref = day 1", r)
# Linear trend over days 1-7 (rate ratio per day).
f7 = full[full.device_day <= 7]
X7, n7 = gp.design(f7, ["device_day"])
r = gp.fit(f7.n_all, X7, np.log(f7.monitored_h.to_numpy()), f7.fid, n7)
log_model("4b Linear trend, device days 1-7 (RR per day), all episodes", r)
day_tab = pd.concat([day_res[k].table().assign(series=k) for k in day_res])
save(day_tab.round(4), "4_course_over_days_glmm")

# ---------------------------------------------------------------- hourly table (sections 5-6)
con.sql(f"""
    CREATE TABLE hr AS
    WITH mm AS (
        SELECT right(m.folder_id, 3) AS fid, m.minute_pc AS t, m.monitor_s, m.feed_basal_rate, m.ree_valid_frac, m.peep
        FROM read_parquet('{T2.as_posix()}/*.parquet') m
        LEFT JOIN art a ON a.fid = right(m.folder_id, 3) AND a.t = m.minute_pc
        WHERE a.fid IS NULL AND right(m.folder_id, 3) NOT IN ({excl})),
    h AS (
        SELECT fid, date_trunc('hour', t) AS h, sum(monitor_s) / 3600.0 AS mon_h,
               median(feed_basal_rate) AS rate, avg(ree_valid_frac) AS ree_frac, median(peep) AS peep
        FROM mm GROUP BY ALL),
    eh AS (
        SELECT fid, date_trunc('hour', t0) AS h, count(*) FILTER (typ = 'minor') AS n_minor,
               count(*) FILTER (typ = 'massive') AS n_massive
        FROM ev GROUP BY ALL)
    SELECT h.*, coalesce(eh.n_minor, 0) AS n_minor, coalesce(eh.n_massive, 0) AS n_massive,
           hour(h.h) AS clock_h, CAST(h.h - INTERVAL {DAY_START_H} HOUR AS DATE) AS study_date,
           {ERA.replace('fid', 'h.fid')} AS era
    FROM h LEFT JOIN eh USING (fid, h)""")
hr = q("""SELECT hr.*, pd.device_day, lag(rate) OVER (PARTITION BY hr.fid ORDER BY hr.h) AS rate_prev,
                 lag(hr.h) OVER (PARTITION BY hr.fid ORDER BY hr.h) AS h_prev
          FROM hr JOIN pd USING (fid, study_date) ORDER BY hr.fid, hr.h""")
hr = hr[hr.mon_h >= MIN_HOUR_MON_H].copy()
hr["n_all"] = hr.n_minor + hr.n_massive
hr["day_cat"] = np.where(hr.device_day >= 8, 8, hr.device_day)
hr["block"] = ((hr.clock_h - DAY_START_H) % 24 // 4).astype(int)   # 0 = 07-11, 1 = 11-15, ...
BLOCK_LAB = {0: "07-11", 1: "11-15", 2: "15-19", 3: "19-23", 4: "23-03", 5: "03-07"}
for k in (1, 2):
    hr[f"sin{k}"] = np.sin(2 * np.pi * k * hr.clock_h / 24)
    hr[f"cos{k}"] = np.cos(2 * np.pi * k * hr.clock_h / 24)
hr["vent"] = (hr.ree_frac >= 0.5).astype(float)
gap_ok = (hr.h - hr.h_prev) == pd.Timedelta(hours=1)
hr["rate_prev10"] = np.where(gap_ok, hr.rate_prev / 10.0, np.nan)
hoff = np.log(hr.mon_h.to_numpy())

# Overdispersion of hourly counts within patient (Fano factor).
fano = hr.groupby("fid").n_all.agg(lambda s: s.var() / s.mean() if s.mean() > 0 else np.nan)
print(f"\nHourly counts: {len(hr)} patient-hours; within-patient Fano factor median {fano.median():.2f} "
      f"(IQR {fano.quantile(.25):.2f}-{fano.quantile(.75):.2f})")
models_txt.append(f"\nHourly counts: {len(hr)} patient-hours (>= {MIN_HOUR_MON_H} monitored h); within-patient "
                  f"Fano factor median {fano.median():.2f} (IQR {fano.quantile(.25):.2f}-{fano.quantile(.75):.2f}). "
                  "Robust (cluster sandwich) SEs are reported throughout.\n")

# ---------------------------------------------------------------- 5 time of day
Xb, nb = gp.design(hr, ["block:C", "day_cat:C"])
nb = [n.replace("block[", "block_vs_07-11[").replace("day_cat[8]", "day_cat[8+]") for n in nb]
for k, v in BLOCK_LAB.items():
    nb = [n.replace(f"block_vs_07-11[{k}]", f"block_vs_07-11[{v}]") for n in nb]
rb = gp.fit(hr.n_all, Xb, hoff, hr.fid, nb)
log_model("5 Time of day, 4-h blocks (ref 07-11), adjusted for device day; hourly counts", rb)
Xc, nc = gp.design(hr, ["sin1", "cos1", "sin2", "cos2", "day_cat:C"])
rc = gp.fit(hr.n_all, Xc, hoff, hr.fid, nc)
log_model("5b Time of day, 24-h + 12-h harmonics, adjusted for device day", rc)
idx = [nc.index(n) for n in ("sin1", "cos1", "sin2", "cos2")]
b = rc.params[idx]
V = rc.cov_robust[np.ix_(idx, idx)]
wald = float(b @ np.linalg.solve(V, b))
p_wald = float(stats.chi2.sf(wald, 4))
grid = np.arange(0, 24, 0.1)
H = np.column_stack([np.sin(2 * np.pi * grid / 24), np.cos(2 * np.pi * grid / 24),
                     np.sin(4 * np.pi * grid / 24), np.cos(4 * np.pi * grid / 24)])
lp = H @ b
lp_se = np.sqrt(np.einsum("ij,jk,ik->i", H, V, H))
rel = np.exp(lp - lp.mean())
circ = {"wald_chi2_4df": round(wald, 2), "p": p_wald, "peak_clock_h": round(float(grid[lp.argmax()]), 1),
        "trough_clock_h": round(float(grid[lp.argmin()]), 1),
        "peak_to_trough_ratio": round(float(np.exp(lp.max() - lp.min())), 3)}
models_txt.append(f"Harmonics joint Wald test: {json.dumps(circ)}\n")
print("time-of-day harmonics:", circ)
save(rb.table().round(4), "5_time_of_day_blocks_glmm")
raw_hour = hr.groupby("clock_h").agg(events=("n_all", "sum"), mon_h=("mon_h", "sum")).reset_index()
raw_hour["rate_per_h"] = raw_hour.events / raw_hour.mon_h
save(raw_hour.round(4), "5_time_of_day_pooled_rates")

# ---------------------------------------------------------------- 6 determinants
d6 = hr.dropna(subset=["rate_prev10"])
X6, n6 = gp.design(d6, ["rate_prev10", "vent", "block:C", "day_cat:C"])
r6 = gp.fit(d6.n_all, X6, np.log(d6.mon_h.to_numpy()), d6.fid, n6)
log_model("6 Programmed feed rate in the previous hour (per 10 mL/h) and invasive ventilation (REE valid >= 50% "
          "of the hour), adjusted for 4-h block and device day", r6)
d6p = hr[(hr.vent == 1) & hr.peep.notna() & (hr.peep > 0)].copy()
d6p["peep5"] = d6p.peep / 5.0
X6p, n6p = gp.design(d6p, ["peep5", "block:C", "day_cat:C"])
r6p = gp.fit(d6p.n_all, X6p, np.log(d6p.mon_h.to_numpy()), d6p.fid, n6p)
log_model("6b PEEP (per 5 cmH2O) among ventilated hours with PEEP recorded (eras 1-4), adjusted for block and day", r6p)
save(pd.concat([r6.table().assign(model="rate_vent"), r6p.table().assign(model="peep")]).round(4),
     "6_determinants_glmm")

# ---------------------------------------------------------------- 7 device coupling
bal_day = q(f"""
    SELECT right(folder_id, 3) AS fid, CAST(minute_pc - INTERVAL {DAY_START_H} HOUR AS DATE) AS study_date,
           sum(balloon_inflated_s) / 3600.0 AS balloon_h
    FROM read_parquet('{T2.as_posix()}/*.parquet') GROUP BY ALL""")
cp = full.merge(bal_day, on=["fid", "study_date"], how="left")
coup = pd.DataFrame([{
    "pump_pause_h_total": round(pdf.pause_h_total.sum(), 1),
    "pump_pause_h_reflux_triggered": round(pdf.pause_h_reflux.sum(), 1),
    "pct_pause_reflux_triggered": round(100 * pdf.pause_h_reflux.sum() / pdf.pause_h_total.sum(), 1),
    "reflux_pause_h_per_full_day": miqr(cp.pause_h_reflux),
    "balloon_h_per_full_day": miqr(cp.balloon_h)}])
save(coup, "7_device_coupling")
print("\n7 coupling\n", coup.T.to_string(header=False))

# ---------------------------------------------------------------- 8 era
pt["era"] = pt.fid.astype(int).map(lambda n: 0 if n <= 20 else 1 if n <= 38 else 2 if n <= 53 else 3 if n <= 82 else 4)
era_rows = []
for a in AX:
    groups = [g[a].dropna() for _, g in pt.groupby("era")]
    kw = stats.kruskal(*groups)
    row = {"axis": a, "kruskal_p": round(kw.pvalue, 4)}
    for e, g in pt.groupby("era"):
        row[f"era{e} (n={len(g)})"] = miqr(g[a], 3 if a == "massive_share" else 2)
    era_rows.append(row)
save(pd.DataFrame(era_rows), "8_axes_by_era")
print("\n8 era\n", pd.DataFrame(era_rows).to_string(index=False))

# ---------------------------------------------------------------- 9 sensitivity
sens = []
for tag, lab in SENS_TAGS.items():
    f = T4 / f"reflux_axes_patient{tag}.csv"
    if not f.exists():
        print(f"missing {f.name}; run build_reflux_axes.py with --tag {tag}")
        continue
    s = pd.read_csv(f, dtype={"patient_id": str})
    m = pt.merge(s, on="patient_id", suffixes=("", "_s"))
    row = {"variant": lab, "patients": len(s)}
    for a in AX:
        row[f"{a}_median"] = round(s[a].median(), 3)
        row[f"{a}_rho"] = round(stats.spearmanr(m[a], m[f"{a}_s"], nan_policy="omit").statistic, 3)
    d = (m.burden_index - m.burden_index_s).abs()
    row["BI_same_%"] = round(100 * (d == 0).mean(), 1)
    row["BI_within1_%"] = round(100 * (d <= 1).mean(), 1)
    lm = pd.read_csv(T4 / f"reflux_landmark_d1_3{tag}.csv")
    row["landmark_eligible"] = int(lm.eligible_landmark.sum())
    sens.append(row)
prim = {"variant": "PRIMARY", "patients": len(pt),
        **{f"{a}_median": round(pt[a].median(), 3) for a in AX},
        "landmark_eligible": int(pd.read_csv(T4 / "reflux_landmark_d1_3.csv").eligible_landmark.sum())}
sens = pd.DataFrame([prim] + sens)
# Height without non-functioning balloons (label-free: >= 10 massive events, < 10% followed by an inflation).
bc = pd.read_csv(VAL / "balloon_coupling_by_folder.csv", dtype={"fid": str})
bc = bc[(bc.table == "reflux_events_7s") & (bc.typ == "massive")]
no_bal = sorted(set(bc[(bc.n_events >= 10) & (bc.p_balloon < 0.10)].fid) | {"028"})
hb = pt[~pt.fid.isin(no_bal)]
sens = pd.concat([sens, pd.DataFrame([{"variant": f"height without non-functioning balloon ({','.join(no_bal)})",
                                       "patients": len(hb), "massive_share_median": round(hb.massive_share.median(), 3)}])])
save(sens, "9_sensitivity")
print("\n9 sensitivity\n", sens.to_string(index=False))

# Correlation between axes (redundancy check for the burden index).
corr_day = full[AX].corr("spearman").round(2)
corr_pt = pt[AX].corr("spearman").round(2)
save(corr_day.reset_index(), "3_axes_spearman_patient_day")
save(corr_pt.reset_index(), "3_axes_spearman_patient")

(OUT / "models.txt").write_text("Descriptive models. Poisson GLMM with a normal random intercept per patient, "
                                "fitted by adaptive Gauss-Hermite quadrature (scripts/glmm_poisson.py).\n"
                                + "".join(models_txt), encoding="utf-8")

# ---------------------------------------------------------------- figures
# F1: the three axes over device days (full days): individual trajectories, median and IQR.
fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.4), sharex=True)
dd = full[full.device_day <= 10]
for ax, a in zip(axs, AX):
    for _, g in dd.groupby("fid"):
        ax.plot(g.device_day, g[a], color=C_IND, lw=0.5, alpha=0.6)
    s = dd.groupby("device_day")[a]
    med, q1, q3 = s.median(), s.quantile(.25), s.quantile(.75)
    ax.fill_between(med.index, q1, q3, color=C_MINOR, alpha=0.18, lw=0)
    ax.plot(med.index, med, color=C_MINOR, lw=2)
    hi = dd[a].quantile(.97)
    ax.set_ylim(0, hi * 1.05)
    ax.set_title(AX_LABEL[a], fontsize=9, loc="left")
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("Device day")
n_risk = dd.groupby("device_day").fid.nunique()
axs[0].text(0, -0.30, "Patients: " + "  ".join(f"{d}:{n}" for d, n in n_risk.items()), transform=axs[0].transAxes,
            fontsize=7, color=INK2)
fig.suptitle("Reflux axes by device day (full days; grey = patients, line and band = median and IQR; "
             "y-axis cut at 97th percentile)", fontsize=9, x=0.01, ha="left")
fig.tight_layout()
fig.savefig(FIG / "F1_axes_by_device_day.png")
plt.close(fig)

# F2: time of day: harmonic model (relative rate, 95% CI) with pooled observed rates.
fig, ax = plt.subplots(figsize=(6, 3.2))
lo, hi_ = np.exp(lp - 1.96 * lp_se - lp.mean()), np.exp(lp + 1.96 * lp_se - lp.mean())
ax.fill_between(grid, lo, hi_, color=C_MINOR, alpha=0.18, lw=0)
ax.plot(grid, rel, color=C_MINOR, lw=2)
obs = raw_hour.rate_per_h / raw_hour.rate_per_h.mean()
ax.plot(raw_hour.clock_h + 0.5, obs, "o", ms=4, color=INK2, mfc=SURF, mew=1)
ax.axhline(1, color=INK2, lw=0.6)
ax.set_xticks(range(0, 25, 3))
ax.set_xlim(0, 24)
ax.set_xlabel("Clock hour (PC clock)")
ax.set_ylabel("Relative episode rate")
ax.set_title(f"Episodes by time of day (GLMM, 95% CI; circles = observed; p = {p_wald:.2f})",
             fontsize=8.5, loc="left")
fig.tight_layout()
fig.savefig(FIG / "F2_time_of_day.png")
plt.close(fig)

# F3: time in state per episode, minor vs massive (ECDF, log x).
fig, ax = plt.subplots(figsize=(5.2, 3.2))
for typ, col in (("minor", C_MINOR), ("massive", C_MASSIVE)):
    v = np.sort(q(f"SELECT in_state_s FROM ev WHERE typ = '{typ}' AND in_state_s > 0").in_state_s.to_numpy())
    y = np.arange(1, len(v) + 1) / len(v)
    ax.step(v, y, where="post", color=col, lw=2, label=f"{typ} (n={len(v):,})")
ax.set_xscale("log")
ax.set_xlabel("Seconds in reflux state per episode (log scale)")
ax.set_ylabel("Cumulative share of episodes")
ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.set_title("Episode duration", fontsize=9, loc="left")
fig.tight_layout()
fig.savefig(FIG / "F3_episode_duration.png")
plt.close(fig)

# F4: patient-level axes (one dot per patient, median line).
fig, axs = plt.subplots(1, 3, figsize=(9, 3.0))
rng = np.random.default_rng(1)
for ax, a in zip(axs, AX):
    v = pt[a].dropna()
    ax.plot(rng.uniform(-0.18, 0.18, len(v)), v, "o", ms=4.5, color=C_MINOR, alpha=0.75, mec=SURF, mew=0.6)
    ax.hlines(v.median(), -0.3, 0.3, color=INK, lw=1.5)
    ax.set_xlim(-0.6, 0.6)
    ax.set_xticks([])
    ax.set_title(AX_LABEL[a], fontsize=9, loc="left")
fig.suptitle(f"Patient-level reflux axes (n={len(pt)}; pooled over full days; line = median)", fontsize=9,
             x=0.01, ha="left")
fig.tight_layout()
fig.savefig(FIG / "F4_patient_axes.png")
plt.close(fig)

print(f"\nWrote {OUT}")
