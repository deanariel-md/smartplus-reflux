"""Sedation dose-response, extubation adjusted for sedative doses (with death as a competing event), and the
hours with an RSBI recorded.

Daily doses (midazolam mg, propofol mg, fentanyl mcg, remifentanil mcg) come from the trial's analysis workbook
(Analysis/Archive/Analysis results 28May23.xlsx, sheet "Treated Group"; read in memory, only ID, start time and dose
columns). Dexmedetomidine and ketamine have no doses there and stay as yes/no days from the master file.

  A  dose-response within patients: daily episodes ~ each drug's daily dose (Mundlak: daily dose + patient mean),
     dexmedetomidine and ketamine days, device day; Poisson GLMM with log monitored hours offset. Also dose categories
     (none / below / above the drug's median non-zero dose) for a trend.
  B  extubation alive, Cox per doubling of the day 1-3 episode rate: + APACHE II + mean daily doses on days 1-3;
     and a Fine-Gray subdistribution model (ICU death competing; censoring weights) with the same covariates.
  C  hours with an RSBI recorded during ventilation (>= 50% of the hour, as explore_extubation.py; and any RSBI value):
     share of the hour with the feed pump running vs other ventilated hours, and the reflux RR with and without
     adjustment for feed running.

Writes results/explore/18_sedation_dose.csv.
"""
import sys
import warnings
from pathlib import Path

import duckdb
import numpy as np
import openpyxl
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
from config import DATA_DIR, OUT_DIR, DOSE_WORKBOOK  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))
import glmm_poisson as gp  # noqa: E402

OUT, CL, T2, T4 = OUT_DIR / "explore", DATA_DIR / "clinical", DATA_DIR / "t2" / "minute", DATA_DIR / "t4"
XL = DOSE_WORKBOOK
DRUGS = {"Midazolam [mg]": ("midazolam_mg", 50), "Propofol [mg]": ("propofol_mg", 1000),
         "Fentanyl [mcg]": ("fentanyl_mcg", 1000), "Remifentanil [mcg]": ("remifentanil_mcg", 5000)}
rows = []

# ---------------------------------------------------------------- doses (ID, start time and dose columns only)
ws = openpyxl.load_workbook(XL, read_only=True, data_only=True)["Treated Group"]
it = ws.iter_rows(values_only=True)
hdr = list(next(it))
ci = {h: hdr.index(h) for h in ["Record & System ID", "Start time (hh:mm)", *DRUGS]}
recs = []
for r in it:
    pid = r[ci["Record & System ID"]]
    if not isinstance(pid, str) or not pid.startswith("RMC1-"):
        continue
    rec = {"patient_id": pid[-3:], "start": r[ci["Start time (hh:mm)"]]}
    for h, (nm, _) in DRUGS.items():
        v = r[ci[h]]
        rec[nm] = float(v) if isinstance(v, (int, float)) else 0.0
    recs.append(rec)
dose = pd.DataFrame(recs).dropna(subset=["start"])
dose["study_date"] = (pd.to_datetime(dose.start) - pd.Timedelta(hours=7)).dt.date.astype(str)
dose = dose.groupby(["patient_id", "study_date"], as_index=False)[[v[0] for v in DRUGS.values()]].sum()

fr = pd.read_parquet(OUT / "frame_patient_day.parquet")
d = fr[fr.full_day & fr.in_analysis_set].merge(dose, on=["patient_id", "study_date"], how="left")
print(f"full days {len(d)}; with a dose row {d.midazolam_mg.notna().sum()}")
d = d.dropna(subset=["midazolam_mg"]).copy()
d["day_cat"] = np.minimum(d.device_day, 8).astype(int)
for h, (nm, unit) in DRUGS.items():
    d[nm + "_u"] = d[nm] / unit
    d[nm + "_pm"] = d.groupby("patient_id")[nm + "_u"].transform("mean")
    on = d[d[nm] > 0]
    rows.append({"part": "0 doses", "item": f"{nm}: days given / patients / median dose on those days",
                 "estimate": len(on), "lo": on.patient_id.nunique(), "hi": on[nm].median()})
for c in ("dexmedetomidine", "ketamine"):
    d[c] = d[c].astype(float)
    d[c + "_pm"] = d.groupby("patient_id")[c].transform("mean")

# ---------------------------------------------------------------- A dose-response
terms = []
for nm, unit in DRUGS.values():
    terms += [nm + "_u", nm + "_pm"]
terms += ["dexmedetomidine", "dexmedetomidine_pm", "ketamine", "ketamine_pm", "day_cat:C"]
X, names = gp.design(d, terms)
res = gp.fit(d.n_all.to_numpy(), X, np.log(d.monitored_h.to_numpy()), d.patient_id.to_numpy(), names).table()
for nm, unit in DRUGS.values():
    q = res[res.term == nm + "_u"].iloc[0]
    rows.append({"part": "A within-patient RR per dose unit (all drugs together)", "item": f"{nm} per {unit}",
                 "estimate": q.RR, "lo": q.RR_lo, "hi": q.RR_hi, "p": q.p, "n": len(d)})
for c in ("dexmedetomidine", "ketamine"):
    q = res[res.term == c].iloc[0]
    rows.append({"part": "A within-patient RR per dose unit (all drugs together)", "item": f"{c} day (yes/no)",
                 "estimate": q.RR, "lo": q.RR_lo, "hi": q.RR_hi, "p": q.p, "n": len(d)})
# yes/no days from the workbook (dose > 0), all six agents together: the master file under-records midazolam
yn = []
for nm, _ in DRUGS.values():
    d[nm + "_yn"] = (d[nm] > 0).astype(float)
    d[nm + "_yn_pm"] = d.groupby("patient_id")[nm + "_yn"].transform("mean")
    yn += [nm + "_yn", nm + "_yn_pm"]
X, names = gp.design(d, yn + ["dexmedetomidine", "dexmedetomidine_pm", "ketamine", "ketamine_pm", "day_cat:C"])
ry = gp.fit(d.n_all.to_numpy(), X, np.log(d.monitored_h.to_numpy()), d.patient_id.to_numpy(), names).table()
for t in [v[0] + "_yn" for v in DRUGS.values()] + ["dexmedetomidine", "ketamine"]:
    q = ry[ry.term == t].iloc[0]
    rows.append({"part": "A within-patient RR, drug day yes/no from the workbook (all agents together)",
                 "item": t.replace("_yn", " > 0"), "estimate": q.RR, "lo": q.RR_lo, "hi": q.RR_hi, "p": q.p, "n": len(d)})
# master-file flags against the workbook on the same days
for yn_c, nm in (("midazolam", "midazolam_mg"), ("propofol", "propofol_mg"), ("fentanyl", "fentanyl_mcg"),
                 ("remifentanil", "remifentanil_mcg")):
    a, b = d[yn_c].fillna(0) > 0, d[nm] > 0
    on_off = d.assign(b=b).groupby("patient_id").b.nunique().gt(1).sum()
    rows.append({"part": "0 master-file flag vs workbook dose > 0 (same days)",
                 "item": f"{yn_c}: master days / workbook days / agreement / patients on and off (workbook)",
                 "estimate": int(a.sum()), "lo": int(b.sum()), "hi": (a == b).mean(), "n": on_off})
# dose categories for the two hypnotics with enough days
for nm in ("midazolam_mg", "propofol_mg"):
    med = d.loc[d[nm] > 0, nm].median()
    d["cat"] = np.select([d[nm] == 0, d[nm] <= med], ["0 none", "1 low"], "2 high")
    X, names = gp.design(d, ["cat:C", "dexmedetomidine", "ketamine", "day_cat:C"])
    rc = gp.fit(d.n_all.to_numpy(), X, np.log(d.monitored_h.to_numpy()), d.patient_id.to_numpy(), names).table()
    for t in ("cat[1 low]", "cat[2 high]"):
        q = rc[rc.term == t].iloc[0]
        rows.append({"part": f"A {nm} dose category vs none (random intercept, device day)",
                     "item": f"{t[4:-1]} (median non-zero {med:.0f})", "estimate": q.RR, "lo": q.RR_lo, "hi": q.RR_hi,
                     "p": q.p, "n": int((d.cat == t[4:-1]).sum())})
    # within patients only: category indicators plus their patient means (Mundlak)
    d["c_low"], d["c_high"] = (d.cat == "1 low").astype(float), (d.cat == "2 high").astype(float)
    d["c_low_pm"] = d.groupby("patient_id").c_low.transform("mean")
    d["c_high_pm"] = d.groupby("patient_id").c_high.transform("mean")
    others = [t for o, _ in DRUGS.values() if o != nm for t in (o + "_yn", o + "_yn_pm")]
    for adj, extra in (("", []), (" + other three drugs yes/no", others)):
        X, names = gp.design(d, ["c_low", "c_high", "c_low_pm", "c_high_pm", "dexmedetomidine", "dexmedetomidine_pm",
                                 "ketamine", "ketamine_pm"] + extra + ["day_cat:C"])
        rw = gp.fit(d.n_all.to_numpy(), X, np.log(d.monitored_h.to_numpy()), d.patient_id.to_numpy(), names).table()
        for t, lab in (("c_low", "low"), ("c_high", "high")):
            q = rw[rw.term == t].iloc[0]
            rows.append({"part": f"A {nm} dose category vs none, within patients (Mundlak, device day{adj})",
                         "item": f"{lab} (median non-zero {med:.0f})", "estimate": q.RR, "lo": q.RR_lo, "hi": q.RR_hi,
                         "p": q.p, "n": int(d.groupby("patient_id").cat.nunique().gt(1).sum())})

# ---------------------------------------------------------------- B extubation
le = pd.read_csv(DATA_DIR / "analysis" / "landmark_exposure.csv", dtype={"patient_id": str})
le = le[le.n_full_d1_3 >= 2]
op = pd.read_parquet(CL / "outcomes" / "outcomes_patient.parquet")
early = (fr[fr.clinical_day.between(1, 3)].merge(dose, on=["patient_id", "study_date"], how="left")
         .groupby("patient_id")[[v[0] for v in DRUGS.values()]].mean())
x = le.merge(op[op.arm == "treated"], on="patient_id").merge(early.reset_index(), on="patient_id", how="left")
x["reflux"] = np.log2(x.episodes_per_h + 0.25)            # as explore_ventilation.py
x["T"] = x.ventilation_days
x["E"] = (~x.icu_death.astype(bool)).astype(int)                     # 1 extubated alive
x["D"] = x.icu_death.astype(bool).astype(int)                        # competing event
for nm, unit in DRUGS.values():
    x[nm + "_u"] = x[nm].fillna(0) / unit
dcols = [v[0] + "_u" for v in DRUGS.values()]


def cox(df, cols, label, weights=None):
    m = CoxPHFitter().fit(df[cols + ["T", "E"] + ([weights] if weights else [])], "T", "E",
                          weights_col=weights, robust=weights is not None)
    s = m.summary.loc["reflux"]
    rows.append({"part": "B extubation alive, HR per doubling of the day 1-3 rate", "item": label,
                 "estimate": s["exp(coef)"], "lo": s["exp(coef) lower 95%"], "hi": s["exp(coef) upper 95%"],
                 "p": s["p"], "n": len(df)})
    for c in cols:                                       # dose terms: HR per dose unit (mg/day or 5 mg/day)
        if c.endswith("_u") and weights is None:
            d = m.summary.loc[c]
            rows.append({"part": "B extubation alive, HR per dose unit of each drug (days 1-3 mean)", "item": f"{label}: {c}",
                         "estimate": d["exp(coef)"], "lo": d["exp(coef) lower 95%"], "hi": d["exp(coef) upper 95%"],
                         "p": d["p"], "n": len(df)})


cox(x, ["reflux", "apache2"], "cause-specific Cox + APACHE II")
cox(x, ["reflux", "apache2"] + dcols, "cause-specific Cox + APACHE II + day 1-3 doses (4 drugs)")

# Fine-Gray: patients who die stay in the risk set after death with weight G(t)/G(t_death), G = censoring KM
cens = KaplanMeierFitter().fit(x["T"], event_observed=((x.E == 0) & (x.D == 0)).astype(int))
G = lambda t: float(cens.survival_function_at_times(t).iloc[0])   # noqa: E731
times = np.sort(x.loc[x.E == 1, "T"].unique())
long = []
for _, r in x.iterrows():
    if r.D == 1:                                                    # expand after the competing event
        starts = [0.0] + [t for t in times if t > r["T"]]
        ends = [r["T"]] + [t for t in times if t > r["T"]][1:] + ([times.max() + 1] if starts[1:] else [])
        for a, b in zip(starts, ends):
            w = 1.0 if a == 0 else G(a) / max(G(r["T"]), 1e-9)
            long.append({**r.to_dict(), "start": a, "stop": b, "Eev": 0, "w": w})
    else:
        long.append({**r.to_dict(), "start": 0.0, "stop": r["T"], "Eev": int(r.E), "w": 1.0})
lg = pd.DataFrame(long)
lg = lg[lg.stop > lg.start]
from lifelines import CoxTimeVaryingFitter  # noqa: E402
for cols, label in ((["reflux", "apache2"], "Fine-Gray + APACHE II"),
                    (["reflux", "apache2"] + dcols, "Fine-Gray + APACHE II + day 1-3 doses (4 drugs)")):
    lg["pid"] = lg.patient_id
    m = CoxTimeVaryingFitter().fit(lg[cols + ["pid", "start", "stop", "Eev", "w"]], id_col="pid", event_col="Eev",
                                   start_col="start", stop_col="stop", weights_col="w")
    s = m.summary.loc["reflux"]
    rows.append({"part": "B extubation alive, HR per doubling of the day 1-3 rate", "item": label,
                 "estimate": s["exp(coef)"], "lo": s["exp(coef) lower 95%"], "hi": s["exp(coef) upper 95%"],
                 "p": s["p"], "n": len(x)})

# check: with no censoring, Fine-Gray equals a Cox model with competing-event patients kept to the end of follow-up
print(f"censored (neither extubated alive nor died): {int(((x.E == 0) & (x.D == 0)).sum())}")
xc = x.copy()
xc.loc[xc.D == 1, "T"] = x["T"].max() + 1
for cols, label in ((["reflux", "apache2"], "Fine-Gray check (deaths kept to end) + APACHE II"),
                    (["reflux", "apache2"] + dcols, "Fine-Gray check (deaths kept to end) + APACHE II + doses")):
    m = CoxPHFitter().fit(xc[cols + ["T", "E"]], "T", "E")
    s_ = m.summary.loc["reflux"]
    rows.append({"part": "B extubation alive, HR per doubling of the day 1-3 rate", "item": label,
                 "estimate": s_["exp(coef)"], "lo": s_["exp(coef) lower 95%"], "hi": s_["exp(coef) upper 95%"],
                 "p": s_["p"], "n": len(xc)})
dd_ = x.groupby("D").episodes_per_h.median()
rows.append({"part": "B extubation alive, HR per doubling of the day 1-3 rate", "item": "median day 1-3 rate: survivors / ICU deaths",
             "estimate": dd_.get(0), "lo": dd_.get(1), "n": len(x)})

# ---------------------------------------------------------------- C hours with an RSBI recorded
con = duckdb.connect()
con.sql("SET enable_progress_bar = false")
h = con.sql(f"""
    WITH m AS (SELECT right(m.folder_id, 3) AS fid, date_trunc('hour', m.minute_pc) AS hr, m.monitor_s, m.ree_valid_frac,
                      m.rsbi, m.fpf20_running_s, m.covered_s
               FROM read_parquet('{T2.as_posix()}/*.parquet') m
               LEFT JOIN read_parquet('{(T4 / 'artefact_minute_list.parquet').as_posix()}') a
                      ON a.fid = right(m.folder_id, 3) AND a.t = m.minute_pc
               WHERE a.fid IS NULL AND right(m.folder_id, 3) <> '002')
    SELECT fid, hr, sum(monitor_s) / 3600 AS mon_h, avg(ree_valid_frac) AS vent,
           avg(CASE WHEN rsbi > 0 THEN 1.0 ELSE 0.0 END) AS rsbi_frac,
           sum(fpf20_running_s) / nullif(sum(covered_s), 0) AS feed_run
    FROM m GROUP BY ALL HAVING sum(monitor_s) >= 1800""").df()
e = con.sql(f"""SELECT right(folder_id, 3) AS fid, date_trunc('hour', CAST(start_t AS TIMESTAMP)) AS hr, count(*) n
                FROM read_parquet('{(DATA_DIR / 't3' / 'reflux_events_7s_drain.parquet').as_posix()}') GROUP BY ALL""").df()
h = h.merge(e, on=["fid", "hr"], how="left").fillna({"n": 0})
h = h[h.vent >= 0.5].copy()
h["day_cat"] = np.minimum((h.hr - h.groupby("fid").hr.transform("min")).dt.days + 1, 8)
h["feed_run"] = h.feed_run.fillna(0)
# >= 50% of the hour is the definition in explore_extubation.py (6_breathing_glmm.csv); any RSBI value is the sensitivity
for dname, flag in (("RSBI in >= 50% of the hour", h.rsbi_frac >= 0.5), ("any RSBI value in the hour", h.rsbi_frac > 0)):
    h["rsbi_h"] = flag.astype(float)
    rs = h[h.rsbi_h == 1]
    rows.append({"part": f"C {dname}", "item": "hours (patients); share of ventilated hours",
                 "estimate": len(rs), "lo": rs.fid.nunique(), "hi": len(rs) / len(h)})
    rows.append({"part": f"C {dname}", "item": "feed pump running, share of hour: RSBI hours vs other",
                 "estimate": rs.feed_run.mean(), "lo": h[h.rsbi_h == 0].feed_run.mean()})
    for adj, cols in (("device day", ["rsbi_h", "day_cat:C"]), ("+ feed running share", ["rsbi_h", "feed_run", "day_cat:C"])):
        X, names = gp.design(h, cols)
        r = gp.fit(h.n.to_numpy(), X, np.log(h.mon_h.to_numpy()), h.fid.to_numpy(), names).table()
        q = r[r.term == "rsbi_h"].iloc[0]
        rows.append({"part": f"C {dname}: reflux RR vs other ventilated hours", "item": adj, "estimate": q.RR,
                     "lo": q.RR_lo, "hi": q.RR_hi, "p": q.p, "n": len(h)})
out = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print(out.round(3).to_string(index=False))
out.round(4).to_csv(OUT / "18_sedation_dose.csv", index=False)
