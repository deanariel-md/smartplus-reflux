"""Battery of analyses: what goes with reflux, what follows it, and what explains its rise over ICU days.

Frame: aligned full monitored days (device day = clinical day; analysis set), joined to the daily
covariates (drugs, SOFA) and daily feeding data (REE, plan, delivery, GRV, formula, feeding stops).

  1  drivers      Poisson GLMM, episodes ~ factor + device day; Mundlak split gives the WITHIN-patient
                  rate ratio (factor + its patient mean)
  2  consequences day-level mixed models: plan / REE, delivered / plan, adequacy, GRV and feeding stops vs
                  reflux the same day and the day before
  3  prognosis    reflux on clinical days 1-3 vs ventilation days, ICU stay, ICU death
  4  the rise     does the day trend survive adjustment for the clinical course; does reflux rise toward
                  the end of monitoring (weaning, extubation) rather than with time since insertion?

Writes results/explore/*.csv and prints a digest.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
from config import DATA_DIR, OUT_DIR  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))
import glmm_poisson as gp  # noqa: E402

CL, T4, OUT = DATA_DIR / "clinical", DATA_DIR / "t4", OUT_DIR / "explore"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- frame
al = pd.read_csv(CL / "day_alignment.csv", dtype={"patient_id": str})
al = al[al.match == "both"][["patient_id", "clinical_day", "study_date", "device_day"]]
pdy = pd.read_parquet(T4 / "reflux_axes_patient_day.parquet")
pdy = pdy[pdy.in_analysis_set].assign(study_date=lambda x: pd.to_datetime(x.study_date).dt.date.astype(str))
cov = pd.read_parquet(CL / "patient_day_covariates.parquet")
cov = cov[cov.arm == "treated"].drop(columns=["arm", "start", "end", "study_date"])
od = pd.read_parquet(CL / "outcomes" / "patient_day_outcomes.parquet")
od = od[od.arm == "treated"][["patient_id", "clinical_day", "ree_kcal_treated", "delivered_kcal", "vtbd_ml",
                              "kcal_per_ml", "nutrition_type", "fluids_ml", "grv_total_ml", "grv_max_ml",
                              "feeding_stops"]]
d = (al.merge(pdy.drop(columns=["device_day"]), on=["patient_id", "study_date"])
       .merge(cov, on=["patient_id", "clinical_day"], how="left")
       .merge(od, on=["patient_id", "clinical_day"], how="left"))
d = d.sort_values(["patient_id", "clinical_day"]).reset_index(drop=True)
d["n_all"] = d.n_minor + d.n_massive
d["day_cat"] = np.minimum(d.device_day, 8).astype(int)
d["ree"] = d.ree_kcal_treated.where(d.ree_kcal_treated > 0)
d["plan_kcal"] = d.vtbd_ml * d.kcal_per_ml
d["plan_frac"] = d.plan_kcal / (d.ree * d.day_ratio)
d["eff"] = d.delivered_kcal / d.plan_kcal
d["adequacy"] = d.delivered_kcal / (d.ree * d.day_ratio)
d["feed_stop"] = (d.feeding_stops.notna() & (d.feeding_stops.astype(str).str.strip() != "")).astype(float)
nt = d.nutrition_type.astype(str).str.strip().str.lower()
d["formula"] = np.select([nt.str.startswith("peptamen"), nt.str.startswith("nephro"), nt.str.startswith("jevity"),
                          nt.str.startswith("glucerna")], ["peptide", "renal", "polymeric", "diabetic"], "unknown")
d["log2_rate"] = np.log2(d.episodes_per_h + 0.25)
d["log2_pct"] = np.log2(d.pct_time_reflux + 0.05)
for c in ("propofol", "midazolam", "dexmedetomidine", "ketamine", "fentanyl", "remifentanil", "morphine",
          "noradrenaline", "vasopressin", "hydrocortisone", "pantoprazole", "lactulose", "any_sedative",
          "any_opioid", "any_vasopressor"):
    d[c] = d[c].astype(float)
d["fluids_L"] = d.fluids_ml / 1000
d["ree_100"] = d.ree / 100
d["grv_100"] = d.grv_total_ml / 100
full = d[d.full_day].copy()
print(f"frame: {len(full)} full aligned days, {full.patient_id.nunique()} patients")

# ---------------------------------------------------------------- 1 drivers (within-patient)
rows = []
FACTORS = ["propofol", "midazolam", "dexmedetomidine", "ketamine", "fentanyl", "remifentanil", "morphine",
           "noradrenaline", "vasopressin", "hydrocortisone", "pantoprazole", "lactulose", "sofa", "ree_100",
           "fluids_L", "kcal_per_ml", "plan_frac", "grv_100"]
for f in FACTORS:
    x = full.dropna(subset=[f]).copy()
    x[f + "_pm"] = x.groupby("patient_id")[f].transform("mean")
    x["dev"] = x[f] - x[f + "_pm"]
    contrast = (x.groupby("patient_id")[f].nunique() > 1).sum()
    if contrast < 5:
        continue
    X, names = gp.design(x, ["dev", f + "_pm", "day_cat:C"])
    r = gp.fit(x.n_all, X, np.log(x.monitored_h.to_numpy()), x.patient_id, names).table().set_index("term")
    rows.append({"factor": f, "days": len(x), "patients_with_contrast": int(contrast),
                 "mean_exposure": x[f].mean(), "RR_within": r.loc["dev", "RR"], "lo": r.loc["dev", "RR_lo"],
                 "hi": r.loc["dev", "RR_hi"], "p": r.loc["dev", "p"],
                 "RR_between": r.loc[f + "_pm", "RR"], "p_between": r.loc[f + "_pm", "p"]})
# formula type (categorical, within patients who switched)
x = full[full.formula != "unknown"].copy()
X, names = gp.design(x, ["formula:C", "day_cat:C"])
ft = gp.fit(x.n_all, X, np.log(x.monitored_h.to_numpy()), x.patient_id, names).table()
ft = ft[ft.term.str.startswith("formula")]
for _, r in ft.iterrows():
    rows.append({"factor": r.term + " vs diabetic", "days": len(x),
                 "patients_with_contrast": int((x.groupby("patient_id").formula.nunique() > 1).sum()),
                 "RR_within": r.RR, "lo": r.RR_lo, "hi": r.RR_hi, "p": r.p})
drivers = pd.DataFrame(rows)
drivers.round(4).to_csv(OUT / "1_drivers_within.csv", index=False)
print("\n1 DRIVERS of episodes (within-patient RR per unit; adjusted for device day)\n",
      drivers[["factor", "days", "patients_with_contrast", "RR_within", "lo", "hi", "p"]].round(3).to_string(index=False))
print("formula days:", x.formula.value_counts().to_dict())

# ---------------------------------------------------------------- 2 consequences (day level)
d["lag_log2_rate"] = d.groupby("patient_id").log2_rate.shift(1).where(
    d.groupby("patient_id").clinical_day.diff() == 1)
d["lag_full"] = d.groupby("patient_id").full_day.shift(1).where(d.groupby("patient_id").clinical_day.diff() == 1)
ok = (d.day_ratio >= 0.5) & (d.ree > 0) & (d.delivered_kcal >= 0) & (d.plan_kcal > 0)
cons = []
for y, xv, label, where in (
        ("plan_frac", "lag_log2_rate", "prescription (plan/REE) day d ~ reflux day d-1", ok & (d.lag_full == True)),
        ("plan_frac", "log2_rate", "prescription (plan/REE) ~ reflux same day", ok & d.full_day),
        ("eff", "log2_rate", "delivered/plan ~ reflux same day", ok & d.full_day),
        ("eff", "lag_log2_rate", "delivered/plan day d ~ reflux day d-1", ok & (d.lag_full == True)),
        ("adequacy", "lag_log2_rate", "adequacy day d ~ reflux day d-1", ok & (d.lag_full == True)),
        ("grv_100", "log2_rate", "GRV (100 mL) ~ reflux same day", d.full_day & d.grv_total_ml.notna())):
    x = d[where].dropna(subset=[y, xv]).copy()
    x = x[np.isfinite(x[y])]
    x["pm"] = x.groupby("patient_id")[xv].transform("mean")
    x["dev"] = x[xv] - x.pm
    try:
        m = smf.mixedlm(f"{y} ~ dev + pm + C(day_cat)", x, groups=x.patient_id).fit()
        cons.append({"model": label, "days": len(x), "patients": x.patient_id.nunique(),
                     "beta_within_per_doubling": m.params["dev"], "lo": m.conf_int().loc["dev", 0],
                     "hi": m.conf_int().loc["dev", 1], "p": m.pvalues["dev"], "outcome_median": x[y].median()})
    except Exception as ex:
        cons.append({"model": label, "note": str(ex)})
# feeding stop (medical order) ~ reflux day before: GEE logistic
x = d[d.lag_full == True].dropna(subset=["lag_log2_rate"]).copy()
g = smf.gee("feed_stop ~ lag_log2_rate + C(day_cat)", "patient_id", x, family=sm.families.Binomial(),
            cov_struct=sm.cov_struct.Exchangeable()).fit()
cons.append({"model": "feeding stop (medical order) day d ~ reflux day d-1 (GEE, OR per doubling)",
             "days": len(x), "patients": x.patient_id.nunique(),
             "beta_within_per_doubling": np.exp(g.params["lag_log2_rate"]),
             "lo": np.exp(g.conf_int().loc["lag_log2_rate", 0]), "hi": np.exp(g.conf_int().loc["lag_log2_rate", 1]),
             "p": g.pvalues["lag_log2_rate"], "outcome_median": x.feed_stop.mean()})
cons = pd.DataFrame(cons)
cons.round(4).to_csv(OUT / "2_consequences.csv", index=False)
print("\n2 CONSEQUENCES (within-patient, per doubling of the episode rate)\n", cons.round(3).to_string(index=False))

# ---------------------------------------------------------------- 3 prognosis (days 1-3 exposure)
op = pd.read_parquet(CL / "outcomes" / "outcomes_patient.parquet")
op = op[op.arm == "treated"].drop(columns="arm")
le = pd.read_csv(DATA_DIR / "analysis" / "landmark_exposure.csv", dtype={"patient_id": str})
le = le[le.n_full_d1_3 >= 2].merge(op, on="patient_id", how="left")
le["vfd28"] = np.where(le.icu_death, 0, np.clip(28 - le.ventilation_days, 0, 28))
prog = []
for xv in ("episodes_per_h", "pct_time_reflux", "massive_share", "bi2", "bi3"):
    for y in ("ventilation_days", "icu_los_days", "vfd28"):
        m = le[[xv, y]].dropna()
        r, p = stats.spearmanr(m[xv], m[y])
        prog.append({"exposure_d1_3": xv, "outcome": y, "n": len(m), "rho": r, "p": p})
    a, b = le[le.icu_death == True][xv].dropna(), le[le.icu_death == False][xv].dropna()
    prog.append({"exposure_d1_3": xv, "outcome": "icu_death (median died vs survived)", "n": len(a) + len(b),
                 "rho": np.nan, "p": stats.mannwhitneyu(a, b).pvalue,
                 "median_died": a.median(), "median_survived": b.median()})
prog = pd.DataFrame(prog)
prog.round(4).to_csv(OUT / "3_prognosis.csv", index=False)
print(f"\n3 PROGNOSIS (reflux on clinical days 1-3; n = {len(le)}, deaths {int(le.icu_death.sum())})\n",
      prog.round(3).to_string(index=False))

# ---------------------------------------------------------------- 4 the rise
rise = []
x = full.dropna(subset=["any_sedative", "sofa", "any_vasopressor"]).copy()
x["dday"] = x.device_day.astype(float)
for label, terms in (("device day only", ["dday"]),
                     ("+ sedation, opioid, vasopressor", ["dday", "any_sedative", "any_opioid", "any_vasopressor"]),
                     ("+ SOFA", ["dday", "any_sedative", "any_opioid", "any_vasopressor", "sofa"]),
                     ("+ feed plan (plan/REE)", ["dday", "any_sedative", "any_opioid", "any_vasopressor", "sofa",
                                                 "plan_frac"])):
    xx = x.dropna(subset=terms)
    X, names = gp.design(xx, terms)
    t = gp.fit(xx.n_all, X, np.log(xx.monitored_h.to_numpy()), xx.patient_id, names).table().set_index("term")
    rise.append({"model": label, "days": len(xx), "RR_per_day": t.loc["dday", "RR"], "lo": t.loc["dday", "RR_lo"],
                 "hi": t.loc["dday", "RR_hi"], "p": t.loc["dday", "p"]})
# Time since start vs time to the end of monitoring.
last = d.groupby("patient_id").device_day.transform("max")
full["days_to_end"] = (last.loc[full.index] - full.device_day).astype(float)
full["dday"] = full.device_day.astype(float)
xx = full[full.groupby("patient_id").device_day.transform("count") >= 4]
for label, terms in (("time since start", ["dday"]), ("time to end of monitoring", ["days_to_end"]),
                     ("both", ["dday", "days_to_end"])):
    X, names = gp.design(xx, terms)
    r = gp.fit(xx.n_all, X, np.log(xx.monitored_h.to_numpy()), xx.patient_id, names)
    t = r.table().set_index("term")
    for term in terms:
        rise.append({"model": f"{label}: {term}", "days": len(xx), "RR_per_day": t.loc[term, "RR"],
                     "lo": t.loc[term, "RR_lo"], "hi": t.loc[term, "RR_hi"], "p": t.loc[term, "p"],
                     "loglik": r.loglik})
# Trend by ICU outcome
xx = full.merge(op[["patient_id", "icu_death"]], on="patient_id")
xx["died"] = xx.icu_death.astype(float)
xx["dday_x_died"] = xx.dday * xx.died
X, names = gp.design(xx, ["dday", "died", "dday_x_died"])
t = gp.fit(xx.n_all, X, np.log(xx.monitored_h.to_numpy()), xx.patient_id, names).table().set_index("term")
for term in ("dday", "dday_x_died"):
    rise.append({"model": f"by ICU death: {term}", "days": len(xx), "RR_per_day": t.loc[term, "RR"],
                 "lo": t.loc[term, "RR_lo"], "hi": t.loc[term, "RR_hi"], "p": t.loc[term, "p"]})
rise = pd.DataFrame(rise)
rise.round(4).to_csv(OUT / "4_rise.csv", index=False)
print("\n4 THE RISE over device days\n", rise.round(3).to_string(index=False))
d.to_parquet(OUT / "frame_patient_day.parquet", index=False)
print(f"\nWrote {OUT}")
