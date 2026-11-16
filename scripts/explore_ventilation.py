"""Follow-up of explore_reflux.py §3: does early reflux predict a shorter ventilation, beyond severity and sedation?

Patients with >= 2 full monitored days on clinical days 1-3. Exposure: episodes per hour on days 1-3
(log2). Outcomes: ventilator-free days to day 28 (0 if died in ICU), and time to extubation alive
(cause-specific Cox; ICU death censored). Confounders considered: APACHE II, age, admission reason, and the
share of days 1-3 on midazolam, propofol, dexmedetomidine, opioids and vasopressors, and SOFA.

Writes results/explore/5_ventilation_*.csv.
"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
from config import DATA_DIR, OUT_DIR  # noqa: E402
CL, OUT = DATA_DIR / "clinical", OUT_DIR / "explore"

le = pd.read_csv(DATA_DIR / "analysis" / "landmark_exposure.csv", dtype={"patient_id": str})
le = le[le.n_full_d1_3 >= 2].copy()
op = pd.read_parquet(CL / "outcomes" / "outcomes_patient.parquet")
op = op[op.arm == "treated"].drop(columns="arm")
base = pd.read_csv(CL / "baseline_patient.csv", dtype={"patient_id": str})
cov = pd.read_parquet(CL / "patient_day_covariates.parquet")
cov = cov[(cov.arm == "treated") & cov.clinical_day.between(1, 3)]
drug = cov.groupby("patient_id")[["midazolam", "propofol", "dexmedetomidine", "any_opioid", "any_vasopressor",
                                  "sofa"]].mean().add_suffix("_d1_3").reset_index()
keep = [c for c in ("patient_id", "sex", "admission_reason") if c in base.columns]
x = (le.drop(columns=[c for c in ("age", "sex", "admission_reason") if c in le.columns])
       .merge(op, on="patient_id", how="left").merge(drug, on="patient_id", how="left")
       .merge(base[keep], on="patient_id", how="left"))
if "age" not in x.columns:
    x = x.merge(base[["patient_id", "age"]], on="patient_id", how="left")
x["log2_rate"] = np.log2(x.episodes_per_h + 0.25)
x["died"] = x.icu_death.astype(float)
x["vfd28"] = np.where(x.icu_death, 0, np.clip(28 - x.ventilation_days, 0, 28))
x["trauma"] = x.admission_reason.astype(str).str.lower().str.contains("trauma").astype(float)
print(f"n = {len(x)}, deaths {int(x.died.sum())}, median ventilation {x.ventilation_days.median():.1f} d")

# Tertiles of early rate
x["rate_tertile"] = pd.qcut(x.episodes_per_h, 3, labels=["low", "mid", "high"])
tert = x.groupby("rate_tertile", observed=True).agg(
    n=("patient_id", "size"), rate_median=("episodes_per_h", "median"), vent_days_median=("ventilation_days", "median"),
    vfd28_median=("vfd28", "median"), deaths=("died", "sum"), apache_median=("apache2", "median"),
    midazolam_share=("midazolam_d1_3", "mean"), dexmed_share=("dexmedetomidine_d1_3", "mean"),
    propofol_share=("propofol_d1_3", "mean")).reset_index()
print("\nBy tertile of early episode rate\n", tert.round(2).to_string(index=False))

# Confounder correlations with the exposure
cc = []
for c in ("apache2", "age", "trauma", "midazolam_d1_3", "propofol_d1_3", "dexmedetomidine_d1_3", "any_opioid_d1_3",
          "any_vasopressor_d1_3", "sofa_d1_3", "massive_share", "era"):
    m = x[["log2_rate", c]].dropna()
    r, p = stats.spearmanr(m.log2_rate, m[c])
    r2, p2 = stats.spearmanr(x.dropna(subset=[c, "vfd28"])[c], x.dropna(subset=[c, "vfd28"]).vfd28)
    cc.append({"covariate": c, "n": len(m), "rho_with_reflux": r, "p": p, "rho_with_vfd28": r2, "p_vfd": p2})
cc = pd.DataFrame(cc)
print("\nCovariates vs early reflux and vs VFD28\n", cc.round(3).to_string(index=False))

# VFD28 regression, adding confounders one at a time and together (HC3)
res = []
for label, extra in (("unadjusted", ""), ("+ APACHE II", " + apache2"), ("+ age", " + age"),
                     ("+ trauma", " + trauma"), ("+ midazolam d1-3", " + midazolam_d1_3"),
                     ("+ propofol d1-3", " + propofol_d1_3"), ("+ dexmedetomidine d1-3", " + dexmedetomidine_d1_3"),
                     ("+ vasopressor d1-3", " + any_vasopressor_d1_3"), ("+ era group", " + C(era_group)"),
                     ("+ APACHE + midazolam + vasopressor", " + apache2 + midazolam_d1_3 + any_vasopressor_d1_3"),
                     ("+ APACHE + all sedatives + vasopressor",
                      " + apache2 + midazolam_d1_3 + propofol_d1_3 + dexmedetomidine_d1_3 + any_vasopressor_d1_3")):
    f = smf.ols(f"vfd28 ~ log2_rate{extra}", x).fit(cov_type="HC3")
    res.append({"model": label, "n": int(f.nobs), "vfd_per_doubling": f.params["log2_rate"],
                "lo": f.conf_int().loc["log2_rate", 0], "hi": f.conf_int().loc["log2_rate", 1],
                "p": f.pvalues["log2_rate"]})
res = pd.DataFrame(res)
print("\nVentilator-free days to day 28 per doubling of the early episode rate\n", res.round(3).to_string(index=False))

# Cause-specific Cox: time to extubation alive (death censored)
cox = []
x["extubated_alive"] = (~x.icu_death).astype(int)
for label, cols in (("unadjusted", ["log2_rate"]), ("+ APACHE II", ["log2_rate", "apache2"]),
                    ("+ APACHE + midazolam + vasopressor", ["log2_rate", "apache2", "midazolam_d1_3",
                                                            "any_vasopressor_d1_3"])):
    m = x.dropna(subset=cols + ["ventilation_days"])
    ph = sm.PHReg(m.ventilation_days, m[cols], status=m.extubated_alive).fit()
    i = cols.index("log2_rate")
    hr, lo, hi = np.exp(ph.params[i]), np.exp(ph.conf_int()[i, 0]), np.exp(ph.conf_int()[i, 1])
    cox.append({"model": label, "n": len(m), "HR_extubation_per_doubling": hr, "lo": lo, "hi": hi,
                "p": ph.pvalues[i]})
cox = pd.DataFrame(cox)
print("\nTime to extubation alive (HR > 1 = earlier extubation), per doubling of the early episode rate\n",
      cox.round(3).to_string(index=False))

tert.round(3).to_csv(OUT / "5_ventilation_tertiles.csv", index=False)
cc.round(4).to_csv(OUT / "5_ventilation_confounders.csv", index=False)
res.round(4).to_csv(OUT / "5_ventilation_vfd_models.csv", index=False)
cox.round(4).to_csv(OUT / "5_ventilation_cox.csv", index=False)
print(f"\nWrote {OUT}/5_ventilation_*.csv")
