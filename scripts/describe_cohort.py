"""Cohort description and clinical-covariate context for the descriptive aim.

Reads covariates only (data/clinical/*.csv|parquet); never reads data/clinical/outcomes/.
Run after build_clinical_tables.py and describe_reflux.py:
    python scripts/describe_cohort.py
Writes results/descriptive/tables/10_baseline.csv, 11_drugs_glmm.csv, 12_art_counts_sensitivity.csv
and appends the models to results/descriptive/models.txt.

  10 baseline   Table 1: treated arm (48) and the device cohort (45)
  11 drugs      episodes per day vs sedative, opioid and vasopressor exposure
  12 ART counts patient-level frequency and height from ART's daily reports,
                adding patients who have no raw logs; agreement with the device-derived values
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import glmm_poisson as gp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
from config import DATA_DIR, OUT_DIR  # noqa: E402
CL = DATA_DIR / "clinical"
T4 = DATA_DIR / "t4"
TAB = OUT_DIR / "descriptive" / "tables"
MODELS = OUT_DIR / "descriptive" / "models.txt"

base = pd.read_csv(CL / "baseline_patient.csv", dtype={"patient_id": str})
cov = pd.read_parquet(CL / "patient_day_covariates.parquet")
al = pd.read_csv(CL / "day_alignment.csv", dtype={"patient_id": str})
pt = pd.read_csv(T4 / "reflux_axes_patient.csv", dtype={"patient_id": str, "fid": str})
pdy = pd.read_parquet(T4 / "reflux_axes_patient_day.parquet")
pdy = pdy[pdy.in_analysis_set]

# ---------------------------------------------------------------- 10 baseline
treated = base[base.arm == "treated"].copy()
treated["device_cohort"] = treated.patient_id.isin(pt.patient_id)


def summarise(g):
    def msd(c):
        return f"{g[c].mean():.1f} ({g[c].std():.1f})"

    def med(c):
        return f"{g[c].median():.1f} ({g[c].quantile(.25):.1f}-{g[c].quantile(.75):.1f})"

    def npct(mask):
        return f"{int(mask.sum())} ({100 * mask.mean():.0f}%)"

    return {
        "n": len(g),
        "Age, years, mean (SD)": msd("age"),
        "Male, n (%)": npct(g.sex == "male"),
        "BMI, kg/m2, mean (SD)": msd("bmi"),
        "APACHE II, mean (SD)": msd("apache2"),
        "Admission: internal medicine, n (%)": npct(g.admission_reason.str.lower().str.startswith("internal")),
        "Admission: trauma, n (%)": npct(g.admission_reason.str.lower().str.startswith("trauma")),
        "Admission: surgical, n (%)": npct(g.admission_reason.str.lower().str.startswith("surg")),
        "Diabetes, n (%)": npct(g.diabetes.astype(bool)),
        "Any GI history, n (%)": npct(g.gi_history.fillna(False).astype(bool)),
        "GERD/oesophageal history, n (%)": npct(g.gerd_or_oesophageal_history.fillna(False).astype(bool)),
        "ICU admission to randomisation, h, median (IQR)": med("hours_icu_to_randomization"),
        "Study days in master, median (IQR)": med("n_clinical_days"),
    }


t1 = pd.DataFrame({"Treated arm": summarise(treated),
                   "Device cohort": summarise(treated[treated.device_cohort])})
dc = pt.merge(treated[["patient_id"]], on="patient_id")
t1.loc["Monitored hours (full days), median (IQR)"] = ["", f"{pt.monitored_h.median():.0f} "
                                                       f"({pt.monitored_h.quantile(.25):.0f}-{pt.monitored_h.quantile(.75):.0f})"]
t1.loc["Full monitored days, median (IQR)"] = ["", f"{pt.full_days.median():.0f} "
                                               f"({pt.full_days.quantile(.25):.0f}-{pt.full_days.quantile(.75):.0f})"]
t1.index.name = "characteristic"
t1.reset_index().to_csv(TAB / "10_baseline.csv", index=False)
print("10 baseline\n", t1.to_string())

# ---------------------------------------------------------------- 11 drugs
d = al[al.match == "both"].merge(
    cov[["patient_id", "clinical_day", "any_sedative", "any_opioid", "any_vasopressor", "pantoprazole"]],
    on=["patient_id", "clinical_day"]).merge(
    pdy[["patient_id", "study_date", "full_day", "n_minor", "n_massive"]].assign(
        study_date=lambda x: pd.to_datetime(x.study_date).dt.date.astype(str)),
    on=["patient_id", "study_date"], suffixes=("", "_t4"))
d = d[d.full_day_t4 if "full_day_t4" in d else d.full_day].copy()
d["n_all"] = d.n_minor + d.n_massive
d["day_cat"] = np.where(d.device_day >= 8, 8, d.device_day).astype(int)
for c in ("any_sedative", "any_opioid", "any_vasopressor", "pantoprazole"):
    d[c] = d[c].astype(float)
expo = d[["any_sedative", "any_opioid", "any_vasopressor", "pantoprazole"]].mean().round(3).to_dict()
X, names = gp.design(d, ["any_sedative", "any_opioid", "any_vasopressor", "day_cat:C"])
res = gp.fit(d.n_all, X, np.log(d.monitored_h.to_numpy()), d.patient_id, names)
txt = ("\n### 11 Drug exposure (master yes/no per clinical day) and episodes per day; aligned full days; "
       f"adjusted for device day. Share of days exposed: {expo}\n{res.summary()}\n")
print(txt)
res.table().round(4).to_csv(TAB / "11_drugs_glmm.csv", index=False)
# Replace this script's section in models.txt (written by describe_reflux.py), so reruns don't duplicate it.
prev = MODELS.read_text(encoding="utf-8") if MODELS.exists() else ""
MODELS.write_text(prev.split("\n### 11 Drug exposure")[0] + txt, encoding="utf-8")

# ---------------------------------------------------------------- 12 ART counts
a = al[al.art_minor_n.notna()].merge(cov[["patient_id", "clinical_day", "day_ratio"]],
                                     on=["patient_id", "clinical_day"])
a = a[a.day_ratio >= 0.5]                      # >= 12 h, as for device full days
art_pt = a.groupby("patient_id").apply(lambda g: pd.Series({
    "art_days": len(g),
    "art_episodes_per_h": (g.art_minor_n + g.art_massive_n).sum() / (24 * g.day_ratio.sum()),
    "art_massive_share": g.art_massive_n.sum() / max((g.art_minor_n + g.art_massive_n).sum(), 1),
}), include_groups=False).reset_index()
m = art_pt.merge(pt[["patient_id", "episodes_per_h", "massive_share"]], on="patient_id", how="left")
rho_f = stats.spearmanr(m.art_episodes_per_h, m.episodes_per_h, nan_policy="omit").statistic
rho_h = stats.spearmanr(m.art_massive_share, m.massive_share, nan_policy="omit").statistic
no_raw = m[m.episodes_per_h.isna()]
summary = pd.DataFrame([{
    "patients_with_art_counts": len(m), "rho_frequency_vs_device": round(rho_f, 3), "rho_height_vs_device": round(rho_h, 3),
    "median_episodes_per_h_art_all": round(m.art_episodes_per_h.median(), 3),
    "median_episodes_per_h_device_45": round(pt.episodes_per_h.median(), 3),
    "median_massive_share_art_all": round(m.art_massive_share.median(), 3),
    "no_raw_logs": "; ".join(f"{r.patient_id}: {r.art_episodes_per_h:.2f}/h, share {r.art_massive_share:.2f}, "
                             f"{int(r.art_days)} d" for r in no_raw.itertuples()),
}])
summary.to_csv(TAB / "12_art_counts_sensitivity.csv", index=False)
print("\n12 ART counts\n", summary.T.to_string(header=False))
