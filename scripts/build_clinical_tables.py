"""Clean clinical tables from the trial master workbook, with outcome tables written to a separate folder.

Source: the master workbook (config.MASTER_WORKBOOK), sheet " Patients info".
Only rows whose ID matches ^RMC1-\\d{3}$ are kept (subtotal and summary rows are dropped).
The workbook holds no names, initials or dates of birth.

Also reads the Castor medical-history export (DM data), only its record ID and gastrointestinal
fields. The Castor demographics export (dates of birth, initials) is never read.

Outputs in data/clinical/:
  baseline_patient.csv            demographics, severity, admission reason, GI history flags (no outcomes)
  patient_day_covariates.parquet  study day, SOFA, insulin, daily drug exposure (no nutrition, no prokinetics)
  day_alignment.csv               clinical study day <-> device day, ART daily reflux counts (labels corrected)
  outcomes/patient_day_outcomes.parquet   nutrition targets/delivery, REE, VTBD, FE, GRV, prokinetics, feeding stops
  outcomes/outcomes_patient.parquet       LOS, ventilation days, ICU mortality

The outcomes/ folder holds the outcome and nutrition-delivery tables.

Reflux columns in the master have minor and massive swapped; here they are
relabelled and checked against the device-derived counts on aligned days.

Run from the project root:  python scripts/build_clinical_tables.py
"""
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
from config import DATA_DIR, MASTER_WORKBOOK, ECRF_MEDICAL_HISTORY  # noqa: E402
MASTER = MASTER_WORKBOOK
SHEET = " Patients info"
CASTOR_DM = ECRF_MEDICAL_HISTORY
T4 = DATA_DIR / "t4"
OUT = DATA_DIR / "clinical"
OUTCOMES = OUT / "outcomes"
DAY_START_H = 7


def col(df, pattern):
    """The single column whose name matches pattern (case-insensitive, whitespace-normalised)."""
    hits = [c for c in df.columns if re.search(pattern, re.sub(r"\s+", " ", str(c)), re.I)]
    if len(hits) != 1:
        raise SystemExit(f"pattern {pattern!r} matched {hits}")
    return hits[0]


def yes(s):
    return s.astype(str).str.strip().str.lower().isin(["yes", "y", "1", "1.0", "true"])


raw = pd.read_excel(MASTER, sheet_name=SHEET)
idc = col(raw, r"^Record & System ID$")
d = raw[raw[idc].astype(str).str.match(r"^RMC1-\d{3}$")].copy()
d["patient_id"] = d[idc].str[-3:]
d["arm"] = d[col(d, r"^Treated=1, Control=2$")].map({1: "treated", 2: "control"})
d["start"] = pd.to_datetime(d[col(d, r"^Start time")])
d["end"] = pd.to_datetime(d[col(d, r"^End time")])
d = d.sort_values(["patient_id", "start"])
d["clinical_day"] = d.groupby("patient_id").cumcount() + 1          # first study-day row = day 1
# Most rows start at 06:xx or 07:00, so start - 7 h can fall on the previous date. The midpoint
# of start and end sits safely inside the 07:00-06:59 study day for full and partial days alike.
mid = d["start"] + (d["end"] - d["start"]) / 2
d["study_date"] = (mid - pd.Timedelta(hours=DAY_START_H)).dt.date
dups = d.duplicated(["patient_id", "study_date"]).sum()
if dups:
    print(f"WARNING: {dups} clinical rows share a study date with another row of the same patient")
print(f"master: {len(d)} patient-day rows, {d.patient_id.nunique()} patients "
      f"({d[d.arm == 'treated'].patient_id.nunique()} treated)")

# ------------------------------------------------------------------ baseline (no outcomes)
first = d.groupby("patient_id").first()
base = pd.DataFrame({
    "arm": first["arm"],
    "age": first[col(d, r"^Age$")],
    "sex": first[col(d, r"^Gender$")].str.strip().str.lower(),
    "weight_kg": first[col(d, r"^Weight$")],
    "height_m": first[col(d, r"^Height$")],
    "admission_reason": first[col(d, r"^Reason for ICU admission")].str.strip(),
    "apache2": first[col(d, r"^APACHEII")],
    "hours_icu_to_randomization": first[col(d, r"^Time from ICU admission to randomization")],
    "diabetes": yes(d[col(d, r"^Diabetes mellitus")]).groupby(d.patient_id).any(),
    "first_study_day_start": first["start"],
    "n_clinical_days": d.groupby("patient_id").size(),
})
base["bmi"] = base.weight_kg / base.height_m ** 2
base = base.reset_index()

# Gastrointestinal history from the Castor medical-history export. Only the record ID and the three
# GI fields are read; the free text is reduced to a keyword flag and not stored.
gi = pd.read_excel(CASTOR_DM, sheet_name="Study results",
                   usecols=["Record Id", "gastrointestinal_med_history", "gastrointestinal_clinically",
                            "gastrointestinal_describe"])
gi = pd.DataFrame({
    "patient_id": gi["Record Id"].astype(str).str[-3:],
    "gi_history": gi["gastrointestinal_med_history"].map({"Yes": True, "No": False}),
    "gi_clinically_significant": gi["gastrointestinal_clinically"].map({"Yes": True, "No": False}),
    "gerd_or_oesophageal_history": gi["gastrointestinal_describe"].astype(str).str.contains(
        r"gerd|reflux|hiatal|hiatus|esophag|oesophag|heartburn|barrett", case=False)
                                   & gi["gastrointestinal_describe"].notna(),
})
base = base.merge(gi, on="patient_id", how="left")

# ------------------------------------------------------------------ patient-day covariates
DRUGS = {
    "midazolam": r"^Midazolam$", "propofol": r"^Propofol$", "dexmedetomidine": r"^Dexmedtomidine",
    "ketamine": r"^Ketamine$", "fentanyl": r"^Fentanyl$", "remifentanil": r"^Remifentanil$",
    "morphine": r"^Morphine$", "noradrenaline": r"^Noradrenaline", "adrenaline": r"^Adrenaline$",
    "vasopressin": r"^Vasopressin$", "hydrocortisone": r"^Hydrocortisone$", "dexamethasone": r"^Dexamethasone$",
    "prednisone": r"^Prednisone$", "pantoprazole": r"^Pantoprazole", "lactulose": r"^Lactulose",
}
cov = d[["patient_id", "arm", "clinical_day", "start", "end", "study_date"]].copy()
cov["day_ratio"] = d[col(d, r"^Day ratio$")]
cov["sofa"] = d[col(d, r"^SOFA score$")]
cov["insulin_units"] = pd.to_numeric(d[col(d, r"Insulin daily dose")], errors="coerce")
for k, pat in DRUGS.items():
    cov[k] = yes(d[col(d, pat)])
cov["any_sedative"] = cov[["midazolam", "propofol", "dexmedetomidine", "ketamine"]].any(axis=1)
cov["any_opioid"] = cov[["fentanyl", "remifentanil", "morphine"]].any(axis=1)
cov["any_vasopressor"] = cov[["noradrenaline", "adrenaline", "vasopressin"]].any(axis=1)

# ------------------------------------------------------------------ outcome files
outcome_cols = {
    "ree_kcal_treated": r"^REE baseline", "target_kcal_control": r"^Daily Total Nutrient Target \[kCal\]",
    "partial_pct_control": r"^If partial", "delivered_kcal": r"^Daily Enteral Delivered \[kCal\]$",
    "delivered_kcal_grv_adj_control": r"consider GRV/2", "nutrition_type": r"^Nutrition Type$",
    "kcal_per_ml": r"^Nutrient Kcal/ml$", "target_protein_g": r"^Target Protein", "protein_delivered_g": r"^Total Protein Delivered",
    "fluids_ml": r"^Estimated fluids", "grv_total_ml": r"^Daily total amount of GRV", "grv_max_ml": r"^Max Daily GRV",
    "target_ml_control": r"^Daily Total Nutrient Target \[ml\]$", "vtbd_ml": r"^VTBD", "delivered_ml": r"^Daily Enteral Delivered \[ml\]$",
    "feeding_efficiency_pct": r"^Feeding efficiency", "deviation_fe": r"^Deviation from 100%",
    "feeding_stops": r"^Feeding stops medical order", "feeding_stops_ratio": r"^Feeding stops discont",
    "metoclopramide_dose": r"^Metoclopramide.*Dose", "metoclopramide_time": r"^Metoclopramide.*time",
    "erythromycin_dose": r"^Erythromycin- Dose", "erythromycin_time": r"^Erythromycin- time",
    "norton": r"^Norton score", "pressure_ulcer": r"^Pressure Ulcer after 48",
}
out_day = d[["patient_id", "arm", "clinical_day", "study_date"]].copy()
for k, pat in outcome_cols.items():
    out_day[k] = d[col(d, pat)]
for k in ("metoclopramide_dose", "metoclopramide_time", "erythromycin_dose", "erythromycin_time",
          "feeding_stops", "nutrition_type", "pressure_ulcer"):
    out_day[k] = out_day[k].astype("string")
out_day["prokinetic_given"] = out_day[["metoclopramide_dose", "erythromycin_dose"]].notna().any(axis=1)
out_pt = pd.DataFrame({
    "arm": first["arm"],
    "icu_los_days": first[col(d, r"^LOS \[days")],
    "ventilation_days": first[col(d, r"^Ventilation \[days")],
    "icu_death": yes(d[col(d, r"^Mortality")]).groupby(d.patient_id).any(),
}).reset_index()

# ------------------------------------------------------------------ ART daily reflux (labels corrected)
# Master columns say "massive" where the device report says minor, and vice versa.
art = d[["patient_id", "clinical_day", "study_date"]].copy()
art["art_minor_n"] = d[col(d, r"^Number of total daily massive reflux")]
art["art_minor_s"] = d[col(d, r"^Duration of total daily massive reflux")]
art["art_massive_n"] = d[col(d, r"^Number of total daily minor reflux")]
art["art_massive_s"] = d[col(d, r"^Duration of total daily minor reflux")]
art = art[d.arm == "treated"]

# ------------------------------------------------------------------ day alignment with the device
dev = duckdb.sql(f"""SELECT patient_id, fid AS folder_id, study_date, device_day, monitored_h, full_day,
                            n_minor AS dev_minor_n, n_massive AS dev_massive_n
                     FROM read_parquet('{(T4 / 'reflux_axes_patient_day.parquet').as_posix()}')
                     WHERE in_analysis_set""").df()
dev["study_date"] = pd.to_datetime(dev.study_date).dt.date
al = art.merge(dev, on=["patient_id", "study_date"], how="outer", indicator=True)
al["match"] = al.pop("_merge").map({"both": "both", "left_only": "clinical_only", "right_only": "device_only"})
al = al.sort_values(["patient_id", "study_date"])

# Swap check: correlation of ART (relabelled) and device counts on aligned days.
both = al[al.match == "both"].dropna(subset=["art_minor_n", "art_massive_n"])
chk = {
    "aligned_days": len(both), "patients": both.patient_id.nunique(),
    "rho_minor_vs_minor": stats.spearmanr(both.art_minor_n, both.dev_minor_n).statistic,
    "rho_massive_vs_massive": stats.spearmanr(both.art_massive_n, both.dev_massive_n).statistic,
    "rho_minor_vs_massive (swap hypothesis)": stats.spearmanr(both.art_minor_n, both.dev_massive_n).statistic,
    "sum_art_minor": both.art_minor_n.sum(), "sum_dev_minor": both.dev_minor_n.sum(),
    "sum_art_massive": both.art_massive_n.sum(), "sum_dev_massive": both.dev_massive_n.sum(),
}
offset = both.assign(off=both.clinical_day - both.device_day).groupby("patient_id").off.agg(["min", "max"])

OUT.mkdir(parents=True, exist_ok=True)
OUTCOMES.mkdir(parents=True, exist_ok=True)
base.round(3).to_csv(OUT / "baseline_patient.csv", index=False)
cov.to_parquet(OUT / "patient_day_covariates.parquet", index=False)
al.to_csv(OUT / "day_alignment.csv", index=False)
out_day.to_parquet(OUTCOMES / "patient_day_outcomes.parquet", index=False)
out_pt.to_parquet(OUTCOMES / "outcomes_patient.parquet", index=False)

print("\nalignment:", al.match.value_counts().to_dict())
print("clinical_day - device_day per patient (aligned days):",
      offset.apply(lambda r: f"{r['min']}" if r['min'] == r['max'] else f"{r['min']}..{r['max']}", axis=1)
      .value_counts().to_dict())
print("swap check on aligned days:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in chk.items()})
print(f"\nWrote {OUT} (outcome files in {OUTCOMES.name}/)")
