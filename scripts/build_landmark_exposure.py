"""Landmark exposure on clinical study days, with covariates. No outcomes are read.

Exposure window: clinical study days 1-3 (= device days 1-3 except for one patient whose clinical day is offset by one device day).
Eligible: >= 2 full monitored days (>= 12 h) among clinical days 1-3, and a master row for clinical day 4
(still in the study when the outcome window opens).
Burden indices use the pooled full-day cut-points of the primary run (data/t4/cutpoints.json):
  bi3 = frequency + duration + height tertiles (3-9),
  bi2 = frequency + height tertiles (2-6)
A missing height (no events) gets tertile 1.

Run after build_reflux_axes.py and build_clinical_tables.py:
    python scripts/build_landmark_exposure.py
Writes data/analysis/landmark_exposure.csv.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
from config import DATA_DIR  # noqa: E402
T4 = DATA_DIR / "t4"
CL = DATA_DIR / "clinical"
OUT = DATA_DIR / "analysis"

cuts = json.loads((T4 / "cutpoints.json").read_text())["cutpoints_full_days"]
al = pd.read_csv(CL / "day_alignment.csv", dtype={"patient_id": str})
cov = pd.read_parquet(CL / "patient_day_covariates.parquet")
base = pd.read_csv(CL / "baseline_patient.csv", dtype={"patient_id": str})
pdy = pd.read_parquet(T4 / "reflux_axes_patient_day.parquet")
pdy = pdy[pdy.in_analysis_set].assign(study_date=lambda x: pd.to_datetime(x.study_date).dt.date.astype(str))

# Clinical day for every device day (aligned rows), with the device numerators and denominators.
d = al[al.match == "both"][["patient_id", "clinical_day", "study_date"]].merge(
    pdy[["patient_id", "fid", "study_date", "device_day", "monitored_h", "full_day", "n_minor", "n_massive",
         "pct_time_reflux", "pause_h_reflux"]], on=["patient_id", "study_date"])
w = d[(d.clinical_day <= 3) & d.full_day]


def tert(v, c):
    return np.where(v.isna(), 1, 1 + (v > c[0]).astype(int) + (v > c[1]).astype(int))


lm = w.groupby("patient_id").apply(lambda g: pd.Series({
    "fid": g.fid.iloc[0],
    "n_full_d1_3": len(g),
    "monitored_h_d1_3": g.monitored_h.sum(),
    "n_minor_d1_3": g.n_minor.sum(), "n_massive_d1_3": g.n_massive.sum(),
    "episodes_per_h": (g.n_minor + g.n_massive).sum() / g.monitored_h.sum(),
    "pct_time_reflux": (g.pct_time_reflux * g.monitored_h).sum() / g.monitored_h.sum(),
    "massive_share": g.n_massive.sum() / (g.n_minor + g.n_massive).sum() if (g.n_minor + g.n_massive).sum() else np.nan,
    "pause_h_reflux_d1_3": g.pause_h_reflux.sum(),
}), include_groups=False).reset_index()

day4 = set(cov[cov.clinical_day == 4].patient_id)
lm["in_study_day4"] = lm.patient_id.isin(day4)
lm["eligible"] = (lm.n_full_d1_3 >= 2) & lm.in_study_day4
t_f = tert(lm.episodes_per_h, cuts["episodes_per_h"])
t_d = tert(lm.pct_time_reflux, cuts["pct_time_reflux"])
t_h = tert(lm.massive_share, cuts["massive_share"])
lm["bi3"] = t_f + t_d + t_h
lm["bi2"] = t_f + t_h

lm = lm.merge(base[["patient_id", "age", "sex", "apache2", "admission_reason"]], on="patient_id", how="left")
fid = lm.fid.astype(int)
lm["era"] = np.select([fid <= 20, fid <= 38, fid <= 53, fid <= 82], [0, 1, 2, 3], 4)
lm["era_group"] = lm.era.map({0: "0-1", 1: "0-1", 2: "2", 3: "3-4", 4: "3-4"})

OUT.mkdir(parents=True, exist_ok=True)
lm.round(4).to_csv(OUT / "landmark_exposure.csv", index=False)
e = lm[lm.eligible]
print(f"patients with >= 1 full day in clinical days 1-3: {len(lm)}; eligible: {len(e)}")
print("not eligible:", lm[~lm.eligible][["patient_id", "n_full_d1_3", "in_study_day4"]].to_dict("records"))
print("bi3:", e.bi3.value_counts().sort_index().to_dict(), " bi2:", e.bi2.value_counts().sort_index().to_dict())
print(f"Spearman bi2 vs bi3 (eligible): {e[['bi2', 'bi3']].corr('spearman').iloc[0, 1]:.2f}")
print(f"Wrote {OUT / 'landmark_exposure.csv'}")
