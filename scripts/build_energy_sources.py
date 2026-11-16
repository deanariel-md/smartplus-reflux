"""Non-enteral energy for the treated arm (energy-source sensitivity).

Source: eCRF export of the treated daily feeding log (config.ECRF_FEEDING_LOG), sheet Patient_Feedi...(A).
 Comment and "specify" free-text columns are not read.

Outputs (outcome side):
  data/clinical/outcomes/iv_hidden_kcal.parquet  patient, study day, daily IV "hidden" kcal (propofol, dextrose, ...)
  data/clinical/outcomes/pn_days.parquet         patient, study day, PN volume (mL), product type and kcal
PN kcal = volume x the product's total energy density from its public datasheet (PN_KCAL_PER_ML below). The
Castor options are n4 / n9 / SMOF; the product behind each is assumed.

Run after build_clinical_tables.py:  python scripts/build_energy_sources.py
"""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
from config import DATA_DIR, ECRF_FEEDING_LOG  # noqa: E402
SRC = ECRF_FEEDING_LOG
OUTCOMES = DATA_DIR / "clinical" / "outcomes"
DAY_START_H = 7
# Total energy (kcal/mL), approximate, from the products' public datasheets:
#   n9   Olimel N9-840(E), Baxter: 1070 kcal/L (NZ data sheet, 17 Sep 2020, section 3)
#   n4   PeriOlimel N4-600E, Baxter: 700 kcal/L (same sheet); Oliclinomel N4-550E would be 610 kcal/L
#   SMOF SmofKabiven (central), Fresenius Kabi: about 1100 kcal per 986 mL bag. Daily volumes up to
#        1.8 L rule out SMOFlipid alone (2.0 kcal/mL, typically 100-500 mL/day).
PN_KCAL_PER_ML = {"n9": 1.07, "n4": 0.70, "SMOF": 1100 / 986}

xl = pd.ExcelFile(SRC)
sheet = next(s for s in xl.sheet_names if s.startswith("Patient_Feedi") and s.endswith("(A)"))
df = pd.read_excel(SRC, sheet_name=sheet)
df["patient_id"] = df["Participant Id"].astype(str).str[-3:]
cols = [str(c) for c in df.columns]


def slots(prefix):
    return sorted({int(m.group(1)) for c in cols if (m := re.match(rf"{prefix}_(\d+)_Date_dd_mm_yyyy_", c))})


def find(prefix, k, pattern):
    hits = [c for c in cols if c.startswith(f"{prefix}_{k}_") and re.search(pattern, c[len(f"{prefix}_{k}_"):])]
    return hits[0] if hits else None


def to_long(prefix, value_pat, extra=None):
    parts = []
    for k in slots(prefix):
        out = pd.DataFrame({"patient_id": df.patient_id,
                            "date": pd.to_datetime(df[f"{prefix}_{k}_Date_dd_mm_yyyy_"], errors="coerce", dayfirst=True),
                            "start_time": df[find(prefix, k, r"^Start_time")].astype(str),
                            "value": pd.to_numeric(df[find(prefix, k, value_pat)], errors="coerce")})
        if extra:
            out["type"] = df[find(prefix, k, extra)].astype("string")
        parts.append(out[out.date.notna()])
    d = pd.concat(parts, ignore_index=True)
    hh = pd.to_numeric(d.start_time.str.extract(r"^(\d{1,2}):")[0], errors="coerce")
    d["study_date"] = (d.date - pd.to_timedelta((hh < DAY_START_H).astype(int), unit="D")).dt.date
    return d.drop(columns="start_time")


cov = pd.read_parquet(DATA_DIR / "clinical" / "patient_day_covariates.parquet")
cov = cov[cov.arm == "treated"][["patient_id", "clinical_day", "study_date"]].assign(
    study_date=lambda x: pd.to_datetime(x.study_date).dt.date)
iv = to_long("Day_IV", r"Daily_IV_kCal").rename(columns={"value": "iv_hidden_kcal"}).merge(
    cov, on=["patient_id", "study_date"], how="left")
pn = to_long("TPN", r"Daily_TPN_ml", extra=r"Select_TPN_type").rename(columns={"value": "pn_ml"}).merge(
    cov, on=["patient_id", "study_date"], how="left")
pn = pn[pn.pn_ml > 0].copy()
pn["pn_kcal"] = pn.pn_ml * pn["type"].map(PN_KCAL_PER_ML)
assert pn.pn_kcal.notna().all(), f"PN type without an energy density: {set(pn['type']) - set(PN_KCAL_PER_ML)}"

OUTCOMES.mkdir(parents=True, exist_ok=True)
iv.to_parquet(OUTCOMES / "iv_hidden_kcal.parquet", index=False)
pn.to_parquet(OUTCOMES / "pn_days.parquet", index=False)
print(f"IV hidden kcal: {len(iv)} patient-days, {iv.patient_id.nunique()} patients, "
      f"{iv.clinical_day.notna().mean():.0%} on a master study day")
print(f"PN: {len(pn)} patient-days in {pn.patient_id.nunique()} patients: {sorted(pn.patient_id.unique())}; "
      f"types {pn['type'].value_counts().to_dict()}; median {pn.pn_kcal.median():.0f} kcal/day")
print(f"Wrote {OUTCOMES}")
