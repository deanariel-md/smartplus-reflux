"""Days with a documented feeding-tube or sensor fault, from ART's device-deficiency export.

Source: the manufacturer's device-deficiency export (config.DEVICE_DEFICIENCY_EXPORT), study-site rows only.
Complaints cover Apr 2021 - Apr 2022 (the later software eras); earlier eras have none.

A complaint counts as a detection fault when its summary or description mentions the tube's sensors,
impedance, position (out of position / out of body / folded), or exposed or loose sensor parts.
Feeding-set, urineFlow, drainage, REE and pump complaints are not detection faults. "The patient vomited"
is real reflux, not a fault.
Complaints without a patient number are assigned to every device folder recording on that date.
Excluded study dates: the complaint date and the day before (a fault before 07:00 belongs to the previous
study day).

Run after build_reflux_axes.py:  python scripts/device_fault_days.py
Writes data/validation/device_fault_days.csv (fid, study_date, reason) for
build_reflux_axes.py --exclude-days, and results/descriptive/tables/14_device_fault_days.csv (reflux on
fault days vs the same patient's other full days, with sign tests).
"""
from pathlib import Path

import duckdb
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
from config import DATA_DIR, OUT_DIR, DEVICE_DEFICIENCY_EXPORT  # noqa: E402
SRC = DEVICE_DEFICIENCY_EXPORT
OUT = DATA_DIR / "validation" / "device_fault_days.csv"
TAB = OUT_DIR / "descriptive" / "tables" / "14_device_fault_days.csv"
FAULT = (r"sensor|impedanc|\bZ\d|electrod|out[- ]of[- ](?:body|position)|\bOOP\b|folded|wire.*expos|"
         r"swollen|stood out|not fastened|recogni[sz]e correct position")

df = pd.read_excel(SRC, sheet_name="Data")
df = df[df.Site.astype(str).str.contains("Belinson", case=False)].copy()
NOT_TUBE = r"(?:REE )?flow sensor|co2 sensor|urine ?flow"   # respiratory and urine sensors, not the feeding tube
text = (df.Summary.astype(str) + " " + df.Description.astype(str)).str.replace(NOT_TUBE, "", case=False, regex=True)
df = df[text.str.contains(FAULT, case=False, regex=True)
        & ~df.Summary.astype(str).str.contains(NOT_TUBE, case=False, regex=True)].copy()
df["date"] = pd.to_datetime(df["Occured On"]).dt.date
df["patient_id"] = df["Patient "].astype(str).str.extract(r"(\d{3})")[0]
df["reason"] = df.Summary.astype(str).str.replace(r"\s+", " ", regex=True).str.strip().str[:90]

mp = pd.read_csv(DATA_DIR / "folder_to_patient.csv", dtype=str)
mp["fid"], mp["pid"] = mp.folder_id.str[-3:], mp.patient_id.str[-3:]
pdy = duckdb.sql(f"""SELECT fid, study_date, monitored_h, full_day, episodes_per_h, pct_time_reflux, in_analysis_set
                     FROM read_parquet('{(DATA_DIR / 't4' / 'reflux_axes_patient_day.parquet').as_posix()}')""").df()
pdy["study_date"] = pd.to_datetime(pdy.study_date).dt.date

rows = []
for r in df.itertuples():
    if isinstance(r.patient_id, str):
        fids = list(mp[mp.pid == r.patient_id].fid)
        how = "patient"
    else:
        fids = sorted(pdy[(pdy.study_date == r.date) & (pdy.monitored_h > 0)].fid.unique())
        how = "date"
    for f in fids:
        for d in (r.date - pd.Timedelta(days=1), r.date):
            rows.append({"fid": f, "study_date": d, "complaint_date": r.date, "assigned_by": how, "reason": r.reason})
fd = pd.DataFrame(rows).drop_duplicates(["fid", "study_date"])
OUT.parent.mkdir(parents=True, exist_ok=True)
fd.to_csv(OUT, index=False)

# Reflux on fault days vs the same patient's other full days.
x = pdy[pdy.in_analysis_set & pdy.full_day].merge(fd[["fid", "study_date"]].assign(fault=True),
                                                  on=["fid", "study_date"], how="left")
x["fault"] = x.fault.astype("boolean").fillna(False).astype(bool)
cmp = x.groupby(["fid", "fault"])[["episodes_per_h", "pct_time_reflux"]].median().unstack("fault")
cmp = cmp.dropna()
print(f"{len(df)} detection-fault complaints -> {len(fd)} folder-days in {fd.fid.nunique()} folders "
      f"({(fd.assigned_by == 'date').sum()} folder-days assigned by date only)")
print(fd.groupby("fid").reason.first().to_string())
print("\nMedian episodes/h and % time on fault days vs other full days (patients with both):")
print(cmp.round(2).to_string())

up = int((cmp[("episodes_per_h", True)] > cmp[("episodes_per_h", False)]).sum())
up_t = int((cmp[("pct_time_reflux", True)] > cmp[("pct_time_reflux", False)]).sum())
summ = pd.DataFrame([{
    "complaints": len(df), "folder_days": len(fd), "folders": fd.fid.nunique(),
    "folder_days_assigned_by_date": int((fd.assigned_by == "date").sum()),
    "patients_compared": len(cmp),
    "higher_rate_on_fault_days": f"{up}/{len(cmp)}",
    "sign_test_p_rate": round(stats.binomtest(up, len(cmp)).pvalue, 3),
    "higher_pct_time_on_fault_days": f"{up_t}/{len(cmp)}",
    "sign_test_p_pct_time": round(stats.binomtest(up_t, len(cmp)).pvalue, 3)}])
TAB.parent.mkdir(parents=True, exist_ok=True)
summ.to_csv(TAB, index=False)
print(summ.T.to_string(header=False))
print(f"\nWrote {OUT} and {TAB}")
