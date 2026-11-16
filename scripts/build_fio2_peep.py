"""Hourly FiO2 and PEEP from the hospital-record exports (source of the trial's VAE endpoint).

Source (read in memory from the zips; nothing extracted to disk), in config.FIO2_PEEP_DIR:
    {Treatment,Control} patients.zip
One folder per patient (study number), one workbook per parameter. Layouts differ between files; some carry a
patient-name column and a hospital number. Only the timestamp and the parameter value are kept, and no text
cell is ever read into the output: the time column is the datetime column with the most distinct values, the
value column is the numeric column whose values are plausible for the parameter (FiO2 21-100 %, or 0.21-1;
PEEP 0-30 cmH2O), and where a label column exists only rows labelled with the parameter are kept.

CDC ventilator-associated condition (VAC), from daily minima (calendar days, as CDC; daily minimum PEEP of 0-5
cmH2O counts as 5): >= 2 days of stable or falling daily minimum PEEP and FiO2, then >= 2 days with the daily minimum PEEP >= 3 cmH2O or FiO2 >= 20 points
above the lowest baseline value. Onset = first day of worsening.

Writes data/clinical/outcomes/fio2_peep_hourly.parquet and data/clinical/outcomes/vac_events.csv.
"""
import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
from config import DATA_DIR, FIO2_PEEP_DIR  # noqa: E402
SRC = FIO2_PEEP_DIR
OUT = DATA_DIR / "clinical" / "outcomes"
RANGE = {"fio2": (0.2, 100), "peep": (0, 30)}


def parse(raw, param, pid):
    d = pd.read_excel(io.BytesIO(raw), sheet_name=0, header=None)
    is_dt = d.map(lambda v: isinstance(v, (pd.Timestamp, np.datetime64)) or hasattr(v, "year"))
    dt_cols = [j for j in d.columns if is_dt[j].mean() > 0.3]
    if not dt_cols:
        return None
    tcol = max(dt_cols, key=lambda j: d.loc[is_dt[j], j].nunique())
    lab_rows = None
    for j in d.columns:                                   # parameter label column, if any (labels only, no names)
        s = d[j].map(lambda v: v.strip().lower() if isinstance(v, str) else None)
        hit = s.str.match(r"^(peep|fio2|fi0|ins)", na=False)
        if hit.mean() > 0.3:
            lab_rows = s.str.match(r"^pee" if param == "peep" else r"^(fio|fi0|ins)", na=False)
    best, score = None, -1
    lo, hi = RANGE[param]
    for j in d.columns:
        if j == tcol:
            continue
        v = pd.to_numeric(d[j].where(~d[j].map(lambda x: isinstance(x, str))), errors="coerce")
        if v.notna().sum() < 5 or (v.dropna() == int(pid)).all():     # the study-number column
            continue
        ok = v.dropna().between(lo, hi).mean()
        if ok > score:
            best, score = v, ok
    if best is None or score < 0.5:
        return None
    out = pd.DataFrame({"t": pd.to_datetime(d[tcol].where(is_dt[tcol]), errors="coerce"), "value": best})
    if lab_rows is not None:
        out = out[lab_rows.to_numpy()]
    out = out.dropna()
    if param == "fio2" and out.value.median() <= 1:
        out["value"] *= 100
    return out[out.value.between(*((21, 100) if param == "fio2" else (0, 30)))]


rows, skipped = [], []
for arm, zn in (("treated", "Treatment patients.zip"), ("control", "Control patients.zip")):
    z = zipfile.ZipFile(SRC / zn)
    for info in z.infolist():
        if info.is_dir() or not info.filename.lower().endswith(".xlsx"):
            continue
        parts = info.filename.split("/")
        pid = re.sub(r"\D", "", parts[-2]).zfill(3)
        name = parts[-1].lower()
        param = "fio2" if re.search(r"fio|fi0", name) else "peep" if "peep" in name else None
        if param is None:
            skipped.append(pid)
            continue
        p = parse(z.read(info), param, pid)
        if p is None or p.empty:
            skipped.append(f"{pid}:{param}")
            continue
        rows.append(p.assign(patient_id=pid, arm=arm, param=param))
h = pd.concat(rows, ignore_index=True).drop_duplicates(["patient_id", "param", "t"])
h = h.sort_values(["patient_id", "param", "t"])[["patient_id", "arm", "param", "t", "value"]]
h.to_parquet(OUT / "fio2_peep_hourly.parquet", index=False)
print(f"patients: {h.patient_id.nunique()} ({h.groupby('arm').patient_id.nunique().to_dict()}); rows {len(h)}; "
      f"files skipped: {skipped}")
print(h.groupby("param").value.describe().round(1))

# ---------------------------------------------------------------- CDC VAC from daily minima
dm = (h.assign(day=h.t.dt.normalize()).groupby(["patient_id", "arm", "param", "day"]).value.min()
      .unstack("param").reset_index().sort_values(["patient_id", "day"]))
ev = []
for (pid, arm), g in dm.groupby(["patient_id", "arm"]):
    g = g.set_index("day").asfreq("D")
    pe, fi = np.maximum(g.peep.to_numpy(), 5), g.fio2.to_numpy()     # CDC: daily minimum PEEP 0-5 counts as 5
    days = g.index
    for i in range(2, len(g) - 1):
        b_pe, b_fi = pe[i - 2:i], fi[i - 2:i]
        if np.isnan(b_pe).any() or np.isnan(b_fi).any():
            continue
        if b_pe[1] > b_pe[0] or b_fi[1] > b_fi[0]:            # baseline must be stable or falling
            continue
        w_pe, w_fi = pe[i:i + 2], fi[i:i + 2]
        if (w_pe >= b_pe.min() + 3).all() or (w_fi >= b_fi.min() + 20).all():
            ev.append({"patient_id": pid, "arm": arm, "vac_date": days[i].date(),
                       "by": "PEEP" if (w_pe >= b_pe.min() + 3).all() else "FiO2"})
            break                                            # first VAC per patient
vac = pd.DataFrame(ev)
vac.to_csv(OUT / "vac_events.csv", index=False)
n = h.groupby("arm").patient_id.nunique()
print(f"\nVAC (first per patient): {vac.groupby('arm').size().to_dict()} of {n.to_dict()}")
print(f"Wrote {OUT / 'fio2_peep_hourly.parquet'} and vac_events.csv")
