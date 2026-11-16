"""Robustness of "reflux rises before a new respiratory infection" (explore_infection.py event study).

1  placebo: patients without a respiratory infection get pseudo-onsets drawn from the observed onset-day
   distribution; the within-patient RR for days -3..-2 vs > 3 days before is re-estimated (N_PERM draws).
   If the real RR is an artefact of the time structure (rising reflux over days), placebo RRs will be as large.
2  minor vs massive episodes separately; linear device-day adjustment instead of categories.

Writes results/explore/7_infection_robustness.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
from config import DATA_DIR, OUT_DIR  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))
import glmm_poisson as gp  # noqa: E402

OUT = OUT_DIR / "explore"
N_PERM = 200
rng = np.random.default_rng(11)

fr = pd.read_parquet(OUT / "frame_patient_day.parquet")
fr = fr[fr.full_day].copy()
ae = pd.read_csv(DATA_DIR / "clinical" / "outcomes" / "adverse_events_ctr.csv", dtype={"patient_id": str},
                 parse_dates=["date"])
base = pd.read_csv(DATA_DIR / "clinical" / "baseline_patient.csv", dtype={"patient_id": str},
                   parse_dates=["first_study_day_start"])
ae = ae.merge(base[base.arm == "treated"][["patient_id", "first_study_day_start"]], on="patient_id")
ae["study_day"] = (ae.date.dt.normalize() - ae.first_study_day_start.dt.normalize()).dt.days + 1
onset = ae[ae.resp].groupby("patient_id").study_day.min()


def within_rr(f, y="n_all", day_term="day_cat:C"):
    f = f.copy()
    f["win"] = np.select([f.rel.between(-3, -2), f.rel.between(-1, 0), f.rel >= 1], ["a", "b", "c"], "0")
    f = f[f.onset.notna()]
    if (f.win == "0").sum() < 5 or (f.win == "a").sum() < 5:
        return np.nan
    X, names = gp.design(f, ["win:C", day_term])
    t = gp.fit(f[y], X, np.log(f.monitored_h.to_numpy()), f.patient_id, names).table().set_index("term")
    return t.loc["win[a]", "RR"], t.loc["win[a]", "RR_lo"], t.loc["win[a]", "RR_hi"], t.loc["win[a]", "p"]


fr["day_cat"] = np.minimum(fr.device_day, 8).astype(int)
fr["dday"] = fr.device_day.astype(float)
real = fr.assign(onset=fr.patient_id.map(onset))
real["rel"] = real.clinical_day - real.onset
rows = []
for label, y, dterm in (("all episodes, day categories", "n_all", "day_cat:C"),
                        ("all episodes, linear device day", "n_all", "dday"),
                        ("minor episodes", "n_minor", "day_cat:C"),
                        ("massive episodes", "n_massive", "day_cat:C")):
    rr, lo, hi, p = within_rr(real, y, dterm)
    rows.append({"analysis": label, "RR_days_-3..-2_vs_earlier": rr, "lo": lo, "hi": hi, "p": p})

# Placebo: pseudo-onsets for never-infected patients, drawn from the real onset-day distribution,
# restricted to days the patient was monitored.
never = sorted(set(fr.patient_id) - set(onset.index))
obs_days = onset[onset.index.isin(fr.patient_id)].to_numpy()
perm = []
for _ in range(N_PERM):
    ps = {}
    for pid in never:
        days = fr[fr.patient_id == pid].clinical_day
        cand = obs_days[(obs_days >= days.min() + 4) & (obs_days <= days.max())]
        if len(cand):
            ps[pid] = rng.choice(cand)
    f = fr[fr.patient_id.isin(ps)].assign(onset=lambda x: x.patient_id.map(ps))
    f["rel"] = f.clinical_day - f.onset
    r = within_rr(f)
    if isinstance(r, tuple):
        perm.append(r[0])
perm = np.array(perm)
real_rr = rows[0]["RR_days_-3..-2_vs_earlier"]
rows.append({"analysis": f"placebo onsets in {len(never)} never-infected patients ({len(perm)} draws): median RR",
             "RR_days_-3..-2_vs_earlier": np.median(perm), "lo": np.quantile(perm, 0.025), "hi": np.quantile(perm, 0.975),
             "p": (perm >= real_rr).mean()})
res = pd.DataFrame(rows)
print(res.round(3).to_string(index=False))
print("(placebo row: 'p' = share of placebo RRs >= the real RR)")
res.round(4).to_csv(OUT / "7_infection_robustness.csv", index=False)
