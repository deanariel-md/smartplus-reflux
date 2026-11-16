"""Does reflux rise before extubation ("the gut wakes before the lungs")?

Extubation time: the last device minute with valid REE (invasive ventilation with the flow sensor in the circuit) in
patients extubated alive per the trial record. Device monitoring usually ends within minutes of it (median 0.4 h).
Checked against the trial's ventilation days.

  A  days before extubation, observed / expected: expected = the device-day-specific rate on full ventilated days
     that are not within 5 days before an extubation (all patients); patient bootstrap
  B  within patients: windows -4..-3 and -2..-1 days before the extubation day vs the patient's days >= 5 days
     before, Poisson GLMM adjusted for device day
  C  readiness: on each full ventilated day from device day 3, is extubation in the next 1-2 days predicted by the
     change in reflux vs the patient's previous days, the absolute rate, the change in daily FiO2, or device day? AUC
     with patient bootstrap

Writes results/explore/15_extubation_alignment.csv.
"""
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
from config import DATA_DIR, OUT_DIR  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))
import glmm_poisson as gp  # noqa: E402

OUT, CL, T2 = OUT_DIR / "explore", DATA_DIR / "clinical", DATA_DIR / "t2" / "minute"
rng = np.random.default_rng(3)

fr = pd.read_parquet(OUT / "frame_patient_day.parquet")
op = pd.read_parquet(CL / "outcomes" / "outcomes_patient.parquet")
base = pd.read_csv(CL / "baseline_patient.csv", dtype={"patient_id": str}, parse_dates=["first_study_day_start"])
v = duckdb.sql(f"""SELECT right(folder_id, 3) AS fid, max(minute_pc) AS t_last,
                          max(minute_pc) FILTER (ree_valid_frac >= 0.5) AS t_last_vent,
                          min(minute_pc) FILTER (ree_valid_frac >= 0.5) AS t_first_vent
                   FROM read_parquet('{T2.as_posix()}/*.parquet') GROUP BY 1""").df()
pid = fr[["fid", "patient_id"]].drop_duplicates()
ex = v.merge(pid, on="fid").merge(op[op.arm == "treated"], on="patient_id").merge(
    base[["patient_id", "first_study_day_start", "hours_icu_to_randomization"]], on="patient_id")
ex["ext"] = ~ex.icu_death.astype(bool)
ex["dev_vent_days"] = (ex.t_last_vent - ex.first_study_day_start).dt.total_seconds() / 86400
r = stats.spearmanr(ex.dev_vent_days, ex.ventilation_days, nan_policy="omit")
print(f"device time from first study day to last ventilated minute vs trial ventilation days: rho {r.statistic:.2f}; "
      f"median difference {np.nanmedian(ex.ventilation_days - ex.dev_vent_days):.1f} d")
ex["ext_date"] = (ex.t_last_vent - pd.Timedelta(hours=7)).dt.date.astype(str)
ext = ex[ex.ext].set_index("patient_id")
print(f"patients extubated alive with a device extubation time: {len(ext)}; died: {int((~ex.ext).sum())}")

d = fr[fr.full_day & fr.in_analysis_set].copy()
d["day_cat"] = np.minimum(d.device_day, 8).astype(int)
d["rel"] = np.nan
m = d.patient_id.isin(ext.index)
d.loc[m, "rel"] = (pd.to_datetime(d.loc[m, "study_date"]) - pd.to_datetime(d.loc[m, "patient_id"].map(ext.ext_date))).dt.days
d = d[~(d.rel > 0)]                                                     # nothing after the extubation day
rows = []

# ---------------------------------------------------------------- A observed / expected
ref = d[~d.rel.between(-5, 0)]                                          # not within 5 days before an extubation
rate_dd = ref.groupby("day_cat").apply(lambda g: g.n_all.sum() / g.monitored_h.sum(), include_groups=False)
d["exp"] = d.day_cat.map(rate_dd) * d.monitored_h
W = {"-7..-6": (-7, -6), "-5..-4": (-5, -4), "-3..-2": (-3, -2), "-1": (-1, -1), "0 (extubation day)": (0, 0)}
for wn, (a, b) in W.items():
    g = d[d.rel.between(a, b)]
    pts = g.patient_id.unique()
    oe = g.n_all.sum() / g.exp.sum()
    bs = []
    for _ in range(2000):
        s = pd.concat([g[g.patient_id == p] for p in rng.choice(pts, len(pts))])
        bs.append(s.n_all.sum() / s.exp.sum())
    rows.append({"analysis": "A observed/expected vs days not before an extubation", "window": wn, "patients": len(pts),
                 "days": len(g), "estimate": oe, "lo": np.quantile(bs, .025), "hi": np.quantile(bs, .975)})

# ---------------------------------------------------------------- B within patients
w = d[d.patient_id.isin(ext.index)].copy()
w["win"] = np.select([w.rel.between(-2, -1), w.rel.between(-4, -3), w.rel == 0], ["b -2..-1", "a -4..-3", "c day 0"],
                     np.where(w.rel <= -5, "0 ref <= -5", "x"))
w = w[w.win != "x"]
X, names = gp.design(w, ["win:C", "day_cat:C"])
res = gp.fit(w.n_all.to_numpy(), X, np.log(w.monitored_h.to_numpy()), w.patient_id.to_numpy(), names).table()
print(f"\nB within patients: {w.patient_id.nunique()} patients, {int((w.win == '0 ref <= -5').sum())} reference days in "
      f"{w[w.win == '0 ref <= -5'].patient_id.nunique()} patients")
for _, q in res[res.term.astype(str).str.startswith("win")].iterrows():
    rows.append({"analysis": "B within patients vs own days <= -5 (device-day adjusted)", "window": q.term,
                 "patients": w.patient_id.nunique(), "days": len(w), "estimate": q.RR, "lo": q.RR_lo, "hi": q.RR_hi})

# ---------------------------------------------------------------- C readiness
ox = pd.read_parquet(CL / "outcomes" / "fio2_peep_hourly.parquet")
ox["study_date"] = (ox.t - pd.Timedelta(hours=7)).dt.date.astype(str)
fio2 = ox[ox.param == "fio2"].groupby(["patient_id", "study_date"]).value.mean().rename("fio2").reset_index()
c = fr[fr.full_day & fr.in_analysis_set].merge(fio2, on=["patient_id", "study_date"], how="left").sort_values(
    ["patient_id", "device_day"]).copy()
c["rel"] = (pd.to_datetime(c.study_date) - pd.to_datetime(c.patient_id.map(ext.ext_date))).dt.days
c["lr"] = np.log(c.episodes_per_h + 0.25)
c["base_lr"] = c.groupby("patient_id").lr.transform(lambda s: s.expanding().median().shift(1))
c["jump"] = c.lr - c.base_lr
c["base_fio2"] = c.groupby("patient_id").fio2.transform(lambda s: s.expanding().median().shift(1))
c["fio2_drop"] = c.base_fio2 - c.fio2
c = c[(c.device_day >= 3) & c.base_lr.notna() & ~(c.rel >= 0)]         # ventilated days before any extubation
c["y"] = c.rel.between(-2, -1).astype(int)
print(f"\nC readiness: {len(c)} patient-days in {c.patient_id.nunique()} patients; extubation within 1-2 days on "
      f"{int(c.y.sum())} days ({c[c.y == 1].patient_id.nunique()} patients)")


def auc(df, col):
    a = roc_auc_score(df.y, df[col])
    pts, bs = df.patient_id.unique(), []
    for _ in range(1000):
        s = pd.concat([df[df.patient_id == p] for p in rng.choice(pts, len(pts))])
        if s.y.nunique() == 2:
            bs.append(roc_auc_score(s.y, s[col]))
    return a, np.quantile(bs, .025), np.quantile(bs, .975)


for col, lab in (("jump", "reflux change vs own baseline"), ("lr", "absolute reflux rate"),
                 ("fio2_drop", "FiO2 fall vs own baseline"), ("device_day", "device day")):
    q = c.dropna(subset=[col])
    a, lo, hi = auc(q, col)
    rows.append({"analysis": "C readiness: extubation within 1-2 days, AUC", "window": lab, "patients": q.patient_id.nunique(),
                 "days": len(q), "estimate": a, "lo": lo, "hi": hi})
out = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(out.round(3).to_string(index=False))
out.round(4).to_csv(OUT / "15_extubation_alignment.csv", index=False)
