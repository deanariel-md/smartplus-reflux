"""Does early reflux add to standard predictors of extubation? Harrell's C for Cox models of time to extubation alive
(ICU death censored), with and without the log2 day 1-3 episode rate, and an optimism-corrected bootstrap.

Models: APACHE II; APACHE II + mean FiO2 days 1-3; each + log2 early rate. n = 41 (as explore_ventilation.py).
Writes results/explore/13_extubation_discrimination.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

ROOT = Path(__file__).resolve().parents[1]
from config import DATA_DIR, OUT_DIR  # noqa: E402
OUT, CL = OUT_DIR / "explore", DATA_DIR / "clinical"
rng = np.random.default_rng(11)

le = pd.read_csv(DATA_DIR / "analysis" / "landmark_exposure.csv", dtype={"patient_id": str})
le = le[le.n_full_d1_3 >= 2]
op = pd.read_parquet(CL / "outcomes" / "outcomes_patient.parquet")
base = pd.read_csv(CL / "baseline_patient.csv", dtype={"patient_id": str}, parse_dates=["first_study_day_start"])
ox = pd.read_parquet(CL / "outcomes" / "fio2_peep_hourly.parquet").merge(base[["patient_id", "first_study_day_start"]],
                                                                       on="patient_id")
ox["cday"] = (ox.t - ox.first_study_day_start.dt.normalize() - pd.Timedelta(hours=7)).dt.total_seconds() // 86400 + 1
f13 = ox[ox.cday.between(1, 3) & (ox.param == "fio2")].groupby("patient_id").value.mean().rename("fio2")
x = (le.merge(op[op.arm == "treated"], on="patient_id")
     .merge(f13.reset_index(), on="patient_id", how="left"))
x["reflux"] = np.log2(x.episodes_per_h + 0.1)
x["T"], x["E"] = x.ventilation_days, (~x.icu_death.astype(bool)).astype(int)
x = x.dropna(subset=["apache2", "fio2", "reflux"])
MODELS = {"APACHE II": ["apache2"], "APACHE II + reflux": ["apache2", "reflux"],
          "APACHE II + FiO2": ["apache2", "fio2"], "APACHE II + FiO2 + reflux": ["apache2", "fio2", "reflux"],
          "reflux alone": ["reflux"]}


def cindex(train, test, cols):
    m = CoxPHFitter().fit(train[cols + ["T", "E"]], "T", "E")
    return concordance_index(test["T"], -m.predict_partial_hazard(test[cols]), test["E"])   # higher hazard, earlier


rows = []
app = {k: cindex(x, x, c) for k, c in MODELS.items()}
boot = {k: [] for k in MODELS}
delta = {"+ reflux over APACHE II": [], "+ reflux over APACHE II + FiO2": []}
for _ in range(500):
    b = x.sample(len(x), replace=True, random_state=int(rng.integers(1e9)))
    if b.E.nunique() < 2:
        continue
    opt = {}
    for k, c in MODELS.items():
        opt[k] = cindex(b, b, c) - cindex(b, x, c)                     # optimism (Harrell)
        boot[k].append(cindex(b, b, c))
    delta["+ reflux over APACHE II"].append((cindex(b, x, MODELS["APACHE II + reflux"]) - cindex(b, x, MODELS["APACHE II"])))
    delta["+ reflux over APACHE II + FiO2"].append(cindex(b, x, MODELS["APACHE II + FiO2 + reflux"])
                                                     - cindex(b, x, MODELS["APACHE II + FiO2"]))
    for k in MODELS:
        boot.setdefault(k + "_opt", []).append(opt[k])
for k in MODELS:
    o = np.mean(boot[k + "_opt"])
    rows.append({"model": k, "n": len(x), "events": int(x.E.sum()), "C_apparent": app[k], "optimism": o,
                 "C_corrected": app[k] - o})
for k, v in delta.items():
    rows.append({"model": f"delta C {k} (bootstrap, tested on original)", "n": len(x), "events": int(x.E.sum()),
                 "C_apparent": np.mean(v), "optimism": np.quantile(v, 0.025), "C_corrected": np.quantile(v, 0.975)})
r = pd.DataFrame(rows)
print(r.round(3).to_string(index=False))
print("(for delta rows: C_apparent = mean delta, optimism/C_corrected columns = 2.5%/97.5% bootstrap quantiles)")
r.round(4).to_csv(OUT / "13_extubation_discrimination.csv", index=False)
