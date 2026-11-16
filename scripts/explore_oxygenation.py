"""Oxygenation (hospital-record FiO2 and PEEP, build_fio2_peep.py) and reflux.

A  Does oxygenation worsen around the culture-based respiratory-infection onset (explore_infection.py)? Daily
   mean FiO2, daily minimum FiO2 and PEEP by window (infected patients, device-day categories, mixed model).
B  Reflux around an objective CDC ventilator-associated condition (VAC: sustained rise in daily minimum PEEP or
   FiO2 after a stable baseline; no cultures involved). Same windows, within patients with a VAC, Poisson GLMM.
C  Hour level: does reflux in one 6-h block go with a rise in FiO2 over the next 6 h (and the reverse)?
   Within-patient deviations, ventilated hours, linear mixed model; lags in 6-h steps from -24 to +24 h.
D  Early reflux -> faster extubation (explore_ventilation.py), adjusted for mean FiO2 and PEEP on days 1-3.

Writes results/explore/10_oxygenation_infection.csv, 10_surge_adjusted_oxygenation.csv, 10_reflux_around_vac.csv,
10_reflux_fio2_lags.csv and 10_extubation_oxygenation.csv.
"""
import sys
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
from config import DATA_DIR, OUT_DIR  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))
import glmm_poisson as gp  # noqa: E402

OUT, CL = OUT_DIR / "explore", DATA_DIR / "clinical"
T2, T3, T4 = DATA_DIR / "t2" / "minute", DATA_DIR / "t3", DATA_DIR / "t4"
WIN = ["a -3..-2", "b -1..0", "c +1..+2", "d >=+3"]

ox = pd.read_parquet(CL / "outcomes" / "fio2_peep_hourly.parquet")
ox["study_date"] = (ox.t - pd.Timedelta(hours=7)).dt.date.astype(str)
daily = ox.pivot_table(index=["patient_id", "study_date"], columns="param", values="value",
                       aggfunc=["mean", "min"]).reset_index()
daily.columns = ["patient_id", "study_date", "fio2_mean", "peep_mean", "fio2_min", "peep_min"]

base = pd.read_csv(CL / "baseline_patient.csv", dtype={"patient_id": str}, parse_dates=["first_study_day_start"])
ae = pd.read_csv(CL / "outcomes" / "adverse_events_ctr.csv", dtype={"patient_id": str}, parse_dates=["date"])
ae = ae.merge(base[base.arm == "treated"][["patient_id", "first_study_day_start"]], on="patient_id")
ae["study_day"] = (ae.date.dt.normalize() - ae.first_study_day_start.dt.normalize()).dt.days + 1
resp_onset = ae[ae.resp].groupby("patient_id").study_day.min()
vac = pd.read_csv(CL / "outcomes" / "vac_events.csv", dtype={"patient_id": str}, parse_dates=["vac_date"])
vac = vac[vac.arm == "treated"].merge(base[["patient_id", "first_study_day_start"]], on="patient_id")
vac_onset = vac.set_index("patient_id").pipe(
    lambda d: (d.vac_date - d.first_study_day_start.dt.normalize()).dt.days + 1)

fr = pd.read_parquet(OUT / "frame_patient_day.parquet")
fr = fr[fr.full_day].merge(daily, on=["patient_id", "study_date"], how="left")
fr["day_cat"] = np.minimum(fr.device_day, 8).astype(int)
print(f"full days with FiO2: {fr.fio2_mean.notna().sum()} of {len(fr)}")


def windows(f, onset):
    f = f.assign(onset=f.patient_id.map(onset))
    f["rel"] = f.clinical_day - f.onset
    f["win"] = np.select([f.rel.between(-3, -2), f.rel.between(-1, 0), f.rel.between(1, 2), f.rel >= 3], WIN, "0 ref")
    return f[f.onset.notna()]


# ---------------------------------------------------------------- A
inf = windows(fr, resp_onset)
rows = []
for y in ("fio2_mean", "fio2_min", "peep_min"):
    g = inf.dropna(subset=[y])
    m = smf.mixedlm(f"{y} ~ C(win, Treatment('0 ref')) + C(day_cat)", g, groups=g.patient_id).fit()
    ci = m.conf_int()
    for w in WIN:
        k = f"C(win, Treatment('0 ref'))[T.{w}]"
        rows.append({"measure": y, "window": w, "change": m.params[k], "lo": ci.loc[k, 0], "hi": ci.loc[k, 1],
                     "p": m.pvalues[k], "ref_mean": g[g.win == "0 ref"][y].mean(), "patients": g.patient_id.nunique()})
a = pd.DataFrame(rows)
print("\nA. Oxygenation around respiratory-infection onset (infected patients, vs > 3 days before)\n",
      a.round(2).to_string(index=False))
a.round(4).to_csv(OUT / "10_oxygenation_infection.csv", index=False)
nv = fr[~fr.patient_id.isin(resp_onset.index)]
same = nv[nv.day_cat.isin(inf[inf.win == "0 ref"].day_cat.unique())]
print("mean FiO2 / PEEP: infected, reference days "
      f"{inf[inf.win == '0 ref'][['fio2_mean', 'peep_mean']].mean().round(1).to_dict()}; -3..-2 "
      f"{inf[inf.win == 'a -3..-2'][['fio2_mean', 'peep_mean']].mean().round(1).to_dict()}; never infected, same device "
      f"days {same[['fio2_mean', 'peep_mean']].mean().round(1).to_dict()}")

# reflux surge adjusted for oxygenation (and sedative agents)
SED = ["midazolam", "propofol", "dexmedetomidine", "ketamine"]
inf = inf.assign(fio2_10=inf.fio2_mean / 10, **{c: inf[c].astype(float) for c in SED})
sa = []
for label, extra in (("reference", []), ("+ FiO2 + PEEP", ["fio2_10", "peep_mean"]),
                     ("+ FiO2 + PEEP + sedative agents", ["fio2_10", "peep_mean"] + SED)):
    g = inf.dropna(subset=extra)
    X, names = gp.design(g, ["win:C", "day_cat:C"] + extra)
    t = gp.fit(g.n_all, X, np.log(g.monitored_h.to_numpy()), g.patient_id, names).table().set_index("term")
    sa.append({"model": label, "days": len(g), **{f"{k}_{w[:8].strip()}": t.loc[f"win[{w}]", c] for w in WIN[:2]
                                                 for k, c in (("RR", "RR"), ("lo", "RR_lo"), ("hi", "RR_hi"))}})
sa = pd.DataFrame(sa)
print("\nReflux surge adjusted for oxygenation\n", sa.round(2).to_string(index=False))
sa.round(4).to_csv(OUT / "10_surge_adjusted_oxygenation.csv", index=False)

# ---------------------------------------------------------------- B
vw = windows(fr, vac_onset)
print(f"\nB. VAC in treated patients: {len(vac_onset)}; with monitored full days: {vw.patient_id.nunique()}; "
      f"window days {vw.win.value_counts().sort_index().to_dict()}")
b = []
if vw.patient_id.nunique() >= 3 and (vw.win == "0 ref").sum() >= 3:
    X, names = gp.design(vw, ["win:C", "day_cat:C"])
    t = gp.fit(vw.n_all, X, np.log(vw.monitored_h.to_numpy()), vw.patient_id, names).table()
    for _, r in t[t.term.str.startswith("win")].iterrows():
        b.append({"window": r.term[4:-1], "RR": r.RR, "lo": r.RR_lo, "hi": r.RR_hi, "p": r.p})
pp = vw.groupby(["patient_id", "win"]).apply(lambda g: g.n_all.sum() / g.monitored_h.sum(), include_groups=False)
b = pd.DataFrame(b)
print(b.round(2).to_string(index=False))
print("per-patient rate by window\n", pp.unstack().round(2).to_string())
both = set(vac_onset.index) & set(resp_onset.index)
print(f"VAC patients who also had a respiratory isolate: {len(both)} of {len(vac_onset)}; "
      f"VAC day minus first isolate day: {sorted((vac_onset[list(both)] - resp_onset[list(both)]).tolist())}")
b.round(4).to_csv(OUT / "10_reflux_around_vac.csv", index=False)

# ---------------------------------------------------------------- C: hour level, 6-h blocks
mp = pd.read_csv(DATA_DIR / "folder_to_patient.csv", dtype=str)
mp["fid"], mp["patient_id"] = mp.folder_id.str[-3:], mp.patient_id.str[-3:]
hh = duckdb.sql(f"""
    WITH m AS (SELECT right(m.folder_id, 3) AS fid, date_trunc('hour', m.minute_pc) AS hr, m.monitor_s
               FROM read_parquet('{T2.as_posix()}/*.parquet') m
               LEFT JOIN read_parquet('{(T4 / 'artefact_minute_list.parquet').as_posix()}') a
                      ON a.fid = right(m.folder_id, 3) AND a.t = m.minute_pc
               WHERE a.fid IS NULL AND right(m.folder_id, 3) <> '002'),
    e AS (SELECT right(folder_id, 3) AS fid, date_trunc('hour', CAST(start_t AS TIMESTAMP)) AS hr, count(*) AS n
          FROM read_parquet('{(T3 / 'reflux_events_7s').as_posix()}/*.parquet') GROUP BY ALL)
    SELECT m.fid, m.hr, sum(m.monitor_s) / 3600 AS mon_h, coalesce(any_value(e.n), 0) AS n
    FROM m LEFT JOIN e ON e.fid = m.fid AND e.hr = m.hr GROUP BY m.fid, m.hr""").df()
hh = hh.merge(mp[["fid", "patient_id"]], on="fid")
fi = ox[ox.param == "fio2"].assign(hr=lambda d: d.t.dt.floor("h")).groupby(["patient_id", "hr"]).value.mean()
hh = hh.merge(fi.rename("fio2").reset_index(), on=["patient_id", "hr"], how="inner")
hh["blk"] = hh.hr.dt.floor("6h")
bk = hh.groupby(["patient_id", "blk"]).agg(n=("n", "sum"), mon_h=("mon_h", "sum"), fio2=("fio2", "mean")).reset_index()
bk = bk[bk.mon_h >= 3].sort_values(["patient_id", "blk"])
bk["lr"] = np.log((bk.n + 0.5) / bk.mon_h)
bk["lr_w"] = bk.lr - bk.groupby("patient_id").lr.transform("mean")
full = bk.set_index(["patient_id", "blk"])
lags = []
for k in range(-4, 5):                                   # FiO2 change from block t-1 to block t+k, vs reflux in block t
    s = bk.copy()
    s["fio2_prev"] = s.set_index(["patient_id", "blk"]).index.map(
        lambda ix: full.fio2.get((ix[0], ix[1] - pd.Timedelta(hours=6)), np.nan))
    s["fio2_k"] = s.set_index(["patient_id", "blk"]).index.map(
        lambda ix: full.fio2.get((ix[0], ix[1] + pd.Timedelta(hours=6 * k)), np.nan))
    s["dfio2"] = s.fio2_k - s.fio2_prev
    s = s.dropna(subset=["dfio2"])
    m = smf.mixedlm("dfio2 ~ lr_w", s, groups=s.patient_id).fit()
    lags.append({"lag_blocks_6h": k, "blocks": len(s), "patients": s.patient_id.nunique(),
                 "fio2_change_per_e_fold_reflux": m.params["lr_w"], "lo": m.conf_int().loc["lr_w", 0],
                 "hi": m.conf_int().loc["lr_w", 1], "p": m.pvalues["lr_w"]})
lg = pd.DataFrame(lags)
print("\nC. FiO2 change (points) from the block before to k blocks after, per e-fold more reflux in the block "
      "(within patient)\n", lg.round(3).to_string(index=False))
lg.round(4).to_csv(OUT / "10_reflux_fio2_lags.csv", index=False)

# ---------------------------------------------------------------- D: early reflux -> extubation, adjusted for early oxygenation
import statsmodels.api as sm  # noqa: E402
from scipy import stats  # noqa: E402

le = pd.read_csv(DATA_DIR / "analysis" / "landmark_exposure.csv", dtype={"patient_id": str})
le = le[le.n_full_d1_3 >= 2]
op = pd.read_parquet(CL / "outcomes" / "outcomes_patient.parquet")
o3 = ox.merge(base[["patient_id", "first_study_day_start"]], on="patient_id")
o3["cday"] = (o3.t - o3.first_study_day_start.dt.normalize() - pd.Timedelta(hours=7)).dt.total_seconds() // 86400 + 1
e3 = (o3[o3.cday.between(1, 3)].pivot_table(index="patient_id", columns="param", values="value", aggfunc="mean")
      .add_suffix("_d1_3").reset_index())
x = (le.drop(columns=[c for c in ("apache2",) if c in le.columns])
     .merge(op[op.arm == "treated"][["patient_id", "ventilation_days", "icu_death"]], on="patient_id")
     .merge(base[["patient_id", "apache2"]], on="patient_id").merge(e3, on="patient_id", how="left"))
x["log2_rate"], x["fio2_10"] = np.log2(x.episodes_per_h + 0.25), x.fio2_d1_3 / 10
x["ext"] = (~x.icu_death.astype(bool)).astype(int)
ex = []
for label, cols in (("+ APACHE II", ["log2_rate", "apache2"]), ("+ APACHE II + FiO2 d1-3", ["log2_rate", "apache2", "fio2_10"]),
                    ("+ APACHE II + FiO2 + PEEP d1-3", ["log2_rate", "apache2", "fio2_10", "peep_d1_3"])):
    m = x.dropna(subset=cols)
    ph = sm.PHReg(m.ventilation_days, m[cols], status=m.ext).fit()
    ex.append({"model": label, "n": len(m), "HR_per_doubling": np.exp(ph.params[0]), "lo": np.exp(ph.conf_int()[0, 0]),
               "hi": np.exp(ph.conf_int()[0, 1]), "p": ph.pvalues[0]})
ex = pd.DataFrame(ex)
r = stats.spearmanr(x.log2_rate, x.fio2_d1_3, nan_policy="omit")
print(f"\nD. Early reflux vs FiO2 d1-3: rho {r.statistic:.2f} (p {r.pvalue:.3f}); time to extubation alive\n",
      ex.round(3).to_string(index=False))
ex.round(4).to_csv(OUT / "10_extubation_oxygenation.csv", index=False)
