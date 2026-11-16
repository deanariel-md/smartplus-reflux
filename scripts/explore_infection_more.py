"""Follow-ups to the respiratory-infection finding (explore_infection.py).

A  What else the device records around onset, within infected patients, with the same windows as the event
   study (ref = > 3 days before onset; -3..-2, -1..0, +1..+2, >= +3) and device-day categories:
   balloon inflations (count per monitored hour), balloon-inflated, drain-open, massive-reflux and pump-paused
   share of time, episode duration, GRV (master), and breathing (respiratory rate, EtCO2, minute ventilation).
   Counts: Poisson GLMM with log monitored-hours offset. Shares and continuous: linear mixed model.
   Also: episodes in bursts (< 2 min after the previous one) vs isolated, and ventilator-interruption minutes
   (REE invalid inside an otherwise ventilated hour: suctioning, disconnection), and episodes within 3 min of one.
B  Does infection explain "early reflux -> faster extubation" (explore_ventilation.py)? Cox models for time to
   extubation alive with respiratory infection added as ever/never, as a time-varying covariate (counting
   process: intervals split at onset), as strata, and within never-infected patients only.
C  Is the surge a rise to abnormal levels, or from an abnormal baseline? Observed / expected episodes per window,
   expected = never-infected patients' rate on the same device day (patient bootstrap CI); the clinical profile
   of each window; the surge adjusted for daily sedative agents and feed volume; tube depth and repositions
   (position-logging era only) and the surge without reposition days.
D  The surge by organism: bacterial vs yeast/Candida (mostly colonisation), Enterobacterales vs non-enteric.

Not done: confirming onset dates from antibiotic starts. The treated-arm medication log has antibiotics for
one patient only (12 tazocin rows), so the CTR adverse-event date stays the onset.

Writes results/explore/9_infection_device.csv, 9_extubation_by_infection.csv, 9_surge_vs_never.csv,
9_surge_profile.csv, 9_surge_adjusted.csv and 9_surge_by_organism.csv.
"""
import sys
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
from config import DATA_DIR, OUT_DIR  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))
import glmm_poisson as gp  # noqa: E402

OUT, T2, T3, T4 = OUT_DIR / "explore", DATA_DIR / "t2" / "minute", DATA_DIR / "t3", DATA_DIR / "t4"
CL = DATA_DIR / "clinical"
WIN = ["a -3..-2", "b -1..0", "c +1..+2", "d >=+3"]

base = pd.read_csv(CL / "baseline_patient.csv", dtype={"patient_id": str}, parse_dates=["first_study_day_start"])
base["patient_id"] = base.patient_id.str.zfill(3)
ae = pd.read_csv(CL / "outcomes" / "adverse_events_ctr.csv", dtype={"patient_id": str}, parse_dates=["date"])
ae = ae.merge(base[base.arm == "treated"][["patient_id", "first_study_day_start"]], on="patient_id")
ae["study_day"] = (ae.date.dt.normalize() - ae.first_study_day_start.dt.normalize()).dt.days + 1
onset = ae[ae.resp].groupby("patient_id").study_day.min()
onset_date = ae[ae.resp].groupby("patient_id").date.min()

# ---------------------------------------------------------------- A: device signals by study day (07:00-06:59)
dev = duckdb.sql(f"""
    WITH m AS (SELECT right(m.folder_id, 3) AS fid, CAST(m.minute_pc - INTERVAL 7 HOUR AS DATE) AS d, m.*
               FROM read_parquet('{T2.as_posix()}/*.parquet') m
               LEFT JOIN read_parquet('{(T4 / 'artefact_minute_list.parquet').as_posix()}') a
                      ON a.fid = right(m.folder_id, 3) AND a.t = m.minute_pc
               WHERE a.fid IS NULL),
    b AS (SELECT right(folder_id, 3) AS fid, CAST(CAST(start_t AS TIMESTAMP) - INTERVAL 7 HOUR AS DATE) AS d,
                 count(*) AS n_balloon
          FROM read_parquet('{(T3 / 'balloon_inflations').as_posix()}/*.parquet') GROUP BY ALL),
    e AS (SELECT right(folder_id, 3) AS fid, CAST(CAST(start_t AS TIMESTAMP) - INTERVAL 7 HOUR AS DATE) AS d,
                 median(in_state_s) AS ep_dur_s
          FROM read_parquet('{(T3 / 'reflux_events_7s').as_posix()}/*.parquet') GROUP BY ALL),
    g AS (SELECT fid, d, sum(monitor_s) / 3600 AS mon_h,
                 sum(balloon_inflated_s) / nullif(sum(monitor_s), 0) AS balloon_share,
                 sum(drain_open_s) / nullif(sum(monitor_s), 0) AS drain_share,
                 sum(ft12_massive_s) / nullif(sum(monitor_s), 0) AS massive_time_share,
                 sum(ft11_minor_s) / nullif(sum(monitor_s), 0) AS minor_time_share,
                 sum(fpf21_paused_s) / nullif(sum(monitor_s), 0) AS paused_share,
                 median(rr_total) FILTER (ree_valid_frac >= 0.5 AND rr_total > 0) AS rr,
                 median(etco2) FILTER (ree_valid_frac >= 0.5 AND etco2 > 0) AS etco2,
                 median(mv_exp) FILTER (ree_valid_frac >= 0.5 AND mv_exp > 0) AS mv
          FROM m GROUP BY ALL)
    SELECT g.*, coalesce(b.n_balloon, 0) AS n_balloon, e.ep_dur_s
    FROM g LEFT JOIN b USING (fid, d) LEFT JOIN e USING (fid, d)""").df()
dev["study_date"] = dev.d.astype(str)

# bursts, ventilator interruptions, and episodes near an interruption
ev = duckdb.sql(f"""SELECT right(folder_id, 3) AS fid, CAST(start_t AS TIMESTAMP) AS t
                    FROM read_parquet('{(T3 / 'reflux_events_7s').as_posix()}/*.parquet') ORDER BY 1, 2""").df()
ev["burst"] = ev.groupby("fid").t.diff().dt.total_seconds() < 120
vi = duckdb.sql(f"""
    WITH m AS (SELECT right(folder_id, 3) AS fid, minute_pc AS t, ree_valid_frac AS v,
                      avg(ree_valid_frac) OVER (PARTITION BY folder_id ORDER BY minute_pc RANGE BETWEEN
                          INTERVAL 30 MINUTE PRECEDING AND INTERVAL 30 MINUTE FOLLOWING) AS vh
               FROM read_parquet('{T2.as_posix()}/*.parquet'))
    SELECT fid, t FROM m WHERE vh >= 0.7 AND v < 0.5""").df()
ev = pd.merge_asof(ev.sort_values("t"), vi.rename(columns={"t": "tv"}).sort_values("tv"), left_on="t", right_on="tv",
                   by="fid", direction="nearest", tolerance=pd.Timedelta(minutes=3))
for d in (ev, vi):
    d["study_date"] = (d.t - pd.Timedelta(hours=7)).dt.date.astype(str)
ev_d = ev.assign(near=ev.tv.notna()).groupby(["fid", "study_date"]).agg(
    n_burst=("burst", "sum"), n_near_interruption=("near", "sum")).reset_index()
vi_d = vi.groupby(["fid", "study_date"]).size().rename("n_interruption").reset_index()

fr = pd.read_parquet(OUT / "frame_patient_day.parquet")
fr = (fr[fr.full_day].merge(dev.drop(columns="d"), on=["fid", "study_date"], how="left")
      .merge(ev_d, on=["fid", "study_date"], how="left").merge(vi_d, on=["fid", "study_date"], how="left"))
for c in ("n_burst", "n_near_interruption", "n_interruption"):
    fr[c] = fr[c].fillna(0)
fr["n_isolated"] = fr.n_all - fr.n_burst
miss = fr.mon_h.isna().sum()
fr["onset"] = fr.patient_id.map(onset)
fr["rel"] = fr.clinical_day - fr.onset
fr["win"] = np.select([fr.rel.between(-3, -2), fr.rel.between(-1, 0), fr.rel.between(1, 2), fr.rel >= 3], WIN, "0 ref")
fr["day_cat"] = np.minimum(fr.device_day, 8).astype(int)
fr["balloon_per_episode"] = fr.n_balloon / fr.n_all.where(fr.n_all > 0)
inf = fr[fr.onset.notna()].copy()
print(f"device days joined: {len(fr) - miss} of {len(fr)}; infected patients: {inf.patient_id.nunique()}, "
      f"{len(inf)} full days")

rows = []
for y, kind, label in (("n_all", "count", "reflux episodes (reference result)"),
                       ("n_burst", "count", "episodes in bursts (< 2 min after the previous)"),
                       ("n_isolated", "count", "isolated episodes"),
                       ("n_near_interruption", "count", "episodes within 3 min of a ventilator interruption"),
                       ("n_interruption", "count", "ventilator-interruption minutes"),
                       ("n_balloon", "count", "balloon inflations"),
                       ("balloon_share", "lmm", "balloon inflated, % of monitored time"),
                       ("drain_share", "lmm", "drain open, % of monitored time"),
                       ("minor_time_share", "lmm", "minor reflux, % of time"),
                       ("massive_time_share", "lmm", "massive reflux, % of time"),
                       ("paused_share", "lmm", "feeding pump paused, % of time"),
                       ("ep_dur_s", "lmm", "median episode duration, s"),
                       ("balloon_per_episode", "lmm", "balloon inflations per episode"),
                       ("grv_total_ml", "lmm", "GRV (master), mL/day"),
                       ("rr", "lmm", "respiratory rate, /min (ventilated)"),
                       ("etco2", "lmm", "EtCO2 (ventilated)"),
                       ("mv", "lmm", "minute ventilation, L/min (ventilated)")):
    g = inf.dropna(subset=[y, "mon_h"] if kind == "count" else [y])
    if kind == "count":
        X, names = gp.design(g, ["win:C", "day_cat:C"])
        t = gp.fit(g[y], X, np.log(g.mon_h.to_numpy()), g.patient_id, names).table().set_index("term")
        for w in WIN:
            k = f"win[{w}]"
            if k in t.index:
                rows.append({"signal": label, "model": "rate ratio", "window": w, "estimate": t.loc[k, "RR"],
                             "lo": t.loc[k, "RR_lo"], "hi": t.loc[k, "RR_hi"], "p": t.loc[k, "p"],
                             "ref_mean": g[g.win == "0 ref"][y].sum() / g[g.win == "0 ref"].mon_h.sum()})
    else:
        scale = 100 if y.endswith("share") else 1
        gg = g.assign(yy=g[y] * scale)
        m = smf.mixedlm("yy ~ C(win, Treatment('0 ref')) + C(day_cat)", gg, groups=gg.patient_id).fit()
        ci = m.conf_int()
        for w in WIN:
            k = f"C(win, Treatment('0 ref'))[T.{w}]"
            if k in m.params.index:
                rows.append({"signal": label, "model": "difference", "window": w, "estimate": m.params[k],
                             "lo": ci.loc[k, 0], "hi": ci.loc[k, 1], "p": m.pvalues[k],
                             "ref_mean": gg[gg.win == "0 ref"].yy.mean()})
dv = pd.DataFrame(rows)
print("\nDevice signals around respiratory-infection onset (infected patients, vs > 3 days before; device-day "
      "adjusted)\n", dv.round(3).to_string(index=False))
dv.round(4).to_csv(OUT / "9_infection_device.csv", index=False)

# ---------------------------------------------------------------- B: extubation by infection status
le = pd.read_csv(DATA_DIR / "analysis" / "landmark_exposure.csv", dtype={"patient_id": str})
le = le[le.n_full_d1_3 >= 2].copy()
op = pd.read_parquet(CL / "outcomes" / "outcomes_patient.parquet")
x = le.merge(op[op.arm == "treated"][["patient_id", "ventilation_days", "icu_death"]], on="patient_id")
x = x.drop(columns=[c for c in ("apache2",) if c in x.columns]).merge(
    base[["patient_id", "apache2", "hours_icu_to_randomization", "first_study_day_start"]], on="patient_id")
x["log2_rate"] = np.log2(x.episodes_per_h + 0.25)
x["ext"] = (~x.icu_death.astype(bool)).astype(int)
# onset in ventilation time (days from ICU admission): AE date at noon, shifted by ICU-to-randomisation hours
od = x.patient_id.map(onset_date)
x["t_onset"] = ((od + pd.Timedelta(hours=12) - x.first_study_day_start).dt.total_seconds() / 86400
                + x.hours_icu_to_randomization.fillna(0) / 24)
x["inf_ever"] = x.t_onset.notna().astype(float)
x["inf_during"] = (x.t_onset < x.ventilation_days).astype(float)
print(f"\nextubation cohort: n = {len(x)}, respiratory infection while ventilated: {int(x.inf_during.sum())}, "
      f"ever: {int(x.inf_ever.sum())}")
print(f"early rate by infection: infected median {x[x.inf_ever == 1].episodes_per_h.median():.2f}/h, "
      f"not {x[x.inf_ever == 0].episodes_per_h.median():.2f}/h")


def cox(d, cols, label, strata=None, entry=None, time="ventilation_days", status="ext"):
    ph = sm.PHReg(d[time], d[cols], status=d[status], strata=strata, entry=entry).fit()
    ci = ph.conf_int()
    out = {"model": label, "n_patients": d.patient_id.nunique(), "events": int(d[status].sum())}
    for i, c in enumerate(cols):
        out[f"HR_{c}"], out[f"lo_{c}"], out[f"hi_{c}"], out[f"p_{c}"] = (
            np.exp(ph.params[i]), np.exp(ci[i, 0]), np.exp(ci[i, 1]), ph.pvalues[i])
    return out


cr = [cox(x, ["log2_rate", "apache2"], "reference: early rate + APACHE II"),
      cox(x, ["log2_rate", "apache2", "inf_ever"], "+ respiratory infection ever (fixed)"),
      cox(x, ["log2_rate", "apache2"], "stratified by infection ever", strata=x.inf_ever.to_numpy())]
# time-varying: split each patient at onset (only if onset falls inside the ventilation period)
sp = []
for _, r in x.iterrows():
    t1, t0 = r.ventilation_days, r.t_onset
    if pd.notna(t0) and 0 < t0 < t1:
        sp.append({**r, "start": 0.0, "stop": t0, "inf_tv": 0.0, "ev": 0})
        sp.append({**r, "start": t0, "stop": t1, "inf_tv": 1.0, "ev": r.ext})
    else:
        sp.append({**r, "start": 0.0, "stop": t1, "inf_tv": float(pd.notna(t0) and t0 <= 0), "ev": r.ext})
sp = pd.DataFrame(sp)
cr.append(cox(sp, ["log2_rate", "apache2", "inf_tv"], "+ respiratory infection, time-varying",
              entry=sp.start.to_numpy(), time="stop", status="ev"))
nv = x[x.inf_ever == 0]
cr.append(cox(nv, ["log2_rate", "apache2"], "never-infected patients only"))
iv = x[x.inf_ever == 1]
cr.append(cox(iv, ["log2_rate", "apache2"], "infected patients only"))
cr = pd.DataFrame(cr)
print("\nTime to extubation alive, HR per doubling of the day 1-3 episode rate\n",
      cr.round(3).to_string(index=False))
cr.round(4).to_csv(OUT / "9_extubation_by_infection.csv", index=False)

# ---------------------------------------------------------------- C: what the surge is
nv = fr[fr.onset.isna()]
rate_nv = nv.groupby("day_cat").n_all.sum() / nv.groupby("day_cat").monitored_h.sum()
inf["expected"] = inf.monitored_h * inf.day_cat.map(rate_nv)


def oe(d):
    return d.groupby("win").n_all.sum() / d.groupby("win").expected.sum()


rng = np.random.default_rng(5)
pids = inf.patient_id.unique()
groups = {p: g for p, g in inf.groupby("patient_id")}
bs = pd.DataFrame([oe(pd.concat([groups[p] for p in rng.choice(pids, len(pids))])) for _ in range(2000)])
ov = pd.DataFrame({"days": inf.groupby("win").size(), "patients": inf.groupby("win").patient_id.nunique(),
                   "observed_over_expected": oe(inf), "lo": bs.quantile(0.025), "hi": bs.quantile(0.975)}).reset_index()
print(f"\nEpisodes observed / expected from {nv.patient_id.nunique()} never-infected patients on the same device day\n",
      ov.round(2).to_string(index=False))
ov.round(4).to_csv(OUT / "9_surge_vs_never.csv", index=False)

SED = ["midazolam", "propofol", "dexmedetomidine", "ketamine"]
for c in SED + ["any_opioid", "any_vasopressor"]:
    fr[c], inf[c] = fr[c].astype(float), inf[c].astype(float)
fr["vtbd_L"], inf["vtbd_L"] = fr.vtbd_ml / 1000, inf.vtbd_ml / 1000
PROF = SED + ["any_opioid", "any_vasopressor", "sofa", "vtbd_ml", "delivered_kcal", "device_day"]
ref_days = inf[inf.win == "0 ref"].day_cat.unique()
prof = pd.concat([inf.groupby("win")[PROF].mean(),
                  nv[nv.day_cat.isin(ref_days)][PROF].mean().rename("never infected, same device days").to_frame().T])
print("\nClinical profile by window (share of days on each drug; means)\n", prof.round(2).to_string())
prof.round(3).reset_index(names="window").to_csv(OUT / "9_surge_profile.csv", index=False)

te = pd.read_csv(T3 / "tube_events.csv", dtype={"fid": str}, parse_dates=["t"])
te["study_date"] = (te.t - pd.Timedelta(hours=7)).dt.date.astype(str)
rp = te[te.event == "reposition"].groupby(["fid", "study_date"]).size().rename("n_repos").reset_index()
seg = pd.read_parquet(T3 / "tube_position_segments.parquet")
seg = seg[seg.cp > 0]
dp = []
for _, s in seg.iterrows():                           # time-weighted confirmed depth per study day
    t = s.ts
    while t < s.te:
        t1 = min(s.te, (t - pd.Timedelta(hours=7)).normalize() + pd.Timedelta(days=1, hours=7))
        dp.append({"fid": s.fid, "study_date": str((t - pd.Timedelta(hours=7)).date()), "cp": s.cp,
                   "h": (t1 - t).total_seconds() / 3600})
        t = t1
dp = pd.DataFrame(dp)
dp = (dp.assign(w=dp.cp * dp.h).groupby(["fid", "study_date"])[["w", "h"]].sum()
      .assign(depth=lambda d: d.w / d.h)[["depth"]].reset_index())
inf = inf.merge(rp, on=["fid", "study_date"], how="left").merge(dp, on=["fid", "study_date"], how="left")
inf["n_repos"] = inf.n_repos.fillna(0)
tube = inf[inf.fid.isin(set(te.fid))]
print(f"\nTube era, infected: {tube.patient_id.nunique()} patients; repositions per day by window "
      f"{(tube.groupby('win').n_repos.sum() / tube.groupby('win').size()).round(2).to_dict()}; never infected "
      f"{fr[fr.onset.isna() & fr.fid.isin(set(te.fid))].merge(rp, on=['fid', 'study_date'], how='left').n_repos.fillna(0).mean():.2f}")


def surge(d, terms, label):
    d = d.dropna(subset=[t.split(":")[0] for t in terms])
    X, names = gp.design(d, terms)
    t = gp.fit(d.n_all, X, np.log(d.monitored_h.to_numpy()), d.patient_id, names).table().set_index("term")
    return {"model": label, "patients": d.patient_id.nunique(), "days": len(d),
            **{f"{k}_{w[:8].strip()}": t.loc[f"win[{w}]", c] for w in WIN[:2] for k, c in
               (("RR", "RR"), ("lo", "RR_lo"), ("hi", "RR_hi"))}}


adj = pd.DataFrame([
    surge(inf, ["win:C", "day_cat:C"], "reference (infected, device-day categories)"),
    surge(inf, ["win:C", "day_cat:C"] + SED, "+ daily sedative agents"),
    surge(inf, ["win:C", "day_cat:C", "vtbd_L"], "+ feed volume (VTBD)"),
    surge(inf, ["win:C", "day_cat:C"] + SED + ["vtbd_L"], "+ sedative agents + VTBD"),
    surge(tube, ["win:C", "day_cat:C"], "tube-logging era only"),
    surge(tube, ["win:C", "day_cat:C", "depth"], "tube-logging era, + confirmed depth"),
    surge(inf[inf.n_repos == 0], ["win:C", "day_cat:C"], "days without a tube reposition")])
print("\nReflux surge (RR vs > 3 days before onset), adjusted\n", adj.round(2).to_string(index=False))
adj.round(4).to_csv(OUT / "9_surge_adjusted.csv", index=False)

# ---------------------------------------------------------------- D: by organism (onset = first isolate of that kind)
r = ae[ae.resp].copy()
e = r.event.str.lower()
r["kind"] = np.select([e.str.contains(r"yeast|candida|fung|asperg"),
                       e.str.contains(r"kleb|citrob|escher|coli\b|enterob|entribacter|serrat|proteus|morgan"),
                       e.str.contains(r"staph|pseud|acinet|stenot|burkh|coryn|haemoph|strep")],
                      ["fungal", "Enterobacterales", "non-enteric bacteria"], "other")
org = []
for label, m in (("any respiratory isolate", r.kind.notna()), ("bacterial (not yeast/Candida)", r.kind != "fungal"),
                 ("yeast/Candida", r.kind == "fungal"), ("Enterobacterales", r.kind == "Enterobacterales"),
                 ("non-enteric bacteria", r.kind == "non-enteric bacteria")):
    on = r[m].groupby("patient_id").study_day.min()
    f = fr.assign(onset=fr.patient_id.map(on))
    f["rel"] = f.clinical_day - f.onset
    f["win"] = np.select([f.rel.between(-3, -2), f.rel.between(-1, 0), f.rel.between(1, 2), f.rel >= 3], WIN, "0 ref")
    org.append(surge(f[f.onset.notna()], ["win:C", "day_cat:C"], label) | {"isolates": int(m.sum())})
org = pd.DataFrame(org)
print("\nSurge by organism of the first isolate\n", org.round(2).to_string(index=False))
org.round(4).to_csv(OUT / "9_surge_by_organism.csv", index=False)
print(f"\nWrote {OUT}/9_*.csv")
