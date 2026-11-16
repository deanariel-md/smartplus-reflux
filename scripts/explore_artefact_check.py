"""Is the reflux that comes with arousal a sensor artefact? Checks on the existing derived tables.

Two alternative readings of "more reflux when the patient is more awake":
  swallowing  a swallowed bolus enters the oesophagus from above and passes the upper sensors before the lower ones.
              A minor episode is detected at the lower sensors only, so a swallow cannot produce one.
  movement    movement or tube artefact would change the physical signature of the episodes that come with arousal
              (shorter, more often coded at the upper sensors, less often escalating from the lower sensors, less
              gastric fluid drained).

  B  the arousal associations recomputed for minor episodes only (and massive only), with the specification of the
     original analysis so that "all episodes" reproduces the published estimate:
     B1 hours with spontaneous breaths (any RSBI value) vs other ventilated hours (18_sedation_dose.csv part C)
     B2 dexmedetomidine days within patients (master-file flag, one factor + device day; 1_drivers_within.csv)
  C  the signature of each episode in an arousal context (an hour with spontaneous breaths; a dexmedetomidine day)
     against the same patients' other episodes: duration in state, share massive, escalation of a massive episode
     from a minor one (lower sensors first, then upper), any fluid drained, volume drained, balloon inflation.
     GEE with patient clusters (independence working correlation), device day adjusted.
The peri-event curves by type (minor vs massive) are already in 14_peri_event_windows.csv (explore_peri_event.py).

Reads only derived tables (t2 minute grid, t3 episodes, t4 artefact list, results/explore/frame_patient_day.parquet).
Writes results/explore/20_artefact_check.csv.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
from config import DATA_DIR, OUT_DIR  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))
import glmm_poisson as gp  # noqa: E402

T2, T3, T4, OUT = DATA_DIR / "t2" / "minute", DATA_DIR / "t3", DATA_DIR / "t4", OUT_DIR / "explore"
rows = []


def add(part, item, est, lo=np.nan, hi=np.nan, p=np.nan, n=np.nan):
    rows.append({"part": part, "item": item, "estimate": est, "lo": lo, "hi": hi, "p": p, "n": n})


# ---------------------------------------------------------------- inputs
m = pd.read_parquet(T2, columns=["folder_id", "minute_pc", "monitor_s", "ree_valid_frac", "rsbi"])
m["fid"] = m.folder_id.str[-3:]
m = m[m.fid != "002"].drop(columns="folder_id")
art = pd.read_parquet(T4 / "artefact_minute_list.parquet")[["fid", "t"]].assign(artefact=True)
m = m.merge(art, left_on=["fid", "minute_pc"], right_on=["fid", "t"], how="left").drop(columns="t")
m["artefact"] = m.artefact.fillna(False).astype(bool)

ev = pd.read_parquet(T3 / "reflux_events_7s_drain.parquet",
                     columns=["folder_id", "type", "start_t", "end_t", "in_state_s", "drained_ml", "balloon_req_max"])
ev["fid"] = ev.folder_id.str[-3:]
ev = ev[ev.fid != "002"].drop(columns="folder_id").sort_values(["fid", "start_t"]).reset_index(drop=True)
ev["end_t"] = pd.to_datetime(ev.end_t).astype("datetime64[ns]")
ev["minute"] = ev.start_t.dt.floor("min")
ev = ev.merge(art.rename(columns={"t": "minute", "artefact": "in_artefact"}), on=["fid", "minute"], how="left")
ev = ev[ev.in_artefact.isna()].drop(columns=["minute", "in_artefact"]).reset_index(drop=True)   # as the paper's counts
ev["hr"] = ev.start_t.dt.floor("h")
ev["study_date"] = (ev.start_t - pd.Timedelta(hours=7)).dt.date.astype(str)
# escalation: a massive episode that a minor episode was still running at, or had ended <= 30 s before, its onset
esc = []
for fid, g in ev.groupby("fid", sort=False):
    mi = g[g.type == "minor"]
    ends, starts = mi.end_t.to_numpy(), mi.start_t.to_numpy()
    for i, r in g[g.type == "massive"].iterrows():
        k = np.searchsorted(starts, r.start_t.to_datetime64(), side="right") - 1
        ok = k >= 0 and (r.start_t.to_datetime64() - ends[k]) <= np.timedelta64(30, "s")
        esc.append((i, bool(ok)))
ev["escalated"] = np.nan
ev.loc[[i for i, _ in esc], "escalated"] = [float(o) for _, o in esc]
ev["massive"] = (ev.type == "massive").astype(float)
ev["drain_any"] = (ev.drained_ml.fillna(0) >= 1).astype(float)
ev["drained"] = ev.drained_ml.where(ev.drained_ml.between(0, 500))      # > 500 mL per episode = counter glitch
ev["balloon"] = (ev.balloon_req_max.fillna(0) > 0).astype(float)
add("0 episodes", "episodes outside artefact minutes: minor / massive", int((ev.type == "minor").sum()),
    int((ev.type == "massive").sum()))
add("0 episodes", "massive episodes escalating from a minor episode (share)", ev.escalated.mean(),
    n=int(ev.escalated.notna().sum()))

# ---------------------------------------------------------------- B1 hours with spontaneous breaths, by type
mm = m[~m.artefact].copy()
mm["hr"] = mm.minute_pc.dt.floor("h")
mm["rs"] = (mm.rsbi > 0).astype(float)
h = mm.groupby(["fid", "hr"]).agg(mon_s=("monitor_s", "sum"), vent=("ree_valid_frac", "mean"),
                                  rsbi_frac=("rs", "mean")).reset_index()
h = h[h.mon_s >= 1800].copy()
h["mon_h"] = h.mon_s / 3600
cnt = ev.pivot_table(index=["fid", "hr"], columns="type", values="in_state_s", aggfunc="size", fill_value=0)
cnt = cnt.reindex(columns=["minor", "massive"], fill_value=0).reset_index()
h = h.merge(cnt, on=["fid", "hr"], how="left").fillna({"minor": 0, "massive": 0})
h["n_all"] = h.minor + h.massive
h = h[h.vent >= 0.5].copy()
h["day_cat"] = np.minimum((h.hr - h.groupby("fid").hr.transform("min")).dt.days + 1, 8)
h["rsbi_h"] = (h.rsbi_frac > 0).astype(float)
for y in ("n_all", "minor", "massive"):
    X, names = gp.design(h, ["rsbi_h", "day_cat:C"])
    q = gp.fit(h[y].to_numpy(), X, np.log(h.mon_h.to_numpy()), h.fid.to_numpy(), names).table()
    q = q[q.term == "rsbi_h"].iloc[0]
    add("B1 hours with spontaneous breaths (any RSBI) vs other ventilated hours, reflux RR", y, q.RR, q.RR_lo,
        q.RR_hi, q.p, len(h))

# ---------------------------------------------------------------- B2 dexmedetomidine days, by type
fr = pd.read_parquet(OUT / "frame_patient_day.parquet")
full = fr[fr.full_day & fr.in_analysis_set].copy()
for f in ("dexmedetomidine", "ketamine"):
    x = full.dropna(subset=[f]).copy()
    x[f] = x[f].astype(float)
    x["pm"] = x.groupby("patient_id")[f].transform("mean")
    x["dev"] = x[f] - x.pm
    for y in ("n_all", "n_minor", "n_massive"):
        X, names = gp.design(x, ["dev", "pm", "day_cat:C"])
        q = gp.fit(x[y].to_numpy(), X, np.log(x.monitored_h.to_numpy()), x.patient_id.to_numpy(), names).table()
        q = q[q.term == "dev"].iloc[0]
        add(f"B2 {f} day within patients (master flag, device day), reflux RR", y, q.RR, q.RR_lo, q.RR_hi, q.p, len(x))

# ---------------------------------------------------------------- C signature of the episodes that come with arousal
def gee(df, outcome, ctx, fam, group):
    d_ = df.dropna(subset=[outcome, ctx]).copy()
    if d_[ctx].nunique() < 2:
        return None
    g = smf.gee(f"{outcome} ~ {ctx} + C(day_cat)", group, d_, family=fam,
                cov_struct=sm.cov_struct.Independence()).fit()
    b, (lo, hi) = g.params[ctx], g.conf_int().loc[ctx]
    return np.exp(b), np.exp(lo), np.exp(hi), g.pvalues[ctx], len(d_), d_.loc[d_[ctx] == 1, outcome].mean(), \
        d_.loc[d_[ctx] == 0, outcome].mean()


OUTCOMES = (("in_state_s", "duration in state, ratio of means", sm.families.Poisson()),
            ("massive", "massive (upper sensors), OR", sm.families.Binomial()),
            ("escalated", "massive episode escalating from a minor one, OR", sm.families.Binomial()),
            ("drain_any", "any fluid drained (>= 1 mL), OR", sm.families.Binomial()),
            ("drained", "volume drained, ratio of means", sm.families.Poisson()),
            ("balloon", "massive episode with balloon inflation, OR", sm.families.Binomial()))
# context 1: the episode starts in a ventilated hour with spontaneous breaths (other ventilated hours as reference)
c1 = ev.merge(h[["fid", "hr", "rsbi_h", "day_cat"]], on=["fid", "hr"], how="inner")
# context 2: the episode falls on a full day with dexmedetomidine (master flag)
c2 = ev.merge(full[["fid", "study_date", "patient_id", "dexmedetomidine", "day_cat"]], on=["fid", "study_date"],
              how="inner").dropna(subset=["dexmedetomidine"])
c2["dex"] = c2.dexmedetomidine.astype(float)
for ctxname, df, ctx, grp in (("spontaneous-breathing hour", c1, "rsbi_h", "fid"),
                              ("dexmedetomidine day", c2, "dex", "patient_id")):
    for col, lab, fam in OUTCOMES:
        sub = df[df.massive == 1] if col == "balloon" else df
        res = gee(sub, col, ctx, fam, grp)
        if res is None:
            continue
        est, lo, hi, p, n, m1, m0 = res
        add(f"C episodes in a {ctxname} vs the same patients' other episodes", lab, est, lo, hi, p, n)
        add(f"C episodes in a {ctxname} vs the same patients' other episodes", lab + " | means in context / other",
            m1, m0)

out = pd.DataFrame(rows)
pd.set_option("display.width", 230)
pd.set_option("display.max_colwidth", 90)
print(out.round(3).to_string(index=False))
out.round(4).to_csv(OUT / "20_artefact_check.csv", index=False)
