"""Early reflux burden (clinical days 1-3) vs feeding success (days 4-7).

    python scripts/association_landmark.py

Outcomes
  adequacy_d4_7      patient median over clinical days 4-7 of delivered kcal / (device REE x day ratio)
  deficit_d4_7       patient mean over clinical days 4-7 of (device REE - delivered kcal / day ratio), kcal per day
  cum_deficit_d1_7   sum over clinical days 1-7 of (device REE x day ratio - delivered kcal), kcal
  feeding_stop_d4_7  any physician-ordered feeding stop on clinical days 4-7 (master), binary
Days count only with day ratio >= 0.5, REE > 0 and delivered kcal >= 0 (negative values are data errors).

Analyses
  Spearman      Spearman(index, outcome) with Bonett-Wright CI
  models        OLS outcome ~ index + APACHE II + era group (HC3 SE), and without era;
                logistic for the binary outcome
  coupling      + reflux-triggered pump-pause hours on days 1-3
  lagged        mixed model, adequacy on day d ~ daily index on day d-1 + APACHE II, random intercept
  energy        exclude PN patients; adequacy including IV hidden kcal; including IV hidden + PN kcal;
                PN on days 4-7 counted as feeding failure (lowest adequacy rank)
  dropout       day 1-3 burden in patients still in the study on day 4 vs those who left
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

PRIMARY_INDEX = "bi2"     # frequency tertile + height tertile; bi3 is the alternative index
ERA_ADJUST = True         # models adjust for software era group; unadjusted is also fitted

ROOT = Path(__file__).resolve().parents[1]
from config import DATA_DIR, OUT_DIR  # noqa: E402
CL = DATA_DIR / "clinical"
OUTCOMES = CL / "outcomes"
T4 = DATA_DIR / "t4"
EXPO = DATA_DIR / "analysis" / "landmark_exposure.csv"
OUT = OUT_DIR / "association"

other = "bi3" if PRIMARY_INDEX == "bi2" else "bi2"
indices, era_opts = [PRIMARY_INDEX, other], [ERA_ADJUST, not ERA_ADJUST]

expo = pd.read_csv(EXPO, dtype={"patient_id": str})
expo = expo[expo.eligible].copy()
cov = pd.read_parquet(CL / "patient_day_covariates.parquet")
cov = cov[cov.arm == "treated"]


# ---------------------------------------------------------------- outcomes
def load_outcomes():
    """Patient-day frame with ree, delivered, day_ratio, feeding_stop."""
    base = cov[["patient_id", "clinical_day", "day_ratio"]]
    o = pd.read_parquet(OUTCOMES / "patient_day_outcomes.parquet")
    o = o[o.arm == "treated"][["patient_id", "clinical_day", "ree_kcal_treated", "delivered_kcal",
                               "feeding_stops"]]
    o["feeding_stop"] = o.feeding_stops.notna() & (o.feeding_stops.astype(str).str.strip() != "")
    iv = pd.read_parquet(OUTCOMES / "iv_hidden_kcal.parquet").dropna(subset=["clinical_day"])
    iv = iv.groupby(["patient_id", "clinical_day"]).iv_hidden_kcal.sum().reset_index()
    pn = pd.read_parquet(OUTCOMES / "pn_days.parquet").dropna(subset=["clinical_day"])
    pn = pn.groupby(["patient_id", "clinical_day"]).pn_kcal.sum().reset_index().assign(pn=True)
    return (base.merge(o.drop(columns="feeding_stops"), on=["patient_id", "clinical_day"], how="left")
            .merge(iv, on=["patient_id", "clinical_day"], how="left")
            .merge(pn, on=["patient_id", "clinical_day"], how="left")
            .rename(columns={"ree_kcal_treated": "ree"})
            .assign(iv_hidden_kcal=lambda x: x.iv_hidden_kcal.fillna(0), pn=lambda x: x.pn.fillna(False),
                    pn_kcal=lambda x: x.pn_kcal.fillna(0)))


o = load_outcomes()
ok = (o.day_ratio >= 0.5) & (o.ree > 0) & (o.delivered_kcal >= 0)
o["adequacy"] = np.where(ok, o.delivered_kcal / (o.ree * o.day_ratio), np.nan)
o["deficit"] = np.where(ok, o.ree * o.day_ratio - o.delivered_kcal, np.nan)
o["deficit_per_day"] = np.where(ok, o.ree - o.delivered_kcal / o.day_ratio, np.nan)   # kcal per 24 h

pt = pd.DataFrame({"patient_id": expo.patient_id})
d47 = o[o.clinical_day.between(4, 7)]
pt = pt.merge(d47.groupby("patient_id").adequacy.agg(adequacy_d4_7="median", n_days_d4_7="count").reset_index(),
              on="patient_id", how="left")
pt["deficit_d4_7"] = pt.patient_id.map(d47.groupby("patient_id").deficit_per_day.mean())
d17 = o[o.clinical_day.between(1, 7)]
pt = pt.merge(d17.groupby("patient_id").deficit.agg(cum_deficit_d1_7="sum", n_days_d1_7="count").reset_index(),
              on="patient_id", how="left")
stop = o[o.clinical_day.between(4, 7)].groupby("patient_id").feeding_stop.any()
pt["feeding_stop_d4_7"] = pt.patient_id.map(stop).astype(float)
df = expo.merge(pt, on="patient_id")
BINARY = ["feeding_stop_d4_7"]
OUTCOME_VARS = ["adequacy_d4_7", "deficit_d4_7", "cum_deficit_d1_7"] + BINARY


def spearman_ci(x, y):
    m = x.notna() & y.notna()
    n = int(m.sum())
    r, p = stats.spearmanr(x[m], y[m])
    se = np.sqrt((1 + r ** 2 / 2) / (n - 3))          # Bonett-Wright
    lo, hi = np.tanh(np.arctanh(r) - 1.96 * se), np.tanh(np.arctanh(r) + 1.96 * se)
    return n, r, lo, hi, p


rows, models = [], []
for idx in indices:
    for y in OUTCOME_VARS:
        n, r, lo, hi, p = spearman_ci(df[idx], df[y])
        rows.append({"index": idx, "primary_index": idx == indices[0], "outcome": y, "n": n, "rho": r,
                     "rho_lo": lo, "rho_hi": hi, "p": p})
    for era in era_opts:
        rhs = f"{idx} + apache2" + (" + C(era_group)" if era else "")
        for y in OUTCOME_VARS:
            dd = df.dropna(subset=[y, "apache2"])
            for label, f in (("adjusted", f"{y} ~ {rhs}"), ("coupling", f"{y} ~ {rhs} + pause_h_reflux_d1_3")):
                try:
                    if y in BINARY:
                        if dd[y].sum() < 5:
                            raise ValueError(f"only {int(dd[y].sum())} events")
                        fit = smf.logit(f, dd).fit(disp=0)
                        eff, se = fit.params[idx], fit.bse[idx]
                        est = {"OR_per_point": np.exp(eff), "lo": np.exp(eff - 1.96 * se), "hi": np.exp(eff + 1.96 * se)}
                    else:
                        fit = smf.ols(f, dd).fit(cov_type="HC3")
                        eff, se = fit.params[idx], fit.bse[idx]
                        est = {"beta_per_point": eff, "lo": eff - 1.96 * se, "hi": eff + 1.96 * se}
                    models.append({"index": idx, "era_adjusted": era, "model": label, "outcome": y,
                                   "primary_spec": idx == indices[0] and era == era_opts[0] and label == "adjusted",
                                   "n": int(fit.nobs), **est, "p": fit.pvalues[idx]})
                except Exception as ex:  # report rather than hide
                    models.append({"index": idx, "era_adjusted": era, "model": label, "outcome": y,
                                   "note": f"not fitted: {ex}"})

# Lagged mixed model on patient-days: adequacy on clinical day d ~ daily index on day d-1.
cuts = json.loads((T4 / "cutpoints.json").read_text())["cutpoints_full_days"]
pdy = pd.read_parquet(T4 / "reflux_axes_patient_day.parquet")
pdy = pdy[pdy.full_day & pdy.in_analysis_set].copy()


def tert(v, c):
    return np.where(v.isna(), 1, 1 + (v > c[0]).astype(int) + (v > c[1]).astype(int))


pdy["bi3_day"] = pdy.burden_index
pdy["bi2_day"] = tert(pdy.episodes_per_h, cuts["episodes_per_h"]) + tert(pdy.massive_share, cuts["massive_share"])
al = pd.read_csv(CL / "day_alignment.csv", dtype={"patient_id": str})
day = al[al.match == "both"][["patient_id", "clinical_day", "study_date"]].merge(
    pdy.assign(study_date=lambda x: pd.to_datetime(x.study_date).dt.date.astype(str))
    [["patient_id", "study_date", "bi3_day", "bi2_day"]], on=["patient_id", "study_date"])
day["clinical_day"] = day.clinical_day + 1                           # exposure on d-1 -> outcome day d
lag = o[o.clinical_day.between(2, 7)].merge(day.drop(columns="study_date"), on=["patient_id", "clinical_day"]).merge(
    expo[["patient_id", "apache2"]], on="patient_id").dropna(subset=["adequacy"])
lag_rows = []
for idx in indices:
    v = f"{idx}_day"
    try:
        mm = smf.mixedlm(f"adequacy ~ {v} + apache2 + C(clinical_day)", lag, groups=lag.patient_id).fit()
        lag_rows.append({"index": idx, "n_days": len(lag), "patients": lag.patient_id.nunique(),
                         "beta_per_point": mm.params[v], "lo": mm.conf_int().loc[v, 0],
                         "hi": mm.conf_int().loc[v, 1], "p": mm.pvalues[v]})
    except Exception as ex:
        lag_rows.append({"index": idx, "note": f"not fitted: {ex}"})
lag_res = pd.DataFrame(lag_rows)

# Energy-source sensitivity: (i) exclude patients with PN on days 1-7;
# (ii) adequacy including IV hidden kcal; (iii) including IV hidden kcal and PN kcal (datasheet densities).
o["adequacy_iv"] = np.where(ok, (o.delivered_kcal + o.iv_hidden_kcal) / (o.ree * o.day_ratio), np.nan)
o["adequacy_all"] = np.where(ok, (o.delivered_kcal + o.iv_hidden_kcal + o.pn_kcal) / (o.ree * o.day_ratio), np.nan)
pn_pts = set(o[o.clinical_day.between(1, 7) & o.pn.astype(bool)].patient_id)
pn_fail = set(o[o.clinical_day.between(4, 7) & o.pn.astype(bool)].patient_id) & set(df.patient_id)
adq_iv = o[o.clinical_day.between(4, 7)].groupby("patient_id").adequacy_iv.median()
adq_all = o[o.clinical_day.between(4, 7)].groupby("patient_id").adequacy_all.median()
energy = []
for idx in indices:
    sub = df[~df.patient_id.isin(pn_pts)]
    n, r, lo, hi, p = spearman_ci(sub[idx], sub["adequacy_d4_7"])
    energy.append({"index": idx, "variant": f"exclude PN patients ({len(pn_pts)})", "n": n, "rho": r,
                   "rho_lo": lo, "rho_hi": hi, "p": p})
    n, r, lo, hi, p = spearman_ci(df[idx], df.patient_id.map(adq_iv))
    energy.append({"index": idx, "variant": "adequacy incl. IV hidden kcal", "n": n, "rho": r,
                   "rho_lo": lo, "rho_hi": hi, "p": p})
    n, r, lo, hi, p = spearman_ci(df[idx], df.patient_id.map(adq_all))
    energy.append({"index": idx, "variant": "adequacy incl. IV hidden + PN kcal", "n": n, "rho": r,
                   "rho_lo": lo, "rho_hi": hi, "p": p})
    # (iv) PN on days 4-7 = feeding failure: those patients take the lowest adequacy rank.
    fail = df.adequacy_d4_7.where(~df.patient_id.isin(pn_fail), df.adequacy_d4_7.min() - 1)
    n, r, lo, hi, p = spearman_ci(df[idx], fail)
    energy.append({"index": idx, "variant": f"PN on days 4-7 = failure ({len(pn_fail)})", "n": n, "rho": r,
                   "rho_lo": lo, "rho_hi": hi, "p": p})
energy = pd.DataFrame(energy)

# Dropout: day 1-3 burden in patients with >=2 full days on clinical days 1-3 who were still in
# the study on clinical day 4 vs those who were not.
allx = pd.read_csv(EXPO, dtype={"patient_id": str})
allx = allx[allx.n_full_d1_3 >= 2]
rows_d = []
for idx in indices:
    a, b = allx[allx.in_study_day4.astype(bool)][idx].dropna(), allx[~allx.in_study_day4.astype(bool)][idx].dropna()
    p = stats.mannwhitneyu(a, b).pvalue if len(a) and len(b) else np.nan
    rows_d.append({"index": idx, "n_stayed": len(a), "n_left": len(b), "median_stayed": a.median(),
                   "median_left": b.median() if len(b) else np.nan, "mann_whitney_p": p})
drop = pd.DataFrame(rows_d)

OUT.mkdir(parents=True, exist_ok=True)
spear = pd.DataFrame(rows)
mod = pd.DataFrame(models)
spear.round(4).to_csv(OUT / "spearman.csv", index=False)
mod.round(4).to_csv(OUT / "supportive_models.csv", index=False)
lag_res.round(4).to_csv(OUT / "lagged_mixed_model.csv", index=False)
energy.round(4).to_csv(OUT / "energy_source_sensitivity.csv", index=False)
drop.round(4).to_csv(OUT / "dropout_comparison.csv", index=False)
print("\ndropout comparison:\n", drop.round(3).to_string(index=False))
print("\nenergy-source sensitivity (primary outcome):\n", energy.round(3).to_string(index=False))
print("Spearman:\n", spear.round(3).to_string(index=False))
print("\nmodels:\n", mod.round(3).to_string(index=False))
print("\nlagged mixed model:\n", lag_res.round(4).to_string(index=False))
print(f"\nWrote {OUT}")
