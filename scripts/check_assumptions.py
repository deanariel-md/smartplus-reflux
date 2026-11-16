"""Assumption checks for every model in the manuscript.

Each analysis is first reproduced with its original specification (the "specification" row must match the published
estimate), then the assumptions it rests on are tested and the matching sensitivity analysis is run.

  M  missing data     Little's MCAR test (EM estimates, multivariate normal) on the day-level and patient-level analysis
                      variables; missingness models (GEE logistic, patient clusters); included vs excluded patients
  P  peri-event       baseline balance of onset and control minutes; control draw (other seeds, five controls)
  H  hour-level GLMM  spontaneous-breathing hours; minute ventilation: random-intercept exogeneity (Mundlak, patient
                      fixed effects), overdispersion (Pearson; negative binomial), serial correlation, linearity,
                      leave-one-patient-out
  D  day-level GLMM   device-day trend (linearity), dexmedetomidine (master flag), slope by ICU death, culture event
                      study (Fig. S2)
  L  linear mixed     EE, feeding, ARR: residual and random-effect normality, residual variance, robust SE (GEE),
                      within estimator (patient fixed effects), linearity, offset constant, leave-one-patient-out
  G  GEE              feeding stop: events, small-sample (bias-reduced) SE, working correlation, within-patient form;
                      episode signature: bias-reduced SE and within-patient (Mundlak) form
  C  Cox, Fine-Gray   proportional hazards (Schoenfeld), linearity (restricted cubic spline), offset constant, ties,
                      events per parameter, influence, bootstrap CI, landmark, selection
  S  Spearman         bootstrap CI against the Bonett-Wright interval
  W  dose workbook    skipped when the workbook is absent: dose GLMM leave-one-out, negative binomial,
                      collinearity, midazolam cut point, missing dose rows; Cox + doses: PH, bootstrap, penalised fit,
                      leave-one-out; Fine-Gray + doses: PH. Reads only ID, start time and dose columns.

    python scripts/check_assumptions.py                 # all sections
    python scripts/check_assumptions.py --sections C,S  # some; their rows in the output are replaced

Writes results/explore/21_assumption_checks.csv (one row per check).
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
from config import DATA_DIR, OUT_DIR, DOSE_WORKBOOK  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))
import glmm_poisson as gp  # noqa: E402

CL, T2, T3, T4 = DATA_DIR / "clinical", DATA_DIR / "t2" / "minute", DATA_DIR / "t3", DATA_DIR / "t4"
OUT = OUT_DIR / "explore"
CSV = OUT / "21_assumption_checks.csv"
XL = DOSE_WORKBOOK
ap = argparse.ArgumentParser()
ap.add_argument("--sections", default="M,P,H,D,L,G,C,S,W")
ap.add_argument("--boot", type=int, default=1000)
args = ap.parse_args()
SECTIONS = args.sections.split(",")
rows = []


def add(sec, analysis, assumption, check, est=np.nan, lo=np.nan, hi=np.nan, p=np.nan, n=np.nan, note=""):
    rows.append({"section": sec, "analysis": analysis, "assumption": assumption, "check": check, "estimate": est,
                 "lo": lo, "hi": hi, "p": p, "n": n, "note": note})


# ---------------------------------------------------------------- shared inputs
_c = {}


def minutes():
    if "m" not in _c:
        cols = ["folder_id", "minute_pc", "monitor_s", "covered_s", "ree_valid_frac", "rsbi", "ree_kcal_d", "vco2_ml_min",
                "rr_total", "mv_exp", "pip", "fpf20_running_s", "feed_delivered_ml_last"]
        m = pd.read_parquet(T2, columns=cols)
        m["fid"] = m.folder_id.str[-3:]
        m = m[m.fid != "002"].drop(columns="folder_id")
        art = pd.read_parquet(T4 / "artefact_minute_list.parquet")[["fid", "t"]].assign(artefact=True)
        m = m.merge(art, left_on=["fid", "minute_pc"], right_on=["fid", "t"], how="left").drop(columns="t")
        m["artefact"] = m.artefact.fillna(False).astype(bool)
        _c["m"] = m
    return _c["m"]


def events(drop_artefact=False):
    k = f"ev{drop_artefact}"
    if k not in _c:
        ev = pd.read_parquet(T3 / "reflux_events_7s_drain.parquet",
                             columns=["folder_id", "type", "start_t", "end_t", "in_state_s", "drained_ml",
                                      "balloon_req_max"])
        ev["fid"] = ev.folder_id.str[-3:]
        ev = ev[ev.fid != "002"].drop(columns="folder_id").sort_values(["fid", "start_t"]).reset_index(drop=True)
        ev["start_t"] = pd.to_datetime(ev.start_t).astype("datetime64[ns]")
        ev["end_t"] = pd.to_datetime(ev.end_t).astype("datetime64[ns]")
        if drop_artefact:
            art = pd.read_parquet(T4 / "artefact_minute_list.parquet")[["fid", "t"]]
            ev["minute"] = ev.start_t.dt.floor("min")
            ev = ev.merge(art.rename(columns={"t": "minute"}).assign(a=True), on=["fid", "minute"], how="left")
            ev = ev[ev.a.isna()].drop(columns=["minute", "a"]).reset_index(drop=True)
        ev["hr"] = ev.start_t.dt.floor("h")
        ev["study_date"] = (ev.start_t - pd.Timedelta(hours=7)).dt.date.astype(str)
        _c[k] = ev
    return _c[k]


def frame():
    if "fr" not in _c:
        _c["fr"] = pd.read_parquet(OUT / "frame_patient_day.parquet")
    return _c["fr"].copy()


def full_days():
    fr = frame()
    return fr[fr.full_day & fr.in_analysis_set].copy()


def ee_days(min_minutes=360):
    m = minutes()
    e = m[(m.ree_valid_frac >= 0.5) & (m.ree_kcal_d > 0) & (m.vco2_ml_min > 0)].copy()
    e["study_date"] = (e.minute_pc - pd.Timedelta(hours=7)).dt.date.astype(str)
    day = e.groupby(["fid", "study_date"]).agg(minutes=("ree_kcal_d", "size"), ee=("ree_kcal_d", "median")).reset_index()
    return day[day.minutes >= min_minutes]


def outcomes():
    op = pd.read_parquet(CL / "outcomes" / "outcomes_patient.parquet")
    return op[op.arm == "treated"].drop(columns="arm")


def ci_text(e, lo, hi, d=2):
    return f"{e:.{d}f} ({lo:.{d}f} to {hi:.{d}f})"


# ---------------------------------------------------------------- generic checks
def glmm_fit(df, y, terms, off_col, grp):
    X, names = gp.design(df, terms)
    return gp.fit(df[y].to_numpy(float), X, np.log(df[off_col].to_numpy(float)), df[grp].to_numpy(), names)


def count_checks(sec, analysis, df, y, terms, term, fe_rhs, fe_term, off_col, grp, order_col=None, loo=True):
    """Poisson GLMM: reproduce; model-based vs robust SE; patient fixed effects (Poisson, NB2); serial correlation; LOO."""
    r = glmm_fit(df, y, terms, off_col, grp)
    t = r.table().set_index("term").loc[term]
    tm = r.table(robust=False).set_index("term").loc[term]
    add(sec, analysis, "specification", "reproduces the reported estimate (Poisson GLMM, cluster-robust SE)",
        t.RR, t.RR_lo, t.RR_hi, t.p, len(df), f"{df[grp].nunique()} patients; converged {r.converged}")
    add(sec, analysis, "overdispersion and within-patient correlation handled",
        "model-based CI against the reported cluster-robust CI", tm.RR, tm.RR_lo, tm.RR_hi, tm.p, len(df),
        f"robust/model-based SE ratio {t.se / tm.se:.2f}; random-intercept SD {r.sigma:.2f}")
    dd = df[df.groupby(grp)[y].transform("sum") > 0].copy()
    g = pd.factorize(dd[grp])[0]
    f = f"{y} ~ {fe_rhs} + C({grp})"
    off = np.log(dd[off_col].to_numpy(float))
    po = smf.glm(f, dd, family=sm.families.Poisson(), offset=off).fit(cov_type="cluster", cov_kwds={"groups": g})
    mu, yv = po.fittedvalues.to_numpy(), dd[y].to_numpy(float)
    disp = ((yv - mu) ** 2 / mu).sum() / po.df_resid
    alpha = max(((yv - mu) ** 2 - yv).sum() / (mu ** 2).sum(), 1e-6)
    b, (lo, hi) = po.params[fe_term], po.conf_int().loc[fe_term]
    add(sec, analysis, "random intercept independent of the exposure (within-patient estimate)",
        "patient fixed-effects Poisson, cluster-robust SE", np.exp(b), np.exp(lo), np.exp(hi), po.pvalues[fe_term],
        len(dd), f"Pearson dispersion {disp:.1f}")
    nb = smf.glm(f, dd, family=sm.families.NegativeBinomial(alpha=alpha), offset=off).fit(
        cov_type="cluster", cov_kwds={"groups": g})
    b, (lo, hi) = nb.params[fe_term], nb.conf_int().loc[fe_term]
    add(sec, analysis, "overdispersion (mean-variance relation)",
        f"negative binomial (alpha {alpha:.2f} by moments), patient fixed effects, cluster-robust SE",
        np.exp(b), np.exp(lo), np.exp(hi), nb.pvalues[fe_term], len(dd))
    if order_col is not None:
        pr = pd.Series((yv - mu) / np.sqrt(mu), index=dd.index)
        o = dd.assign(pr=pr).sort_values([grp, order_col])
        step = o.groupby(grp)[order_col].diff()
        lag = o.groupby(grp).pr.shift(1).where(step == step.dropna().min())
        ac = o.pr.corr(lag)
        add(sec, analysis, "serial correlation handled", "lag-1 autocorrelation of Pearson residuals (fixed-effects "
            "Poisson); cluster-robust SE allow any within-patient correlation", ac, n=int(lag.notna().sum()))
    if loo:
        est, sig = [], 0
        for p_ in df[grp].unique():
            try:
                q = glmm_fit(df[df[grp] != p_], y, terms, off_col, grp).table().set_index("term").loc[term]
            except Exception:
                continue
            est.append(q.RR)
            sig += int((q.RR_lo > 1) or (q.RR_hi < 1))
        est = np.array(est)
        add(sec, analysis, "no single patient drives the estimate", "leave-one-patient-out: range of the estimate",
            t.RR, est.min(), est.max(), np.nan, len(est), f"CI excludes 1 in {sig} of {len(est)} fits")
    return r


def lmm_checks(sec, analysis, df, formula, term="dev", grp="patient_id", tf=lambda b: b, loo=True, quad=True):
    """Linear mixed model: reproduce; residuals; random effects; GEE robust SE; within estimator; linearity; LOO."""
    r = smf.mixedlm(formula, df, groups=df[grp]).fit()
    lo, hi = r.conf_int().loc[term]
    add(sec, analysis, "specification", "reproduces the reported estimate (linear mixed model)",
        tf(r.params[term]), tf(lo), tf(hi), r.pvalues[term], len(df), f"{df[grp].nunique()} patients")
    res, fit = np.asarray(r.resid), np.asarray(r.fittedvalues)
    add(sec, analysis, "normal residuals (model-based SE)", "Shapiro-Wilk on conditional residuals",
        p=stats.shapiro(res).pvalue, n=len(res),
        note=f"skewness {stats.skew(res):.2f}, excess kurtosis {stats.kurtosis(res):.2f}")
    h = stats.spearmanr(fit, np.abs(res))
    add(sec, analysis, "constant residual variance", "Spearman of |residual| with fitted value", h.statistic,
        p=h.pvalue, n=len(res))
    re = np.array([v.iloc[0] for v in r.random_effects.values()])
    add(sec, analysis, "normal random intercepts", "Shapiro-Wilk on predicted random intercepts",
        p=stats.shapiro(re).pvalue, n=len(re))
    g = smf.gee(formula, grp, df, family=sm.families.Gaussian(), cov_struct=sm.cov_struct.Exchangeable()).fit(
        cov_type="bias_reduced")
    lo, hi = g.conf_int().loc[term]
    add(sec, analysis, "inference robust to non-normal, heteroscedastic residuals",
        "GEE, Gaussian, exchangeable, bias-reduced robust SE", tf(g.params[term]), tf(lo), tf(hi), g.pvalues[term],
        len(df))
    fe = smf.ols(formula.replace(" + pm", "") + f" + C({grp})", df).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(df[grp])[0]})
    lo, hi = fe.conf_int().loc[term]
    add(sec, analysis, "random intercept independent of the exposure (within-patient estimate)",
        "patient fixed effects (within estimator), cluster-robust SE", tf(fe.params[term]), tf(lo), tf(hi),
        fe.pvalues[term], len(df))
    if quad:
        q = smf.mixedlm(formula + f" + I({term} ** 2)", df, groups=df[grp]).fit()
        k = [c for c in q.params.index if c.startswith("I(")][0]
        add(sec, analysis, "linear in log2 episode rate", "quadratic term added (coefficient, p)", q.params[k],
            q.conf_int().loc[k, 0], q.conf_int().loc[k, 1], q.pvalues[k], len(df),
            f"linear term with quadratic present {tf(q.params[term]):.3g}")
    if loo:
        est, sig = [], 0
        for p_ in df[grp].unique():
            sub = df[df[grp] != p_]
            try:
                q = smf.mixedlm(formula, sub, groups=sub[grp]).fit()
            except Exception:
                continue
            est.append(tf(q.params[term]))
            lo, hi = q.conf_int().loc[term]
            sig += int(lo > 0 or hi < 0)
        est = np.array(est)
        add(sec, analysis, "no single patient drives the estimate", "leave-one-patient-out: range of the estimate",
            tf(r.params[term]), est.min(), est.max(), np.nan, len(est), f"CI excludes 0 in {sig} of {len(est)} fits")
    return r


def little_mcar(df):
    """Little (1988): EM estimates of mean and covariance, then d2 = sum over patterns n_j (ybar_j - mu)' S^-1 (.)."""
    X = df.to_numpy(float)
    X = (X - np.nanmean(X, 0)) / np.where(np.nanstd(X, 0) > 0, np.nanstd(X, 0), 1)
    n, p = X.shape
    miss = np.isnan(X)
    mu, S = np.nanmean(X, 0), np.diag(np.nanvar(X, 0))
    for _ in range(2000):
        T1, T2 = np.zeros(p), np.zeros((p, p))
        for i in range(n):
            o, mi = ~miss[i], miss[i]
            x, C = X[i].copy(), np.zeros((p, p))
            if mi.any():
                B = S[np.ix_(mi, o)] @ np.linalg.pinv(S[np.ix_(o, o)]) if o.any() else np.zeros((mi.sum(), 0))
                x[mi] = mu[mi] + (B @ (X[i, o] - mu[o]) if o.any() else 0)
                C[np.ix_(mi, mi)] = S[np.ix_(mi, mi)] - (B @ S[np.ix_(o, mi)] if o.any() else 0)
            T1 += x
            T2 += np.outer(x, x) + C
        mu1, S1 = T1 / n, T2 / n - np.outer(T1 / n, T1 / n)
        done = max(np.abs(mu1 - mu).max(), np.abs(S1 - S).max()) < 1e-8
        mu, S = mu1, S1
        if done:
            break
    pats = {}
    for i in range(n):
        pats.setdefault(tuple(miss[i]), []).append(i)
    d2, dof = 0.0, 0
    for pat, ids in pats.items():
        o = ~np.array(pat)
        if not o.any():
            continue
        xb = X[np.ix_(ids, np.flatnonzero(o))].mean(0) - mu[o]
        d2 += len(ids) * xb @ np.linalg.pinv(S[np.ix_(o, o)]) @ xb
        dof += int(o.sum())
    dof -= p
    return d2, dof, stats.chi2.sf(d2, dof) if dof > 0 else np.nan, len(pats)


def gee_miss(df, flag, rhs, grp="patient_id"):
    g = smf.gee(f"{flag} ~ {rhs}", grp, df, family=sm.families.Binomial(),
                cov_struct=sm.cov_struct.Exchangeable()).fit()
    return g


# ---------------------------------------------------------------- feeding / Spearman inputs
def adequacy_patients():
    cov = pd.read_parquet(CL / "patient_day_covariates.parquet")
    cov = cov[cov.arm == "treated"]
    o = pd.read_parquet(CL / "outcomes" / "patient_day_outcomes.parquet")
    o = o[o.arm == "treated"][["patient_id", "clinical_day", "ree_kcal_treated", "delivered_kcal"]]
    o = cov[["patient_id", "clinical_day", "day_ratio"]].merge(o, on=["patient_id", "clinical_day"], how="left")
    ok = (o.day_ratio >= 0.5) & (o.ree_kcal_treated > 0) & (o.delivered_kcal >= 0)
    o["adequacy"] = np.where(ok, o.delivered_kcal / (o.ree_kcal_treated * o.day_ratio), np.nan)
    return o[o.clinical_day.between(4, 7)].groupby("patient_id").adequacy.median()


def cox_frame():
    le = pd.read_csv(DATA_DIR / "analysis" / "landmark_exposure.csv", dtype={"patient_id": str})
    x = le[le.n_full_d1_3 >= 2].merge(outcomes(), on="patient_id", how="left")
    x["log2_rate"] = np.log2(x.episodes_per_h + 0.25)
    x["ext"] = (~x.icu_death.astype(bool)).astype(int)
    return le, x


def fio2_d1_3():
    ox = pd.read_parquet(CL / "outcomes" / "fio2_peep_hourly.parquet")
    base = pd.read_csv(CL / "baseline_patient.csv", dtype={"patient_id": str}, parse_dates=["first_study_day_start"])
    o3 = ox.merge(base[["patient_id", "first_study_day_start"]], on="patient_id")
    o3["cday"] = (o3.t - o3.first_study_day_start.dt.normalize() - pd.Timedelta(hours=7)).dt.total_seconds() // 86400 + 1
    return (o3[o3.cday.between(1, 3)].pivot_table(index="patient_id", columns="param", values="value", aggfunc="mean")
            .add_suffix("_d1_3").reset_index())


# ================================================================ M  missing data
def sec_M():
    # day level: the variables of the day-level models, on the 287 full aligned days
    d = full_days().merge(ee_days()[["fid", "study_date", "ee"]], on=["fid", "study_date"], how="left")
    ok = (d.day_ratio >= 0.5) & (d.ree > 0) & (d.delivered_kcal >= 0) & (d.plan_kcal > 0)
    d["eff_ok"] = d.eff.where(ok & np.isfinite(d.eff))
    d["plan_ok"] = d.plan_frac.where(ok & np.isfinite(d.plan_frac))
    vars_ = {"log2_rate": "log2 episode rate", "device_day": "device day", "ee": "device EE (daily median)",
             "plan_ok": "prescribed/EE", "eff_ok": "delivered/prescribed", "sofa": "SOFA",
             "any_sedative": "any sedative (master)"}
    miss = {k: int(d[k].isna().sum()) for k in vars_}
    d2, dof, p, npat = little_mcar(d[list(vars_)])
    add("M", "day-level models (287 full days)", "missing completely at random (MCAR)",
        "Little's MCAR test on log2 rate, device day, EE, prescribed/EE, delivered/prescribed, SOFA, any sedative",
        d2, p=p, n=len(d), note=f"chi2 {d2:.1f} on {dof} df; {npat} patterns; missing: " +
        ", ".join(f"{vars_[k]} {v}" for k, v in miss.items() if v) +
        "; days within patients are not independent, so the test is anti-conservative")
    for flag, lab in (("ee", "device EE missing (no >= 6 h of ventilation with valid VCO2)"),
                      ("eff_ok", "feeding ratios missing"), ("sofa", "SOFA missing")):
        d["miss"] = d[flag].isna().astype(float)
        if d.miss.sum() < 3:
            continue
        g = gee_miss(d, "miss", "log2_rate + device_day")
        lo, hi = g.conf_int().loc["log2_rate"]
        add("M", "day-level models", "missingness unrelated to the exposure (complete-case validity)",
            f"GEE logistic: {lab} ~ log2 rate + device day; OR per doubling", np.exp(g.params["log2_rate"]),
            np.exp(lo), np.exp(hi), g.pvalues["log2_rate"], len(d),
            f"{int(d.miss.sum())} missing; device day OR {np.exp(g.params['device_day']):.2f} "
            f"(p {g.pvalues['device_day']:.3f})")
    # hour level: ventilated hours without a minute-ventilation value
    v = hours_vent()
    v["miss"] = v.mv.isna().astype(float)
    share = v.groupby("fid").miss.mean()
    partial = int(((share > 0) & (share < 1)).sum())
    rate = v.groupby("fid").apply(lambda q: q.n.sum() / q.mon_h.sum())
    a, b = rate[share == 1], rate[share == 0]
    add("M", "minute-ventilation model (ventilated hours)", "missingness unrelated to the exposure",
        "patients with no minute-ventilation export vs others: reflux rate in ventilated hours (medians, Mann-Whitney)",
        a.median(), b.median(), p=stats.mannwhitneyu(a, b).pvalue if len(a) and len(b) else np.nan, n=int(len(a)),
        note=f"{int(v.miss.sum())} of {len(v)} hours missing; {len(a)} patients missing all hours, {partial} partly; "
             "whole-patient missingness does not bias a within-patient estimate")
    # patient level: landmark cohort
    le, x = cox_frame()
    x = x.merge(fio2_d1_3(), on="patient_id", how="left")
    x["adequacy_d4_7"] = x.patient_id.map(adequacy_patients())
    pv = {"log2_rate": "log2 early rate", "apache2": "APACHE II", "age": "age", "fio2_d1_3": "FiO2 d1-3",
          "peep_d1_3": "PEEP d1-3", "adequacy_d4_7": "adequacy d4-7", "ventilation_days": "ventilation days"}
    pv = {k: v for k, v in pv.items() if k in x.columns}
    d2, dof, p, npat = little_mcar(x[list(pv)])
    miss = {k: int(x[k].isna().sum()) for k in pv}
    add("M", "patient-level models (41 landmark patients)", "missing completely at random (MCAR)",
        "Little's MCAR test on " + ", ".join(pv.values()), d2, p=p, n=len(x),
        note=f"chi2 {d2:.1f} on {dof} df; {npat} patterns; missing: " +
        ", ".join(f"{pv[k]} {v}" for k, v in miss.items() if v))
    for k, lab in (("adequacy_d4_7", "adequacy d4-7"), ("fio2_d1_3", "FiO2 d1-3")):
        mm = x[k].isna()
        if mm.sum() == 0:
            continue
        for v in ("log2_rate", "ventilation_days", "apache2"):
            a, b = x.loc[mm, v].dropna(), x.loc[~mm, v].dropna()
            add("M", "patient-level models", "missingness unrelated to exposure, outcome and severity",
                f"{lab} missing vs present: {pv.get(v, v)} (medians, Mann-Whitney)", a.median(), b.median(),
                p=stats.mannwhitneyu(a, b).pvalue if len(a) and len(b) else np.nan, n=int(mm.sum()),
                note="estimate = median if missing; lo = median if present")
    # selection into the landmark cohort
    ex = le[le.n_full_d1_3 < 2].merge(outcomes(), on="patient_id", how="left")
    for v in ("apache2", "ventilation_days"):
        add("M", "extubation and feeding (landmark cohort)", "inclusion unrelated to severity and outcome",
            f"excluded (< 2 full days on ICU days 1-3) vs included: {v} (medians)", ex[v].median(), x[v].median(),
            p=stats.mannwhitneyu(ex[v].dropna(), x[v].dropna()).pvalue if len(ex) else np.nan, n=len(ex),
            note=f"excluded deaths {int(ex.icu_death.sum())} of {len(ex)}; included {int(x.icu_death.sum())} of {len(x)}")


# ================================================================ P  peri-event
PRE, POST, ISO_MIN, BASE, CTL_WIN = 20, 20, 15, (-20, -11), 180
LAGS = np.arange(-PRE, POST + 1)
PSIG = ["vco2_ml_min", "rr_total", "mv_exp", "pip", "feed_run"]


def peri_rows(seed=7, n_ctl=1):
    """explore_peri_event.py's onset and control selection; also keeps each window's raw baseline level."""
    rng = np.random.default_rng(seed)
    m, ev = minutes(), events()
    out = []
    for fid, g in m.groupby("fid"):
        g = g.sort_values("minute_pc")
        t_first = g.minute_pc.min()
        idx = ((g.minute_pc - t_first).dt.total_seconds() // 60).astype(int).to_numpy()
        n = idx.max() + 1
        vent = np.zeros(n, bool)
        vent[idx] = (g.ree_valid_frac.to_numpy() >= 0.5) & (g.rr_total.fillna(0).to_numpy() > 0) & ~g.artefact.to_numpy()
        arr = {}
        for s in PSIG:
            a = np.full(n, np.nan)
            if s == "feed_run":
                v = (g.fpf20_running_s / g.covered_s.where(g.covered_s != 0)).to_numpy(float)
            else:
                v = g[s].to_numpy(float)
                v = np.where(v > 0, v, np.nan)
            a[idx] = v
            a[~vent] = np.nan
            arr[s] = a
        e = ev[ev.fid == fid]
        if e.empty:
            continue
        on = ((e.start_t - t_first).dt.total_seconds() // 60).astype(int).to_numpy()
        off = ((e.end_t - t_first).dt.total_seconds() // 60).astype(int).to_numpy()
        prev_end = np.concatenate([[-10 ** 9], np.maximum.accumulate(off)[:-1]])
        iso = (on - prev_end >= ISO_MIN) & (on - PRE >= 0) & (on + POST < n)
        busy = np.zeros(n + 1, int)
        np.add.at(busy, np.clip(on, 0, n), 1)
        cs = np.concatenate([[0], np.cumsum(busy)])
        types = e.type.to_numpy()[iso]
        for k, o in enumerate(on[iso]):
            if not vent[o - 3:o + 1].all() or np.isnan(arr["rr_total"][o + BASE[0]:o + BASE[1] + 1]).all():
                continue
            same = np.flatnonzero(vent[max(0, o - CTL_WIN):o + CTL_WIN + 1]) + max(0, o - CTL_WIN)
            same = same[(same - PRE >= 0) & (same + POST < n)]
            ok = [c for c in rng.choice(same, size=min(len(same), 60), replace=False)
                  if cs[c + POST + 1] - cs[c - ISO_MIN] == 0 and vent[c - 3:c + 1].all()] if len(same) else []
            if not ok:
                continue
            row = {"fid": fid, "type": types[k]}
            for s in PSIG:
                def curve(z):
                    bw = arr[s][z + BASE[0]:z + BASE[1] + 1]
                    b = np.nanmean(bw) if not np.isnan(bw).all() else np.nan
                    return arr[s][z - PRE:z + POST + 1] - b, b
                c_on, b_on = curve(o)
                cc = [curve(c) for c in ok[:n_ctl]]
                row[s + "|on"], row[s + "|b_on"] = c_on, b_on
                row[s + "|ctl"] = np.nanmean(np.array([c[0] for c in cc]), 0)
                row[s + "|b_ctl"] = np.nanmean([c[1] for c in cc])
            out.append(row)
    return out


def peri_summary(rows_, s, win=(-5, -1), typ="all", seed=11):
    rng = np.random.default_rng(seed)
    sub = [r for r in rows_ if typ == "all" or r["type"] == typ]
    D, B = [], []
    for f in sorted({r["fid"] for r in sub}):
        on_ = np.array([r[s + "|on"] for r in sub if r["fid"] == f])
        if np.isnan(on_).all():
            continue
        ct_ = np.array([r[s + "|ctl"] for r in sub if r["fid"] == f])
        D.append(np.nanmean(on_, 0) - np.nanmean(ct_, 0))
        B.append(np.nanmean([r[s + "|b_on"] for r in sub if r["fid"] == f]) -
                 np.nanmean([r[s + "|b_ctl"] for r in sub if r["fid"] == f]))
    D, B = np.array(D), np.array(B)
    sel = (LAGS >= win[0]) & (LAGS <= win[1])
    dw = np.nanmean(D[:, sel], 1)
    bw = np.array([np.nanmean(dw[rng.integers(0, len(dw), len(dw))]) for _ in range(2000)])
    bb = np.array([np.nanmean(B[rng.integers(0, len(B), len(B))]) for _ in range(2000)])
    return (np.nanmean(dw), np.nanquantile(bw, .025), np.nanquantile(bw, .975), len(dw),
            np.nanmean(B), np.nanquantile(bb, .025), np.nanquantile(bb, .975), len(sub))


def sec_P():
    base = peri_rows(7, 1)
    an = "peri-event (VCO2 and ventilation around isolated onsets)"
    e, lo, hi, npt, b, blo, bhi, non = peri_summary(base, "vco2_ml_min")
    add("P", an, "specification", "reproduces VCO2 onset minus control, -5 to -1 min (mL/min)", e, lo, hi, n=non,
        note=f"{npt} patients")
    for s, lab in (("vco2_ml_min", "VCO2, mL/min"), ("rr_total", "respiratory rate, /min"),
                   ("mv_exp", "minute ventilation, L/min"), ("pip", "peak pressure, cmH2O"),
                   ("feed_run", "feed pump running, share of minute")):
        _, _, _, npt, b, blo, bhi, non = peri_summary(base, s)
        add("P", an, "control minutes exchangeable with onset minutes (no regression to the mean)",
            f"baseline level (-20 to -11 min), onset minus control: {lab}", b, blo, bhi, n=non, note=f"{npt} patients")
    for seed in (1, 2, 3):
        e, lo, hi, npt, *_ = peri_summary(peri_rows(seed, 1), "vco2_ml_min")
        add("P", an, "estimate not an artefact of the random control draw",
            f"VCO2 -5 to -1 min with control draw seed {seed}", e, lo, hi, note=f"{npt} patients")
    five = peri_rows(7, 5)
    for s, lab in (("vco2_ml_min", "VCO2"), ("rr_total", "respiratory rate"), ("mv_exp", "minute ventilation")):
        e, lo, hi, npt, b, blo, bhi, non = peri_summary(five, s)
        add("P", an, "estimate not an artefact of the random control draw",
            f"{lab} -5 to -1 min, mean of up to 5 controls per onset", e, lo, hi, n=non,
            note=f"{npt} patients; baseline difference {b:.2f} ({blo:.2f} to {bhi:.2f})")


# ================================================================ H  hour-level GLMMs
def hours_rsbi():
    m, ev = minutes(), events(drop_artefact=True)
    mm = m[~m.artefact].copy()
    mm["hr"] = mm.minute_pc.dt.floor("h")
    mm["rs"] = (mm.rsbi > 0).astype(float)
    h = mm.groupby(["fid", "hr"]).agg(mon_s=("monitor_s", "sum"), vent=("ree_valid_frac", "mean"),
                                      rsbi_frac=("rs", "mean")).reset_index()
    h = h[h.mon_s >= 1800].copy()
    h["mon_h"] = h.mon_s / 3600
    h = h.merge(ev.groupby(["fid", "hr"]).size().rename("n_all").reset_index(), on=["fid", "hr"], how="left")
    h["n_all"] = h.n_all.fillna(0)
    h = h[h.vent >= 0.5].copy()
    h["day_cat"] = np.minimum((h.hr - h.groupby("fid").hr.transform("min")).dt.days + 1, 8)
    h["rsbi_h"] = (h.rsbi_frac > 0).astype(float)
    h["rsbi_pm"] = h.groupby("fid").rsbi_h.transform("mean")
    return h


def hours_vent():
    m, ev = minutes(), events()
    mm = m[~m.artefact].copy()
    mm["hr"] = mm.minute_pc.dt.floor("h")
    mm["w"] = mm.ree_valid_frac * mm.monitor_s
    mm["mvp"] = mm.mv_exp.where(mm.mv_exp > 0)
    h = mm.groupby(["fid", "hr"]).agg(mon_s=("monitor_s", "sum"), w=("w", "sum"), mv=("mvp", "median")).reset_index()
    h = h[h.mon_s >= 1800].copy()
    h["mon_h"] = h.mon_s / 3600
    h["ree_valid"] = h.w / h.mon_s
    h = h.merge(ev.groupby(["fid", "hr"]).size().rename("n").reset_index(), on=["fid", "hr"], how="left")
    h["n"] = h.n.fillna(0)
    v = h[h.ree_valid >= 0.5].copy()
    v["dday"] = v.groupby("fid").hr.transform(lambda s: (s - s.min()).dt.total_seconds() / 86400)
    return v


def hours_mv():
    x = hours_vent().dropna(subset=["mv"]).copy()
    x["xv"] = x.mv / 2
    x["pm"] = x.groupby("fid").xv.transform("mean")
    x["dev"] = x.xv - x.pm
    x["dev2"] = x.dev ** 2
    return x


def sec_H():
    h = hours_rsbi()
    an = "spontaneous-breathing hours (any RSBI) vs other ventilated hours"
    count_checks("H", an, h, "n_all", ["rsbi_h", "day_cat:C"], "rsbi_h", "rsbi_h + C(day_cat)", "rsbi_h", "mon_h",
                 "fid", order_col="hr")
    t = glmm_fit(h, "n_all", ["rsbi_h", "rsbi_pm", "day_cat:C"], "mon_h", "fid").table().set_index("term").loc["rsbi_h"]
    add("H", an, "random intercept independent of the exposure (within-patient estimate)",
        "Mundlak: patient share of spontaneous-breathing hours added", t.RR, t.RR_lo, t.RR_hi, t.p, len(h))
    k = h[h.rsbi_h == 1].groupby("fid").size()
    add("H", an, "enough exposed hours", "spontaneous-breathing hours per patient (max share of all such hours)",
        k.max() / k.sum(), n=int(k.sum()), note=f"{len(k)} patients; median {k.median():.0f} h per patient")
    x = hours_mv()
    an = "minute ventilation per 2 L/min, within patients"
    count_checks("H", an, x, "n", ["dev", "pm", "dday"], "dev", "dev + dday", "dev", "mon_h", "fid", order_col="hr")
    r = glmm_fit(x, "n", ["dev", "dev2", "pm", "dday"], "mon_h", "fid").table().set_index("term")
    add("H", an, "log-linear in minute ventilation", "quadratic term added (RR per (2 L/min)^2, p)",
        r.loc["dev2", "RR"], r.loc["dev2", "RR_lo"], r.loc["dev2", "RR_hi"], r.loc["dev2", "p"], len(x),
        f"linear term with quadratic present {r.loc['dev', 'RR']:.2f}")
    x["q"] = pd.qcut(x.dev, 5, labels=False)
    r = glmm_fit(x, "n", ["q:C", "pm", "dday"], "mon_h", "fid").table()
    r = r[r.term.str.startswith("q")]
    add("H", an, "log-linear in minute ventilation", "quintiles of the within-patient deviation vs lowest (RR)",
        note="; ".join(f"Q{int(float(t[2:-1])) + 1} {a:.2f}" for t, a in zip(r.term, r.RR)), n=len(x))


# ================================================================ D  day-level GLMMs
def sec_D():
    a = pd.read_parquet(T4 / "reflux_axes_patient_day.parquet")
    a = a[a.full_day & a.in_analysis_set].copy()
    a["n_all"] = a.n_minor + a.n_massive
    f7 = a[a.device_day <= 7].copy()
    f7["dd"] = f7.device_day.astype(float)
    an = "rise over device days 1-7 (per day)"
    count_checks("D", an, f7, "n_all", ["dd"], "dd", "dd", "dd", "monitored_h", "fid")
    rr1 = glmm_fit(f7, "n_all", ["dd"], "monitored_h", "fid").table().set_index("term").loc["dd", "RR"]
    f7["dc"] = f7.device_day.astype(int)
    rc = glmm_fit(f7, "n_all", ["dc:C"], "monitored_h", "fid")
    tc = rc.table()
    # linear trend plus indicators for days 3-7 spans the categorical model; robust Wald test of the indicators
    for k in range(3, 8):
        f7[f"i{k}"] = (f7.device_day == k).astype(float)
    ind = [f"i{k}" for k in range(3, 8)]
    rw = glmm_fit(f7, "n_all", ["dd"] + ind, "monitored_h", "fid")
    j = [rw.names.index(c) for c in ind]
    b, V = rw.params[j], rw.cov_robust[np.ix_(j, j)]
    w = float(b @ np.linalg.solve(V, b))
    add("D", an, "log-linear trend over days 1-7",
        "robust Wald test of departure from linearity (categorical vs linear device day, 5 df)", w,
        p=stats.chi2.sf(w, 5), n=len(f7),
        note="categorical RR vs day 1: " + "; ".join(f"d{t[3:-1]} {v:.2f}" for t, v in zip(tc.term, tc.RR)
                                                      if t != "(Intercept)") +
        "; linear: " + "; ".join(f"d{k} {rr1 ** (k - 1):.2f}" for k in (3, 5, 7)))
    full = full_days()
    x = full.dropna(subset=["dexmedetomidine"]).copy()
    x["pm"] = x.groupby("patient_id").dexmedetomidine.transform("mean")
    x["dev"] = x.dexmedetomidine - x.pm
    an = "dexmedetomidine day, within patients (master flag; dose-workbook model in section W)"
    count_checks("D", an, x, "n_all", ["dev", "pm", "day_cat:C"], "dev", "dev + C(day_cat)", "dev", "monitored_h",
                 "patient_id")
    c = x.groupby("patient_id").dexmedetomidine.agg(["min", "max"])
    add("D", an, "within-patient contrast", "patients with days both on and off dexmedetomidine",
        int(((c["min"] == 0) & (c["max"] == 1)).sum()), n=x.patient_id.nunique(),
        note=f"{int(x.dexmedetomidine.sum())} dexmedetomidine days of {len(x)}")
    xx = full.merge(outcomes()[["patient_id", "icu_death"]], on="patient_id")
    xx["dday"] = xx.device_day.astype(float)
    xx["died"] = xx.icu_death.astype(float)
    xx["dday_x_died"] = xx.dday * xx.died
    count_checks("D", "slope over device days, ICU deaths vs survivors (interaction)", xx, "n_all",
                 ["dday", "died", "dday_x_died"], "dday_x_died", "dday + dday_x_died", "dday_x_died", "monitored_h",
                 "patient_id")
    # tube repositioning: episode rate 6 h after vs 6 h before (describe_tube_position.py)
    tv = pd.read_csv(T3 / "tube_events.csv", dtype={"fid": str}, parse_dates=["t"])
    rp = tv[tv.event == "reposition"][["fid", "t"]].reset_index(drop=True)
    mm, e = minutes(), events()
    mm = mm[~mm.artefact]
    w6 = pd.Timedelta(hours=6)
    rr_ = []
    for _, r in rp.iterrows():
        g = mm[(mm.fid == r.fid) & mm.minute_pc.between(r.t - w6, r.t + w6)]
        ee = e[(e.fid == r.fid) & e.start_t.between(r.t - w6, r.t + w6)]
        rr_.append({"fid": r.fid, "h_pre": g.loc[g.minute_pc < r.t, "monitor_s"].sum() / 3600,
                    "h_post": g.loc[g.minute_pc >= r.t, "monitor_s"].sum() / 3600,
                    "n_pre": int((ee.start_t < r.t).sum()), "n_post": int((ee.start_t >= r.t).sum())})
    res = pd.DataFrame(rr_)
    res = res[(res.h_pre >= 3) & (res.h_post >= 3)].copy()
    pooled = lambda q: (q.n_post.sum() / q.h_post.sum()) / (q.n_pre.sum() / q.h_pre.sum())  # noqa: E731
    pw = stats.wilcoxon(res.n_pre / res.h_pre, res.n_post / res.h_post).pvalue
    an = "tube repositioning: rate 6 h after vs 6 h before"
    add("D", an, "specification", "reproduces the pooled post/pre ratio and the Wilcoxon test over repositions",
        pooled(res), p=pw, n=len(res), note=f"{res.fid.nunique()} patients")
    res["lr"] = np.log((res.n_post + 0.5) / res.h_post) - np.log((res.n_pre + 0.5) / res.h_pre)
    per = res.groupby("fid").lr.mean()
    fids = res.fid.unique()
    rng = np.random.default_rng(2026)
    bs = [pooled(pd.concat([res[res.fid == f] for f in rng.choice(fids, len(fids))])) for _ in range(2000)]
    add("D", an, "independent pairs (repositions cluster within patients)",
        "patient level: Wilcoxon signed-rank on each patient's mean log ratio; patient-bootstrap CI of the pooled ratio",
        pooled(res), np.quantile(bs, .025), np.quantile(bs, .975), stats.wilcoxon(per).pvalue, len(per),
        f"geometric mean of patient ratios {np.exp(per.mean()):.2f}; rate fell in {int((per < 0).sum())} of {len(per)}")
    # Fig. S2: reflux around clinically triggered positive respiratory cultures, infected patients
    base = pd.read_csv(CL / "baseline_patient.csv", dtype={"patient_id": str}, parse_dates=["first_study_day_start"])
    base["patient_id"] = base.patient_id.str.zfill(3)
    ae = pd.read_csv(CL / "outcomes" / "adverse_events_ctr.csv", dtype={"patient_id": str}, parse_dates=["date"])
    ae = ae.merge(base[base.arm == "treated"][["patient_id", "first_study_day_start"]], on="patient_id")
    ae["study_day"] = (ae.date.dt.normalize() - ae.first_study_day_start.dt.normalize()).dt.days + 1
    onset = ae[ae.resp].groupby("patient_id").study_day.min()
    fr = frame()
    f = fr[fr.full_day].copy()
    f["onset"] = f.patient_id.map(onset)
    f = f[f.onset.notna()].copy()
    f["rel"] = f.clinical_day - f.onset
    f["win"] = np.select([f.rel.between(-3, -2), f.rel.between(-1, 0), f.rel.between(1, 2), f.rel >= 3],
                         ["a -3..-2", "b -1..0", "c +1..+2", "d >=+3"], "0 ref")
    f["day_cat"] = np.minimum(f.device_day, 8).astype(int)
    count_checks("D", "Fig. S2: days -3 to -2 before a positive respiratory culture vs > 3 days before", f, "n_all",
                 ["win:C", "day_cat:C"], "win[a -3..-2]", "C(win) + C(day_cat)", "C(win)[T.a -3..-2]",
                 "monitored_h", "patient_id")


# ================================================================ L  linear mixed models
def arr_days():
    m = minutes()
    g = m[~m.artefact][["fid", "minute_pc", "monitor_s", "feed_delivered_ml_last"]].rename(
        columns={"minute_pc": "t", "feed_delivered_ml_last": "fd"})
    r = pd.read_parquet(T3 / "arr_minute.parquet", columns=["folder_id", "minute", "arr_ml"])
    r["fid"] = r.folder_id.str[-3:]
    g = g.merge(r[["fid", "minute", "arr_ml"]].rename(columns={"minute": "t"}), on=["fid", "t"], how="left")
    g = g.sort_values(["fid", "t"]).reset_index(drop=True)
    g["arr"] = g.arr_ml.fillna(0)
    g["fd_inc"] = g.groupby("fid").fd.diff().clip(lower=0).where(lambda s: s < 50, 0).fillna(0)
    g["study_date"] = (g.t - pd.Timedelta(hours=7)).dt.date.astype(str)
    dd = g.groupby(["fid", "study_date"]).agg(arr=("arr", "sum"), feed=("fd_inc", "sum")).reset_index()
    day = full_days()[["patient_id", "fid", "study_date", "device_day", "episodes_per_h", "n_all"]].merge(
        dd, on=["fid", "study_date"], how="left").fillna({"arr": 0, "feed": 0})
    day["per_ep"] = day.arr / day.n_all.where(day.n_all > 0)
    day["day_cat"] = np.minimum(day.device_day, 8).astype(int)
    day["feed_100"] = day.feed / 100
    day["log_per_ep"] = np.log2(day.per_ep + 0.5)
    return day


def within(df, xv, grp="patient_id"):
    df = df.copy()
    df["pm"] = df.groupby(grp)[xv].transform("mean")
    df["dev"] = df[xv] - df.pm
    return df


def sec_L():
    # energy expenditure
    z = full_days().merge(ee_days(), on=["fid", "study_date"], how="inner")
    z["lr"] = np.log2(z.episodes_per_h + 0.1)
    an = "EE per doubling of the day's episode rate (kcal/day)"
    lmm_checks("L", an, within(z, "lr"), "ee ~ dev + pm + C(day_cat)")
    for c in (0.25, 0.5):
        q = within(z.assign(lr=np.log2(z.episodes_per_h + c)), "lr")
        r = smf.mixedlm("ee ~ dev + pm + C(day_cat)", q, groups=q.patient_id).fit()
        add("L", an, "result not driven by the constant added before log2",
            f"log2(rate + {c}) instead of + 0.1", r.params["dev"], *r.conf_int().loc["dev"], r.pvalues["dev"], len(q))
    zz = full_days().merge(ee_days(1080), on=["fid", "study_date"], how="inner")
    zz = within(zz.assign(lr=np.log2(zz.episodes_per_h + 0.1)), "lr")
    r = smf.mixedlm("ee ~ dev + pm + C(day_cat)", zz, groups=zz.patient_id).fit()
    add("L", an, "partly ventilated days (extubation, interruptions) do not drive it",
        "days with >= 18 h of valid VCO2 only", r.params["dev"], *r.conf_int().loc["dev"], r.pvalues["dev"], len(zz),
        f"{zz.patient_id.nunique()} patients")
    # feeding under closed-loop delivery
    d = frame()
    ok = (d.day_ratio >= 0.5) & (d.ree > 0) & (d.delivered_kcal >= 0) & (d.plan_kcal > 0)
    for y, xv, where, lab in (
            ("eff", "log2_rate", ok & d.full_day, "delivered/prescribed, same day (percentage points)"),
            ("eff", "lag_log2_rate", ok & (d.lag_full == True), "delivered/prescribed, next day (points)"),  # noqa: E712
            ("plan_frac", "lag_log2_rate", ok & (d.lag_full == True), "prescribed/EE, next day (points)")):  # noqa: E712
        x = d[where].dropna(subset=[y, xv]).copy()
        x = x[np.isfinite(x[y])]
        lmm_checks("L", lab, within(x, xv), f"{y} ~ dev + pm + C(day_cat)", tf=lambda b: 100 * b)
        if xv == "log2_rate":
            for c in (0.1, 0.5):
                q = within(x.assign(lr=np.log2(x.episodes_per_h + c)), "lr")
                r = smf.mixedlm(f"{y} ~ dev + pm + C(day_cat)", q, groups=q.patient_id).fit()
                lo, hi = r.conf_int().loc["dev"]
                add("L", lab, "result not driven by the constant added before log2",
                    f"log2(rate + {c}) instead of + 0.25", 100 * r.params["dev"], 100 * lo, 100 * hi, r.pvalues["dev"],
                    len(q))
            add("L", lab, "outcome bounded near 1 (ceiling)", "share of days with delivered/prescribed >= 0.98",
                (x[y] >= 0.98).mean(), n=len(x))
    # ARR
    day = arr_days()
    an = "ARR per doubling of the day's episode rate (mL/day)"
    q = within(day.assign(lr=np.log2(day.episodes_per_h + 0.1)), "lr")
    lmm_checks("L", an, q, "arr ~ dev + pm + C(day_cat) + feed_100")
    g = smf.gee("arr ~ dev + pm + C(day_cat) + feed_100", "patient_id", q, family=sm.families.Poisson(),
                cov_struct=sm.cov_struct.Exchangeable()).fit(cov_type="bias_reduced")
    lo, hi = g.conf_int().loc["dev"]
    add("L", an, "skewed outcome (mean model)", "GEE with log link (quasi-Poisson on mL), ratio of daily ARR per "
        "doubling, bias-reduced SE", np.exp(g.params["dev"]), np.exp(lo), np.exp(hi), g.pvalues["dev"], len(q))
    qq = q.dropna(subset=["log_per_ep"])
    lmm_checks("L", "ARR per episode, fold per doubling", qq, "log_per_ep ~ dev + pm + C(day_cat) + feed_100",
               tf=lambda b: 2 ** b)


# ================================================================ G  GEE
def sec_G():
    d = frame()
    x = d[d.lag_full == True].dropna(subset=["lag_log2_rate"]).copy()  # noqa: E712
    an = "physician-ordered feeding stop next day (OR per doubling)"
    f = "feed_stop ~ lag_log2_rate + C(day_cat)"
    for lab, cs, ct in (("specification: exchangeable, robust SE (reported)", sm.cov_struct.Exchangeable(), "robust"),
                        ("exchangeable, bias-reduced SE (45 clusters)", sm.cov_struct.Exchangeable(), "bias_reduced"),
                        ("independence working correlation", sm.cov_struct.Independence(), "robust")):
        g = smf.gee(f, "patient_id", x, family=sm.families.Binomial(), cov_struct=cs).fit(cov_type=ct)
        lo, hi = g.conf_int().loc["lag_log2_rate"]
        add("G", an, "specification" if lab.startswith("spec") else "small-sample SE and working correlation", lab,
            np.exp(g.params["lag_log2_rate"]), np.exp(lo), np.exp(hi), g.pvalues["lag_log2_rate"], len(x),
            f"{x.patient_id.nunique()} patients")
    ev_days, ev_pts = int(x.feed_stop.sum()), x[x.feed_stop == 1].patient_id.nunique()
    add("G", an, "enough events for the parameters", "days with a stop / patients with a stop / parameters",
        ev_days, ev_pts, np.nan, n=len(x), note=f"{ev_days} event days, {ev_pts} patients, 1 exposure + day categories")
    g = smf.gee("feed_stop ~ lag_log2_rate + device_day", "patient_id", x, family=sm.families.Binomial(),
                cov_struct=sm.cov_struct.Exchangeable()).fit(cov_type="bias_reduced")
    lo, hi = g.conf_int().loc["lag_log2_rate"]
    add("G", an, "enough events for the parameters", "device day as one linear term instead of 7 categories "
        "(2 parameters for the events), bias-reduced SE", np.exp(g.params["lag_log2_rate"]), np.exp(lo), np.exp(hi),
        g.pvalues["lag_log2_rate"], len(x))
    x = within(x, "lag_log2_rate")
    g = smf.gee("feed_stop ~ dev + pm + C(day_cat)", "patient_id", x, family=sm.families.Binomial(),
                cov_struct=sm.cov_struct.Exchangeable()).fit(cov_type="bias_reduced")
    lo, hi = g.conf_int().loc["dev"]
    add("G", an, "within-patient form", "Mundlak (day's deviation from the patient mean), bias-reduced SE",
        np.exp(g.params["dev"]), np.exp(lo), np.exp(hi), g.pvalues["dev"], len(x))
    # episode signature
    ev = events(drop_artefact=True).copy()
    esc = []
    for fid, gg in ev.groupby("fid", sort=False):
        mi = gg[gg.type == "minor"]
        ends, starts = mi.end_t.to_numpy(), mi.start_t.to_numpy()
        for i, r in gg[gg.type == "massive"].iterrows():
            k = np.searchsorted(starts, r.start_t.to_datetime64(), side="right") - 1
            esc.append((i, float(k >= 0 and (r.start_t.to_datetime64() - ends[k]) <= np.timedelta64(30, "s"))))
    ev["escalated"] = np.nan
    ev.loc[[i for i, _ in esc], "escalated"] = [o for _, o in esc]
    ev["massive"] = (ev.type == "massive").astype(float)
    ev["drain_any"] = (ev.drained_ml.fillna(0) >= 1).astype(float)
    ev["drained"] = ev.drained_ml.where(ev.drained_ml.between(0, 500))
    h = hours_rsbi()
    c1 = ev.merge(h[["fid", "hr", "rsbi_h", "day_cat"]], on=["fid", "hr"], how="inner")
    full = full_days()
    c2 = ev.merge(full[["fid", "study_date", "patient_id", "dexmedetomidine", "day_cat"]], on=["fid", "study_date"],
                  how="inner").dropna(subset=["dexmedetomidine"])
    c2["dex"] = c2.dexmedetomidine.astype(float)
    outc = (("in_state_s", "duration, ratio of means", sm.families.Poisson()),
            ("massive", "massive, OR", sm.families.Binomial()),
            ("escalated", "escalation from a minor episode, OR", sm.families.Binomial()),
            ("drain_any", "any fluid drained, OR", sm.families.Binomial()),
            ("drained", "volume drained, ratio of means", sm.families.Poisson()))
    for ctxname, df, ctx, grp in (("spontaneous-breathing hour", c1, "rsbi_h", "fid"),
                                  ("dexmedetomidine day", c2, "dex", "patient_id")):
        for col, lab, fam in outc:
            q = df.dropna(subset=[col, ctx]).copy()
            q["ctx_pm"] = q.groupby(grp)[ctx].transform("mean")
            an = f"episode signature in a {ctxname}: {lab}"
            G = q[grp].nunique()
            for clab, rhs, small in (("specification: independence, robust SE (reported)", f"{ctx} + C(day_cat)", False),
                                     ("small-sample correction: SE x sqrt(G/(G-1)), t with G-1 df", f"{ctx} + C(day_cat)",
                                      True),
                                     ("within patients (Mundlak), small-sample correction", f"{ctx} + ctx_pm + C(day_cat)",
                                      True)):
                g = smf.gee(f"{col} ~ {rhs}", grp, q, family=fam, cov_struct=sm.cov_struct.Independence()).fit()
                b, se = g.params[ctx], g.bse[ctx]
                if small:
                    se *= np.sqrt(G / (G - 1))
                    z, p_ = stats.t.ppf(.975, G - 1), 2 * stats.t.sf(abs(b / se), G - 1)
                else:
                    z, p_ = 1.959964, g.pvalues[ctx]
                add("G", an, "specification" if clab.startswith("spec") else
                    ("small-sample SE" if clab.startswith("small") else "within-patient comparison"), clab,
                    np.exp(b), np.exp(b - z * se), np.exp(b + z * se), p_, len(q), f"{G} patients")


# ================================================================ C  Cox and Fine-Gray
def rcs(x, knots):
    k = np.asarray(knots, float)
    norm = (k[-1] - k[0]) ** 2
    cols = []
    for j in range(len(k) - 2):
        cols.append((np.maximum(x - k[j], 0) ** 3
                     - np.maximum(x - k[-2], 0) ** 3 * (k[-1] - k[j]) / (k[-1] - k[-2])
                     + np.maximum(x - k[-1], 0) ** 3 * (k[-2] - k[j]) / (k[-1] - k[-2])) / norm)
    return np.column_stack(cols)


def phreg(d, cols, ties="breslow", time="ventilation_days", status="ext"):
    return sm.PHReg(d[time].to_numpy(float), d[cols].to_numpy(float), status=d[status].to_numpy(), ties=ties).fit()


def ph_test(d, cols, time="ventilation_days", status="ext"):
    from lifelines import CoxPHFitter
    from lifelines.statistics import proportional_hazard_test
    cph = CoxPHFitter().fit(d[cols + [time, status]], time, status)
    out = {}
    for tt in ("rank", "km"):
        s = proportional_hazard_test(cph, d[cols + [time, status]], time_transform=tt).summary
        for c in cols:
            out[(c, tt)] = (float(s.loc[c, "test_statistic"]), float(s.loc[c, "p"]))
    return cph, out


def cox_checks(an, x, cols, boot=1000, loo=True, spline=True, sec="C"):
    ph = phreg(x, cols)
    lo, hi = np.exp(ph.conf_int()[0])
    ev = int(x.ext.sum())
    add(sec, an, "specification", "reproduces the reported HR (statsmodels PHReg, Breslow ties)", np.exp(ph.params[0]),
        lo, hi, ph.pvalues[0], len(x), f"{ev} events")
    add(sec, an, "events per parameter >= 10", "events / parameters", ev / len(cols), n=len(x),
        note=f"{ev} events, {len(cols)} parameters")
    _, t = ph_test(x, cols)
    for c in cols:
        add(sec, an, "proportional hazards", f"Schoenfeld residual test, {c} (rank time; km time in note)",
            t[(c, "rank")][0], p=t[(c, "rank")][1], n=len(x), note=f"km-transformed time p {t[(c, 'km')][1]:.2f}")
    pe = phreg(x, cols, ties="efron")
    lo, hi = np.exp(pe.conf_int()[0])
    add(sec, an, "tie handling does not matter", "Efron ties instead of Breslow", np.exp(pe.params[0]), lo, hi,
        pe.pvalues[0], len(x))
    if spline:
        k = np.quantile(x.log2_rate, [.05, .35, .65, .95])
        S = rcs(x.log2_rate.to_numpy(), k)
        xs = x.copy()
        xs["s1"], xs["s2"] = S[:, 0], S[:, 1]
        lin, spl = phreg(xs, cols), phreg(xs, cols + ["s1", "s2"])
        lr = 2 * (spl.llf - lin.llf)
        add(sec, an, "log-linear in log2 early rate", "restricted cubic spline (4 knots) vs linear, LR test (2 df)",
            lr, p=stats.chi2.sf(lr, 2), n=len(x))
        for c in (0.1, 0.5, 1.0):
            xc = x.assign(log2_rate=np.log2(x.episodes_per_h + c))
            q = phreg(xc, cols)
            lo, hi = np.exp(q.conf_int()[0])
            add(sec, an, "result not driven by the constant added before log2", f"log2(rate + {c}) instead of + 0.25",
                np.exp(q.params[0]), lo, hi, q.pvalues[0], len(x))
    if loo:
        est = []
        for i in range(len(x)):
            q = phreg(x.drop(index=x.index[i]), cols)
            est.append((np.exp(q.params[0]), np.exp(q.conf_int()[0][0])))
        est = np.array(est)
        add(sec, an, "no single patient drives the estimate", "leave-one-patient-out: range of the HR",
            np.exp(ph.params[0]), est[:, 0].min(), est[:, 0].max(), n=len(est),
            note=f"lower CI limit > 1 in {int((est[:, 1] > 1).sum())} of {len(est)} fits")
    if boot:
        rng = np.random.default_rng(2026)
        bs = []
        for _ in range(boot):
            q = x.iloc[rng.integers(0, len(x), len(x))]
            try:
                bs.append(np.exp(phreg(q, cols).params[0]))
            except Exception:
                pass
        bs = np.array(bs)
        add(sec, an, "Wald CI valid in a small sample", f"patient bootstrap percentile CI ({len(bs)} resamples)",
            np.exp(ph.params[0]), np.quantile(bs, .025), np.quantile(bs, .975), n=len(x),
            note=f"share of resamples with HR <= 1: {(bs <= 1).mean():.3f}")
    return ph


def sec_C():
    le, x = cox_frame()
    cox_checks("extubation alive, HR per doubling of the day 1-3 rate, + APACHE II", x, ["log2_rate", "apache2"],
               boot=args.boot)
    ox = x.merge(fio2_d1_3(), on="patient_id", how="left")
    ox["fio2_10"] = ox.fio2_d1_3 / 10
    ox = ox.dropna(subset=["fio2_10"]).reset_index(drop=True)
    cox_checks("extubation alive, + APACHE II + mean FiO2 days 1-3", ox, ["log2_rate", "apache2", "fio2_10"],
               boot=0, loo=False, spline=False)
    # landmark: time from the end of day 3 (exposure window), patients still ventilated
    lm = x[x.ventilation_days > 3].copy()
    q = phreg(lm, ["log2_rate", "apache2"])
    lo, hi = np.exp(q.conf_int()[0])
    add("C", "extubation alive, + APACHE II", "no immortal time (exposure on days 1-3, follow-up from ICU admission)",
        "landmark at day 3: patients still ventilated after day 3", np.exp(q.params[0]), lo, hi, q.pvalues[0], len(lm),
        f"{len(x) - len(lm)} patients extubated or dead by day 3")
    # Fine-Gray: no censoring other than death, so the subdistribution model is a Cox model with deaths kept to the end
    add("C", "competing risk", "independent censoring", "patients censored other than by ICU death",
        int(((x.ext == 0) & (~x.icu_death.astype(bool))).sum()), n=len(x))
    xc = x.copy()
    xc.loc[xc.ext == 0, "ventilation_days"] = x.ventilation_days.max() + 1
    cph, t = ph_test(xc, ["log2_rate", "apache2"])
    s = cph.summary.loc["log2_rate"]
    add("C", "Fine-Gray, + APACHE II", "specification", "reproduces the reported subdistribution HR (lifelines, deaths "
        "kept in the risk set)", s["exp(coef)"], s["exp(coef) lower 95%"], s["exp(coef) upper 95%"], s["p"], len(xc))
    for c in ("log2_rate", "apache2"):
        add("C", "Fine-Gray, + APACHE II", "proportional subdistribution hazards",
            f"Schoenfeld residual test, {c} (rank time)", t[(c, "rank")][0], p=t[(c, "rank")][1], n=len(xc),
            note=f"km-transformed time p {t[(c, 'km')][1]:.2f}")
    fg_split("C", "Fine-Gray, + APACHE II", x, xc, ["apache2"])
    fg_split("C", "extubation alive, HR per doubling of the day 1-3 rate, + APACHE II", x, x, ["apache2"],
             kind="cause-specific")


def fg_split(sec, an, x, xc, adj, kind="subdistribution"):
    """Subdistribution HR before and after the median time to extubation alive (time-varying coefficient)."""
    from lifelines import CoxTimeVaryingFitter
    cut = float(np.median(x.loc[x.ext == 1, "ventilation_days"]))
    sp = []
    for _, r in xc.iterrows():
        T, base = r.ventilation_days, {c: r[c] for c in adj}
        sp.append({"id": r.patient_id, "start": 0.0, "stop": min(T, cut), "ev": int(r.ext == 1 and T <= cut),
                   "x_early": r.log2_rate, "x_late": 0.0, **base})
        if T > cut:
            sp.append({"id": r.patient_id, "start": cut, "stop": T, "ev": int(r.ext), "x_early": 0.0,
                       "x_late": r.log2_rate, **base})
    ctv = CoxTimeVaryingFitter().fit(pd.DataFrame(sp), id_col="id", event_col="ev", start_col="start", stop_col="stop")
    for k, lab in (("x_early", f"up to day {cut:.1f}"), ("x_late", f"after day {cut:.1f}")):
        s = ctv.summary.loc[k]
        add(sec, an, f"proportional {kind} hazards",
            f"{kind} HR {lab} (median time to extubation alive)", s["exp(coef)"], s["exp(coef) lower 95%"],
            s["exp(coef) upper 95%"], s["p"], len(x),
            f"extubations: {int(((x.ext == 1) & (x.ventilation_days <= cut)).sum())} early, "
            f"{int(((x.ext == 1) & (x.ventilation_days > cut)).sum())} late")


# ================================================================ S  Spearman
def sec_S():
    le = pd.read_csv(DATA_DIR / "analysis" / "landmark_exposure.csv", dtype={"patient_id": str})
    x = le[le.eligible].copy()
    x["adequacy_d4_7"] = x.patient_id.map(adequacy_patients())
    m = x.dropna(subset=["bi2", "adequacy_d4_7"])
    r = stats.spearmanr(m.bi2, m.adequacy_d4_7).statistic
    se = np.sqrt((1 + r ** 2 / 2) / (len(m) - 3))
    add("S", "early reflux (burden index bi2) vs energy adequacy days 4-7", "specification",
        "reproduces rho with the Bonett-Wright CI", r, np.tanh(np.arctanh(r) - 1.96 * se),
        np.tanh(np.arctanh(r) + 1.96 * se), n=len(m), note=f"{len(x)} eligible, {len(x) - len(m)} without adequacy")
    rng = np.random.default_rng(2026)
    bs = [stats.spearmanr(*m.iloc[rng.integers(0, len(m), len(m))][["bi2", "adequacy_d4_7"]].T.to_numpy()).statistic
          for _ in range(2000)]
    add("S", "early reflux (burden index bi2) vs energy adequacy days 4-7", "CI valid with ties and n = 35",
        "patient bootstrap percentile CI (2000)", r, np.nanquantile(bs, .025), np.nanquantile(bs, .975), n=len(m),
        note=f"ties in bi2: {m.bi2.nunique()} distinct values")


# ================================================================ W  dose workbook
def sec_W():
    if not XL.exists():
        print("W: dose workbook not found (config.DOSE_WORKBOOK); section skipped")
        return False
    import openpyxl
    from lifelines import CoxPHFitter
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    drugs = {"Midazolam [mg]": ("midazolam_mg", 50), "Propofol [mg]": ("propofol_mg", 1000),
             "Fentanyl [mcg]": ("fentanyl_mcg", 1000), "Remifentanil [mcg]": ("remifentanil_mcg", 5000)}
    ws = openpyxl.load_workbook(XL, read_only=True, data_only=True)["Treated Group"]
    it = ws.iter_rows(values_only=True)
    hdr = list(next(it))
    ci = {h: hdr.index(h) for h in ["Record & System ID", "Start time (hh:mm)", *drugs]}
    recs = []
    for r in it:
        pid = r[ci["Record & System ID"]]
        if not isinstance(pid, str) or not pid.startswith("RMC1-"):
            continue
        rec = {"patient_id": pid[-3:], "start": r[ci["Start time (hh:mm)"]]}
        for h_, (nm, _) in drugs.items():
            v = r[ci[h_]]
            rec[nm] = float(v) if isinstance(v, (int, float)) else 0.0
        recs.append(rec)
    dose = pd.DataFrame(recs).dropna(subset=["start"])
    dose["study_date"] = (pd.to_datetime(dose.start) - pd.Timedelta(hours=7)).dt.date.astype(str)
    dose = dose.groupby(["patient_id", "study_date"], as_index=False)[[v[0] for v in drugs.values()]].sum()
    fr = frame()
    d = fr[fr.full_day & fr.in_analysis_set].merge(dose, on=["patient_id", "study_date"], how="left")
    d["miss"] = d.midazolam_mg.isna().astype(float)
    g = gee_miss(d, "miss", "log2_rate + device_day")
    lo, hi = g.conf_int().loc["log2_rate"]
    add("W", "dose-workbook models (264 of 287 full days have a dose row)",
        "missingness unrelated to the exposure (complete-case validity)",
        "GEE logistic: dose row missing ~ log2 rate + device day; OR per doubling", np.exp(g.params["log2_rate"]),
        np.exp(lo), np.exp(hi), g.pvalues["log2_rate"], len(d), f"{int(d.miss.sum())} days missing")
    d = d.dropna(subset=["midazolam_mg"]).copy()
    d["day_cat"] = np.minimum(d.device_day, 8).astype(int)
    yn = []
    for nm, _ in drugs.values():
        d[nm + "_yn"] = (d[nm] > 0).astype(float)
        d[nm + "_yn_pm"] = d.groupby("patient_id")[nm + "_yn"].transform("mean")
        yn += [nm + "_yn", nm + "_yn_pm"]
    for c in ("dexmedetomidine", "ketamine"):
        d[c] = d[c].astype(float)
        d[c + "_pm"] = d.groupby("patient_id")[c].transform("mean")
    terms = yn + ["dexmedetomidine", "dexmedetomidine_pm", "ketamine", "ketamine_pm", "day_cat:C"]
    fe = " + ".join([v[0] + "_yn" for v in drugs.values()] + ["dexmedetomidine", "ketamine", "C(day_cat)"])
    count_checks("W", "dexmedetomidine day, within patients (dose workbook, all agents)", d, "n_all", terms,
                 "dexmedetomidine", fe, "dexmedetomidine", "monitored_h", "patient_id")
    dev = pd.DataFrame({c: d[c] - d.groupby("patient_id")[c].transform("mean")
                        for c in [v[0] + "_yn" for v in drugs.values()] + ["dexmedetomidine", "ketamine"]})
    Xv = sm.add_constant(dev.to_numpy())
    vif = {c: variance_inflation_factor(Xv, i + 1) for i, c in enumerate(dev.columns)}
    add("W", "all-agents sedation model", "no harmful collinearity between drug days",
        "variance inflation factors of the within-patient drug-day terms (max)", max(vif.values()), n=len(d),
        note="; ".join(f"{k.replace('_yn', '')} {v:.2f}" for k, v in vif.items()))
    for cut, lab in ((0.5, "median (published)"), (2 / 3, "upper tertile")):
        med = d.loc[d.midazolam_mg > 0, "midazolam_mg"].quantile(cut)
        d["c_low"] = ((d.midazolam_mg > 0) & (d.midazolam_mg <= med)).astype(float)
        d["c_high"] = (d.midazolam_mg > med).astype(float)
        for c in ("c_low", "c_high"):
            d[c + "_pm"] = d.groupby("patient_id")[c].transform("mean")
        others = [t for o, _ in drugs.values() if o != "midazolam_mg" for t in (o + "_yn", o + "_yn_pm")]
        terms = ["c_low", "c_high", "c_low_pm", "c_high_pm", "dexmedetomidine", "dexmedetomidine_pm", "ketamine",
                 "ketamine_pm"] + others + ["day_cat:C"]
        an = f"high-dose midazolam day (> {med:.0f} mg, {lab}), within patients, + other drugs"
        if cut == 0.5:
            fe = " + ".join(["c_low", "c_high", "dexmedetomidine", "ketamine"] +
                            [o + "_yn" for o, _ in drugs.values() if o != "midazolam_mg"] + ["C(day_cat)"])
            count_checks("W", an, d, "n_all", terms, "c_high", fe, "c_high", "monitored_h", "patient_id")
        else:
            t = glmm_fit(d, "n_all", terms, "monitored_h", "patient_id").table().set_index("term").loc["c_high"]
            add("W", "high-dose midazolam day, within patients", "result not driven by the cut point",
                f"high dose = above the {lab} of non-zero daily doses ({med:.0f} mg)", t.RR, t.RR_lo, t.RR_hi, t.p, len(d))
    # Cox and Fine-Gray with early doses
    le = pd.read_csv(DATA_DIR / "analysis" / "landmark_exposure.csv", dtype={"patient_id": str})
    le = le[le.n_full_d1_3 >= 2]
    early = (fr[fr.clinical_day.between(1, 3)].merge(dose, on=["patient_id", "study_date"], how="left")
             .groupby("patient_id")[[v[0] for v in drugs.values()]].mean())
    x = le.merge(outcomes(), on="patient_id").merge(early.reset_index(), on="patient_id", how="left")
    x["log2_rate"] = np.log2(x.episodes_per_h + 0.25)
    x["ext"] = (~x.icu_death.astype(bool)).astype(int)
    dcols = []
    for nm, unit in drugs.values():
        x[nm + "_u"] = x[nm].fillna(0) / unit
        dcols.append(nm + "_u")
    cols = ["log2_rate", "apache2"] + dcols
    an = "extubation alive, + APACHE II + day 1-3 doses (4 drugs)"
    cph = CoxPHFitter().fit(x[cols + ["ventilation_days", "ext"]], "ventilation_days", "ext")
    s = cph.summary.loc["log2_rate"]
    add("W", an, "specification", "reproduces the reported HR (lifelines, Efron ties)", s["exp(coef)"],
        s["exp(coef) lower 95%"], s["exp(coef) upper 95%"], s["p"], len(x))
    cox_checks(an + " [statsmodels]", x.reset_index(drop=True), cols, boot=args.boot, spline=False, sec="W")
    for pen in (0.1, 0.5):
        cp = CoxPHFitter(penalizer=pen).fit(x[cols + ["ventilation_days", "ext"]], "ventilation_days", "ext")
        s = cp.summary.loc["log2_rate"]
        add("W", an, "too few events per parameter (33 events, 6 parameters)", f"ridge-penalised Cox (penalizer {pen})",
            s["exp(coef)"], s["exp(coef) lower 95%"], s["exp(coef) upper 95%"], s["p"], len(x))
    xc = x.copy()
    xc.loc[xc.ext == 0, "ventilation_days"] = x.ventilation_days.max() + 1
    cph, t = ph_test(xc, cols)
    s = cph.summary.loc["log2_rate"]
    add("W", "Fine-Gray, + APACHE II + doses", "specification", "reproduces the reported subdistribution HR",
        s["exp(coef)"], s["exp(coef) lower 95%"], s["exp(coef) upper 95%"], s["p"], len(xc))
    for c in cols:
        add("W", "Fine-Gray, + APACHE II + doses", "proportional subdistribution hazards",
            f"Schoenfeld residual test, {c} (rank time)", t[(c, "rank")][0], p=t[(c, "rank")][1], n=len(xc))
    fg_split("W", "Fine-Gray, + APACHE II + doses", x, xc, ["apache2"] + dcols)
    return True


# ================================================================ run
SEC = {"M": sec_M, "P": sec_P, "H": sec_H, "D": sec_D, "L": sec_L, "G": sec_G, "C": sec_C, "S": sec_S, "W": sec_W}
ran = []
for s in SECTIONS:
    n0 = len(rows)
    try:
        ok = SEC[s]()
        if ok is not False:
            ran.append(s)
    except Exception as ex:  # keep the other sections
        add(s, "section failed", "", f"{type(ex).__name__}: {str(ex)[:200]}")
        ran.append(s)
    print(f"section {s}: {len(rows) - n0} rows")
new = pd.DataFrame(rows)
if CSV.exists():
    old = pd.read_csv(CSV)
    new = pd.concat([old[~old.section.isin(ran)], new], ignore_index=True)
order = {k: i for i, k in enumerate(SEC)}
new = new.sort_values("section", key=lambda s: s.map(order), kind="stable")
new.round(4).to_csv(CSV, index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 70)
print(pd.DataFrame(rows).drop(columns=["note"]).round(3).to_string(index=False))
