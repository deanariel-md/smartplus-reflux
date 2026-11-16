"""Reflux and new respiratory infection (a VAP proxy), from the trial report's adverse-event table.

Source: text extract of the adverse-event table of the final trial report (config.ADVERSE_EVENT_TEXT),
Table 11 "Adverse events": patient, AE term, anticipated, start date, device relation.

Respiratory infection = a new infection in sputum, bronchial lavage / BAL or tracheal aspirate, or
"pneumonia". Day of onset = calendar days from the patient's first study day (master workbook); onset on
study day >= 4 counts as acquired after the exposure window (ventilated > 48 h, so VAP-compatible).

Landmark analysis: reflux on clinical days 1-3 (>= 2 full days) vs a new respiratory infection on day >= 4,
in patients still in the study on day 4. Also: new bloodstream / urine infection as a negative-control
outcome (reflux should not cause these).

Writes data/clinical/outcomes/adverse_events_ctr.csv and results/explore/7_infection*.csv.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
from config import DATA_DIR, OUT_DIR, ADVERSE_EVENT_TEXT  # noqa: E402
SRC = ADVERSE_EVENT_TEXT
OUT = OUT_DIR / "explore"

L = SRC.read_text(encoding="utf-8", errors="replace").splitlines()
start = next(i for i, l in enumerate(L) if l.strip().startswith("Table 11. Adverse events"))
rows, i = [], start
while i < len(L) - 5:
    if re.fullmatch(r"RMC1-\d{3}", L[i].strip()):
        rows.append([L[i + k].strip() for k in range(5)])
        i += 5
    else:
        i += 1
ae = pd.DataFrame(rows, columns=["pid", "event", "anticipated", "date", "relation"])
ae["patient_id"] = ae.pid.str[-3:]
ae["date"] = pd.to_datetime(ae.date, dayfirst=True, errors="coerce")
ev = ae.event.str.lower()
# the table has typos ("sputom", "spurum", "sputuim", "broncial", "branchial")
ae["resp"] = ev.str.contains(r"sput|spurum|bron|branch|lavage|\bbal\b|trache|\bvap\b") | (
    ev.str.contains("pneumonia") & ~ev.str.contains("kleb"))
ae["blood_urine"] = ev.str.contains(r"in blood|blood infection|bacteremia|sepsis|in urine")

base = pd.read_csv(DATA_DIR / "clinical" / "baseline_patient.csv", dtype={"patient_id": str},
                   parse_dates=["first_study_day_start"])
base["patient_id"] = base.patient_id.str.zfill(3)

# Date typos: a date outside the stay (study day < -3 or > n_clinical_days + 30) or after the patient's
# death is retried with the year shifted by one or the month set to the study's start month (or the next);
# the first candidate inside the stay and before death is kept and flagged.
st = ae.merge(base[["patient_id", "first_study_day_start", "n_clinical_days"]], on="patient_id", how="left")
death = ae[ev.str.fullmatch(r"\s*death\s*")].groupby("patient_id").date.min()
ae["date_fixed"] = False
for i, r in st.iterrows():
    if pd.isna(r.date) or pd.isna(r.first_study_day_start):
        continue
    s0, dmax = r.first_study_day_start.normalize(), death.get(r.patient_id, pd.NaT)

    def ok(d):
        sd = (d - s0).days + 1
        return -3 <= sd <= r.n_clinical_days + 30 and (pd.isna(dmax) or d <= dmax)

    if ok(r.date):
        continue
    cands = [r.date + pd.DateOffset(years=k) for k in (-1, 1)]
    for mo in (s0.month, s0.month % 12 + 1):
        try:
            cands.append(r.date.replace(year=s0.year if mo >= s0.month else s0.year + 1, month=mo))
        except ValueError:
            pass
    fix = next((d for d in cands if ok(d)), pd.NaT)
    ae.loc[i, ["date", "date_fixed"]] = [fix, True]
print(f"AE dates fixed: {int(ae.date_fixed.sum())} "
      f"({', '.join(sorted(ae[ae.date_fixed].patient_id.unique()))}); unusable: {int(ae.date.isna().sum())}")
(DATA_DIR / "clinical" / "outcomes").mkdir(parents=True, exist_ok=True)
ae.drop(columns="pid").to_csv(DATA_DIR / "clinical" / "outcomes" / "adverse_events_ctr.csv", index=False)

tr = base[base.arm == "treated"][["patient_id", "first_study_day_start", "apache2"]]
a = ae.merge(tr, on="patient_id")
a["study_day"] = (a.date.dt.normalize() - a.first_study_day_start.dt.normalize()).dt.days + 1


def first_day(mask):
    return a[mask].groupby("patient_id").study_day.min()


resp_day, bu_day = first_day(a.resp), first_day(a.blood_urine)
print(f"treated AEs: {len(a)}; respiratory infections: {int(a.resp.sum())} in {a[a.resp].patient_id.nunique()} patients; "
      f"onset day median {a[a.resp].study_day.median():.0f}")

le = pd.read_csv(DATA_DIR / "analysis" / "landmark_exposure.csv", dtype={"patient_id": str})
x = le[(le.n_full_d1_3 >= 2) & le.in_study_day4.astype(bool)].copy()
x["resp_day"] = x.patient_id.map(resp_day)
x["bu_day"] = x.patient_id.map(bu_day)
x["resp_before4"] = x.resp_day < 4
x["resp_after3"] = (x.resp_day >= 4).astype(float)
x["bu_after3"] = (x.bu_day >= 4).astype(float)
x["log2_rate"] = np.log2(x.episodes_per_h + 0.25)
x["log2_massive_rate"] = np.log2(x.n_massive_d1_3 / x.monitored_h_d1_3 + 0.1)
x["log2_pct"] = np.log2(x.pct_time_reflux + 0.05)
xx = x[~x.resp_before4.fillna(False)]            # drop infections already present on days 1-3
print(f"landmark: {len(x)} patients; {int(x.resp_before4.fillna(False).sum())} with a respiratory infection on "
      f"days 1-3 excluded; outcome on day >= 4: {int(xx.resp_after3.sum())} of {len(xx)}")

res = []
for y, label in (("resp_after3", "new respiratory infection, day >= 4"),
                 ("bu_after3", "negative control: new blood/urine infection, day >= 4")):
    for xv in ("episodes_per_h", "pct_time_reflux", "massive_share", "log2_massive_rate", "bi2", "bi3"):
        m = xx.dropna(subset=[xv])
        a1, a0 = m[m[y] == 1][xv], m[m[y] == 0][xv]
        p = stats.mannwhitneyu(a1, a0).pvalue if len(a1) and len(a0) else np.nan
        row = {"outcome": label, "exposure_d1_3": xv, "events": int(m[y].sum()), "n": len(m),
               "median_with": a1.median(), "median_without": a0.median(), "mann_whitney_p": p}
        if xv in ("log2_massive_rate",) or xv == "episodes_per_h":
            v = "log2_rate" if xv == "episodes_per_h" else xv
            try:
                f = smf.logit(f"{y} ~ {v} + apache2", m).fit(disp=0)
                row.update(OR_per_doubling_adj_apache=np.exp(f.params[v]),
                           lo=np.exp(f.conf_int().loc[v, 0]), hi=np.exp(f.conf_int().loc[v, 1]), p_logit=f.pvalues[v])
            except Exception as ex:
                row["note"] = str(ex)[:60]
        res.append(row)
res = pd.DataFrame(res)
print(res.round(3).to_string(index=False))
res.round(4).to_csv(OUT / "7_infection_landmark.csv", index=False)
x[["patient_id", "episodes_per_h", "massive_share", "resp_day", "bu_day"]].to_csv(
    DATA_DIR / "clinical" / "outcomes" / "infection_landmark_frame.csv", index=False)
print(f"\nWrote {OUT / '7_infection_landmark.csv'}")

# ---------------------------------------------------------------- event study: reflux around infection onset
import sys  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
import glmm_poisson as gp  # noqa: E402

fr = pd.read_parquet(OUT / "frame_patient_day.parquet")
fr = fr[fr.full_day].copy()
es_rows = []
for name, onset in (("respiratory infection", resp_day), ("blood/urine infection", bu_day),
                    ("any new infection", first_day(a.event.str.lower().str.contains("infection")))):
    f = fr.copy()
    f["onset"] = f.patient_id.map(onset)
    f["rel"] = f.clinical_day - f.onset
    f["win"] = np.select([f.rel.between(-3, -2), f.rel.between(-1, 0), f.rel.between(1, 2), f.rel >= 3],
                         ["a -3..-2", "b -1..0", "c +1..+2", "d >=+3"], "0 ref")     # ref: >3 days before, or none
    f["day_cat"] = np.minimum(f.device_day, 8).astype(int)
    for scope, g in (("all patients", f), ("infected patients only (within-patient)", f[f.onset.notna()])):
        X, names = gp.design(g, ["win:C", "day_cat:C"])
        t = gp.fit(g.n_all, X, np.log(g.monitored_h.to_numpy()), g.patient_id, names).table()
        t = t[t.term.str.startswith("win")]
        for _, r in t.iterrows():
            es_rows.append({"onset": name, "scope": scope, "window": r.term[4:-1],
                            "patients_with_onset": int(g[g.onset.notna()].patient_id.nunique()),
                            "ref_days": int((g.win == "0 ref").sum()),
                            "days_in_window": int((g.win == r.term[4:-1]).sum()), "RR_vs_ref": r.RR, "lo": r.RR_lo,
                            "hi": r.RR_hi, "p": r.p})
es = pd.DataFrame(es_rows)
print("\nEpisode rate around infection onset (days relative to onset; ref = days > 3 before onset or never "
      "infected; adjusted for device day)\n", es.round(3).to_string(index=False))
es.round(4).to_csv(OUT / "7_infection_event_study.csv", index=False)
