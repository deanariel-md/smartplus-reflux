"""Descriptive companion to association_landmark.py: outcome distributions and a figure.

Descriptive only, not a test. It recomputes the per-patient outcomes with the same
definitions as scripts/association_landmark.py (and checks that the primary Spearman matches its output),
then summarises them and plots adequacy on days 4-7 against the burden index.

Run after association_landmark.py:  python scripts/describe_outcomes.py
Writes results/association/outcome_summary.csv, adequacy_by_bi2.csv and fig_bi2_adequacy.png.
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
from config import DATA_DIR, OUT_DIR  # noqa: E402
CL, OUTCOMES, OUT = DATA_DIR / "clinical", DATA_DIR / "clinical" / "outcomes", OUT_DIR / "association"
BLUE, GREY, INK = "#2a78d6", "#8a8f98", "#2b2f36"

expo = pd.read_csv(DATA_DIR / "analysis" / "landmark_exposure.csv", dtype={"patient_id": str})
expo = expo[expo.eligible]
cov = pd.read_parquet(CL / "patient_day_covariates.parquet")
cov = cov[cov.arm == "treated"][["patient_id", "clinical_day", "day_ratio"]]
o = pd.read_parquet(OUTCOMES / "patient_day_outcomes.parquet")
o = o[o.arm == "treated"][["patient_id", "clinical_day", "ree_kcal_treated", "delivered_kcal", "feeding_stops"]]
o["feeding_stop"] = o.feeding_stops.notna() & (o.feeding_stops.astype(str).str.strip() != "")
o = cov.merge(o.drop(columns="feeding_stops"), on=["patient_id", "clinical_day"], how="left").rename(
    columns={"ree_kcal_treated": "ree"})
ok = (o.day_ratio >= 0.5) & (o.ree > 0) & (o.delivered_kcal >= 0)
o["adequacy"] = np.where(ok, o.delivered_kcal / (o.ree * o.day_ratio), np.nan)
o["deficit_per_day"] = np.where(ok, o.ree - o.delivered_kcal / o.day_ratio, np.nan)
o["deficit"] = np.where(ok, o.ree * o.day_ratio - o.delivered_kcal, np.nan)

d47, d17 = o[o.clinical_day.between(4, 7)], o[o.clinical_day.between(1, 7)]
pt = expo[["patient_id", "bi2", "bi3"]].copy()
pt["adequacy_d4_7"] = pt.patient_id.map(d47.groupby("patient_id").adequacy.median())
pt["n_days_d4_7"] = pt.patient_id.map(d47.groupby("patient_id").adequacy.count()).fillna(0).astype(int)
pt["deficit_d4_7"] = pt.patient_id.map(d47.groupby("patient_id").deficit_per_day.mean())
pt["cum_deficit_d1_7"] = pt.patient_id.map(d17.groupby("patient_id").deficit.sum())
pt["feeding_stop_d4_7"] = pt.patient_id.map(d47.groupby("patient_id").feeding_stop.any()).astype(float)

# Consistency with association_landmark.py.
m = pt.adequacy_d4_7.notna()
rho = stats.spearmanr(pt.bi2[m], pt.adequacy_d4_7[m])[0]
assoc = pd.read_csv(OUT / "spearman.csv")
ref = assoc[(assoc["index"] == "bi2") & (assoc.outcome == "adequacy_d4_7")].rho.iloc[0]
assert abs(rho - ref) < 1e-3, f"recomputed rho {rho:.4f} differs from the association_landmark.py output {ref:.4f}"


def q(s):
    s = s.dropna()
    return f"{s.median():.2f} ({s.quantile(.25):.2f}-{s.quantile(.75):.2f})", len(s)


summ = []
for col, label in (("adequacy_d4_7", "adequacy, days 4-7 (fraction of REE)"),
                   ("deficit_d4_7", "mean daily deficit, days 4-7 (kcal/day)"),
                   ("cum_deficit_d1_7", "cumulative deficit, days 1-7 (kcal)"),
                   ("n_days_d4_7", "valid outcome days, days 4-7")):
    v, n = q(pt[col])
    summ.append({"outcome": label, "median_IQR": v, "n": n})
summ.append({"outcome": "feeding stop on days 4-7", "median_IQR": f"{int(pt.feeding_stop_d4_7.sum())} patients",
             "n": int(pt.feeding_stop_d4_7.notna().sum())})
summ.append({"outcome": "adequacy >= 0.7 on days 4-7", "median_IQR": f"{int((pt.adequacy_d4_7 >= 0.7).sum())} patients",
             "n": int(m.sum())})
summ = pd.DataFrame(summ)
by = pt.groupby("bi2").agg(patients=("patient_id", "size"), adequacy_median=("adequacy_d4_7", "median"),
                           deficit_median=("deficit_d4_7", "median")).reset_index()
summ.to_csv(OUT / "outcome_summary.csv", index=False)
by.round(3).to_csv(OUT / "adequacy_by_bi2.csv", index=False)
print(summ.to_string(index=False))
print("\nBy burden index (bi2):\n", by.round(2).to_string(index=False))

# Figure: adequacy on days 4-7 by bi2, one dot per patient, median per level.
fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
rng = np.random.default_rng(1)
x = pt.bi2[m] + rng.uniform(-0.12, 0.12, int(m.sum()))
ax.scatter(x, pt.adequacy_d4_7[m], s=42, color=BLUE, edgecolor="white", linewidth=1.2, zorder=3)
for _, r in by.dropna(subset=["adequacy_median"]).iterrows():
    ax.plot([r.bi2 - 0.28, r.bi2 + 0.28], [r.adequacy_median] * 2, color=INK, lw=2, solid_capstyle="round", zorder=4)
ax.axhline(1.0, color=GREY, lw=1, ls=(0, (3, 3)), zorder=1)
ax.set_xticks(sorted(pt.bi2.unique()))
ax.set_xlabel("Reflux burden index on days 1-3 (frequency + height tertiles)", color=INK)
ax.set_ylabel("Energy adequacy, days 4-7\n(delivered / measured EE)", color=INK)
fr = assoc[(assoc["index"] == "bi2") & (assoc.outcome == "adequacy_d4_7")].iloc[0]
ax.set_title(f"Early reflux burden vs later energy adequacy\n"
             f"n = {int(m.sum())}; Spearman ρ = {rho:.2f} (95% CI {fr.rho_lo:.2f} to {fr.rho_hi:.2f}); "
             f"bars = medians", fontsize=9.5, color=INK, loc="left")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.spines["left"].set_color(GREY)
ax.spines["bottom"].set_color(GREY)
ax.tick_params(colors=INK, labelsize=9)
ax.grid(axis="y", color="#e6e8eb", lw=0.8, zorder=0)
fig.tight_layout()
fig.savefig(OUT / "fig_bi2_adequacy.png")
print(f"\nWrote {OUT / 'outcome_summary.csv'}, adequacy_by_bi2.csv and fig_bi2_adequacy.png")
