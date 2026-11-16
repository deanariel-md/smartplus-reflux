"""Small descriptive numbers quoted in the paper that no other script writes.

  feeding on clinical days 4-7 (analysis-set days): prescription as a share of measured EE (plan/REE) and delivery as
  a share of the prescription (delivered/plan); day-level medians and the median of patient means.

Writes results/explore/19_paper_extra.csv.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
from config import OUT_DIR  # noqa: E402
OUT = OUT_DIR / "explore"

fr = pd.read_parquet(OUT / "frame_patient_day.parquet")
d = fr[fr.in_analysis_set & fr.clinical_day.between(4, 7)]
rows = []
for c, lab in (("plan_frac", "prescribed kcal / measured EE, days 4-7"), ("eff", "delivered / prescribed, days 4-7")):
    pt = d.groupby("patient_id")[c].mean().dropna()
    rows.append({"item": lab, "day_median": d[c].median(), "days": int(d[c].notna().sum()),
                 "patient_mean_median": pt.median(), "q25": pt.quantile(.25), "q75": pt.quantile(.75), "patients": len(pt)})
out = pd.DataFrame(rows)
print(out.round(3).to_string(index=False))
out.round(4).to_csv(OUT / "19_paper_extra.csv", index=False)
