"""Event scale: what the ventilator shows in the minutes around each reflux onset.

For every reflux onset during invasive ventilation that follows >= ISO_MIN minutes without reflux ("isolated"), the
minute signals from -PRE to +POST minutes are expressed as change from the onset's own baseline (-20..-11 min). Each
onset gets one control: a random ventilated minute within +-3 h in the same patient with no reflux onset from -ISO_MIN
to +POST (so time of day and the patient's state that day are matched).
Inference treats the patient as the unit: per-patient mean curves (onset minus control), then the mean over patients
with a patient-bootstrap 95% CI.

Signals: respiratory rate, minute ventilation, tidal volume, peak pressure, VCO2, EtCO2, dynamic compliance; and the
feed pump (share of the minute running) as context. Artefact minutes are excluded.

Writes results/explore/14_peri_event_curves.csv and 14_peri_event_windows.csv.
"""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
from config import DATA_DIR, OUT_DIR  # noqa: E402
T2, T3, T4, OUT = DATA_DIR / "t2" / "minute", DATA_DIR / "t3", DATA_DIR / "t4", OUT_DIR / "explore"
PRE, POST, ISO_MIN, BASE, CTL_WIN = 20, 20, 15, (-20, -11), 180
SIG = {"rr_total": "respiratory rate (/min)", "mv_exp": "minute ventilation (L/min)", "vte": "tidal volume (mL)",
       "pip": "peak pressure (cmH2O)", "vco2_ml_min": "VCO2 (mL/min)", "etco2": "EtCO2", "cdyn": "compliance",
       "feed_run": "feed pump running (share of minute)"}
rng = np.random.default_rng(7)

con = duckdb.connect()
con.sql("SET enable_progress_bar = false")
m = con.sql(f"""
    SELECT right(m.folder_id, 3) AS fid, m.minute_pc AS t, m.ree_valid_frac,
           m.rr_total, m.mv_exp, m.vte, m.pip, m.vco2_ml_min, m.etco2, m.cdyn,
           m.fpf20_running_s / nullif(m.covered_s, 0) AS feed_run, (a.fid IS NOT NULL) AS artefact
    FROM read_parquet('{T2.as_posix()}/*.parquet') m
    LEFT JOIN read_parquet('{(T4 / 'artefact_minute_list.parquet').as_posix()}') a
           ON a.fid = right(m.folder_id, 3) AND a.t = m.minute_pc
    WHERE right(m.folder_id, 3) <> '002'""").df()
ev = con.sql(f"""SELECT right(folder_id, 3) AS fid, type, CAST(start_t AS TIMESTAMP) AS t0, CAST(end_t AS TIMESTAMP) AS t1
                 FROM read_parquet('{(T3 / 'reflux_events_7s_drain.parquet').as_posix()}')
                 WHERE right(folder_id, 3) <> '002' ORDER BY fid, t0""").df()

curves, n_on, n_pt = [], 0, set()
lags = np.arange(-PRE, POST + 1)
for fid, g in m.groupby("fid"):
    g = g.sort_values("t")
    t_first = g.t.min()
    idx = ((g.t - t_first).dt.total_seconds() // 60).astype(int).to_numpy()
    n = idx.max() + 1
    vent = np.zeros(n, bool)
    vent[idx] = (g.ree_valid_frac.to_numpy() >= 0.5) & (g.rr_total.fillna(0).to_numpy() > 0) & ~g.artefact.to_numpy()
    arr = {}
    for s in SIG:
        a = np.full(n, np.nan)
        v = g[s].to_numpy(dtype=float)
        if s != "feed_run":
            v = np.where(v > 0, v, np.nan)
        a[idx] = v
        a[~vent] = np.nan
        arr[s] = a
    e = ev[ev.fid == fid]
    if e.empty:
        continue
    on = ((e.t0 - t_first).dt.total_seconds() // 60).astype(int).to_numpy()
    off = ((e.t1 - t_first).dt.total_seconds() // 60).astype(int).to_numpy()
    prev_end = np.concatenate([[-10 ** 9], np.maximum.accumulate(off)[:-1]])
    iso = (on - prev_end >= ISO_MIN) & (on - PRE >= 0) & (on + POST < n)
    busy = np.zeros(n + 1, int)                                    # any reflux onset per minute, for control picking
    np.add.at(busy, np.clip(on, 0, n), 1)
    cs = np.concatenate([[0], np.cumsum(busy)])
    day = ((g.t - pd.Timedelta(hours=7)).dt.normalize().to_numpy())
    dmap = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    dmap[idx] = day
    for k, o in enumerate(on[iso]):
        if not vent[o - 3:o + 1].all() or np.isnan(arr["rr_total"][o + BASE[0]:o + BASE[1] + 1]).all():
            continue
        # control: same study day, ventilated, no onset in [c - ISO_MIN, c + POST]
        same = np.flatnonzero(vent[max(0, o - CTL_WIN):o + CTL_WIN + 1]) + max(0, o - CTL_WIN)   # within +-3 h
        same = same[(same - PRE >= 0) & (same + POST < n)]
        ok = [c for c in rng.choice(same, size=min(len(same), 60), replace=False)
              if cs[c + POST + 1] - cs[c - ISO_MIN] == 0 and vent[c - 3:c + 1].all()] if len(same) else []
        if not ok:
            continue
        c = ok[0]
        row = {"fid": fid, "type": e.type.to_numpy()[iso][k]}
        for s in SIG:
            for lab, z in (("on", o), ("ctl", c)):
                w = arr[s][z - PRE:z + POST + 1]
                b = np.nanmean(arr[s][z + BASE[0]:z + BASE[1] + 1]) if not np.isnan(
                    arr[s][z + BASE[0]:z + BASE[1] + 1]).all() else np.nan
                row[f"{s}|{lab}"] = w - b
        curves.append(row)
        n_on += 1
        n_pt.add(fid)
print(f"isolated ventilated onsets with a matched control: {n_on} in {len(n_pt)} patients")

# per patient mean curves, then mean over patients with patient bootstrap
rows, wins = [], []
WINDOWS = {"pre -10..-6": (-10, -6), "pre -5..-1": (-5, -1), "onset 0": (0, 0), "post +1..+5": (1, 5),
           "post +6..+15": (6, 15)}
for typ in ("all", "minor", "massive"):
    sub = [r for r in curves if typ == "all" or r["type"] == typ]
    fids = sorted({r["fid"] for r in sub})
    for s, lab in SIG.items():
        per = []
        for f in fids:
            on_ = np.array([r[f"{s}|on"] for r in sub if r["fid"] == f])
            ct_ = np.array([r[f"{s}|ctl"] for r in sub if r["fid"] == f])
            if np.isnan(on_).all():
                continue
            per.append((np.nanmean(on_, 0), np.nanmean(ct_, 0), len(on_)))
        if len(per) < 5:
            continue
        D = np.array([p[0] - p[1] for p in per])
        ON = np.array([p[0] for p in per])
        boot = np.array([np.nanmean(D[rng.integers(0, len(D), len(D))], 0) for _ in range(2000)])
        mean = np.nanmean(D, 0)
        lo, hi = np.nanquantile(boot, 0.025, 0), np.nanquantile(boot, 0.975, 0)
        for i, L in enumerate(lags):
            rows.append({"type": typ, "signal": lab, "lag_min": L, "onset_minus_control": mean[i], "lo": lo[i],
                         "hi": hi[i], "onset_change": np.nanmean(ON, 0)[i], "patients": len(per)})
        for wn, (a, b) in WINDOWS.items():
            sel = (lags >= a) & (lags <= b)
            dw = np.nanmean(D[:, sel], 1)
            bw = np.array([np.nanmean(dw[rng.integers(0, len(dw), len(dw))]) for _ in range(2000)])
            wins.append({"type": typ, "signal": lab, "window": wn, "diff": np.nanmean(dw),
                         "lo": np.nanquantile(bw, 0.025), "hi": np.nanquantile(bw, 0.975),
                         "patients_up": int((dw > 0).sum()), "patients": int((~np.isnan(dw)).sum()),
                         "onsets": int(sum(p[2] for p in per))})
cur, win = pd.DataFrame(rows), pd.DataFrame(wins)
cur.round(4).to_csv(OUT / "14_peri_event_curves.csv", index=False)
win.round(4).to_csv(OUT / "14_peri_event_windows.csv", index=False)
pd.set_option("display.width", 200)
print(win[win.type == "all"].round(3).to_string(index=False))
print(win[win.type != "all"].query("window in ['pre -5..-1', 'post +1..+5']").round(3).to_string(index=False))
