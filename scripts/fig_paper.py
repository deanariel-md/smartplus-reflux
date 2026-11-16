"""Figures for the clinical paper (Figs 1-4) and Additional file 1 (Figs S1-S3).

Reads only tables the pipeline has written (results/explore, results/descriptive/tables, data/validation), the T2 minute
grid and T3 event tables (for the example day, the hourly heat map and the PEEP check), and the outcome and
FiO2/PEEP tables. pandas and pyarrow only, so it runs anywhere the pipeline runs.
Writes results/paper_figures/Fig1-4 and FigS1-S3 as .png (300 dpi), .pdf and .jpg.
"""
import glob
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
from config import DATA_DIR, OUT_DIR  # noqa: E402
EX, DT, VA = OUT_DIR / "explore", OUT_DIR / "descriptive" / "tables", DATA_DIR / "validation"
T2, T3, T4, CL = DATA_DIR / "t2" / "minute", DATA_DIR / "t3", DATA_DIR / "t4", DATA_DIR / "clinical"
OUT = OUT_DIR / "paper_figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"     # categorical slots 1-3
INK, MUTED, GRID, LIGHT = "#1f2328", "#57606a", "#e6e5e1", "#9db9e3"
SEQ = LinearSegmentedColormap.from_list("seq", ["#e1e8f3", "#9db9e3", "#2a78d6", "#123f78"])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titlesize": 8.5, "axes.titleweight": "bold",
                     "axes.titlelocation": "left", "legend.frameon": False, "axes.linewidth": 0.8})
SUB2 = {"VCO2": "VCO$_2$", "FiO2": "FiO$_2$", "cmH2O": "cmH$_2$O"}


def tag(ax, letter, x=-0.14):
    ax.text(x, 1.07, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", color=INK, va="bottom")


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def ref1(ax, axis="x"):
    (ax.axvline if axis == "x" else ax.axhline)(1, color=MUTED, lw=0.8, ls=(0, (3, 3)))


def logticks(ax, ticks, axis="x"):
    getattr(ax, f"set_{axis}scale")("log")
    getattr(ax, f"set_{axis}ticks")(ticks)
    getattr(ax, f"set_{axis}ticklabels")([f"{t:g}" for t in ticks])
    ax.minorticks_off()


def save(fig, name):
    for ext, kw in (("png", {"dpi": 300, "metadata": {"Software": None}}),
                    ("pdf", {"metadata": {"Creator": None, "Producer": None}}),
                    ("jpg", {"dpi": 200, "pil_kwargs": {"quality": 92}})):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", facecolor="white", **kw)
    plt.close(fig)
    print("wrote", name)


def forest(ax, rows, colour_of=None, xlim=None, ticks=None, label_x=None, fmt="{:.2f} ({:.2f}–{:.2f})"):
    """rows: (label, est, lo, hi); top to bottom."""
    n = len(rows)
    for i, (lab, e, lo, hi) in enumerate(rows):
        y = n - 1 - i
        c = colour_of(i, e, lo, hi) if colour_of else BLUE
        ax.plot([lo, hi], [y, y], color=c, lw=1.6, solid_capstyle="round")
        ax.plot(e, y, "o", color=c, ms=4.5, mec="white", mew=0.8)
        if label_x is not None:
            ax.text(label_x, y, fmt.format(e, lo, hi), fontsize=6.5, va="center", ha="left", color=INK,
                    transform=ax.get_yaxis_transform())
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=7)
    ax.set_ylim(-0.6, n - 0.4)
    if ticks:
        logticks(ax, ticks)
    if xlim:
        ax.set_xlim(*xlim)
    ref1(ax)
    grid(ax, "x")


def pick(df, col, contains):
    hit = df[df[col].astype(str).str.contains(contains, regex=False)]
    assert len(hit) >= 1, contains
    return hit.iloc[-1]


# ---------------------------------------------------------------- shared inputs
fr = pd.read_parquet(EX / "frame_patient_day.parquet")
full = fr[fr.full_day & fr.in_analysis_set].copy()
sd = pd.read_csv(EX / "18_sedation_dose.csv")
ac = pd.read_csv(EX / "21_assumption_checks.csv")


def minutes(cols):
    m = pd.read_parquet(T2, columns=["folder_id", "minute_pc"] + cols)
    m["fid"] = m.folder_id.str[-3:]
    m = m[m.fid != "002"].drop(columns="folder_id")
    art = pd.read_parquet(T4 / "artefact_minute_list.parquet")[["fid", "t"]].assign(artefact=True)
    m = m.merge(art, left_on=["fid", "minute_pc"], right_on=["fid", "t"], how="left").drop(columns="t")
    m["artefact"] = m.artefact.fillna(False).astype(bool)
    return m


ev = pd.read_parquet(T3 / "reflux_events_7s_drain.parquet", columns=["folder_id", "type", "start_t", "end_t"])
ev["fid"] = ev.folder_id.str[-3:]
ev["start_t"] = pd.to_datetime(ev.start_t).astype("datetime64[ns]")
ev["end_t"] = pd.to_datetime(ev.end_t).astype("datetime64[ns]")

# ================================================================ Fig 1: continuous reflux over days
cand = full[(full.fid.astype(int) >= 59) & (full.artefact_h == 0) & full.device_day.between(3, 6)]
ex = cand.iloc[(cand.episodes_per_h - full.episodes_per_h.median()).abs().argsort().iloc[0]]
d0 = pd.Timestamp(ex.study_date) + pd.Timedelta(hours=7)
d1 = d0 + pd.Timedelta(days=1)
folder = f"RMC1-{ex.fid}"
evs = ev[(ev.folder_id == folder) & (ev.start_t >= d0) & (ev.start_t < d1)].rename(columns={"start_t": "t0", "end_t": "t1"})
drn = pd.read_parquet(T3 / "drain_runs.parquet")
drn = drn[(drn.folder_id == folder) & (drn.start_t >= d0) & (drn.start_t < d1)].rename(columns={"start_t": "t0", "end_t": "t1"})
bal = pd.concat([pd.read_parquet(f, columns=["folder_id", "start_t", "end_t"])
                 for f in glob.glob(str(T3 / "balloon_inflations" / "*.parquet"))])
bal["start_t"], bal["end_t"] = pd.to_datetime(bal.start_t), pd.to_datetime(bal.end_t)
bal = bal[(bal.folder_id == folder) & (bal.start_t >= d0) & (bal.start_t < d1)].rename(columns={"start_t": "t0", "end_t": "t1"})

fig = plt.figure(figsize=(7.2, 7.4))
outer = fig.add_gridspec(2, 1, height_ratios=[1.45, 2.1], hspace=0.28)
top = outer[0].subgridspec(2, 1, height_ratios=[1.0, 0.45], hspace=0.08)
bot = outer[1].subgridspec(1, 2, width_ratios=[2.05, 1], wspace=0.62)
ax = fig.add_subplot(top[0])
hrs = lambda t: (pd.to_datetime(t) - d0).dt.total_seconds() / 3600   # noqa: E731
tracks = [("Lower sensors (minor)", evs[evs.type == "minor"], BLUE), ("Upper sensors (massive)", evs[evs.type == "massive"], ORANGE),
          ("Drain open", drn, MUTED), ("Balloon inflated", bal, AQUA)]
for i, (lab, g, c) in enumerate(tracks):
    y = len(tracks) - 1 - i
    a, b = hrs(g.t0), hrs(g.t1)
    ax.broken_barh(list(zip(a, np.maximum(b - a, 0.02))), (y - 0.32, 0.64), color=c, lw=0)
    ax.text(-0.3, y, lab, ha="right", va="center", color=INK, fontsize=7.5)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.tick_params(labelbottom=False)
ax.set_xlim(0, 24)
ax.set_title(f"One monitored day (device day {int(ex.device_day)}, {ex.episodes_per_h:.1f} episodes/h, close to the "
             f"cohort median)", fontweight="normal", color=INK)
tag(ax, "a", -0.02)
axh = fig.add_subplot(top[1], sharex=ax)
hc = np.histogram(hrs(evs.t0), bins=np.arange(0, 25))[0]
axh.bar(np.arange(24) + 0.5, hc, width=0.9, color=BLUE, lw=0)
axh.set_ylabel("Episodes\nper hour")
grid(axh)
axh.set_xticks(range(0, 25, 3))
axh.set_xticklabels([f"{(7 + h) % 24:02d}:00" for h in range(0, 25, 3)])

# b: hourly rate, six patients, first six full days
m = minutes(["monitor_s"])
m = m[~m.artefact]
m["hr"] = m.minute_pc.dt.floor("h")
hh = m.groupby(["fid", "hr"]).monitor_s.sum().div(3600).rename("mon_h").reset_index()
hh = hh[hh.mon_h >= 0.75]
ev["hr"] = ev.start_t.dt.floor("h")
hh = hh.merge(ev.groupby(["fid", "hr"]).size().rename("n").reset_index(), on=["fid", "hr"], how="left").fillna({"n": 0})
hh["study_date"] = (hh.hr - pd.Timedelta(hours=7)).dt.date.astype(str)
hh["slot"] = (hh.hr - pd.Timedelta(hours=7)).dt.hour
hh = hh.merge(full[["fid", "study_date", "device_day", "patient_id"]], on=["fid", "study_date"])
pt = full.groupby("patient_id").agg(nd=("full_day", "size"), r=("episodes_per_h", "median"))
pt = pt[pt.nd >= 6].sort_values("r")
six = pt.index[np.linspace(0, len(pt) - 1, 6).round().astype(int)]
ax = fig.add_subplot(bot[0])
mat, breaks = [], []
for k, p in enumerate(six):
    g = hh[hh.patient_id == p]
    for dd in sorted(g.device_day.unique())[:6]:
        r = np.full(24, np.nan)
        gg = g[g.device_day == dd]
        r[gg.slot.values] = (gg.n / gg.mon_h).values
        mat.append(r)
    breaks.append(len(mat))
cmap = SEQ.copy()
cmap.set_bad("white")
im = ax.imshow(np.log10(np.array(mat) + 1), aspect="auto", cmap=cmap, vmin=0, vmax=np.log10(61), interpolation="nearest")
for yb in breaks[:-1]:
    ax.axhline(yb - 0.5, color="white", lw=2.5)
starts = [0] + breaks[:-1]
ax.set_yticks([(a + b - 1) / 2 for a, b in zip(starts, breaks)])
ax.set_yticklabels([f"P{k + 1}" for k in range(len(six))])
ax.set_ylabel("Patient (rows: full days 1–6)")
ax.set_xticks(range(0, 24, 6))
ax.set_xticklabels([f"{(7 + h) % 24:02d}:00" for h in range(0, 24, 6)])
ax.set_title("Hourly rate, six patients, first six full days")
ax.spines[["left", "bottom"]].set_visible(False)
ax.tick_params(length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.015)
cb.set_ticks(np.log10(np.array([0, 2, 5, 10, 20, 60]) + 1))
cb.set_ticklabels(["0", "2", "5", "10", "20", "60"])
cb.set_label("Episodes/h", fontsize=7)
cb.outline.set_visible(False)
tag(ax, "b", -0.08)

# c: rate ratio by device day
ax = fig.add_subplot(bot[1])
co = pd.read_csv(DT / "4_course_over_days_glmm.csv")
co = co[(co.series == "all episodes") & co.term.str.startswith("day_cat")]
dd = [1] + [int(t.split("[")[1].rstrip("]").rstrip("+")) for t in co.term]
rr, lo, hi = [1] + list(co.RR), [1] + list(co.RR_lo), [1] + list(co.RR_hi)
ax.errorbar(dd, rr, yerr=[np.subtract(rr, lo), np.subtract(hi, rr)], fmt="o", color=BLUE, ms=4.5, lw=1.5, capsize=0,
            mec="white", mew=0.8)
logticks(ax, [0.5, 1, 2, 4, 8], "y")
ref1(ax, "y")
ax.set_xticks(dd)
ax.set_xticklabels([str(x) if x < 8 else "8+" for x in dd])
ax.set_xlabel("Device day")
ax.set_ylabel("Rate ratio vs day 1")
ax.set_title("Rise over the stay")
grid(ax)
tag(ax, "c")
save(fig, "Fig1")

# ================================================================ Fig 2: reflux begins within bursts of activity
cu = pd.read_csv(EX / "14_peri_event_curves.csv")
cu = cu[cu.type == "all"]
feed = cu[cu.signal.str.startswith("feed")].set_index("lag_min").onset_minus_control
paused = feed[feed < -0.1].index
fig = plt.figure(figsize=(7.2, 4.9))
gs = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.68)
row2 = gs[1, :].subgridspec(1, 3, width_ratios=[0.62, 1.38, 0.5], wspace=0.0)
panels = [("respiratory rate (/min)", "Respiratory rate, /min", "a"),
          ("minute ventilation (L/min)", "Minute ventilation, L/min", "b"), ("peak pressure (cmH2O)", "Peak pressure, cmH2O", "c")]
for k, (sig, ylab, letter) in enumerate(panels):
    ax = fig.add_subplot(gs[0, k])
    g = cu[cu.signal == sig].sort_values("lag_min")
    if len(paused):
        ax.axvspan(paused.min(), paused.max() + 1, color="#eef0f2", lw=0, zorder=0)
    ax.fill_between(g.lag_min, g.lo, g.hi, color=LIGHT, alpha=0.55, lw=0)
    ax.plot(g.lag_min, g.onset_minus_control, color=BLUE, lw=1.8)
    ax.axhline(0, color=MUTED, lw=0.8, ls=(0, (3, 3)))
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_xlim(-20, 20)
    ax.set_xticks([-20, -10, 0, 10, 20])
    ax.set_xlabel("Minutes from reflux onset")
    for a_, b_ in SUB2.items():
        ylab = ylab.replace(a_, b_)
    ax.set_title(ylab, fontweight="bold")
    ax.set_ylabel("Onset − control")
    grid(ax)
    tag(ax, letter, -0.3)
ax = fig.add_subplot(row2[1])
rs = pick(sd[sd.part.str.contains("any RSBI value in the hour: reflux RR")], "item", "device day")
mv = pd.read_csv(EX / "6_breathing_glmm.csv").set_index("factor").loc["mv (per 2)"]
forest(ax, [("Spontaneous\nbreaths", rs.estimate, rs.lo, rs.hi),
            ("Minute ventilation,\nper 2 L/min", mv.RR_within, mv.lo, mv.hi)],
       xlim=(0.8, 4), ticks=[1, 1.5, 2, 3], label_x=1.02)
ax.set_xlabel("Rate ratio, within patients")
ax.set_title("Hour-level reflux")
tag(ax, "d", -0.42)
save(fig, "Fig2")

# ================================================================ Fig 3: sedative and analgesic days
yn = sd[sd.part.str.startswith("A within-patient RR, drug day yes/no")].set_index("item")
cat = lambda drug, lev: pick(sd[sd.part.str.contains(f"A {drug} dose category vs none, within patients") &   # noqa: E731
                                sd.part.str.contains("other three drugs")], "item", lev)
mlo, mhi, plo, phi = cat("midazolam_mg", "low"), cat("midazolam_mg", "high"), cat("propofol_mg", "low"), cat("propofol_mg", "high")
mmed = mlo["item"].split("non-zero ")[1].rstrip(")")
pmed = plo["item"].split("non-zero ")[1].rstrip(")")
rows = [("Dexmedetomidine day", *yn.loc["dexmedetomidine", ["estimate", "lo", "hi"]]),
        ("Ketamine day", *yn.loc["ketamine", ["estimate", "lo", "hi"]]),
        (f"Midazolam ≤ {mmed} mg/day", mlo.estimate, mlo.lo, mlo.hi),
        (f"Midazolam > {mmed} mg/day", mhi.estimate, mhi.lo, mhi.hi),
        (f"Propofol ≤ {int(pmed):,} mg/day", plo.estimate, plo.lo, plo.hi),
        (f"Propofol > {int(pmed):,} mg/day", phi.estimate, phi.lo, phi.hi),
        ("Fentanyl day", *yn.loc["fentanyl_mcg > 0", ["estimate", "lo", "hi"]]),
        ("Remifentanil day", *yn.loc["remifentanil_mcg > 0", ["estimate", "lo", "hi"]])]
fig, ax = plt.subplots(figsize=(4.6, 3.0))
forest(ax, rows, colour_of=lambda i, e, lo, hi: BLUE if (lo > 1 or hi < 1) else LIGHT,
       xlim=(0.25, 4), ticks=[0.25, 0.5, 1, 2, 4], label_x=1.02)
ax.set_xlabel("Rate ratio vs the same patient's other days")
ax.set_title("Sedative and analgesic days")
save(fig, "Fig3")

# ================================================================ Fig 4: early reflux and extubation
le = pd.read_csv(DATA_DIR / "analysis" / "landmark_exposure.csv", dtype={"patient_id": str})
le = le[le.n_full_d1_3 >= 2]
op = pd.read_parquet(CL / "outcomes" / "outcomes_patient.parquet")
x = le.merge(op[op.arm == "treated"], on="patient_id")
x["tertile"] = pd.qcut(x.episodes_per_h, 3, labels=["Low", "Middle", "High"])
x["event"] = (~x.icu_death.astype(bool)).astype(int)


def km(t, e):
    t, e = np.asarray(t), np.asarray(e)
    s, out = 1.0, [(0.0, 0.0)]
    for u in np.sort(np.unique(t[e == 1])):
        s *= 1 - ((t == u) & (e == 1)).sum() / (t >= u).sum()
        out.append((u, 1 - s))
    out.append((t.max(), out[-1][1]))
    return np.array(out)


fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.3), gridspec_kw={"wspace": 1.0, "width_ratios": [1, 1.05]})
ax = axs[0]
for lab, c in (("Low", "#9db9e3"), ("Middle", "#5a92dd"), ("High", "#123f78")):
    g = x[x.tertile == lab]
    k_ = km(g.ventilation_days, g.event)
    ax.step(k_[:, 0], k_[:, 1], where="post", color=c, lw=2,
            label=f"{lab} (median {g.episodes_per_h.median():.2f}/h, n = {len(g)})")
ax.set_xlim(0, 30)
ax.set_ylim(0, 1.02)
ax.set_xlabel("Days of mechanical ventilation")
ax.set_ylabel("Extubated alive (cumulative)")
ax.set_title("By tertile of reflux on ICU days 1–3")
ax.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(-0.02, -0.2), title="Reflux, days 1–3", title_fontsize=6.5,
          alignment="left")
grid(ax)
tag(ax, "a")

ax = axs[1]
cx = pd.read_csv(EX / "5_ventilation_cox.csv").set_index("model")
ox = pd.read_csv(EX / "10_extubation_oxygenation.csv").set_index("model")
ib = pd.read_csv(EX / "9_extubation_by_infection.csv")
ib = ib.set_index(ib.columns[0])
sb = sd[sd.part.str.startswith("B extubation")].set_index("item")
cxr = lambda r: (r.HR_extubation_per_doubling, r.lo, r.hi)   # noqa: E731
oxr = lambda r: (r.HR_per_doubling, r.lo, r.hi)              # noqa: E731
pen = pick(ac[ac.section == "W"], "check", "ridge-penalised Cox (penalizer 0.1)")
lm = pick(ac[ac.section == "C"], "check", "landmark at day 3")
tv = ib.loc["+ respiratory infection ever (fixed)"]
rows = [("Unadjusted", *cxr(cx.loc["unadjusted"])),
        ("+ APACHE II", *cxr(cx.loc["+ APACHE II"])),
        ("+ midazolam, vasopressor days", *cxr(cx.loc["+ APACHE + midazolam + vasopressor"])),
        ("+ FiO2", *oxr(ox.loc["+ APACHE II + FiO2 d1-3"])),
        ("+ FiO2 and PEEP", *oxr(ox.loc["+ APACHE II + FiO2 + PEEP d1-3"])),
        ("+ positive respiratory culture", tv.HR_log2_rate, tv.lo_log2_rate, tv.hi_log2_rate),
        ("+ sedative doses", *sb.loc["cause-specific Cox + APACHE II + day 1-3 doses (4 drugs)", ["estimate", "lo", "hi"]]),
        ("+ sedative doses, penalised", pen.estimate, pen.lo, pen.hi),
        ("Ventilated after day 3", lm.estimate, lm.lo, lm.hi),
        ("Fine–Gray, + APACHE II", *sb.loc["Fine-Gray + APACHE II", ["estimate", "lo", "hi"]]),
        ("Fine–Gray, + sedative doses", *sb.loc["Fine-Gray + APACHE II + day 1-3 doses (4 drugs)", ["estimate", "lo", "hi"]])]
rows = [(r[0].replace("FiO2", "FiO$_2$"), *map(float, r[1:])) for r in rows]
forest(ax, rows, colour_of=lambda i, e, lo, hi: ORANGE if i >= 9 else BLUE, xlim=(0.8, 3.4), ticks=[1, 1.5, 2, 3],
       label_x=1.02)
ax.axhline(1.5, color=GRID, lw=0.8)
ax.set_xlabel("Hazard ratio for extubation alive, per doubling")
ax.set_title("Across models")
ax.text(-0.9, -0.2, "Blue: cause-specific Cox (ICU death censored); rows 3–9 also adjusted for APACHE II.\n"
        "Orange: Fine–Gray (ICU death as competing risk).", transform=ax.transAxes, fontsize=6.2, color=MUTED, va="top")
tag(ax, "b", -1.0)
save(fig, "Fig4")

# ================================================================ Fig S1: verification of the device states
fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.8), gridspec_kw={"wspace": 0.62})
ax = axs[0]
bc = pd.read_csv(VA / "balloon_coupling_pooled.csv").query("table == 'reflux_events_7s'").set_index("typ")
dr = pd.read_csv(VA / "drainage_rule_by_era.csv")
dr_p = dr.groupby("state").apply(lambda g: (g.n_3_15s * g.drain_open_3_15s).sum() / g.n_3_15s.sum(), include_groups=False)
vals = {"Balloon\nwithin 30 s": (bc.loc["minor", "pooled_p_balloon"], bc.loc["massive", "pooled_p_balloon"]),
        "Drain opens\non a 3–15 s run": (dr_p[11], dr_p[12])}
xx = np.arange(len(vals))
for j, (c, lab) in enumerate(((BLUE, "minor state (lower sensors)"), (ORANGE, "massive state (upper sensors)"))):
    v = [p[j] for p in vals.values()]
    ax.bar(xx + (j - 0.5) * 0.36, v, width=0.34, color=c, label=lab, lw=0)
    for xi, vi in zip(xx, v):
        ax.text(xi + (j - 0.5) * 0.36, vi + 0.02, f"{vi:.0%}", ha="center", fontsize=6.5, color=INK)
ax.set_xticks(xx)
ax.set_xticklabels(vals.keys(), fontsize=7)
ax.set_ylim(0, 1.25)
ax.set_yticks([0, 0.5, 1])
ax.set_yticklabels(["0%", "50%", "100%"])
ax.set_title("Device response by state")
ax.legend(fontsize=6.3, loc="upper left")
grid(ax)
tag(ax, "a")
ax = axs[1]
dc = pd.read_csv(VA / "reflux_daily_comparison.csv")
smy = pd.read_csv(VA / "reflux_validation_summary.csv").query("clock == 'pc'").set_index("type")
for c, typ, rep, ours in ((BLUE, "minor", "report_minor", "ours_minor_pc"), (ORANGE, "massive", "report_massive", "ours_massive_pc")):
    ax.scatter(np.sqrt(dc[rep]), np.sqrt(dc[ours]), s=7, color=c, alpha=0.6, lw=0,
               label=f"{typ}: r = {smy.loc[typ, 'pearson_r']:.3f}")
mx = np.sqrt(max(dc.report_minor.max(), dc.ours_minor_pc.max()))
ax.plot([0, mx], [0, mx], color=MUTED, lw=0.8, ls=(0, (3, 3)))
tk = [0, 25, 100, 225, 400]
ax.set_xticks(np.sqrt([0, 100, 400]))
ax.set_xticklabels([0, 100, 400])
ax.set_yticks(np.sqrt(tk))
ax.set_yticklabels(tk)
ax.set_xlabel("Manufacturer's report, episodes/day")
ax.set_ylabel("This analysis, episodes/day")
ax.set_title(f"Daily counts, {len(dc)} days")
ax.legend(fontsize=6.3, loc="upper left")
grid(ax, "both")
tag(ax, "b")
ax = axs[2]
mp = minutes(["peep"])
mp = mp[~mp.artefact & (mp.peep > 0)]
mp["hr"] = mp.minute_pc.dt.floor("h")
dp = mp.groupby(["fid", "hr"]).peep.median().rename("device").reset_index()
hp = pd.read_parquet(CL / "outcomes" / "fio2_peep_hourly.parquet")
hp = hp[(hp.arm == "treated") & (hp.param == "peep")].copy()
hp["hr"] = hp.t.dt.floor("h")
hp = hp.groupby(["patient_id", "hr"]).value.mean().rename("hospital").reset_index()
idmap = fr[["patient_id", "fid"]].drop_duplicates()
hp = hp.merge(idmap, on="patient_id")
pp = dp.merge(hp, on=["fid", "hr"])
eq = ((pp.device - pp.hospital).abs() <= 1).mean()
print(f"S1c PEEP: {len(pp)} matched hours, {pp.fid.nunique()} patients, within 1 cmH2O {eq:.1%}")
cnt = pp.assign(d=pp.device.round(), h=pp.hospital.round()).groupby(["h", "d"]).size().reset_index(name="n")
ax.scatter(cnt.h, cnt.d, s=np.sqrt(cnt.n) * 4, color=BLUE, alpha=0.7, lw=0)
lim = [0, 20]
cnt = cnt[(cnt.h <= 20) & (cnt.d <= 20)]
ax.plot(lim, lim, color=MUTED, lw=0.8, ls=(0, (3, 3)))
ax.set_xlim(*lim)
ax.set_ylim(*lim)
ax.set_xticks([0, 5, 10, 15, 20])
ax.set_yticks([0, 5, 10, 15, 20])
ax.set_xlabel("Hospital record, cmH$_2$O")
ax.set_ylabel("Device log, cmH$_2$O")
ax.set_title(f"PEEP, {pp.fid.nunique()} patients")
ax.text(0.04, 0.96, f"within 1 cmH$_2$O in {eq:.0%}\nof {len(pp):,} hours", transform=ax.transAxes, fontsize=6.5,
        color=INK, va="top")
grid(ax, "both")
tag(ax, "c")
save(fig, "FigS1")

# ================================================================ Fig S2: burstiness
fig, ax = plt.subplots(figsize=(3.6, 2.8))
sn = pd.read_csv(EX / "12_snapshot_vs_continuous.csv")
for col, c, lab in (("within_factor_2", BLUE, "within twofold of the rest of the day"),
                    ("same_tertile", ORANGE, "same tertile as the rest of the day")):
    ax.plot(sn.window_h, sn[col], color=c, lw=2, marker="o", ms=4, mec="white", mew=0.8, label=lab)
    ax.text(sn.window_h.iloc[-1] + 0.4, sn[col].iloc[-1], f"{sn[col].iloc[-1]:.0%}", color=INK, fontsize=6.5, va="center")
ax.set_xticks(sn.window_h)
ax.set_xlabel("Window length, contiguous monitored hours")
ax.set_ylim(0, 1)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
ax.set_ylabel("Share of windows")
ax.set_title("A short window against the rest of the day")
ax.legend(fontsize=6.5, loc="lower right")
grid(ax)
save(fig, "FigS2")

# ================================================================ Fig S3: reflux before positive respiratory cultures
win = ["a -3..-2", "b -1..0", "c +1..+2", "d >=+3"]
wl = ["−3 to −2", "−1 to 0", "+1 to +2", "≥ +3"]
es = pd.read_csv(EX / "7_infection_event_study.csv")
es = es[es.scope.str.startswith("infected patients only")]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"wspace": 0.5, "width_ratios": [1.6, 1]})
ax = axs[0]
for j, (onset, c, lab) in enumerate((("respiratory infection", BLUE, "positive sputum or lavage culture"),
                                     ("blood/urine infection", MUTED, "blood or urine culture (negative control)"))):
    g = es[es.onset == onset].set_index("window").loc[win]
    xx = np.arange(4) + (j - 0.5) * 0.25
    ax.errorbar(xx, g.RR_vs_ref, yerr=[g.RR_vs_ref - g.lo, g.hi - g.RR_vs_ref], fmt="o", color=c, ms=4.5, lw=1.5,
                capsize=0, mec="white", mew=0.8, label=f"{lab} ({int(g.patients_with_onset.iloc[0])} patients)")
logticks(ax, [0.03, 0.1, 0.3, 1, 3, 10], "y")
ax.set_ylim(0.015, 20)
ref1(ax, "y")
ax.set_xticks(range(4))
ax.set_xticklabels(wl)
ax.set_xlabel("Days from the culture")
ax.set_ylabel("Rate ratio vs > 3 days before")
ax.set_title("Within patients, by window")
ax.legend(fontsize=6.3, loc="upper left", bbox_to_anchor=(0, -0.22))
grid(ax)
tag(ax, "a")
ax = axs[1]
rb = pd.read_csv(EX / "7_infection_robustness.csv")
obs = es[(es.onset == "respiratory infection") & (es.window == win[0])].iloc[0]
plc = pick(rb, "analysis", "placebo onsets")
pts = [("Observed\nonsets", obs.RR_vs_ref, obs.lo, obs.hi, BLUE), ("Placebo\nonsets", plc.iloc[1], plc.lo, plc.hi, MUTED)]
for i, (lab, e, lo_, hi_, c) in enumerate(pts):
    ax.errorbar(i, e, yerr=[[e - lo_], [hi_ - e]], fmt="o", color=c, ms=5, lw=1.5, capsize=0, mec="white", mew=0.8)
logticks(ax, [0.3, 1, 3, 10], "y")
ax.set_ylim(0.25, 12)
ref1(ax, "y")
ax.set_xlim(-0.6, 1.6)
ax.set_xticks([0, 1])
ax.set_xticklabels([p[0] for p in pts])
ax.set_title("Days −3 to −2")
ax.text(1, -0.33, "Placebo: median and 95% range\nof 200 draws in 26 patients", transform=ax.transAxes, fontsize=6,
        color=MUTED, ha="right", va="top")
grid(ax)
tag(ax, "b")
save(fig, "FigS3")
