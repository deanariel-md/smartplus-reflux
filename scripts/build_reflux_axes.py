"""Reflux burden per patient-day and per patient: the three axes.

  episodes_per_h   (minor + massive events) / monitored hours     [frequency]
  pct_time_reflux  seconds in state 11 or 12 / monitored seconds  [duration]
  massive_share    massive / (minor + massive) events             [height]
  burden_index     sum of tertile ranks of the three axes (3-9), cut-points
                   fixed on pooled full days of the analysis set

Events come from t3/reflux_events_7s (the rule validated against ART's report).
Time and denominators come from the t2 minute grid. States 11 and 12 are
mutually exclusive per row, so time in reflux is not double-counted.
Study day = 07:00-06:59 local (Com_Time); device_day counts calendar study days
from the first recorded one, so gaps do not shift later days. Long continuous
reflux states (>= ARTEFACT_RUN_MIN saturated minutes) are treated as sensor
artefact and removed from numerator and denominator.

Height on a day without events is undefined: massive_share is left missing and
the day gets the lowest height rank in the burden index. Folder 002 (an
excluded patient) is kept in the patient-day file but flagged out of the
analysis set, so it never enters cut-points or summaries.

Run from the project root (default = primary definition):
    python scripts/build_reflux_axes.py
Sensitivity variants write files with a suffix, e.g.
    python scripts/build_reflux_axes.py --events raw --tag _raw
    python scripts/build_reflux_axes.py --artefact-min 15 --tag _art15
    python scripts/build_reflux_axes.py --exclude-era 1 --tag _noera1
    python scripts/build_reflux_axes.py --denominator covered --tag _covered
    python scripts/build_reflux_axes.py --exclude-days data/validation/device_fault_days.csv --tag _nofault
Writes data/t4/reflux_axes_patient_day{tag}.{parquet,csv}, reflux_axes_patient{tag}.csv,
reflux_landmark_d1_3{tag}.csv, artefact_minutes_by_folder{tag}.csv, artefact_minute_list{tag}.parquet,
cutpoints{tag}.json.
If detection fails, fill COLS and rerun.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
from config import DATA_DIR  # noqa: E402
T2 = DATA_DIR / "t2" / "minute"
T3 = DATA_DIR / "t3"
MAP = DATA_DIR / "folder_to_patient.csv"
OUT = DATA_DIR / "t4"

DAY_START_H = 7
FULL_DAY_H = 12          # patient-days with fewer monitored hours are flagged partial
ARTEFACT_RUN_MIN = 30    # consecutive saturated minutes in one reflux state
SATURATED_S = 55         # seconds in state within a minute to count as saturated
PAUSE_WINDOW_S = 30      # pump pause starting within this many s after an event onset = reflux-triggered
EXCLUDED_FOLDERS = {"002"}   # randomised patient excluded from the analysis; no manufacturer report

# Leave None to auto-detect; set to override. "monitored" may be a list of
# per-state second columns to sum (e.g. states 10, 11, 12, 14).
COLS = {
    "minute": {"id": None, "time": None, "sec_minor": None, "sec_massive": None, "monitored": None,
               "covered": None},
    "events": {"id": None, "type": None, "start": None},
    "map": {"folder": None, "patient": None},
}

ID_PATS = [r"^folder_id$", r"^folder$", r"folder", r"^patient_id$", r"patient", r"^pid$"]
TIME_PATS = [r"^minute$", r"^ts$", r"^t$", r"^time$", r"minute", r"time", r"^ts"]
MINOR_PATS = [r"^s(ec)?_?(state_?)?11$", r"(sec|_s$|^s_).*(minor|11)", r"(minor|11).*(sec|_s$)", r"minor.*(dur|time)"]
MASSIVE_PATS = [r"^s(ec)?_?(state_?)?12$", r"(sec|_s$|^s_).*(massive|12)", r"(massive|12).*(sec|_s$)", r"massive.*(dur|time)"]
MON_PATS = [r"^s(ec)?_?monitor", r"monitor.*(sec|_s$)", r"(sec|_s$|^s_).*monitor", r"valid.*sec", r"sec.*valid"]
COV_PATS = [r"^covered", r"cover.*(sec|_s$)", r"recorded.*(sec|_s$)"]
STATE_SEC = r"^(s|sec|secs|seconds)_?(state_?)?(10|11|12|14)$"
START_PATS = [r"^start$", r"^t_start$", r"^start_time$", r"^start_ts$", r"^start", r"onset", r"begin", r"start"]
TYPE_PATS = [r"^type$", r"^event_type$", r"^reflux_type$", r"kind", r"severity", r"^state$", r"type", r"state"]

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--events", choices=["7s", "raw"], default="7s",
                help="7s = reflux_events_7s (primary); raw = reflux_runs_raw without drop/merge")
ap.add_argument("--artefact-min", type=int, default=ARTEFACT_RUN_MIN, help="0 disables the artefact rule")
ap.add_argument("--denominator", choices=["monitor", "covered"], default="monitor",
                help="monitor = time in FT states >= 10 (primary); covered = all recorded time")
ap.add_argument("--exclude-era", type=int, nargs="*", default=[], help="software eras to drop")
ap.add_argument("--exclude-days", default=None,
                help="CSV with fid,study_date to drop from numerator and denominator (e.g. device fault days)")
ap.add_argument("--tag", default="", help="suffix for output files; empty for the primary analysis")
args = ap.parse_args()

con = duckdb.connect()


def show_and_exit(table, cols, what):
    print(f"\nCould not detect '{what}' in {table}. Columns:")
    for c, t in cols.items():
        print(f"  {c:40s} {t}")
    sys.exit(f"Set COLS['{table}']['{what}'] at the top of this script.")


def pick(cols, pats, override, what, table, exclude=()):
    if override:
        return override
    for pat in pats:
        hits = [c for c in cols if re.search(pat, c, re.I) and c not in exclude
                and not re.search(r"onset|count|^n_", c, re.I)]
        if hits:
            return hits[0]
    show_and_exit(table, cols, what)


def schema(src):
    return {r[0]: r[1].upper() for r in con.sql(f"DESCRIBE SELECT * FROM {src}").fetchall()}


def ts(col, typ):
    if "TIMESTAMP" in typ:
        return f"CAST(\"{col}\" AS TIMESTAMP)"
    if any(k in typ for k in ("INT", "DOUBLE", "FLOAT", "DECIMAL")):
        return f"to_timestamp(\"{col}\")::TIMESTAMP"
    return f"try_cast(\"{col}\" AS TIMESTAMP)"


def fid(col):
    return f"regexp_extract(CAST(\"{col}\" AS VARCHAR), '(\\d{{3}})[^0-9]*$', 1)"


def table_src(name):
    for p in (T3 / f"{name}.parquet", T3 / f"{name}.csv"):
        if p.exists():
            return ("read_parquet" if p.suffix == ".parquet" else "read_csv_auto") + f"('{p.as_posix()}')"
    if (T3 / name).is_dir():
        return f"read_parquet('{(T3 / name).as_posix()}/**/*.parquet', union_by_name=true)"
    sys.exit(f"{name} not found in {T3}")


# Software era from the folder number (README).
ERA_SQL = ("CASE WHEN fid::INT BETWEEN 2 AND 20 THEN 0 WHEN fid::INT BETWEEN 24 AND 38 THEN 1 "
           "WHEN fid::INT BETWEEN 44 AND 53 THEN 2 WHEN fid::INT BETWEEN 59 AND 82 THEN 3 "
           "WHEN fid::INT BETWEEN 84 AND 111 THEN 4 END")
keep_era = (f"{ERA_SQL} NOT IN ({','.join(str(e) for e in args.exclude_era)})"
            if args.exclude_era else "true")

# --- minute grid -------------------------------------------------------------
m_src = f"read_parquet('{T2.as_posix()}/**/*.parquet', union_by_name=true, filename=true)"
mc = schema(m_src)
mc.pop("filename", None)
C = COLS["minute"]
m_id = C["id"] or next((c for p in ID_PATS for c in mc if re.search(p, c, re.I)), None)
m_t = pick(mc, TIME_PATS, C["time"], "time", "minute")
s11 = pick(mc, MINOR_PATS, C["sec_minor"], "sec_minor", "minute")
s12 = pick(mc, MASSIVE_PATS, C["sec_massive"], "sec_massive", "minute", exclude=(s11,))
if args.denominator == "covered":
    mon = pick(mc, COV_PATS, C["covered"], "covered", "minute")
else:
    mon = C["monitored"]
    if mon is None:
        state_cols = [c for c in mc if re.search(STATE_SEC, c, re.I)]
        mon = state_cols if len(state_cols) >= 3 else pick(mc, MON_PATS, None, "monitored", "minute")
mon_expr = " + ".join(f"coalesce(\"{c}\", 0)" for c in mon) if isinstance(mon, list) else f"coalesce(\"{mon}\", 0)"
# Folder id from a column if present, otherwise from the parquet file name.
id_expr = fid(m_id) if m_id else "regexp_extract(filename, '(\\d{3})[^0-9/\\\\]*\\.parquet$', 1)"
print(f"minute grid: id={m_id or '<filename>'}, time={m_t}, minor={s11}, massive={s12}, denominator={mon}")
print(f"options: events={args.events}, artefact_min={args.artefact_min}, "
      f"exclude_era={args.exclude_era or 'none'}, tag='{args.tag}'")

con.sql(f"""
    CREATE TEMP TABLE m AS
    SELECT * FROM (
        SELECT {id_expr} AS fid, {ts(m_t, mc[m_t])} AS t,
               coalesce("{s11}", 0) AS s11, coalesce("{s12}", 0) AS s12, {mon_expr} AS smon
        FROM {m_src})
    WHERE {keep_era}""")

# Artefact: islands of saturated minutes in one state.
art_on = args.artefact_min > 0
con.sql(f"""
    CREATE TEMP TABLE m2 AS
    WITH x AS (
        SELECT *, CASE WHEN s12 >= {SATURATED_S} THEN 12 WHEN s11 >= {SATURATED_S} THEN 11 ELSE 0 END AS sat,
               epoch(t) / 60 AS mi FROM m),
    g AS (
        SELECT *, mi - row_number() OVER (PARTITION BY fid, sat ORDER BY t) AS grp FROM x),
    runs AS (
        SELECT fid, sat, grp, count(*) AS len FROM g WHERE sat > 0 GROUP BY ALL)
    SELECT g.fid, g.t, g.s11, g.s12, g.smon,
           {'coalesce(r.len >= ' + str(args.artefact_min) + ', false)' if art_on else 'false'} AS artefact,
           CAST(g.t - INTERVAL {DAY_START_H} HOUR AS DATE) AS study_date
    FROM g LEFT JOIN runs r USING (fid, sat, grp)""")

if args.exclude_days:
    con.sql(f"""CREATE TEMP TABLE xd AS
                SELECT lpad(CAST(fid AS VARCHAR), 3, '0') AS fid, CAST(study_date AS DATE) AS study_date
                FROM read_csv_auto('{Path(args.exclude_days).resolve().as_posix()}', all_varchar=true)""")
    con.sql("UPDATE m2 SET artefact = true FROM xd WHERE m2.fid = xd.fid AND m2.study_date = xd.study_date")
    print(f"excluded days from {args.exclude_days}: "
          f"{con.sql('SELECT count(DISTINCT (m2.fid, m2.study_date)) FROM m2 JOIN xd USING (fid, study_date)').fetchone()[0]}")

art = con.sql("""
    SELECT fid, count(*) FILTER (artefact) AS artefact_min, count(*) AS total_min,
           round(100.0 * count(*) FILTER (artefact) / count(*), 2) AS pct
    FROM m2 GROUP BY fid HAVING count(*) FILTER (artefact) > 0 ORDER BY artefact_min DESC""").df()
print(f"\nArtefact minutes (runs >= {args.artefact_min} saturated min):")
print(art.to_string(index=False) if len(art) else "  none")

# --- events ------------------------------------------------------------------
e_src = table_src("reflux_events_7s" if args.events == "7s" else "reflux_runs_raw")
ec = schema(e_src)
E = COLS["events"]
e_id = pick(ec, ID_PATS, E["id"], "id", "events")
e_ty = pick(ec, TYPE_PATS, E["type"], "type", "events")
e_t = pick(ec, START_PATS, E["start"], "start", "events")
raw = f"lower(trim(CAST(\"{e_ty}\" AS VARCHAR)))"
con.sql(f"""
    CREATE TEMP TABLE e AS
    SELECT * FROM (
        SELECT {fid(e_id)} AS fid, {ts(e_t, ec[e_t])} AS t0, date_trunc('minute', {ts(e_t, ec[e_t])}) AS tm,
               CASE WHEN {raw} LIKE '%mas%' OR {raw} IN ('12','12.0') THEN 'massive'
                    WHEN {raw} LIKE '%min%' OR {raw} IN ('11','11.0') THEN 'minor' END AS typ
        FROM {e_src})
    WHERE typ IS NOT NULL AND {keep_era}""")
print(f"\nevents ({args.events}): id={e_id}, type={e_ty}, start={e_t}")
print(con.sql("SELECT typ, count(*) n FROM e GROUP BY 1 ORDER BY 1"))

# Pump pauses that start within PAUSE_WINDOW_S after a reflux onset (coupling covariate).
p_src = table_src("pump_paused")
con.sql(f"""
    CREATE TEMP TABLE pz AS
    SELECT p.fid, CAST(p.t - INTERVAL {DAY_START_H} HOUR AS DATE) AS study_date, p.dur_s,
           EXISTS (SELECT 1 FROM e WHERE e.fid = p.fid
                   AND e.t0 BETWEEN p.t - INTERVAL {PAUSE_WINDOW_S} SECOND AND p.t + INTERVAL 2 SECOND) AS reflux_trig
    FROM (SELECT {fid('folder_id')} AS fid, CAST(start_t AS TIMESTAMP) AS t, dur_s FROM {p_src}) p""")

# --- patient-day axes --------------------------------------------------------
con.sql(f"""
    CREATE TEMP TABLE pd AS
    WITH first AS (SELECT fid, min(study_date) AS d0 FROM m2 GROUP BY fid),
    mins AS (
        SELECT fid, study_date,
               coalesce(sum(smon) FILTER (NOT artefact), 0) / 3600.0 AS monitored_h,
               coalesce(sum(smon) FILTER (artefact), 0) / 3600.0 AS artefact_h,
               coalesce(sum(s11) FILTER (NOT artefact), 0) AS sec_minor,
               coalesce(sum(s12) FILTER (NOT artefact), 0) AS sec_massive,
               coalesce(sum(smon) FILTER (NOT artefact), 0) AS sec_mon
        FROM m2 GROUP BY ALL),
    ev AS (
        SELECT e.fid, m2.study_date,
               count(*) FILTER (e.typ = 'minor') AS n_minor,
               count(*) FILTER (e.typ = 'massive') AS n_massive
        FROM e JOIN m2 ON m2.fid = e.fid AND m2.t = e.tm
        WHERE NOT m2.artefact GROUP BY ALL),
    pause AS (
        SELECT fid, study_date, sum(dur_s) / 3600.0 AS pause_h_total,
               coalesce(sum(dur_s) FILTER (reflux_trig), 0) / 3600.0 AS pause_h_reflux
        FROM pz GROUP BY ALL)
    SELECT mins.fid, mins.study_date, datediff('day', first.d0, mins.study_date) + 1 AS device_day,
           mins.fid NOT IN ({",".join(f"'{x}'" for x in EXCLUDED_FOLDERS)}) AS in_analysis_set,
           round(monitored_h, 2) AS monitored_h, round(artefact_h, 2) AS artefact_h,
           monitored_h >= {FULL_DAY_H} AS full_day,
           coalesce(n_minor, 0) AS n_minor, coalesce(n_massive, 0) AS n_massive,
           (coalesce(n_minor, 0) + coalesce(n_massive, 0)) / nullif(monitored_h, 0) AS episodes_per_h,
           100.0 * (sec_minor + sec_massive) / nullif(sec_mon, 0) AS pct_time_reflux,
           n_massive / nullif(coalesce(n_minor, 0) + n_massive, 0) AS massive_share,
           round(coalesce(pause_h_total, 0), 3) AS pause_h_total,
           round(coalesce(pause_h_reflux, 0), 3) AS pause_h_reflux
    FROM mins JOIN first USING (fid)
    LEFT JOIN ev USING (fid, study_date)
    LEFT JOIN pause USING (fid, study_date)""")

# Tertile cut-points on pooled full days of the analysis set (height: days with >= 1 event).
AX = ("episodes_per_h", "pct_time_reflux", "massive_share")
cuts = {a: con.sql(f"SELECT quantile_cont({a}, [1/3, 2/3]) FROM pd WHERE full_day AND in_analysis_set").fetchone()[0]
        for a in AX}


def rank_expr(src_alias=""):
    """Burden index: missing height (no events, but monitored) gets rank 1."""
    p = src_alias
    parts = []
    for a in AX:
        v = f"{p}{a}"
        if a == "massive_share":
            v = f"CASE WHEN {p}episodes_per_h IS NOT NULL THEN coalesce({p}{a}, -1) END"
        parts.append(f"(1 + ({v} > {cuts[a][0]})::INT + ({v} > {cuts[a][1]})::INT)")
    return " + ".join(parts)


con.sql(f"CREATE TEMP TABLE pd2 AS SELECT *, {rank_expr()} AS burden_index FROM pd")

# Map folders to true patient IDs.
if MAP.exists():
    mp = schema(f"read_csv_auto('{MAP.as_posix()}', all_varchar=true)")
    M = COLS["map"]
    f_col = pick(mp, [r"folder"], M["folder"], "folder", "map")
    p_col = pick(mp, [r"patient", r"subject", r"true"], M["patient"], "patient", "map", exclude=(f_col,))
    con.sql(f"""CREATE TEMP TABLE mp AS SELECT {fid(f_col)} AS fid, {fid(p_col)} AS patient_id
                FROM read_csv_auto('{MAP.as_posix()}', all_varchar=true)""")
    con.sql("CREATE TEMP TABLE out AS SELECT mp.patient_id, pd2.* FROM pd2 LEFT JOIN mp USING (fid) "
            "ORDER BY patient_id, study_date")
else:
    print(f"\nWARNING: {MAP} not found; output keeps folder ids only.")
    con.sql("CREATE TEMP TABLE out AS SELECT NULL AS patient_id, * FROM pd2 ORDER BY fid, study_date")


def patient_agg(where):
    """Pooled ratios over the selected patient-days, ranked against the day-level cut-points."""
    base = f"""
        SELECT patient_id, fid, count(*) AS full_days, round(sum(monitored_h), 1) AS monitored_h,
               sum(n_minor) AS n_minor, sum(n_massive) AS n_massive,
               sum(n_minor + n_massive) / sum(monitored_h) AS episodes_per_h,
               sum(pct_time_reflux * monitored_h) / sum(monitored_h) AS pct_time_reflux,
               sum(n_massive) / nullif(sum(n_minor + n_massive), 0) AS massive_share,
               sum(pause_h_reflux) AS pause_h_reflux
        FROM out WHERE in_analysis_set AND full_day AND {where} GROUP BY ALL"""
    return con.sql(f"SELECT *, {rank_expr()} AS burden_index FROM ({base}) ORDER BY patient_id").df()


pt = patient_agg("true")

# Landmark exposure: device days 1-3; eligible with >= 2 full days among them and
# any monitoring on day 4. Final eligibility is set in build_landmark_exposure.py.
lm = patient_agg("device_day <= 3").rename(columns={"full_days": "full_days_d1_3"})
d4 = con.sql("SELECT DISTINCT patient_id FROM out WHERE device_day = 4 AND monitored_h > 0").df()
lm["monitored_day4"] = lm["patient_id"].isin(d4["patient_id"])
lm["eligible_landmark"] = (lm["full_days_d1_3"] >= 2) & lm["monitored_day4"]

OUT.mkdir(parents=True, exist_ok=True)
tag = args.tag
con.sql(f"COPY out TO '{(OUT / f'reflux_axes_patient_day{tag}.parquet').as_posix()}' (FORMAT parquet)")
con.sql(f"COPY out TO '{(OUT / f'reflux_axes_patient_day{tag}.csv').as_posix()}' (HEADER)")
pt.round(4).to_csv(OUT / f"reflux_axes_patient{tag}.csv", index=False)
lm.round(4).to_csv(OUT / f"reflux_landmark_d1_3{tag}.csv", index=False)
art.to_csv(OUT / f"artefact_minutes_by_folder{tag}.csv", index=False)
con.sql(f"COPY (SELECT fid, t FROM m2 WHERE artefact ORDER BY fid, t) "
        f"TO '{(OUT / f'artefact_minute_list{tag}.parquet').as_posix()}' (FORMAT parquet)")
(OUT / f"cutpoints{tag}.json").write_text(json.dumps(
    {"options": vars(args), "full_day_h": FULL_DAY_H,
     "cutpoints_full_days": {a: [round(v, 4) for v in cuts[a]] for a in AX}}, indent=2))

s = con.sql("""SELECT count(DISTINCT fid) AS folders, count(*) AS n_days, count(*) FILTER (full_day) AS full_days,
                      round(median(episodes_per_h) FILTER (full_day), 2) med_ep_h,
                      round(median(pct_time_reflux) FILTER (full_day), 2) med_pct_time,
                      round(median(massive_share) FILTER (full_day), 3) med_massive_share
               FROM out WHERE in_analysis_set""").df()
print("\nSummary, analysis set (medians over full days):")
print(s.to_string(index=False))
print("Tertile cut-points (full days):", {a: [round(v, 3) for v in cuts[a]] for a in AX})
print(f"Patients with >= 1 full day: {len(pt)}; landmark-eligible (days 1-3): {int(lm['eligible_landmark'].sum())}")
print(f"\nWrote {OUT} (tag '{tag}')")
