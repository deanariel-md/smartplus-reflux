"""Minute grid (T2) and run tables (T3) from the full-rate device logs (T0).

Per device folder (RMC1-<id>) this writes:
  t2/minute/<folder>.parquet               one row per patient-minute on the PC clock
  t3/reflux_runs_raw/<folder>.parquet      contiguous runs of FT_System_State 11, 12 or 14
  t3/balloon_inflations/<folder>.parquet   contiguous runs of PSF_System_State 5
  t3/pump_paused/<folder>.parquet          contiguous runs of FPF_System_State 21
Reflux episodes (t3/reflux_events_7s) are made from the run table by build_reflux_events.py.

Conventions:
  rows are ordered by the PC clock (t_com_f); row duration = time to the next row, capped at 2 s;
  a gap of more than 5 s between consecutive rows breaks a run.
  FT_System_State 10 monitoring, 11 minor, 12 massive, 14 drainage open, < 10 setup or positioning.
  PSF_System_State 5 balloon inflated; FPF_System_State 20 running, 21 paused; REE_State 4 measuring;
  DB_State 2 drainage open. 65535 = unset; Balloon_Pressure |value| >= 255 = glitch; ventilator values <= 0 are
  invalid breaths.

The script prints which T0 column feeds each output column. Output column names, order and types are fixed in
MINUTE_SCHEMA and RUN_SCHEMA.

    python scripts/build_t2_t3.py                    # all folders under T0_DIR, tables under DATA_DIR
    python scripts/build_t2_t3.py --out <folder>     # write under another root (verify_rebuild.py uses this)
    python scripts/build_t2_t3.py --folders RMC1-024 RMC1-028
"""
import argparse
import re
from pathlib import Path

import duckdb

from config import DATA_DIR, T0_DIR

ROW_CAP_S = 2.0       # row duration cap
GAP_BREAK_S = 5.0     # a gap above this between rows breaks a run
UNSET = 65535
BALLOON_GLITCH = 255

# T0 columns read by the other release scripts or documented with the T0 conventions.
KNOWN = {
    "ft": "FT_System_State", "db": "DB_State", "psf": "PSF_System_State", "fpf": "FPF_System_State",
    "ree_state": "REE_State", "ree": "REE_CurrentREE", "vco2": "VCO2", "balloon_p": "Balloon_Pressure",
    "grv_reflux": "GRVReflux_Volume", "session": "Session_Volume",
    "feed_delivered": "NutrientsPlan_Delivered_Volume",
    "feed_grv": "NutrientsPlan_GRV_Volume",      # reconstructed: feed_grv_ml_last = last NutrientsPlan_GRV_Volume
}
# reconstructed: no surviving script names the T0 columns below; each takes the first T0 column matching one of
# its patterns (case-insensitive). Check the printed mapping and set an exact name here if it is wrong.
GUESS = {
    "balloon_req": [r"^balloon_?(req|request|requested|target|set)", r"balloon.*(request|target|setpoint)"],
    "feed_expected": [r"^nutrientsplan_expected", r"expected.*vol"],
    "feed_basal_rate": [r"basal.*rate"],
    "feed_max_rate": [r"max.*rate"],
    "feed_vtbd": [r"vtbd"],
    "feed_plan_efficiency": [r"efficien"],
    "fluids_delivered": [r"fluid.*deliver", r"water.*deliver"],
    "ree_baseline": [r"^ree_.*baseline", r"baseline.*ree"],
    "etco2": [r"etco2", r"et_co2", r"end_?tidal"],
    "peep": [r"(^|_)peep($|_)"],
    "pip": [r"(^|_)pip($|_)", r"peak.*press"],
    "pplat": [r"pplat", r"plat.*press"],
    "rr_total": [r"(^|_)rr_?tot(al)?$", r"(^|_)rr$", r"resp.*rate"],
    "vte": [r"(^|_)vte($|_)", r"tidal.*exp"],
    "mv_exp": [r"(^|_)mv(e|_?exp)?$", r"minute_?vol"],
    "cdyn": [r"cdyn", r"dyn.*compl"],
    "rsbi": [r"rsbi"],
    "urine_rate": [r"urine.*(hour|rate)"],
    "urine_accum": [r"urine.*(accum|total|vol)"],
    "ft_type": [r"^ft_?type", r"tube_?type"],
}
ERROR_PAT = r"error"          # reconstructed: any_error = any column whose name contains "error" is non-zero
META = {"folder_id", "era", "hour", "t_com", "t_ssf", "t_com_f", "t_ssf_f", "PacketCount"}

MINUTE_SCHEMA = [
    ("folder_id", "VARCHAR"), ("era", "TINYINT"), ("minute_pc", "TIMESTAMP"), ("minute_start_device", "TIMESTAMP_NS"),
    ("n_rows", "BIGINT"), ("covered_s", "DOUBLE"), ("monitor_s", "DOUBLE"), ("setup_s", "DOUBLE"),
    ("ft10_s", "DOUBLE"), ("ft11_minor_s", "DOUBLE"), ("ft12_massive_s", "DOUBLE"), ("ft14_drain_s", "DOUBLE"),
    ("minor_onsets_raw", "DOUBLE"), ("massive_onsets_raw", "DOUBLE"),
    ("balloon_inflated_s", "DOUBLE"), ("balloon_p_max", "FLOAT"), ("drain_open_s", "DOUBLE"),
    ("drain_session_ml_last", "FLOAT"), ("fpf20_running_s", "DOUBLE"), ("fpf21_paused_s", "DOUBLE"),
    ("feed_delivered_ml_last", "FLOAT"), ("feed_expected_ml_last", "FLOAT"), ("feed_grv_ml_last", "FLOAT"),
    ("feed_basal_rate", "FLOAT"), ("feed_max_rate", "FLOAT"), ("feed_vtbd", "FLOAT"), ("feed_plan_efficiency", "FLOAT"),
    ("fluids_delivered_ml_last", "FLOAT"), ("ree_valid_frac", "DOUBLE"), ("ree_kcal_d", "FLOAT"),
    ("vco2_ml_min", "FLOAT"), ("ree_baseline", "FLOAT"), ("etco2", "FLOAT"), ("peep", "FLOAT"), ("pip", "FLOAT"),
    ("pplat", "FLOAT"), ("rr_total", "FLOAT"), ("vte", "FLOAT"), ("mv_exp", "FLOAT"), ("cdyn", "FLOAT"),
    ("rsbi", "FLOAT"), ("urine_hourly_rate", "FLOAT"), ("urine_accum_ml", "FLOAT"), ("any_error", "INTEGER"),
    ("ft_type", "FLOAT"),
]
RUN_SCHEMA = [
    ("folder_id", "VARCHAR"), ("era", "TINYINT"), ("run_id", "DOUBLE"), ("state", "INTEGER"),
    ("start_t", "TIMESTAMP_NS"), ("end_t", "TIMESTAMP"), ("start_t_device", "TIMESTAMP_NS"), ("dur_s", "DOUBLE"),
    ("n_rows", "BIGINT"), ("balloon_req_max", "FLOAT"), ("psf5_frac", "DOUBLE"), ("balloon_p_max", "FLOAT"),
    ("db_open_frac", "DOUBLE"), ("grv_reflux_ml_max", "FLOAT"), ("fpf_state_at_start", "FLOAT"),
    ("ree_state_at_start", "FLOAT"),
]
RUN_TABLES = {   # table -> run key (state value while the condition holds, else NULL)
    "reflux_runs_raw": "CASE WHEN ft IN (11, 12, 14) THEN ft END",
    "balloon_inflations": "CASE WHEN psf = 5 THEN 5 END",
    "pump_paused": "CASE WHEN fpf = 21 THEN 21 END",
}


def era_of(folder):
    n = int(folder[-3:])
    for lo, hi, e in ((2, 20, 0), (24, 38, 1), (44, 53, 2), (59, 82, 3), (84, 111, 4)):
        if lo <= n <= hi:
            return e
    return None


def cast_select(schema, src):
    return "SELECT " + ", ".join(f'CAST("{c}" AS {t}) AS "{c}"' for c, t in schema) + f" FROM {src}"


def resolve(have):
    lower = {c.lower(): c for c in have}
    col = {k: (v if v in have else lower.get(v.lower())) for k, v in KNOWN.items()}
    used = set(META) | {v for v in col.values() if v}
    for k, pats in GUESS.items():
        col[k] = next((c for p in pats for c in have if c not in used and re.search(p, c, re.I)), None)
        if col[k]:
            used.add(col[k])
    col["errors"] = [c for c in have if c not in used and re.search(ERROR_PAT, c, re.I)]
    return col


def load_rows(con, src, col, off):
    def v(k):
        return f'CAST(NULLIF("{col[k]}", {UNSET}) AS FLOAT)' if col.get(k) else "CAST(NULL AS FLOAT)"
    vals = {k: v(k) for k in list(KNOWN) + list(GUESS)}
    vals["balloon_p"] = f"CASE WHEN abs({vals['balloon_p']}) < {BALLOON_GLITCH} THEN {vals['balloon_p']} END"
    errs = [f'coalesce(NULLIF("{c}", {UNSET}), 0) <> 0' for c in col["errors"]]
    vals["err"] = f"CASE WHEN {' OR '.join(errs)} THEN 1 ELSE 0 END" if errs else "CAST(NULL AS INTEGER)"
    sel = ", ".join(f"{e} AS {k}" for k, e in vals.items())
    con.sql(f"""
        CREATE OR REPLACE TEMP TABLE r0 AS
        SELECT coalesce(t_com_f, t_ssf_f + to_seconds({off})) AS t, t_ssf_f AS t_dev, {sel}
        FROM {src}""")
    con.sql(f"""
        CREATE OR REPLACE TEMP TABLE r AS
        WITH o AS (SELECT *, row_number() OVER (ORDER BY t, t_dev) AS rn FROM r0 WHERE t IS NOT NULL)
        SELECT *,
               least(coalesce((epoch_ns(lead(t) OVER w) - epoch_ns(t)) / 1e9, 0), {ROW_CAP_S}) AS dt,
               (epoch_ns(t) - epoch_ns(lag(t) OVER w)) / 1e9 AS gap_prev
        FROM o WINDOW w AS (ORDER BY rn)""")
    # reconstructed: the last row of a folder gets duration 0


def build_runs(con, key, folder, era):
    con.sql(f"""
        CREATE OR REPLACE TEMP TABLE runs AS
        WITH k AS (SELECT *, {key} AS key FROM r),
        f AS (SELECT *, CASE WHEN key IS NOT NULL AND (lag(key) OVER w IS DISTINCT FROM key OR gap_prev > {GAP_BREAK_S})
                             THEN 1 ELSE 0 END AS new
              FROM k WINDOW w AS (ORDER BY rn)),
        g AS (SELECT *, sum(new) OVER (ORDER BY rn ROWS UNBOUNDED PRECEDING) AS run_id FROM f)
        SELECT '{folder}' AS folder_id, {era} AS era, run_id, any_value(key) AS state,
               min(t) AS start_t,
               -- reconstructed: end_t = last row of the run plus its duration; dur_s = summed row durations
               max(CAST(t AS TIMESTAMP) + to_microseconds(CAST(round(dt * 1e6) AS BIGINT))) AS end_t,
               first(t_dev ORDER BY rn) AS start_t_device,
               sum(dt) AS dur_s, count(*) AS n_rows,
               max(balloon_req) AS balloon_req_max,
               coalesce(sum(dt) FILTER (psf = 5), 0) / nullif(sum(dt), 0) AS psf5_frac,
               max(balloon_p) AS balloon_p_max,
               coalesce(sum(dt) FILTER (db = 2), 0) / nullif(sum(dt), 0) AS db_open_frac,
               max(grv_reflux) AS grv_reflux_ml_max,
               first(fpf ORDER BY rn) AS fpf_state_at_start,
               first(ree_state ORDER BY rn) AS ree_state_at_start
        FROM g WHERE key IS NOT NULL GROUP BY run_id""")


def last(k):
    return f"last({k} ORDER BY rn) FILTER ({k} IS NOT NULL)"


def build_minutes(con, folder, era):
    secs = lambda cond: f"coalesce(sum(dt) FILTER ({cond}), 0)"
    vent = lambda k: f"median({k}) FILTER ({k} > 0)"
    # reconstructed: a row's whole duration counts in the PC-clock minute in which the row starts
    # reconstructed: an onset is the first row of a raw run (reflux_runs_raw) of state 11 or 12
    # reconstructed: feed and urine readings are the last non-missing value in the minute
    con.sql(f"""
        CREATE OR REPLACE TEMP TABLE minute AS
        WITH m AS (
            SELECT date_trunc('minute', t) AS minute_pc, min(t_dev) AS minute_start_device, count(*) AS n_rows,
                   sum(dt) AS covered_s, {secs('ft >= 10')} AS monitor_s, {secs('ft < 10')} AS setup_s,
                   {secs('ft = 10')} AS ft10_s, {secs('ft = 11')} AS ft11_minor_s,
                   {secs('ft = 12')} AS ft12_massive_s, {secs('ft = 14')} AS ft14_drain_s,
                   {secs('psf = 5')} AS balloon_inflated_s, max(balloon_p) AS balloon_p_max,
                   {secs('db = 2')} AS drain_open_s, {last('session')} AS drain_session_ml_last,
                   {secs('fpf = 20')} AS fpf20_running_s, {secs('fpf = 21')} AS fpf21_paused_s,
                   {last('feed_delivered')} AS feed_delivered_ml_last, {last('feed_expected')} AS feed_expected_ml_last,
                   {last('feed_grv')} AS feed_grv_ml_last, {last('feed_basal_rate')} AS feed_basal_rate,
                   {last('feed_max_rate')} AS feed_max_rate, {last('feed_vtbd')} AS feed_vtbd,
                   {last('feed_plan_efficiency')} AS feed_plan_efficiency,
                   {last('fluids_delivered')} AS fluids_delivered_ml_last,
                   -- reconstructed: share of the minute's rows with REE_State 4
                   avg(CASE WHEN ree_state = 4 THEN 1.0 ELSE 0.0 END) AS ree_valid_frac,
                   median(ree) FILTER (ree_state = 4 AND ree > 0) AS ree_kcal_d,
                   median(vco2) FILTER (ree_state = 4 AND vco2 > 0) AS vco2_ml_min,
                   {last('ree_baseline')} AS ree_baseline,
                   {vent('etco2')} AS etco2, {vent('peep')} AS peep, {vent('pip')} AS pip, {vent('pplat')} AS pplat,
                   {vent('rr_total')} AS rr_total, {vent('vte')} AS vte, {vent('mv_exp')} AS mv_exp,
                   {vent('cdyn')} AS cdyn, {vent('rsbi')} AS rsbi,
                   {last('urine_rate')} AS urine_hourly_rate, {last('urine_accum')} AS urine_accum_ml,
                   max(err) AS any_error, {last('ft_type')} AS ft_type
            FROM r GROUP BY 1),
        o AS (SELECT date_trunc('minute', start_t) AS minute_pc,
                     count(*) FILTER (state = 11) AS minor_onsets_raw, count(*) FILTER (state = 12) AS massive_onsets_raw
              FROM runs_reflux GROUP BY 1)
        SELECT '{folder}' AS folder_id, {era} AS era, m.*,
               coalesce(o.minor_onsets_raw, 0) AS minor_onsets_raw, coalesce(o.massive_onsets_raw, 0) AS massive_onsets_raw
        FROM m LEFT JOIN o USING (minute_pc)""")


def write(con, schema, src, path, order):
    path.parent.mkdir(parents=True, exist_ok=True)
    con.sql(f"COPY ({cast_select(schema, src)} ORDER BY {order}) TO '{path.as_posix()}' (FORMAT parquet)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(DATA_DIR), help="root that receives t2/ and t3/ (default DATA_DIR)")
    ap.add_argument("--folders", nargs="*", help="folder names (default: every RMC1-* folder under T0_DIR)")
    args = ap.parse_args()
    out = Path(args.out)
    con = duckdb.connect()
    con.sql("SET enable_progress_bar = false")
    folders = args.folders or sorted(p.name for p in T0_DIR.glob("RMC1-*") if p.is_dir())
    printed = None
    for folder in folders:
        if not any((T0_DIR / folder).glob("*.parquet")):
            print(folder, "no T0 files, skipped")
            continue
        src = f"read_parquet('{(T0_DIR / folder).as_posix()}/*.parquet', union_by_name=true)"
        have = con.sql(f"SELECT * FROM {src} LIMIT 0").columns
        if "FT_System_State" not in have:
            print(folder, "no FT_System_State, skipped")
            continue
        col = resolve(have)
        if col != printed:
            print(f"{folder}: T0 columns used: " + ", ".join(f"{k}={v}" for k, v in col.items()))
            printed = col
        off = con.sql(f"SELECT median(epoch(t_com) - epoch(t_ssf)) FROM {src} "
                      "WHERE t_com IS NOT NULL AND t_ssf IS NOT NULL").fetchone()[0] or 0
        era = era_of(folder)
        era_sql = "NULL" if era is None else str(era)
        load_rows(con, src, col, off)
        counts = {}
        for table, key in RUN_TABLES.items():
            build_runs(con, key, folder, era_sql)
            if table == "reflux_runs_raw":
                con.sql("CREATE OR REPLACE TEMP TABLE runs_reflux AS SELECT * FROM runs")
            write(con, RUN_SCHEMA, "runs", out / "t3" / table / f"{folder}.parquet", "start_t")
            counts[table] = con.sql("SELECT count(*) FROM runs").fetchone()[0]
        build_minutes(con, folder, era_sql)
        write(con, MINUTE_SCHEMA, "minute", out / "t2" / "minute" / f"{folder}.parquet", "minute_pc")
        counts["minutes"] = con.sql("SELECT count(*) FROM minute").fetchone()[0]
        print(folder, ", ".join(f"{k} {v:,}" for k, v in counts.items()), flush=True)
    print(f"\nWrote {out / 't2' / 'minute'} and {out / 't3'}")


if __name__ == "__main__":
    main()
