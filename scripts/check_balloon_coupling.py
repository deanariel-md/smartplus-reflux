"""Label-free check of minor/massive orientation.

Massive reflux (reaching the oesophageal balloon) should trigger a balloon
inflation; minor reflux should not. For every reflux event, flag whether a
balloon inflation starts within [onset - PRE_S, onset + POST_S] in the same
folder. Reverse direction: attribute each inflation to the most recent event
onset. Uses device timing only, independent of ART's labels or reports.

Run from the project root:
    python scripts/check_balloon_coupling.py
Writes data/validation/balloon_coupling_by_folder.csv and _pooled.csv.
If a column can't be detected, the script prints the schema; fill COLS below.
"""
import re
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
from config import DATA_DIR  # noqa: E402
T3 = DATA_DIR / "t3"
OUT = DATA_DIR / "validation"

PRE_S, POST_S = 2, 30
# Balloon disabled or tube without balloon; excluded from pooled rates.
NO_BALLOON = {"003", "028", "038", "071", "092"}

# Leave None to auto-detect; set a column name to override.
COLS = {
    "events": {"id": None, "type": None, "start": None},
    "balloon": {"id": None, "start": None},
}

ID_PATS = [r"^folder_id$", r"^folder$", r"folder", r"^patient_id$", r"patient", r"^pid$", r"subject"]
START_PATS = [r"^start$", r"^t_start$", r"^start_time$", r"^start_ts$", r"^start", r"onset", r"begin", r"^t0$", r"start"]
TYPE_PATS = [r"^type$", r"^event_type$", r"^reflux_type$", r"kind", r"severity", r"class", r"^state$", r"type", r"state"]

con = duckdb.connect()


def source(name):
    for p in (T3 / f"{name}.parquet", T3 / f"{name}.csv"):
        if p.exists():
            fn = "read_parquet" if p.suffix == ".parquet" else "read_csv_auto"
            return f"{fn}('{p.as_posix()}')"
    d = T3 / name
    if d.is_dir():
        return f"read_parquet('{d.as_posix()}/**/*.parquet', union_by_name=true)"
    return None


def schema(src):
    return {r[0]: r[1].upper() for r in con.sql(f"DESCRIBE SELECT * FROM {src}").fetchall()}


def pick(cols, pats, override, what, table):
    if override:
        return override
    for pat in pats:
        hits = [c for c in cols if re.search(pat, c, re.I)]
        if hits:
            return hits[0]
    print(f"\nCould not detect the {what} column in {table}. Columns:")
    for c, t in cols.items():
        print(f"  {c:40s} {t}")
    sys.exit(f"Set COLS['{table}']['{what}'] at the top of this script.")


def secs(col, typ):
    """Expression giving event time in epoch seconds."""
    if "TIMESTAMP" in typ or typ == "DATE":
        return f"epoch(\"{col}\")"
    if any(k in typ for k in ("INT", "DOUBLE", "FLOAT", "DECIMAL", "REAL")):
        return f"\"{col}\""
    return f"epoch(try_cast(\"{col}\" AS TIMESTAMP))"


def fid(col):
    return f"regexp_extract(CAST(\"{col}\" AS VARCHAR), '(\\d{{3}})[^0-9]*$', 1)"


def era(f):
    n = int(f) if f.isdigit() else -1
    for lo, hi, e in ((2, 20, 0), (24, 38, 1), (44, 53, 2), (59, 82, 3), (84, 111, 4)):
        if lo <= n <= hi:
            return e
    return None


b_src = source("balloon_inflations")
if b_src is None:
    sys.exit(f"balloon_inflations not found in {T3}: {sorted(x.name for x in T3.iterdir())}")
b_cols = schema(b_src)
b_id = pick(b_cols, ID_PATS, COLS["balloon"]["id"], "id", "balloon")
b_t = pick(b_cols, START_PATS, COLS["balloon"]["start"], "start", "balloon")
con.sql(f"CREATE TEMP TABLE b AS SELECT {fid(b_id)} AS fid, {secs(b_t, b_cols[b_t])} AS t "
        f"FROM {b_src} WHERE \"{b_t}\" IS NOT NULL")
print(f"balloon_inflations: id={b_id}, start={b_t}, n={con.sql('SELECT count(*) FROM b').fetchone()[0]}")

OUT.mkdir(parents=True, exist_ok=True)
by_folder, pooled = [], []

for table in ("reflux_events_7s", "reflux_runs_raw"):
    src = source(table)
    if src is None:
        print(f"\n{table}: not found, skipped")
        continue
    cols = schema(src)
    e_id = pick(cols, ID_PATS, COLS["events"]["id"], "id", "events")
    e_ty = pick(cols, TYPE_PATS, COLS["events"]["type"], "type", "events")
    e_t = pick(cols, START_PATS, COLS["events"]["start"], "start", "events")
    print(f"\n=== {table}: id={e_id}, type={e_ty}, start={e_t}")
    print("Raw type values:")
    print(con.sql(f"SELECT \"{e_ty}\" AS raw, count(*) n FROM {src} GROUP BY 1 ORDER BY 2 DESC LIMIT 12"))

    raw = f"lower(trim(CAST(\"{e_ty}\" AS VARCHAR)))"
    norm = (f"CASE WHEN {raw} LIKE '%mas%' OR {raw} IN ('12','12.0') THEN 'massive' "
            f"WHEN {raw} LIKE '%min%' OR {raw} IN ('11','11.0') THEN 'minor' END")
    con.sql(f"""
        CREATE OR REPLACE TEMP TABLE e AS
        SELECT row_number() OVER () AS eid, {fid(e_id)} AS fid, {norm} AS typ, {secs(e_t, cols[e_t])} AS t
        FROM {src}""")
    con.sql("DELETE FROM e WHERE typ IS NULL OR t IS NULL")

    con.sql(f"""
        CREATE OR REPLACE TEMP TABLE hit AS
        SELECT e.*, eid IN (
            SELECT DISTINCT e2.eid FROM e e2 JOIN b
              ON b.fid = e2.fid AND b.t BETWEEN e2.t - {PRE_S} AND e2.t + {POST_S}
        ) AS balloon
        FROM e""")
    f = con.sql("""
        SELECT fid, typ, count(*) AS n_events, avg(balloon::INT) AS p_balloon
        FROM hit GROUP BY ALL ORDER BY fid, typ""").df()
    f["era"] = f["fid"].map(era)
    f["table"] = table
    by_folder.append(f)

    # Reverse: which event type precedes each inflation (within POST_S s)?
    rev = con.sql(f"""
        SELECT CASE WHEN e.t IS NULL OR b.t - e.t > {POST_S} THEN 'none' ELSE e.typ END AS preceding,
               count(*) AS n_inflations
        FROM b ASOF LEFT JOIN (SELECT fid, t - {PRE_S} AS t, typ FROM e) e
          ON b.fid = e.fid AND b.t >= e.t
        WHERE b.fid NOT IN ({",".join(f"'{x}'" for x in NO_BALLOON)})
        GROUP BY 1 ORDER BY 2 DESC""").df()

    keep = f[~f["fid"].isin(NO_BALLOON)]
    for typ, g in keep.groupby("typ"):
        w = (g["p_balloon"] * g["n_events"]).sum() / g["n_events"].sum()
        pooled.append({"table": table, "typ": typ, "folders": len(g), "events": int(g["n_events"].sum()),
                       "pooled_p_balloon": round(w, 3),
                       "median_folder_p": round(g["p_balloon"].median(), 3),
                       "iqr_folder_p": f"{g['p_balloon'].quantile(.25):.2f}-{g['p_balloon'].quantile(.75):.2f}"})
    print(f"\nShare of events followed by a balloon inflation within {POST_S} s "
          f"(excluding folders {sorted(NO_BALLOON)}):")
    for r in (p for p in pooled if p["table"] == table):
        print(f"  {r['typ']:8s} events={r['events']:>7}  pooled={r['pooled_p_balloon']:.3f}  "
              f"median folder={r['median_folder_p']:.3f} (IQR {r['iqr_folder_p']})")
    print("\nBy era (median folder share):")
    print(keep.groupby(["era", "typ"])["p_balloon"].median().unstack().round(3).to_string())
    print("\nInflations by preceding event type:")
    print(rev.to_string(index=False))

import pandas as pd  # noqa: E402

if by_folder:
    pd.concat(by_folder).to_csv(OUT / "balloon_coupling_by_folder.csv", index=False)
    pd.DataFrame(pooled).to_csv(OUT / "balloon_coupling_pooled.csv", index=False)
    print(f"\nWrote {OUT / 'balloon_coupling_by_folder.csv'} and _pooled.csv")
    print("Expected if labels are oriented correctly: massive >> minor, and most inflations "
          "preceded by 'massive'. The reverse pattern means the labels in t3 are swapped.")
