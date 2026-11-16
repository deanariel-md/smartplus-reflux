r"""Rebuild T2 and T3 from T0 into a temporary folder and compare them with the tables under DATA_DIR.

Checks, per table: column names, order and types; row counts per folder; totals against the published counts.
For reflux_events_7s it also matches every episode exactly on folder_id, type, start_t and end_t.
Prints PASS or FAIL per check and exits with code 1 on any FAIL.

    py -3.12 release\scripts\verify_rebuild.py                 # rebuild, compare, delete the temporary folder
    py -3.12 release\scripts\verify_rebuild.py --keep          # keep the rebuilt tables (path printed)
    py -3.12 release\scripts\verify_rebuild.py --rebuilt DIR   # compare an earlier rebuild without rebuilding
Set DATA_DIR (folder holding t2/ and t3/) and T0_DIR (full-rate logs) first; see config.py.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import duckdb

from config import DATA_DIR, T0_DIR

HERE = Path(__file__).resolve().parent
TABLES = ["t2/minute", "t3/reflux_runs_raw", "t3/reflux_events_7s", "t3/balloon_inflations", "t3/pump_paused"]
EXPECTED_ROWS = {"t2/minute": 426_652, "t3/reflux_runs_raw": 161_996, "t3/reflux_events_7s": 32_769,
                 "t3/balloon_inflations": 20_146, "t3/pump_paused": 54_546}
EXPECTED_EVENTS = {"minor": 24_530, "massive": 8_239}
EXPECTED_MINUTE_FOLDERS = 46
EXPECTED_BALLOON_FOLDERS = 42

con = duckdb.connect()
con.sql("SET enable_progress_bar = false")
results = []


def report(ok, what, detail):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {what}: {detail}")


def src(root, table):
    return f"read_parquet('{(Path(root) / table).as_posix()}/*.parquet')"


def schema(root, table):
    files = sorted((Path(root) / table).glob("*.parquet"))
    if not files:
        return None
    return [tuple(r[:2]) for r in con.sql(f"DESCRIBE SELECT * FROM read_parquet('{files[0].as_posix()}')").fetchall()]


def per_folder(root, table):
    rows = {}
    for f in sorted((Path(root) / table).glob("*.parquet")):
        rows[f.stem] = con.sql(f"SELECT count(*) FROM read_parquet('{f.as_posix()}')").fetchone()[0]
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--keep", action="store_true", help="keep the rebuilt tables")
    ap.add_argument("--rebuilt", help="compare this earlier rebuild instead of rebuilding")
    args = ap.parse_args()
    for t in TABLES:
        if not (DATA_DIR / t).is_dir():
            sys.exit(f"{DATA_DIR / t} not found; set DATA_DIR to the folder holding t2/ and t3/.")

    if args.rebuilt:
        new = Path(args.rebuilt)
    else:
        if not T0_DIR.is_dir():
            sys.exit(f"T0_DIR {T0_DIR} not found; set T0_DIR to the folder holding RMC1-<id>/*.parquet.")
        new = Path(tempfile.mkdtemp(prefix="t2t3_rebuild_"))
        print(f"Rebuilding into {new}", flush=True)
        env = dict(os.environ, PYTHONUTF8="1")
        for script in ("build_t2_t3.py", "build_reflux_events.py"):
            subprocess.run([sys.executable, str(HERE / script), "--out", str(new)], check=True, env=env)
    print()

    try:
        for t in TABLES:
            so, sn = schema(DATA_DIR, t), schema(new, t)
            diff = [] if so == sn else [f"{a} vs {b}" for a, b in zip(so or [], sn or []) if a != b]
            if so != sn and not diff:
                diff = [f"{len(so or [])} vs {len(sn or [])} columns"]
            report(so == sn, f"{t} schema", "identical" if so == sn else "; ".join(diff[:5]))

            po, pn = per_folder(DATA_DIR, t), per_folder(new, t)
            bad = sorted(f for f in set(po) | set(pn) if po.get(f) != pn.get(f))
            detail = (f"{len(pn)} folder files, rows {sum(pn.values()):,} (existing {sum(po.values()):,})"
                      + (f"; differ in {len(bad)}: " + ", ".join(f"{f} {po.get(f)}->{pn.get(f)}" for f in bad[:8])
                         if bad else ""))
            report(not bad, f"{t} rows per folder", detail)
            report(sum(pn.values()) == EXPECTED_ROWS[t], f"{t} total",
                   f"{sum(pn.values()):,} (expected {EXPECTED_ROWS[t]:,})")

        n_min = con.sql(f"SELECT count(DISTINCT folder_id) FROM {src(new, 't2/minute')}").fetchone()[0]
        report(n_min == EXPECTED_MINUTE_FOLDERS, "t2/minute folders", f"{n_min} (expected {EXPECTED_MINUTE_FOLDERS})")
        n_bal = con.sql(f"SELECT count(DISTINCT folder_id) FROM {src(new, 't3/balloon_inflations')}").fetchone()[0]
        report(n_bal == EXPECTED_BALLOON_FOLDERS, "balloon_inflations patients",
               f"{n_bal} (expected {EXPECTED_BALLOON_FOLDERS})")

        by_type = dict(con.sql(f"SELECT type, count(*) FROM {src(new, 't3/reflux_events_7s')} GROUP BY 1").fetchall())
        for k, v in EXPECTED_EVENTS.items():
            report(by_type.get(k, 0) == v, f"reflux_events_7s {k}", f"{by_type.get(k, 0):,} (expected {v:,})")

        key = "folder_id, type, CAST(start_t AS TIMESTAMP_NS) AS start_t, CAST(end_t AS TIMESTAMP) AS end_t"
        m = con.sql(f"""
            WITH o AS (SELECT {key} FROM {src(DATA_DIR, 't3/reflux_events_7s')}),
                 n AS (SELECT {key} FROM {src(new, 't3/reflux_events_7s')})
            SELECT count(*) FILTER (o.folder_id IS NOT NULL AND n.folder_id IS NOT NULL),
                   count(*) FILTER (n.folder_id IS NULL), count(*) FILTER (o.folder_id IS NULL)
            FROM o FULL OUTER JOIN n ON o.folder_id = n.folder_id AND o.type = n.type
                 AND o.start_t = n.start_t AND o.end_t = n.end_t""").fetchone()
        report(m[1] == 0 and m[2] == 0, "reflux_events_7s exact match (folder_id, type, start_t, end_t)",
               f"matched {m[0]:,}; mismatches {m[1] + m[2]:,} (existing only {m[1]:,}, rebuilt only {m[2]:,})")
        if m[1] or m[2]:
            near = con.sql(f"""
                SELECT count(*) FROM {src(DATA_DIR, 't3/reflux_events_7s')} o
                WHERE EXISTS (SELECT 1 FROM {src(new, 't3/reflux_events_7s')} n
                              WHERE n.folder_id = o.folder_id AND n.type = o.type
                                AND abs(epoch(n.start_t) - epoch(o.start_t)) <= 1
                                AND abs(epoch(n.end_t) - epoch(o.end_t)) <= 1)""").fetchone()[0]
            print(f"      info: existing episodes with a rebuilt episode within 1 s at both ends: {near:,}")
    finally:
        if not args.rebuilt and not args.keep:
            shutil.rmtree(new, ignore_errors=True)
        elif not args.rebuilt:
            print(f"\nRebuilt tables kept in {new}")

    ok = all(results)
    print(f"\n{'PASS' if ok else 'FAIL'}: {sum(results)} of {len(results)} checks passed")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
