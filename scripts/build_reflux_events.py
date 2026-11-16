"""Reflux episodes from the raw reflux runs: the episode rule of the paper.

Rule (reproduces the manufacturer's daily-report counts): runs of FT state 11 (minor) or 12 (massive) shorter
than 1.5 s are dropped; then runs of the same state whose gap (start of a run minus end of the previous kept
run of that state) is at most 7.5 s are merged into one episode. Runs of the other state or of state 14 in
between do not interrupt a merge.

Reads t3/reflux_runs_raw/<folder>.parquet and writes t3/reflux_events_7s/<folder>.parquet.
    python scripts/build_reflux_events.py                 # under DATA_DIR
    python scripts/build_reflux_events.py --out <folder>  # runs and episodes under another root
    python scripts/build_reflux_events.py --self-test     # check the rule on a small made-up example
"""
import argparse
from pathlib import Path

import duckdb
import pandas as pd

from config import DATA_DIR

MIN_RUN_S = 1.5
MAX_GAP_S = 7.5
TYPES = {11: "minor", 12: "massive"}

EVENT_SCHEMA = [
    ("folder_id", "VARCHAR"), ("era", "TINYINT"), ("state", "INTEGER"), ("type", "VARCHAR"), ("ev_id", "DOUBLE"),
    ("start_t", "TIMESTAMP_NS"), ("end_t", "TIMESTAMP"), ("start_t_device", "TIMESTAMP_NS"), ("span_s", "DOUBLE"),
    ("in_state_s", "DOUBLE"), ("n_runs", "BIGINT"), ("balloon_req_max", "FLOAT"), ("psf5_frac_max", "DOUBLE"),
    ("balloon_p_max", "FLOAT"), ("grv_reflux_ml_max", "FLOAT"),
]


def runs_to_events(runs, min_run_s=MIN_RUN_S, max_gap_s=MAX_GAP_S):
    """Drop runs shorter than min_run_s, then merge same-state runs at most max_gap_s apart.

    runs: DataFrame with folder_id, state, start_t, end_t, dur_s (and optionally era, start_t_device,
    balloon_req_max, psf5_frac, balloon_p_max, grv_reflux_ml_max). Returns one row per episode.
    """
    r = runs[runs["state"].isin(list(TYPES)) & (runs["dur_s"] >= min_run_s)].copy()
    for c in ("era", "start_t_device", "balloon_req_max", "psf5_frac", "balloon_p_max", "grv_reflux_ml_max"):
        if c not in r:
            r[c] = None
    r["start_t"] = pd.to_datetime(r["start_t"]).astype("datetime64[ns]")
    r["end_t"] = pd.to_datetime(r["end_t"]).astype("datetime64[ns]")
    r = r.sort_values(["folder_id", "state", "start_t"], kind="mergesort")
    prev_end = r.groupby(["folder_id", "state"])["end_t"].shift()
    # gaps are compared at millisecond precision, so a gap of exactly 7.5 s merges despite floating-point error
    gap = (r["start_t"] - prev_end).dt.total_seconds().round(3)
    r["episode"] = (gap.isna() | (gap > max_gap_s)).cumsum()
    ev = r.groupby("episode").agg(
        folder_id=("folder_id", "first"), era=("era", "first"), state=("state", "first"),
        start_t=("start_t", "min"), end_t=("end_t", "max"), start_t_device=("start_t_device", "first"),
        in_state_s=("dur_s", "sum"), n_runs=("dur_s", "size"), balloon_req_max=("balloon_req_max", "max"),
        psf5_frac_max=("psf5_frac", "max"), balloon_p_max=("balloon_p_max", "max"),
        grv_reflux_ml_max=("grv_reflux_ml_max", "max")).reset_index(drop=True)
    ev["type"] = ev["state"].map(TYPES)
    ev["span_s"] = (ev["end_t"] - ev["start_t"]).dt.total_seconds()
    ev = ev.sort_values(["folder_id", "start_t", "state"], kind="mergesort").reset_index(drop=True)
    ev["ev_id"] = ev.groupby("folder_id").cumcount() + 1.0
    return ev[[c for c, _ in EVENT_SCHEMA]]


def self_test():
    t = pd.Timestamp("2021-01-01 10:00:00")
    s = lambda x: t + pd.Timedelta(seconds=x)
    runs = pd.DataFrame({
        "folder_id": "RMC1-000", "state": [11, 11, 12, 11, 11, 11, 14],
        "start_t": [s(0), s(5), s(7), s(10), s(30), s(40), s(50)],
        "end_t": [s(3), s(8), s(9), s(12), s(31), s(43), s(60)],
        "dur_s": [3, 3, 2, 2, 1, 3, 10]})
    ev = runs_to_events(runs)
    # minor: 0-3, 5-8, 10-12 merge (gaps 2 s, 2 s; the massive run in between does not interrupt);
    # 30-31 is dropped (1 s); 40-43 is 28 s after 12 s, a new episode. massive: 7-9. State 14 is ignored.
    got = [(x.type, (x.start_t - t).total_seconds(), (x.end_t - t).total_seconds(), x.n_runs) for x in ev.itertuples()]
    want = [("minor", 0, 12, 3), ("massive", 7, 9, 1), ("minor", 40, 43, 1)]
    assert got == want, got
    assert runs_to_events(runs.assign(dur_s=runs.dur_s.where(runs.start_t != s(5), 1.4))).n_runs.iloc[0] == 2
    print("self-test passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(DATA_DIR), help="root holding t3/reflux_runs_raw (default DATA_DIR)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    t3 = Path(args.out) / "t3"
    dst = t3 / "reflux_events_7s"
    dst.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.sql("SET enable_progress_bar = false")
    totals = {"minor": 0, "massive": 0}
    for f in sorted((t3 / "reflux_runs_raw").glob("*.parquet")):
        runs = con.sql(f"SELECT * FROM read_parquet('{f.as_posix()}')").df()
        ev = runs_to_events(runs)
        con.register("ev", ev)
        sel = ", ".join(f'CAST("{c}" AS {t}) AS "{c}"' for c, t in EVENT_SCHEMA)
        con.sql(f"COPY (SELECT {sel} FROM ev ORDER BY ev_id) TO '{(dst / f.name).as_posix()}' (FORMAT parquet)")
        con.unregister("ev")
        n = ev["type"].value_counts()
        for k in totals:
            totals[k] += int(n.get(k, 0))
        print(f.stem, f"runs {len(runs):,}, episodes minor {int(n.get('minor', 0)):,} massive {int(n.get('massive', 0)):,}",
              flush=True)
    print(f"\nTotal episodes: minor {totals['minor']:,}, massive {totals['massive']:,}. Wrote {dst}")


if __name__ == "__main__":
    main()
