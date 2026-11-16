"""Check the episode rule against the existing tables, without the full-rate logs.

Applies runs_to_events() to DATA_DIR/t3/reflux_runs_raw and compares the result with
DATA_DIR/t3/reflux_events_7s (counts by type and an exact match on folder, type, start and end).
Read-only: writes nothing.
    py -3.12 scripts/verify_events.py
"""
import sys
from pathlib import Path

import duckdb
import pandas as pd

from build_reflux_events import runs_to_events
from config import DATA_DIR

KEY = ["folder_id", "type", "start_t", "end_t"]


def main():
    t3 = Path(DATA_DIR) / "t3"
    runs_dir, ev_dir = t3 / "reflux_runs_raw", t3 / "reflux_events_7s"
    if not runs_dir.is_dir() or not ev_dir.is_dir():
        sys.exit(f"Missing {runs_dir} or {ev_dir}; set DATA_DIR.")
    con = duckdb.connect()
    tot = {"rebuilt": {"minor": 0, "massive": 0}, "existing": {"minor": 0, "massive": 0}}
    only_new = only_old = 0
    bad = []
    for f in sorted(runs_dir.glob("*.parquet")):
        new = runs_to_events(con.sql(f"SELECT * FROM read_parquet('{f.as_posix()}')").df())
        g = ev_dir / f.name
        old = con.sql(f"SELECT * FROM read_parquet('{g.as_posix()}')").df() if g.exists() else new.iloc[0:0]
        for name, d in (("rebuilt", new), ("existing", old)):
            n = d["type"].value_counts()
            for k in ("minor", "massive"):
                tot[name][k] += int(n.get(k, 0))
        a, b = new[KEY].copy(), old[KEY].copy()
        for d in (a, b):
            d["start_t"] = pd.to_datetime(d["start_t"]).astype("datetime64[ns]").dt.round("ms")
            d["end_t"] = pd.to_datetime(d["end_t"]).astype("datetime64[ns]").dt.round("ms")
        m = a.merge(b, on=KEY, how="outer", indicator=True)
        x, y = int((m._merge == "left_only").sum()), int((m._merge == "right_only").sum())
        only_new += x; only_old += y
        if x or y:
            bad.append((f.stem, len(a), len(b), x, y))
    print("Episodes  rebuilt: minor {minor:,}, massive {massive:,}".format(**tot["rebuilt"]))
    print("Episodes existing: minor {minor:,}, massive {massive:,}".format(**tot["existing"]))
    print(f"Unmatched episodes: {only_new:,} only in rebuild, {only_old:,} only in existing")
    for b in bad[:15]:
        print("  folder {}: rebuilt {}, existing {}, only-rebuilt {}, only-existing {}".format(*b))
    ok = only_new == 0 and only_old == 0
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
