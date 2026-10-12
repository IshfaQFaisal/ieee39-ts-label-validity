"""Repair a data folder after a crash or a full disk.

Usage:  python repair_data.py data_v3

What it does (it never touches good data):
  1. Backs up results.csv to results.csv.bak.
  2. Drops rows that are unusable: label -1 with cause 'exception' (e.g. 'No space left'),
     and any row whose trajectory file is missing, empty or cannot be read.
  3. Skips a half-written last line of results.csv, if there is one.
  4. Deletes trajectory files that have no valid row (partial writes).
Then rerun the SAME generator command: it will redo only the scenarios that were dropped.
"""
import os, sys, shutil
import numpy as np
import pandas as pd

out = sys.argv[1] if len(sys.argv) > 1 else "data_v3"
csv = os.path.join(out, "results.csv")
tdir = os.path.join(out, "traj")
shutil.copy(csv, csv + ".bak")
d = pd.read_csv(csv, on_bad_lines="skip")
n0 = len(d)

def traj_ok(sid):
    p = os.path.join(tdir, sid + ".npz")
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        return False
    try:
        z = np.load(p)
        return z["delta"].shape[1] == 10 and z["omega"].shape[1] == 10
    except Exception:
        return False

bad_exc = (d.label == -1) & (d.cause == "exception")
need = d.label >= 0                                   # valid runs must have a trajectory
ok_tr = pd.Series([traj_ok(s) if n else True for s, n in zip(d.sid, need)], index=d.index)
keep = ~bad_exc & ok_tr
print(f"rows: {n0}; exception rows dropped: {int(bad_exc.sum())}; "
      f"rows with bad/missing trajectory dropped: {int((~ok_tr & ~bad_exc).sum())}")
d[keep].to_csv(csv, index=False)

valid = set(d[keep].sid)
removed = 0
for f in os.listdir(tdir):
    if f.endswith(".npz") and f[:-4] not in valid:
        os.remove(os.path.join(tdir, f)); removed += 1
print(f"orphan trajectory files removed: {removed}")
print(f"rows kept: {int(keep.sum())}  (backup: {csv}.bak)")
