"""Re-simulate every solver_failure scenario (and every audit disagreement) at finer steps.

Usage:  python rerun_failures.py data_v3 --workers 6
        python rerun_failures.py data_v2 --workers 6
        python rerun_failures.py data_v3 --workers 6 --select angle_lowxf --xf-below 0.002 \
               --steps 0.0005 --outfile recheck_angle_lowxf.csv

Reads <out>/results.csv, picks rows with cause == solver_failure or agree == 0,
reruns each at the steps below, and writes <out>/rerun_failures.csv with the label
and cause at each step. Resumable: sids already in the output are skipped.
"""
import argparse, os, shutil, sys, time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd

DEFAULT_STEPS = [0.0005, 0.00025]    # seconds; the original run used 0.001


def work(args):
    sid, bus, load, dur, xf, t_end, thresh, steps = args
    if shutil.disk_usage(os.getcwd()).free < 1.5e9:          # protect the disk: do nothing, row is not recorded
        return None
    from gen_dataset import simulate, label_run          # imported inside the worker
    out = dict(sid=sid)
    for ts in steps:
        tag = f"{ts*1000:g}ms"
        t0 = time.time()
        try:
            res, err = simulate(load, bus, dur, xf, ts, t_end)
            if res is None:
                out.update({f"label_{tag}": -1, f"cause_{tag}": err})
            else:
                t, d, _ = res
                lab, cause, mx = label_run(t, d, t_end, thresh)
                out.update({f"label_{tag}": lab, f"cause_{tag}": cause,
                            f"spread_{tag}": round(mx, 2), f"tlast_{tag}": round(float(t[-1]), 4)})
        except Exception as e:
            out.update({f"label_{tag}": -1, f"cause_{tag}": f"{type(e).__name__}: {str(e)[:80]}"})
        out[f"secs_{tag}"] = round(time.time() - t0, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--thresh", type=float, default=180.0)
    ap.add_argument("--select", choices=["failures", "angle_lowxf"], default="failures",
                    help="failures: solver_failure rows and audit disagreements (default); "
                         "angle_lowxf: angle_separation rows with xf below --xf-below")
    ap.add_argument("--xf-below", type=float, default=0.002)
    ap.add_argument("--steps", type=float, nargs="+", default=DEFAULT_STEPS)
    ap.add_argument("--outfile", default="rerun_failures.csv")
    a = ap.parse_args()

    d = pd.read_csv(os.path.join(a.out, "results.csv"))
    if a.select == "failures":
        sel = d[(d.cause == "solver_failure") | (d.agree == 0)].copy()
    else:
        sel = d[(d.cause == "angle_separation") & (d.xf < a.xf_below)].copy()
    path = os.path.join(a.out, a.outfile)
    done = set(pd.read_csv(path).sid) if os.path.exists(path) else set()
    sel = sel[~sel.sid.isin(done)]
    print(f"{len(sel)} scenarios to rerun at steps {a.steps} (skipping {len(done)} already done)", flush=True)
    if sel.empty:
        return
    jobs = [(r.sid, int(r.bus), float(r.load_scale), float(r.duration), float(r.xf),
             float(r.t_end), a.thresh, a.steps) for r in sel.itertuples()]
    t_start = time.time(); n = 0; skipped = 0
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn")) as ex:
        futs = [ex.submit(work, j) for j in jobs]
        for f in as_completed(futs):
            row = f.result()
            if row is None:
                skipped += 1
                print("skipped (disk below 1.5 GB free); rerun the same command later", flush=True)
                continue
            pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
            n += 1
            print(f"[{n}/{len(jobs)}] {row['sid']} | {(time.time()-t_start)/60:.1f} min", flush=True)
    if skipped:
        print(f"\n{skipped} scenarios skipped because the disk was nearly full; rerun the same command to finish them.")
    # summary
    r = pd.read_csv(path).merge(d[["sid", "bus", "duration", "xf", "label", "cause"]], on="sid")
    print("\noriginal label (1 ms) vs finer steps")
    tags = [f"{ts*1000:g}ms" for ts in a.steps]
    for tag in tags:
        print(f"  {tag}: stable {int((r[f'label_{tag}']==0).sum())}, unstable {int((r[f'label_{tag}']==1).sum())}, "
              f"failed {int((r[f'label_{tag}']==-1).sum())}  (of {len(r)})")
    if len(tags) > 1:
        print(f"agree {tags[0]} vs {tags[-1]}:", int((r[f"label_{tags[0]}"] == r[f"label_{tags[-1]}"]).sum()), "of", len(r))
    print("original label (1 ms) unchanged at", tags[0], ":", int((r[f"label_{tags[0]}"] == r["label"]).sum()), "of", len(r))


if __name__ == "__main__":
    main()
