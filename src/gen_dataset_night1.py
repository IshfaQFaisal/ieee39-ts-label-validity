#!/usr/bin/env python
"""
gen_dataset.py -- seeded, resumable scenario generator for IEEE 39-bus
transient stability (ANDES 2.0.0).

Each scenario = (fault bus, fault duration, fault impedance xf, load scale).
For every scenario it stores
  * one row in  <out>/results.csv   (inputs, label, cause, flags, audit result)
  * one file    <out>/traj/<sid>.npz (rotor angle + speed on a fixed 10 ms grid)
and a static description of the system in <out>/static.json.

Design points
  * Scenario i of seed s is a pure function of (s, i): reruns, resumes and
    parallel workers always produce identical scenarios.
  * Fault buses are cycled through a seeded shuffle of all 39 buses, so every
    bus gets (almost) the same number of scenarios.
  * A deterministic ~audit_pct % of scenarios is rerun at half the time step;
    the two labels are compared and stored (columns label_half / agree).
  * Resumable: scenarios whose sid is already in results.csv are skipped.
  * A settings guard refuses to resume into a folder created with different
    simulation settings, so data produced with different settings never mixes.

Usage (Anaconda Prompt, andes env active):
  python gen_dataset.py --out data_v2 --n 20  --seed 1 --workers 1     # smoke test
  python gen_dataset.py --out data_v2 --n 2000 --seed 1 --workers 3    # real run
  python gen_dataset.py --out data_v2 --summary                         # progress / stats
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # one thread per worker process

import argparse
import json
import multiprocessing as mp
import platform
import sys
import time
import zlib
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np
import pandas as pd

CASE = "ieee39/ieee39_full.xlsx"
T_FAULT = 1.0           # fault inception time (s)
GRID_START = 0.9        # trajectory grid starts here (s)

ROW_KEYS = [
    "sid", "seed", "idx", "bus", "gen_bus", "load_scale", "duration", "xf",
    "t_clear", "tstep", "t_end", "label", "cause", "max_spread", "t_last",
    "post_window", "no_postfault_signal", "runtime", "audit", "label_half",
    "cause_half", "max_spread_half", "t_last_half", "agree", "error",
]

GUARD_KEYS = ["tstep", "t_end", "load_range", "dur_range", "xf_range",
              "thresh", "grid_dt"]


# --------------------------------------------------------------------------
# simulation
# --------------------------------------------------------------------------
def simulate(load_scale, bus, dur, xf, tstep, t_end):
    """Run one fault scenario. Returns ((t, delta_deg, omega_pu) or None, error)."""
    import andes
    andes.config_logger(stream_level=50)
    ss = andes.load(andes.get_case(CASE), setup=False, no_output=True)
    ss.add("Fault", dict(bus=bus, tf=T_FAULT, tc=round(T_FAULT + dur, 4), xf=xf))
    ss.setup()

    # scale loads and non-slack generation together (slack covers only losses)
    ss.PQ.p0.v[:] = ss.PQ.p0.v * load_scale
    ss.PQ.q0.v[:] = ss.PQ.q0.v * load_scale
    ss.PV.p0.v[:] = ss.PV.p0.v * load_scale

    ss.PFlow.run()
    if not ss.PFlow.converged:
        return None, "pflow_failure"

    ss.TDS.config.tf = t_end
    ss.TDS.config.no_tqdm = 1
    ss.TDS.config.tstep = tstep
    ss.TDS.run()

    # ANDES keeps simulating after a failed dynamic initialization (it only prints
    # a message and sets test_ok = False), so check the flag explicitly.
    if not ss.TDS.test_ok:
        return None, "init_check_failed"
    t = np.asarray(ss.dae.ts.t, dtype=float)
    if len(t) < 2:
        return None, "init_failure"
    delta = np.rad2deg(np.asarray(ss.dae.ts.x[:, ss.GENROU.delta.a]))
    omega = np.asarray(ss.dae.ts.x[:, ss.GENROU.omega.a])
    return (t, delta, omega), None


def label_run(t, delta, t_end, thresh):
    """1 = unstable. Cause is kept for auditing; analysis should use the binary label."""
    spread = delta.max(axis=1) - delta.min(axis=1)
    mx = float(spread.max())
    if mx > thresh:
        return 1, "angle_separation", mx
    if t[-1] < t_end - 0.01:
        return 1, "solver_failure", mx
    return 0, "stable", mx


def resample(t, arr, grid):
    """Linear interpolation onto `grid`; NaN after the run ended."""
    tu, k = np.unique(t, return_index=True)
    out = np.full((len(grid), arr.shape[1]), np.nan, dtype=np.float32)
    ok = grid <= tu[-1] + 1e-9
    for j in range(arr.shape[1]):
        out[ok, j] = np.interp(grid[ok], tu, arr[k, j])
    return out


# --------------------------------------------------------------------------
# scenario sampling (pure function of seed and index)
# --------------------------------------------------------------------------
def make_scenario(seed, i, buses, cfg):
    order = np.random.default_rng(seed).permutation(buses)
    rng = np.random.default_rng([seed, i])
    bus = int(order[i % len(order)])
    load = round(float(rng.uniform(*cfg["load_range"])), 3)
    dur = round(float(rng.uniform(*cfg["dur_range"])), 3)
    lo, hi = np.log(cfg["xf_range"][0]), np.log(cfg["xf_range"][1])
    xf = float(f"{np.exp(rng.uniform(lo, hi)):.3g}")
    sid = f"s{seed}_{i:05d}"
    return dict(sid=sid, seed=seed, idx=i, bus=bus, load_scale=load,
                duration=dur, xf=xf)


def is_audited(sid, pct):
    return (zlib.crc32(sid.encode()) % 100) < pct


# --------------------------------------------------------------------------
# worker
# --------------------------------------------------------------------------
def run_scenario(args):
    sc, cfg, outdir = args
    row = dict.fromkeys(ROW_KEYS)
    row.update(sc)
    row.update(gen_bus=int(sc["bus"] >= 30), t_clear=round(T_FAULT + sc["duration"], 4),
               tstep=cfg["tstep"], t_end=cfg["t_end"])
    t0 = time.time()
    try:
        res, err = simulate(sc["load_scale"], sc["bus"], sc["duration"], sc["xf"],
                            cfg["tstep"], cfg["t_end"])
        if res is None:
            row.update(label=-1, cause=err)
        else:
            t, delta, omega = res
            lab, cause, mx = label_run(t, delta, cfg["t_end"], cfg["thresh"])
            row.update(label=lab, cause=cause, max_spread=mx, t_last=float(t[-1]))
            row["post_window"] = float(t[-1] - row["t_clear"])
            row["no_postfault_signal"] = int(lab == 1 and row["post_window"] < 0.05)
            grid = np.arange(GRID_START, cfg["t_end"] + 1e-9, cfg["grid_dt"])
            np.savez_compressed(os.path.join(outdir, "traj", sc["sid"] + ".npz"),
                                t=grid.astype(np.float32),
                                delta=resample(t, delta, grid),
                                omega=resample(t, omega, grid))
            if is_audited(sc["sid"], cfg["audit_pct"]):
                row["audit"] = 1
                res2, err2 = simulate(sc["load_scale"], sc["bus"], sc["duration"],
                                      sc["xf"], cfg["tstep"] / 2, cfg["t_end"])
                if res2 is None:
                    row.update(label_half=-1, cause_half=err2)
                else:
                    t2, d2, _ = res2
                    l2, c2, m2 = label_run(t2, d2, cfg["t_end"], cfg["thresh"])
                    row.update(label_half=l2, cause_half=c2, max_spread_half=m2,
                               t_last_half=float(t2[-1]), agree=int(l2 == lab))
            else:
                row["audit"] = 0
    except Exception as e:                                   # keep the batch alive
        row.update(label=-1, cause="exception", error=f"{type(e).__name__}: {str(e)[:150]}")
    row["runtime"] = round(time.time() - t0, 1)
    return row


# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------
def summary(outdir):
    path = os.path.join(outdir, "results.csv")
    if not os.path.exists(path):
        print("no results.csv in", outdir)
        return
    df = pd.read_csv(path)
    print(f"{len(df)} scenarios in {path}")
    print("\nlabel / cause counts (label -1 = no valid result):")
    print(df.groupby(["label", "cause"]).size().to_string())
    ok = df[df.label >= 0]
    if len(ok):
        print(f"\nunstable fraction (valid runs): {ok.label.mean():.3f}")
        print("unstable fraction by bus type (gen_bus 1 = buses 30-39):")
        print(ok.groupby("gen_bus").label.agg(["count", "mean"]).round(3).to_string())
        print(f"\nunstable runs with no post-fault signal: "
              f"{int(ok.no_postfault_signal.fillna(0).sum())} of {int(ok.label.sum())}")
    au = df[(df.audit == 1) & df.agree.notna()]
    if len(au):
        print(f"\nhalf-step audit: {len(au)} scenarios, label agreement {au.agree.mean():.3f}")
        au = au.assign(xf_bin=pd.cut(au.xf, [0, 2e-3, 5e-3, 1e-1]))
        print(au.groupby("xf_bin", observed=True).agree.agg(["count", "mean"]).round(3).to_string())
        bad = au[au.agree == 0][["sid", "bus", "load_scale", "duration", "xf",
                                 "label", "cause", "label_half", "cause_half"]]
        if len(bad):
            print("\ndisagreements:")
            print(bad.to_string(index=False))
    print(f"\nruntime: mean {df.runtime.mean():.1f} s/scenario, total {df.runtime.sum()/3600:.2f} h of CPU time")
    err = df[df.error.notna()]
    if len(err):
        print("\nerrors:")
        print(err.groupby("error").size().to_string())


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data_v2")
    ap.add_argument("--n", type=int, default=20, help="scenarios for this seed")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--tstep", type=float, default=0.001)
    ap.add_argument("--t-end", type=float, default=3.0)
    ap.add_argument("--load-range", type=float, nargs=2, default=[0.8, 1.3])
    ap.add_argument("--dur-range", type=float, nargs=2, default=[0.05, 0.40])
    ap.add_argument("--xf-range", type=float, nargs=2, default=[1e-3, 1e-2])
    ap.add_argument("--thresh", type=float, default=180.0)
    ap.add_argument("--grid-dt", type=float, default=0.01)
    ap.add_argument("--audit-pct", type=int, default=10)
    ap.add_argument("--max-minutes", type=float, default=0,
                    help="stop submitting new scenarios after this long (0 = no limit)")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--quiet", action="store_true",
                    help="no per-scenario console output (progress goes to <out>/run.log only). "
                         "Use for long runs: clicking in a Windows console can freeze a program "
                         "that is printing to it.")
    a = ap.parse_args()

    if a.summary:
        summary(a.out)
        return

    cfg = dict(tstep=a.tstep, t_end=a.t_end, load_range=a.load_range,
               dur_range=a.dur_range, xf_range=a.xf_range, thresh=a.thresh,
               grid_dt=a.grid_dt, audit_pct=a.audit_pct)

    os.makedirs(os.path.join(a.out, "traj"), exist_ok=True)

    # settings guard
    cfg_path = os.path.join(a.out, "config.json")
    if os.path.exists(cfg_path):
        old = json.load(open(cfg_path))["settings"]
        diff = {k: (old.get(k), cfg[k]) for k in GUARD_KEYS if old.get(k) != cfg[k]}
        if diff:
            sys.exit(f"Refusing to resume: settings differ from those in {cfg_path}:\n"
                     f"  {diff}\nUse a new --out folder, or delete the old one.")

    # warm-up in the main process (builds ANDES code once, records static data)
    import andes
    andes.config_logger(stream_level=50)
    ss = andes.load(andes.get_case(CASE), setup=True, no_output=True)
    buses = [int(b) for b in ss.Bus.idx.v]
    static = dict(buses=buses, gen_buses=[int(b) for b in ss.GENROU.bus.v],
                  M=[float(m) for m in ss.GENROU.M.v], grid_start=GRID_START,
                  grid_dt=a.grid_dt, t_fault=T_FAULT,
                  columns="trajectory columns follow gen_buses order")
    json.dump(static, open(os.path.join(a.out, "static.json"), "w"), indent=1)
    meta = dict(settings=cfg, andes=andes.__version__, python=platform.python_version(),
                numpy=np.__version__, pandas=pd.__version__, platform=platform.platform())
    old_meta = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    meta["seeds_used"] = sorted(set(old_meta.get("seeds_used", [])) | {a.seed})
    json.dump(meta, open(cfg_path, "w"), indent=1)
    del ss

    res_path = os.path.join(a.out, "results.csv")
    done = set(pd.read_csv(res_path, usecols=["sid"]).sid) if os.path.exists(res_path) else set()
    todo = [i for i in range(a.n) if f"s{a.seed}_{i:05d}" not in done]
    print(f"seed {a.seed}: {a.n} requested, {len(done)} rows already in folder, "
          f"{len(todo)} to run, workers={a.workers}", flush=True)
    if not todo:
        print("nothing to do")
        return
    if a.quiet:
        print(f"running quietly; progress is written to {os.path.join(a.out, 'run.log')} "
              f"(check with --summary)", flush=True)

    t_start = time.time()
    n_done = 0

    def write(row):
        pd.DataFrame([row], columns=ROW_KEYS).to_csv(
            res_path, mode="a", header=not os.path.exists(res_path), index=False)

    log_path = os.path.join(a.out, "run.log")

    def log(msg):
        with open(log_path, "a") as fh:          # file writes cannot be frozen by the console
            fh.write(msg + "\n")
        if not a.quiet:
            print(msg, flush=True)

    def report(row):
        nonlocal n_done
        n_done += 1
        el = (time.time() - t_start) / 60
        log(f"[{n_done}/{len(todo)}] {row['sid']} bus={row['bus']} dur={row['duration']} "
            f"xf={row['xf']} load={row['load_scale']} -> {row['label']} {row['cause']} "
            f"({row['runtime']} s) | {el:.1f} min elapsed")

    def args_for(i):
        return (make_scenario(a.seed, i, buses, cfg), cfg, a.out)

    if a.workers <= 1:
        for i in todo:
            if a.max_minutes and (time.time() - t_start) / 60 > a.max_minutes:
                print("time limit reached, stopping (rerun the same command to resume)")
                break
            row = run_scenario(args_for(i))
            write(row)
            report(row)
    else:
        it = iter(todo)
        running = {}
        stop = False
        with ProcessPoolExecutor(max_workers=a.workers,
                                 mp_context=mp.get_context("spawn")) as ex:
            def submit_next():
                try:
                    i = next(it)
                except StopIteration:
                    return
                running[ex.submit(run_scenario, args_for(i))] = i
            for _ in range(a.workers):
                submit_next()
            while running:
                finished, _ = wait(list(running), return_when=FIRST_COMPLETED)
                for f in finished:
                    running.pop(f)
                    row = f.result()
                    if a.max_minutes and (time.time() - t_start) / 60 > a.max_minutes:
                        if not stop:
                            log("time limit reached, finishing running scenarios "
                                "(rerun the same command to resume)")
                        stop = True
                    if not stop:
                        submit_next()        # hand out new work before any printing
                    write(row)
                    report(row)

    print(f"\nfinished {n_done} scenarios in {(time.time() - t_start) / 60:.1f} min")
    summary(a.out)


if __name__ == "__main__":
    main()
