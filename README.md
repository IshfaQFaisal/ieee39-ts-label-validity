# How much can you trust reported accuracy in ML-based transient stability assessment?

Label validity and evaluation validity for machine-learning transient stability classification on the IEEE 39-bus system, using time-domain simulations from [ANDES](https://github.com/CURENT/andes) 2.0.0.

**Status:** work in progress; a preprint is in preparation. Results below are from the current dataset and may change before the preprint.

## Question

Papers on ML-based transient stability assessment often report accuracy above 99%. We ask how much of that survives two checks:

1. **Label validity.** Are the stable/unstable labels themselves reliable? Labels come from a numerical simulation, so the time step and fault impedance can flip them.
2. **Evaluation validity.** Does the accuracy hold when the test set is not a random sample of the same scenarios, for example at buses the model never saw, or for scenarios near the stability boundary?

## Data

7,000 fault scenarios on the IEEE 39-bus case (`ieee39/ieee39_full.xlsx` shipped with ANDES): three-phase fault at t = 1.0 s on a bus, load scaling 0.8-1.3, fault duration 0.05-0.40 s, fault impedance xf 1e-3 to 1e-2 p.u. (log-uniform), 1 ms step, 3 s window.

| Set | Scenarios | Sampling |
|---|---|---|
| Night 1 (`data/night1`) | 3,000 | uniform over the parameter ranges |
| Night 2 (`data/night2`) | 4,000 | concentrated near each bus's estimated stability boundary (`src/boundary.json`) |

A run is labelled unstable if the rotor-angle spread exceeds 180 degrees or the solver fails. Solver-failure labels and low-impedance cases were re-simulated at 0.5 ms (and 0.25 ms) and the finer-step label is used where available.

The rotor angle and speed trajectories (about 40 MB per night) are not in this repository; they are archived on Zenodo (link to be added). Unzip them into `data/night1/traj/` and `data/night2/traj/`.

## Main findings (current data)

- **Labels.** Of 74 solver-failure labels at 1 ms, 20 (27%) turned out stable at a finer step. Angle-separation labels are reliable: of 402 low-impedance cases rechecked at 0.5 ms, 2 flipped (0.5%).
- **Random split vs. held-out buses.** A model that sees only scenario parameters and bus id (no dynamics) scores balanced accuracy 0.97 on a random split of the uniform data, close to a trajectory model (0.98). On near-boundary scenarios at unseen buses it is at chance (AUC 0.50).
- **Near-boundary scenarios.** The trajectory model drops from about 0.98 balanced accuracy on the usual benchmark to 0.76-0.84 on near-boundary scenarios at unseen buses.

Full tables with bootstrap confidence intervals are in `results/`.

## Repository layout

```
src/        scenario generator, boundary model, repair and rerun scripts
analysis/   feature extraction and evaluation scripts
data/       scenario tables (results.csv, rerun files); trajectories are not included
results/    evaluation output tables
```

## Reproduce

```
conda env create -f environment.yml
conda activate andes39

# 1. generate scenarios (hours; resumable)
python src/gen_dataset.py --out data_v3 --boundary src/boundary.json

# 2. re-simulate uncertain labels at finer steps
python src/rerun_failures.py data_v3 --workers 4

# 3. evaluation (needs the trajectory files in data/*/traj)
python analysis/eval1.py
python analysis/eval2.py
```

Notes: night 1 was generated with an earlier version of the generator (uniform sampling, no boundary model); see the preprint for details. Generation time depends on the machine (night 1 took about 4.5 hours on an office Windows PC).

## Limitations

One system, one simulator, finite fault impedance (not a bolted fault), a 3 s first-swing window, a 180-degree separation criterion, constant-impedance loads, and no line-trip clearing. The near-boundary subset is defined by a logistic boundary model fitted on the night-1 data. Only gradient boosting has been evaluated so far; results may differ for other architectures.

## Citation

If you use this code or data, please cite the preprint (link to be added) and the ANDES paper: H. Cui, F. Li, K. Tomsovic, "Hybrid Symbolic-Numeric Framework for Power System Modeling and Analysis," IEEE Trans. Power Systems, 2021.

## License

MIT
