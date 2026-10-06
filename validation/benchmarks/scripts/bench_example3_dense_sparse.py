"""Example 3: dense versus sparse backend, pointwise versus factorized evaluation, fifth-order 2Q response.

Model, UFSS pathways, and protocol from models.py (as in examples/example3_chi5_2q.ipynb:
N_max = 3, D = 10, eta = 8 meV); the 31 x 31 grid of the former benchmark over the same window, and
the published 151 x 151 grid for the factorized evaluation. The model is closed and in its eigenbasis,
so the sparse backend uses the diagonal preconditioner (preconditioner="auto").
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # single-thread BLAS, before NumPy is imported
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))   # validation/ (models.py)
from models import DATA
import json
import time

import numpy as np
from qudpy_fdgf import SpectroscopySolver
from models import EX3, EX3_AXES, EX3_FIXED, EX3_PROTOCOL, example3_model, example3_pathways

ETA = 0.008
SMALL = {"omega_2q": np.linspace(-3.20, -2.99, 31), "omega_emit": np.linspace(1.42, 1.66, 31)}


def calculate(backend, strategy, axes):
    model = example3_model()
    pathways = example3_pathways()
    solver = SpectroscopySolver(backend=backend, eta=ETA, max_cache_entries=4096)
    solver.feed_model(model)
    solver.set_pathways(pathways)
    start = time.perf_counter()
    result = solver.generate_nq_spectrum(2, EX3_PROTOCOL, axes=axes, fixed_coordinates=EX3_FIXED,
                                         pathways=pathways, observables=("polarization_h", "polarization_l"),
                                         strategy=strategy)
    D = sum(solver.backend.layout.dimensions.values())
    return result, time.perf_counter() - start, D, len(pathways)


def worst_difference(first, second):
    worst, details = 0.0, {}
    for name in ("polarization_h", "polarization_l"):
        a = sum(first.observables[name].values())
        b = sum(second.observables[name].values())
        details[name] = float(np.max(abs(a - b)) / np.max(abs(a)))
        worst = max(worst, details[name])
    return worst, details


results, runs, D, n_path = {}, {}, None, None
for backend in ("dense", "sparse_sector"):
    for strategy in ("pointwise", "factorized"):
        result, elapsed, D, n_path = calculate(backend, strategy, SMALL)
        results[(backend, strategy)] = result
        runs[f"{backend}_{strategy}_31x31"] = {"time_s": elapsed, "resolvent_solves": result.diagnostics["resolvent_solves"],
                                               "propagations": result.diagnostics["propagations"]}
        print(f"{backend:14s} {strategy:10s} 31x31 {elapsed:8.3f} s, {result.diagnostics['resolvent_solves']} solves", flush=True)
    result, elapsed, _, _ = calculate(backend, "factorized", EX3_AXES)
    runs[f"{backend}_factorized_151x151"] = {"time_s": elapsed, "resolvent_solves": result.diagnostics["resolvent_solves"],
                                             "propagations": result.diagnostics["propagations"]}
    print(f"{backend:14s} factorized 151x151 {elapsed:8.3f} s, {result.diagnostics['resolvent_solves']} solves", flush=True)

worst, details = worst_difference(results[("dense", "factorized")], results[("sparse_sector", "factorized")])
strategy_difference = {backend: worst_difference(results[(backend, "pointwise")], results[(backend, "factorized")])[0]
                       for backend in ("dense", "sparse_sector")}
report = {"D": D, "pathways": n_path, "grid": [31, 31], "full_grid": [151, 151], "eta": ETA, "runs": runs,
          "time_dense_s": runs["dense_factorized_31x31"]["time_s"], "time_sparse_s": runs["sparse_sector_factorized_31x31"]["time_s"],
          "relative_difference": details, "strategy_difference": strategy_difference}
print(f"D = {D}, {n_path} pathways; dense versus sparse {details}; pointwise versus factorized {strategy_difference}")
out = DATA
out.mkdir(parents=True, exist_ok=True)
(out / "example3_dense_sparse.json").write_text(json.dumps(report, indent=2))
