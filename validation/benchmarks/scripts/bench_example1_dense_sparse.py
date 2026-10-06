"""Example 1: dense versus sparse backend, pointwise versus factorized evaluation.

Same model (models.py), pathways, protocol, grid, and observables as Listing 1 of the manuscript
(polarization, action-detected population, integrated fluorescence), computed with
backend="dense" and backend="sparse_sector" (default preconditioner), each with
strategy="pointwise" and strategy="factorized". Reports the maximum difference between the
backends and between the strategies, relative to the maximum of each map, and for every run the
wall-clock time and the number of resolvent solves.
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
from qudpy_fdgf import ObservableSpec, SpectroscopySolver
from models import EX1, EX1_AXES, EX1_PATHWAYS, EX1_PROTOCOL, example1_model

gamma_1 = EX1["gamma_1"]
protocol, axes = EX1_PROTOCOL, EX1_AXES


def run(backend, strategy):
    solver = SpectroscopySolver(backend=backend, eta=EX1["eta"])
    solver.feed_model(example1_model())
    radiative = solver.jump_channel_names()[0]
    observables = {
        "polarization": "polarization",
        "population": ObservableSpec.action(
            "population", fourth_interaction="Bu", operator="excited_population"),
        "fluorescence": ObservableSpec.mean_jump(
            "fluorescence", radiative, time_window=(0.0, 5 / gamma_1),
            fourth_interaction="Bu"),
    }
    start = time.perf_counter()
    result = solver.generate_spectrum(protocol, axes, pathways=EX1_PATHWAYS,
                                      fixed_coordinates={"t2": EX1["t2"]},
                                      observables=observables, strategy=strategy)
    return result, time.perf_counter() - start


def worst_difference(first, second):
    worst, details = 0.0, {}
    for name in ("polarization", "population", "fluorescence"):
        for path in ("R1", "R2", "R3"):
            a, b = first.observables[name][path], second.observables[name][path]
            rel = float(np.max(np.abs(a - b)) / np.max(np.abs(a)))
            details[f"{name}/{path}"] = rel
            worst = max(worst, rel)
    return worst, details


results, runs = {}, {}
for backend in ("dense", "sparse_sector"):
    for strategy in ("pointwise", "factorized"):
        result, elapsed = run(backend, strategy)
        results[(backend, strategy)] = result
        runs[f"{backend}_{strategy}"] = {"time_s": elapsed, "resolvent_solves": result.diagnostics["resolvent_solves"],
                                         "propagations": result.diagnostics["propagations"]}
        print(f"{backend:14s} {strategy:10s} {elapsed:8.3f} s, {result.diagnostics['resolvent_solves']} solves", flush=True)

worst, details = worst_difference(results[("dense", "factorized")], results[("sparse_sector", "factorized")])
strategy_difference = {backend: worst_difference(results[(backend, "pointwise")], results[(backend, "factorized")])[0]
                       for backend in ("dense", "sparse_sector")}
report = {"grid": [41, 41], "t2": 10.0, "eta": 0.002, "runs": runs,
          "time_dense_s": runs["dense_factorized"]["time_s"], "time_sparse_s": runs["sparse_sector_factorized"]["time_s"],
          "relative_difference": details, "worst": worst, "strategy_difference": strategy_difference}
print(f"dense versus sparse: worst = {worst:.2e}; pointwise versus factorized: {strategy_difference}")
out = DATA
out.mkdir(parents=True, exist_ok=True)
(out / "example1_dense_sparse.json").write_text(json.dumps(report, indent=2))
