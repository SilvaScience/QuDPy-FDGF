"""Example 2 (two coupled dissipative modes): dense versus sparse backend, pointwise versus factorized evaluation.

Two problems, as in the article:

    open     third-order rephasing response R_GSB + R_SE + R_ESA of the Perspective model (D = 9, eta = 1e-5 eV), polarization
    closed   fifth-order 2Q response of the same modes without dissipation, N_max = 3 (D = 10, seven pathways, eta = 2 meV)

On a small grid (31 x 31 over the window of the example) the four combinations of backend and strategy are compared; on the
full grid of the example the factorized evaluation is run with both backends. The agreement is the largest deviation of the sum
over the pathways relative to its largest value. Results: results/data/modes_dense_sparse.json.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # single-thread BLAS, before NumPy is imported
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))   # validation/ (models.py)
import json
import time

import numpy as np
from qudpy_fdgf import SpectroscopySolver
from models import (DATA, EX4, EX4_AXES, EX4_CHI5_AXES, EX4_CHI5_ETA, EX4_CHI5_FIXED, EX4_CHI5_PROTOCOL, EX4_PATHWAYS,
                    EX4_PROTOCOL, example4_chi5_model, example4_chi5_pathways, example4_model)

SMALL = 31


def subgrid(axes, n):
    return {name: np.linspace(values[0], values[-1], n) for name, values in axes.items()}


PROBLEMS = {
    "open": dict(eta=EX4["eta"], model=example4_model, protocol=EX4_PROTOCOL, axes=EX4_AXES, fixed={"t2": EX4["t2"]},
                 pathways=lambda: EX4_PATHWAYS),
    "closed": dict(eta=EX4_CHI5_ETA, model=example4_chi5_model, protocol=EX4_CHI5_PROTOCOL, axes=EX4_CHI5_AXES,
                   fixed=EX4_CHI5_FIXED, pathways=example4_chi5_pathways),
}


def calculate(problem, backend, strategy, axes):
    solver = SpectroscopySolver(backend=backend, eta=problem["eta"], max_cache_entries=4096)
    solver.feed_model(problem["model"]())
    pathways = problem["pathways"]()
    start = time.perf_counter()
    result = solver.generate_spectrum(problem["protocol"], axes, pathways=pathways, fixed_coordinates=problem["fixed"],
                                      strategy=strategy, warn_null=False)
    elapsed = time.perf_counter() - start
    dimension = sum(solver.backend.layout.dimensions.values())
    return result, elapsed, dimension, len(pathways)


def total(result):
    return sum(result.pathways.values())


def difference(first, second):
    a, b = total(first), total(second)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(a)))


report = {}
for name, problem in PROBLEMS.items():
    small, results, runs, dimension, count = subgrid(problem["axes"], SMALL), {}, {}, None, None
    for backend in ("dense", "sparse_sector"):
        for strategy in ("pointwise", "factorized"):
            result, elapsed, dimension, count = calculate(problem, backend, strategy, small)
            results[(backend, strategy)] = result
            runs[f"{backend}_{strategy}_{SMALL}x{SMALL}"] = {"time_s": elapsed, "resolvent_solves": result.diagnostics["resolvent_solves"],
                                                          "propagations": result.diagnostics["propagations"]}
            print(f"{name:7s} {backend:14s} {strategy:10s} {SMALL}x{SMALL} {elapsed:8.3f} s, {result.diagnostics['resolvent_solves']} solves", flush=True)
        result, elapsed, _, _ = calculate(problem, backend, "factorized", problem["axes"])
        shape = "x".join(str(len(values)) for values in problem["axes"].values())
        runs[f"{backend}_factorized_{shape}"] = {"time_s": elapsed, "resolvent_solves": result.diagnostics["resolvent_solves"],
                                                 "propagations": result.diagnostics["propagations"]}
        results[(backend, "full")] = result
        print(f"{name:7s} {backend:14s} factorized {shape} {elapsed:8.3f} s, {result.diagnostics['resolvent_solves']} solves", flush=True)
    report[name] = {
        "D": dimension, "pathways": count, "grid": [SMALL, SMALL], "full_grid": [len(v) for v in problem["axes"].values()],
        "eta": problem["eta"], "runs": runs,
        "dense_versus_sparse": difference(results[("dense", "factorized")], results[("sparse_sector", "factorized")]),
        "dense_versus_sparse_full_grid": difference(results[("dense", "full")], results[("sparse_sector", "full")]),
        "strategy": {backend: difference(results[(backend, "pointwise")], results[(backend, "factorized")])
                     for backend in ("dense", "sparse_sector")},
    }
    print(name, "D =", dimension, {k: v for k, v in report[name].items() if k in ("dense_versus_sparse", "dense_versus_sparse_full_grid", "strategy")}, flush=True)
out = DATA
out.mkdir(parents=True, exist_ok=True)
(out / "modes_dense_sparse.json").write_text(json.dumps(report, indent=2))
