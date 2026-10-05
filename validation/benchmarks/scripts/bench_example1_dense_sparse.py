"""Example 1: dense versus sparse backend for a nonlinear, dissipative calculation.

Same model (models.py), pathways, protocol, grid, and observables as Listing 1 of the manuscript
(polarization, action-detected population, integrated fluorescence), computed with
backend="dense" and backend="sparse_sector". Reports the maximum difference relative
to the maximum of each map, and the wall-clock time of each backend.
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


def run(backend):
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
                                      observables=observables)
    return result, time.perf_counter() - start


dense, t_dense = run("dense")
sparse, t_sparse = run("sparse_sector")
report = {"grid": [41, 41], "t2": 10.0, "eta": 0.002,
          "time_dense_s": t_dense, "time_sparse_s": t_sparse, "relative_difference": {}}
worst = 0.0
for name in ("polarization", "population", "fluorescence"):
    for path in ("R1", "R2", "R3"):
        a = dense.observables[name][path]
        b = sparse.observables[name][path]
        rel = float(np.max(np.abs(a - b)) / np.max(np.abs(a)))
        report["relative_difference"][f"{name}/{path}"] = rel
        worst = max(worst, rel)
        print(f"{name:13s} {path}: max|dense - sparse| / max|dense| = {rel:.2e}")
report["worst"] = worst
print(f"worst = {worst:.2e};  time dense = {t_dense:.1f} s, sparse = {t_sparse:.1f} s")
out = DATA
out.mkdir(parents=True, exist_ok=True)
(out / "example1_dense_sparse.json").write_text(json.dumps(report, indent=2))
