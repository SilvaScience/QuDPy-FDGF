"""Example 3: dense versus sparse backend for the fifth-order 2Q response.

Model, UFSS pathways, and protocol from models.py (as in examples/example3_chi5_2q.ipynb:
N_max = 3, D = 10, eta = 8 meV); grid reduced to 31 x 31 over the same window.
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
from pathlib import Path

import numpy as np
from qudpy_fdgf import SpectroscopySolver
from models import EX3, EX3_FIXED, EX3_PROTOCOL, example3_model, example3_pathways

ETA = 0.008
omega_2q = np.linspace(-3.20, -2.99, 31)
omega_emit = np.linspace(1.42, 1.66, 31)


def calculate(backend):
    model = example3_model()
    pathways = example3_pathways()
    solver = SpectroscopySolver(backend=backend, eta=ETA, max_cache_entries=4096)
    solver.feed_model(model)
    solver.set_pathways(pathways)
    start = time.perf_counter()
    result = solver.generate_nq_spectrum(2, EX3_PROTOCOL,
                                         axes={"omega_2q": omega_2q, "omega_emit": omega_emit},
                                         fixed_coordinates=EX3_FIXED, pathways=pathways,
                                         observables=("polarization_h", "polarization_l"))
    D = sum(solver.backend.layout.dimensions.values())
    return result, time.perf_counter() - start, D, len(pathways)


dense, t_d, D, n_path = calculate("dense")
sparse, t_s, _, _ = calculate("sparse_sector")
report = {"D": D, "pathways": n_path, "grid": [31, 31], "eta": ETA,
          "time_dense_s": t_d, "time_sparse_s": t_s, "relative_difference": {}}
for name in ("polarization_h", "polarization_l"):
    a = sum(dense.observables[name].values())
    b = sum(sparse.observables[name].values())
    rel = float(np.max(abs(a - b)) / np.max(abs(a)))
    report["relative_difference"][name] = rel
    print(f"{name}: max|dense - sparse| / max|dense| = {rel:.2e}")
print(f"D = {D}, {n_path} pathways; time dense = {t_d:.1f} s, sparse = {t_s:.1f} s")
out = DATA
out.mkdir(parents=True, exist_ok=True)
(out / "example3_dense_sparse.json").write_text(json.dumps(report, indent=2))
