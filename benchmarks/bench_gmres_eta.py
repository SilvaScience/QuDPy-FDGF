"""GMRES iterations of the sparse backend versus the regularization eta.

Model: Frenkel chain (frenkel_model.py) with n = 6 sites, D = 22, with and without
dissipation. Task: third-order rephasing (GSB + SE + ESA), 3 x 3 grid, t2 = 10 eV^-1;
54 shifted Liouville solves per eta. The GMRES call of the sparse backend is wrapped
to count inner iterations (callback_type="pr_norm"); the code itself is not modified.
Default solver settings: rtol = 1e-10, restart = 20 (SciPy default).
The dense backend at the same eta gives the reference for the accuracy column.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import time
from pathlib import Path

import numpy as np
import qudpy_fdgf.backends.sparse_sector as ss
from qudpy_fdgf import SpectroscopySolver
from qudpy_fdgf.exceptions import ConvergenceError
from qudpy_fdgf.protocols import standard_nq_protocol
from frenkel_model import REPHASING, frenkel_model, hilbert_dimension

N_SITES = 6
ETAS = (0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005)
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

iterations = []
_original = ss.gmres


def counting(*args, **kwargs):
    count = [0]

    def callback(_):
        count[0] += 1
    out = _original(*args, callback=callback, callback_type="pr_norm", **kwargs)
    iterations.append(count[0])
    return out


ss.gmres = counting
protocol = standard_nq_protocol(order=1, n_interactions=3, nq_interval=1, detection_interval=3,
                                nq_axis="w1", detection_axis="w3")
axes = {"w1": np.linspace(-2.15, -1.85, 3), "w3": np.linspace(1.85, 2.15, 3)}

rows = []
for dissipative in (True, False):
    for eta in ETAS:
        ref_solver = SpectroscopySolver(backend="dense", eta=eta)
        ref_solver.feed_model(frenkel_model(N_SITES, dissipative=dissipative))
        ref = ref_solver.generate_spectrum(protocol, axes, pathways=REPHASING,
                                           fixed_coordinates={"t2": 10.0}).components["rephasing"]
        solver = SpectroscopySolver(backend="sparse_sector", eta=eta)
        solver.feed_model(frenkel_model(N_SITES, dissipative=dissipative))
        iterations.clear()
        start = time.perf_counter()
        row = {"dissipative": dissipative, "eta": eta, "D": hilbert_dimension(N_SITES)}
        try:
            res = solver.generate_spectrum(protocol, axes, pathways=REPHASING,
                                           fixed_coordinates={"t2": 10.0})
            S = res.components["rephasing"]
            row.update({
                "converged": True,
                "relative_difference_to_dense": float(np.max(abs(S - ref)) / np.max(abs(ref))),
            })
        except ConvergenceError as exc:
            row.update({"converged": False, "error": str(exc)[:200]})
        row.update({"time_s": time.perf_counter() - start, "solves": len(iterations),
                    "mean_iterations": float(np.mean(iterations)) if iterations else None,
                    "max_iterations": int(max(iterations)) if iterations else None})
        rows.append(row)
        print(row, flush=True)
        (OUT / "gmres_eta.json").write_text(json.dumps(rows, indent=2))
