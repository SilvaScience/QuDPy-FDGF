"""GMRES iterations of the sparse backend versus eta, for Example 2 at its published size.

XXZ chain (L = 6, N_max = 4, D = 57, canonical state at 4 K), rephasing GSB + SE + ESA,
t2 = 0, 3 x 3 subgrid of the published window, eta from 0.3 to 0.003 meV (production:
0.03 meV). The generator is closed, so eta alone keeps the shifted systems away from
singularity. The GMRES call of the sparse backend is wrapped to count inner iterations;
the solver code is not modified. The dense backend at the same eta is the reference; its
spectra are saved and reused by later runs.

Usage:  python bench_gmres_eta.py            (plain GMRES, writes results/gmres_eta.json)
        python bench_gmres_eta.py diagonal   (diagonal preconditioner, results/gmres_eta_diagonal.json)
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # single-thread BLAS, before NumPy is imported
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))   # validation/ (models.py)
from models import DATA
import json
import sys
import time
from pathlib import Path

import numpy as np
import qudpy_fdgf.backends.sparse_sector as ss
from qudpy_fdgf import SpectroscopySolver
from qudpy_fdgf.exceptions import ConvergenceError
from models import EX2, EX2_PATHWAYS, EX2_PROTOCOL, example2_context, example2_model

ETAS = (0.3, 0.1, 0.03, 0.01, 0.003)
PRECONDITIONER = sys.argv[1] if len(sys.argv) > 1 else None
RESULT = "gmres_eta.json" if PRECONDITIONER is None else f"gmres_eta_{PRECONDITIONER}.json"
OUT = DATA
OUT.mkdir(parents=True, exist_ok=True)
AXES = {"omega_1q": np.linspace(-1.55, -0.55, 3), "omega_emit": np.linspace(0.55, 1.55, 3)}

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
rows = []
for eta in ETAS:
    model, _ = example2_model()
    reference_file = OUT / f"gmres_eta_dense_{eta}.npy"
    if reference_file.exists():
        ref = np.load(reference_file)
    else:
        dense = SpectroscopySolver(backend="dense", eta=eta, cache_resolvents=False)   # memory
        dense.feed_model(model, context=example2_context())
        ref = dense.generate_spectrum(EX2_PROTOCOL, AXES, pathways=EX2_PATHWAYS,
                                      fixed_coordinates={"t2": EX2["t2"]}).components["rephasing"]
        del dense
        np.save(reference_file, ref)
    sparse = SpectroscopySolver(backend="sparse_sector", eta=eta, krylov_tolerance=EX2["krylov_tolerance"],
                                preconditioner=PRECONDITIONER)
    sparse.feed_model(model, context=example2_context())
    iterations.clear()
    start = time.perf_counter()
    row = {"eta_meV": eta, "D": 57, "preconditioner": PRECONDITIONER}
    try:
        S = sparse.generate_spectrum(EX2_PROTOCOL, AXES, pathways=EX2_PATHWAYS,
                                     fixed_coordinates={"t2": EX2["t2"]}).components["rephasing"]
        row.update({"converged": True,
                    "relative_difference_to_dense": float(np.max(abs(S - ref)) / np.max(abs(ref)))})
    except ConvergenceError as exc:
        row.update({"converged": False, "error": str(exc)[:200]})
    row.update({"time_s": time.perf_counter() - start, "solves": len(iterations),
                "mean_iterations": float(np.mean(iterations)) if iterations else None,
                "max_iterations": int(max(iterations)) if iterations else None})
    rows.append(row)
    print(json.dumps(row), flush=True)
    (OUT / RESULT).write_text(json.dumps(rows, indent=2))
