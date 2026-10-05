"""Cost of one application of the Liouvillian: matrix products versus one call per column.

Example 2 model (XXZ chain, N_max = 4, no dissipative channel) at D = 57 (L = 6) and D = 163 (L = 8),
sparse sector backend, single-thread BLAS. The reference is the column-by-column action that the code
used before the vectorization (``tests/helpers.py``); both are checked to give the same matrix.
Then one GMRES resolvent of the preconditioned solver (2 x 2 grid, the three rephasing pathways of
Example 2, D = 57) shows what the gain means for a spectrum.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # single-thread BLAS, before NumPy is imported
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))               # validation/ (models.py)
sys.path.insert(0, str(_Path(__file__).resolve().parents[3] / "tests"))     # reference_generator_action
from models import DATA
import json
import time

import numpy as np
from qudpy_fdgf import SpectroscopySolver
from helpers import reference_generator_action
from models import EX2, EX2_AXES, EX2_PATHWAYS, EX2_PROTOCOL, example2_context, example2_model


def timed(function, repetitions):
    start = time.perf_counter()
    for _ in range(repetitions):
        function()
    return (time.perf_counter() - start) / repetitions


report = {"actions": {}}
for n_sites in (6, 8):
    model, _ = example2_model(n_sites=n_sites, max_magnons=4)
    solver = SpectroscopySolver(backend="sparse_sector", eta=EX2["eta"], krylov_tolerance=EX2["krylov_tolerance"])
    solver.feed_model(model, context=example2_context())
    generator = solver.backend.generator
    size = generator.dimension
    rho = np.random.default_rng(1).standard_normal((size, size)) + 1j * np.random.default_rng(2).standard_normal((size, size))
    vector = rho.reshape(-1, order="F")
    new = generator.matvec(vector).reshape((size, size), order="F")
    old = reference_generator_action(generator, rho)
    repetitions = 20 if size < 100 else 5
    t_new = timed(lambda: generator.matvec(vector), repetitions)
    t_old = timed(lambda: reference_generator_action(generator, rho), max(repetitions // 4, 2))
    report["actions"][f"D{size}"] = {
        "matvec_vectorized_ms": 1e3 * t_new, "matvec_column_loop_ms": 1e3 * t_old, "speedup": t_old / t_new,
        "relative_difference": float(np.max(abs(new - old)) / np.max(abs(old))),
    }

model, _ = example2_model()
solver = SpectroscopySolver(backend="sparse_sector", eta=EX2["eta"], krylov_tolerance=EX2["krylov_tolerance"])
solver.feed_model(model, context=example2_context())
pick = [20, 40]
axes = {"omega_1q": EX2_AXES["omega_1q"][pick], "omega_emit": EX2_AXES["omega_emit"][pick]}
start = time.perf_counter()
solver.generate_spectrum(EX2_PROTOCOL, axes, pathways=EX2_PATHWAYS, fixed_coordinates={"t2": EX2["t2"]}, warn_null=False)
report["spectrum_D57_2x2_s"] = time.perf_counter() - start
print(json.dumps(report, indent=1))
DATA.mkdir(parents=True, exist_ok=True)
(DATA / "generator_action.json").write_text(json.dumps(report, indent=2))
