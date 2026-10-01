"""Example 2 at its published size: dense versus sparse backend.

XXZ chain (L = 6, N_max = 4, D = 57, canonical state at 4 K), rephasing GSB + SE + ESA,
t2 = 0, eta = 0.03 meV, on a 5 x 5 subgrid of the published 60 x 60 window. The dense
backend inverts a 3249 x 3249 matrix per frequency; resolvent caching is disabled to keep
its memory within the test machine.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import time
from pathlib import Path

import numpy as np
from qudpy_fdgf import SpectroscopySolver
from example_models import EX2, EX2_AXES, EX2_PATHWAYS, EX2_PROTOCOL, example2_context, example2_model

pick = np.linspace(0, 59, 5).round().astype(int)
AXES = {"omega_1q": EX2_AXES["omega_1q"][pick], "omega_emit": EX2_AXES["omega_emit"][pick]}
results, times = {}, {}
for backend, options in (("dense", {"cache_resolvents": False}),
                         ("sparse_sector", {"krylov_tolerance": EX2["krylov_tolerance"]})):
    model, _ = example2_model()
    solver = SpectroscopySolver(backend=backend, eta=EX2["eta"], **options)
    solver.feed_model(model, context=example2_context())
    start = time.perf_counter()
    res = solver.generate_spectrum(EX2_PROTOCOL, AXES, pathways=EX2_PATHWAYS,
                                   fixed_coordinates={"t2": EX2["t2"]})
    times[backend] = time.perf_counter() - start
    results[backend] = res
report = {"D": 57, "grid": [5, 5], "eta_meV": EX2["eta"], "time_dense_s": times["dense"],
          "time_sparse_s": times["sparse_sector"], "relative_difference": {}}
for name in ("GSB", "SE", "ESA"):
    a, b = results["dense"].pathways[name], results["sparse_sector"].pathways[name]
    report["relative_difference"][name] = float(np.max(abs(a - b)) / np.max(abs(a)))
report["worst"] = max(report["relative_difference"].values())
print(json.dumps(report, indent=1))
out = Path(__file__).with_name("results")
out.mkdir(exist_ok=True)
(out / "example2_dense_sparse.json").write_text(json.dumps(report, indent=2))
