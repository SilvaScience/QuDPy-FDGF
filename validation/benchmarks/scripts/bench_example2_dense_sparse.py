"""Example 2 at its published size: dense versus sparse backend, pointwise versus factorized evaluation.

XXZ chain (L = 6, N_max = 4, D = 57, canonical state at 4 K), rephasing GSB + SE + ESA,
t2 = 0, eta = 0.03 meV, on a 5 x 5 subgrid of the published 60 x 60 window. Runs:

    dense_factorized, dense_pointwise    dense backend (resolvent caching off, to keep its memory within
                                         the test machine; the pointwise run is slow, 150 inversions)
    sparse_plain_*                       sparse backend, preconditioner=None, diagonal_solve=False (plain GMRES)
    sparse_*                             sparse backend, defaults (this generator is diagonal: solved by division)

with strategy "factorized" and "pointwise". ``python bench_example2_dense_sparse.py skip-dense-pointwise``
leaves out the slow run.
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
from models import EX2, EX2_AXES, EX2_PATHWAYS, EX2_PROTOCOL, example2_context, example2_model

pick = np.linspace(0, 59, 5).round().astype(int)
AXES = {"omega_1q": EX2_AXES["omega_1q"][pick], "omega_emit": EX2_AXES["omega_emit"][pick]}
RUNS = {
    "dense": ("dense", {"cache_resolvents": False}),
    "sparse_plain": ("sparse_sector", {"krylov_tolerance": EX2["krylov_tolerance"], "preconditioner": None, "diagonal_solve": False}),
    "sparse": ("sparse_sector", {"krylov_tolerance": EX2["krylov_tolerance"]}),
}
skip_dense_pointwise = "skip-dense-pointwise" in sys.argv

results, runs = {}, {}
for name, (backend, options) in RUNS.items():
    for strategy in ("factorized", "pointwise"):
        if name == "dense" and strategy == "pointwise" and skip_dense_pointwise:
            continue
        model, _ = example2_model()
        solver = SpectroscopySolver(backend=backend, eta=EX2["eta"], **options)
        solver.feed_model(model, context=example2_context())
        start = time.perf_counter()
        res = solver.generate_spectrum(EX2_PROTOCOL, AXES, pathways=EX2_PATHWAYS,
                                       fixed_coordinates={"t2": EX2["t2"]}, strategy=strategy)
        elapsed = time.perf_counter() - start
        results[(name, strategy)] = res
        statistics = res.diagnostics["solver"]
        runs[f"{name}_{strategy}"] = {
            "time_s": elapsed, "resolvent_solves": res.diagnostics["resolvent_solves"],
            "resolvent_factorizations": res.diagnostics.get("resolvent_factorizations"),
            "iterations_mean": statistics.get("iterations_mean"), "preconditioner": statistics.get("preconditioner")}
        print(f"{name:13s} {strategy:10s} {elapsed:9.2f} s, {res.diagnostics['resolvent_solves']} solves", flush=True)


def difference(first, second):
    return {p.name: float(np.max(abs(first.pathways[p.name] - second.pathways[p.name]))
                          / np.max(abs(first.pathways[p.name]))) for p in EX2_PATHWAYS}


report = {"D": 57, "grid": [5, 5], "eta_meV": EX2["eta"], "runs": runs,
          "time_dense_s": runs["dense_factorized"]["time_s"], "time_sparse_s": runs["sparse_factorized"]["time_s"],
          "time_sparse_plain_s": runs["sparse_plain_factorized"]["time_s"],
          "relative_difference": difference(results[("dense", "factorized")], results[("sparse", "factorized")]),
          "relative_difference_plain": difference(results[("dense", "factorized")], results[("sparse_plain", "factorized")]),
          "strategy_difference": {name: max(difference(results[(name, "pointwise")], results[(name, "factorized")]).values())
                                  for name in RUNS if (name, "pointwise") in results}}
report["worst"] = max(max(report["relative_difference"].values()), max(report["relative_difference_plain"].values()))
print(json.dumps(report["strategy_difference"]), f"worst dense versus sparse = {report['worst']:.2e}")
out = DATA
out.mkdir(parents=True, exist_ok=True)
previous = out / "example2_dense_sparse.json"
if skip_dense_pointwise and "dense_pointwise" not in runs and previous.exists():
    old_runs = json.loads(previous.read_text()).get("runs", {})
    if "dense_pointwise" in old_runs:                 # the 15-minute dense pointwise run is kept from the previous result
        runs["dense_pointwise"] = old_runs["dense_pointwise"]
        report["strategy_difference_note"] = "dense_pointwise kept from an earlier run"
(out / "example2_dense_sparse.json").write_text(json.dumps(report, indent=2))
