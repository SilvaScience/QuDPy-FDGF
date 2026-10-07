"""Cost versus the number of grid points per axis N: pointwise (about 2 N^2 solves per pathway) against factorized (about 2 N).

Four problems of the article, each on N x N grids over its published window:

    quickstart open two-level system of the listing, R1 + R2 + R3, polarization (sparse backend, defaults)
    example1   XXZ chain L = 6 (D = 57), GSB + SE + ESA, sparse backend, preconditioner="auto"
               (diagonal) and preconditioner=None (plain GMRES, N <= 9 pointwise)
    example2   two coupled dissipative modes (D = 9), third order, R_GSB + R_SE + R_ESA, sparse backend with the defaults
    example2_closed   the same modes without dissipation, fifth-order 2Q response, D = 10, seven pathways, defaults

Each row: wall-clock time, resolvent solves and propagations from result.diagnostics, and for the
factorized evaluation the number of stored Liouville vectors. The log-log slopes of time against N are
printed at the end. Results: results/data/grid_size.json.
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
from models import (EX1, EX1_AXES, EX1_PATHWAYS, EX1_PROTOCOL, EX2, EX2_AXES, EX2_PATHWAYS, EX2_PROTOCOL, EX4, EX4_AXES,
                    EX4_CHI5_AXES, EX4_CHI5_ETA, EX4_CHI5_FIXED, EX4_CHI5_PROTOCOL, EX4_PATHWAYS, EX4_PROTOCOL,
                    example1_model, example2_context, example2_model, example4_chi5_model, example4_chi5_pathways,
                    example4_model)


def subgrid(axes, n):
    """Axes of the published window -> n points per axis over the same window."""
    return {name: np.linspace(values[0], values[-1], n) for name, values in axes.items()}


def problems():
    yield ("quickstart", "sparse", [5, 11, 21, 41, 81],
           lambda: (SpectroscopySolver(backend="sparse_sector", eta=EX1["eta"]), example1_model(), None),
           lambda solver, n, strategy: solver.generate_spectrum(
               EX1_PROTOCOL, subgrid(EX1_AXES, n), pathways=EX1_PATHWAYS, fixed_coordinates={"t2": EX1["t2"]},
               strategy=strategy, warn_null=False), {"pointwise": 81, "factorized": 81})
    for label, options, limits in (("sparse", {}, {"pointwise": 33, "factorized": 65}),
                                   ("sparse_plain", {"preconditioner": None, "diagonal_solve": False}, {"pointwise": 5, "factorized": 33})):
        yield ("example1", label, [3, 5, 9, 17, 33, 65],
               lambda options=options: (SpectroscopySolver(backend="sparse_sector", eta=EX2["eta"],
                                                           krylov_tolerance=EX2["krylov_tolerance"], **options),
                                        example2_model()[0], example2_context()),
               lambda solver, n, strategy: solver.generate_spectrum(
                   EX2_PROTOCOL, subgrid(EX2_AXES, n), pathways=EX2_PATHWAYS, fixed_coordinates={"t2": EX2["t2"]},
                   strategy=strategy, warn_null=False), limits)
    yield ("example2", "sparse", [11, 31, 61, 151],
           lambda: (SpectroscopySolver(backend="sparse_sector", eta=EX4["eta"]), example4_model(), None),
           lambda solver, n, strategy: solver.generate_spectrum(
               EX4_PROTOCOL, subgrid(EX4_AXES, n), pathways=EX4_PATHWAYS, fixed_coordinates={"t2": EX4["t2"]},
               strategy=strategy, warn_null=False), {"pointwise": 61, "factorized": 151})
    pathways = example4_chi5_pathways()
    yield ("example2_closed", "sparse", [11, 31, 61, 151],
           lambda: (SpectroscopySolver(backend="sparse_sector", eta=EX4_CHI5_ETA), example4_chi5_model(), None),
           lambda solver, n, strategy: solver.generate_spectrum(
               EX4_CHI5_PROTOCOL, subgrid(EX4_CHI5_AXES, n), pathways=pathways, fixed_coordinates=EX4_CHI5_FIXED,
               strategy=strategy, warn_null=False), {"pointwise": 61, "factorized": 151})


rows = []
for name, label, sizes, build, evaluate, limits in problems():
    solver, model, context = build()
    solver.feed_model(model, context=context) if context is not None else solver.feed_model(model)
    for strategy in ("pointwise", "factorized"):
        for n in sizes:
            if n > limits[strategy]:
                continue
            start = time.perf_counter()
            result = evaluate(solver, n, strategy)
            elapsed = time.perf_counter() - start
            diagnostics = result.diagnostics
            row = {"example": name, "backend": label, "strategy": strategy, "N": n, "time_s": elapsed,
                   "resolvent_solves": diagnostics["resolvent_solves"], "propagations": diagnostics["propagations"],
                   "stored_vectors": diagnostics.get("factorization", {}).get("stored_vectors")}
            rows.append(row)
            print(json.dumps(row), flush=True)

slopes = {}
for name, label in sorted({(r["example"], r["backend"]) for r in rows}):
    for strategy in ("pointwise", "factorized"):
        points = [(r["N"], r["time_s"], r["resolvent_solves"]) for r in rows
                  if r["example"] == name and r["backend"] == label and r["strategy"] == strategy and r["N"] >= 9]
        if len(points) >= 2:
            n, t, s = (np.array(column, dtype=float) for column in zip(*points))
            slopes[f"{name}/{label}/{strategy}"] = {"time": float(np.polyfit(np.log(n), np.log(t), 1)[0]),
                                                    "solves": float(np.polyfit(np.log(n), np.log(s), 1)[0])}
print(json.dumps(slopes, indent=1))
out = DATA
out.mkdir(parents=True, exist_ok=True)
(out / "grid_size.json").write_text(json.dumps({"rows": rows, "log_log_slopes_for_N_at_least_9": slopes}, indent=2))
