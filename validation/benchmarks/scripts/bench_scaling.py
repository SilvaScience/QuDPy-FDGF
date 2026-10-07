"""Time and memory versus Hilbert dimension: Example 1 scaled in chain length.

The XXZ chain of Example 1 at its published settings (magnon cutoff N_max = 4, canonical
state at 4 K, eta = 0.03 meV, rephasing GSB + SE + ESA, t2 = 0) is enlarged from L = 4 to
L = 10 sites, D = sum_{N<=4} C(L, N) = 16, 31, 57, 99, 163, 256, 386. The task is a
9 x 9 subgrid of the published frequency window. Four configurations, each in its own process
per L (memory is the peak working set of that process minus its working set after the imports):

    dense             dense backend, factorized evaluation (L <= 6: 0.17 GB per Liouville matrix at D = 57)
    sparse_plain      sparse backend, preconditioner=None and diagonal_solve=False (plain GMRES), factorized (L <= 8)
    sparse            sparse backend with the defaults (diagonal_solve=True: this generator is diagonal, so the
                      shifted systems are solved by division; preconditioner="auto", i.e. diagonal for this
                      model in its eigenbasis), factorized evaluation
    sparse_pointwise  the same, with strategy="pointwise" (one propagation per grid point)

The iterations and solves come from result.diagnostics. Spectra are compared with the dense ones
where they exist and, beyond, with the plain sparse ones.

Usage:  python bench_scaling.py                    (driver: all configurations, writes results/data/scaling.json)
        python bench_scaling.py CONFIG             (driver for one configuration, merged into scaling.json)
        python bench_scaling.py worker CONFIG L    (one measurement, prints a JSON line)
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # single-thread BLAS, before NumPy is imported
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))   # validation/ (models.py)
from models import DATA
import json
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = DATA
GRID = 9
CONFIGS = {
    "dense": {"backend": "dense", "lengths": (4, 5, 6), "options": {"cache_resolvents": False}, "strategy": "auto"},
    "sparse_plain": {"backend": "sparse_sector", "lengths": (4, 5, 6, 7, 8), "options": {"preconditioner": None, "diagonal_solve": False},
                     "strategy": "auto"},
    "sparse": {"backend": "sparse_sector", "lengths": (4, 5, 6, 7, 8, 9, 10), "options": {}, "strategy": "auto"},
    "sparse_pointwise": {"backend": "sparse_sector", "lengths": (4, 5, 6, 7, 8, 9, 10), "options": {},
                         "strategy": "pointwise"},
}


def worker(config, n_sites):
    import numpy as np
    import psutil
    from qudpy_fdgf import SpectroscopySolver
    from models import (EX1, EX1_PATHWAYS, EX1_PROTOCOL, example1_context,
                        example1_dimension, example1_model)

    settings = CONFIGS[config]
    options = dict(settings["options"])
    if settings["backend"] == "sparse_sector":
        options["krylov_tolerance"] = EX1["krylov_tolerance"]
    process = psutil.Process()
    base = process.memory_info().wset
    axes = {"omega_1q": np.linspace(-1.55, -0.55, GRID), "omega_emit": np.linspace(0.55, 1.55, GRID)}
    t0 = time.perf_counter()
    model, _ = example1_model(n_sites=n_sites)
    solver = SpectroscopySolver(backend=settings["backend"], eta=EX1["eta"], **options)
    solver.feed_model(model, context=example1_context())
    t1 = time.perf_counter()
    result = solver.generate_spectrum(EX1_PROTOCOL, axes, pathways=EX1_PATHWAYS,
                                      fixed_coordinates={"t2": EX1["t2"]}, strategy=settings["strategy"])
    t2 = time.perf_counter()
    peak = process.memory_info().peak_wset - base
    np.save(OUT / f"scaling_{config}_{n_sites}.npy", result.components["rephasing"])
    diagnostics = result.diagnostics
    statistics = diagnostics["solver"]
    print(json.dumps({
        "config": config, "backend": settings["backend"], "L": n_sites,
        "D": example1_dimension(n_sites, EX1["max_magnons"]), "grid": [GRID, GRID],
        "strategy": diagnostics["strategy"], "preconditioner": statistics.get("preconditioner"),
        "build_s": t1 - t0, "spectrum_s": t2 - t1, "peak_MB": peak / 1e6,
        "resolvent_solves": diagnostics["resolvent_solves"],
        "resolvent_factorizations": diagnostics.get("resolvent_factorizations"),
        "iterations_mean": statistics.get("iterations_mean"), "iterations_max": statistics.get("iterations_max"),
        "relative_residual_max": statistics.get("relative_residual_max"),
    }))


def compare(rows):
    import numpy as np
    for row in rows:
        if row.get("failed"):
            continue
        reference = None
        for config in ("dense", "sparse_plain"):
            candidate = OUT / f"scaling_{config}_{row['L']}.npy"
            if candidate.exists() and config != row["config"]:
                reference = candidate
                row["reference"] = config
                break
        if reference is not None:
            a = np.load(reference)
            b = np.load(OUT / f"scaling_{row['config']}_{row['L']}.npy")
            row["relative_difference_to_reference"] = float(np.max(abs(a - b)) / np.max(abs(a)))


def driver(configs):
    OUT.mkdir(parents=True, exist_ok=True)
    result_file = OUT / "scaling.json"
    rows = [row for row in (json.loads(result_file.read_text()) if result_file.exists() else [])
            if row["config"] not in configs]
    for config in configs:
        for n_sites in CONFIGS[config]["lengths"]:
            env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
            proc = subprocess.run([sys.executable, __file__, "worker", config, str(n_sites)],
                                  cwd=HERE, capture_output=True, text=True, timeout=7200, env=env)
            line = [text for text in proc.stdout.splitlines() if text.startswith("{")]
            if proc.returncode != 0 or not line:
                row = {"config": config, "L": n_sites, "failed": True, "stderr": proc.stderr[-400:]}
            else:
                row = json.loads(line[-1])
            rows.append(row)
            print(json.dumps(row), flush=True)
            result_file.write_text(json.dumps(rows, indent=2))
    rows.sort(key=lambda row: (list(CONFIGS).index(row["config"]), row["L"]))
    compare(rows)
    result_file.write_text(json.dumps(rows, indent=2))
    print("done", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        worker(sys.argv[2], int(sys.argv[3]))
    else:
        driver(sys.argv[1:] or list(CONFIGS))
