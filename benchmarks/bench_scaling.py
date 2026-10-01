"""Time and memory versus Hilbert dimension D, dense versus sparse backend.

Model: dissipative Frenkel chain (frenkel_model.py), n sites, D = 1 + n + n(n-1)/2.
Task: third-order rephasing (GSB + SE + ESA), 3 x 3 frequency grid, t2 = 10 eV^-1,
eta = 0.005 eV. Each (backend, n) runs in its own process; memory is the peak
working set of that process minus its working set after the imports.

Usage:  python bench_scaling.py            (driver, writes results/scaling.json)
        python bench_scaling.py worker BACKEND N
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
GRID = 3
ETA = 0.005


def worker(backend, n):
    import numpy as np
    import psutil
    import qudpy_fdgf.backends.sparse_sector as ss
    from qudpy_fdgf import SpectroscopySolver
    from qudpy_fdgf.protocols import standard_nq_protocol
    from frenkel_model import REPHASING, frenkel_model, hilbert_dimension

    iterations = []
    original = ss.gmres

    def counting(*args, **kwargs):
        count = [0]

        def callback(_):
            count[0] += 1
        out = original(*args, callback=callback, callback_type="pr_norm", **kwargs)
        iterations.append(count[0])
        return out
    ss.gmres = counting

    process = psutil.Process()
    base = process.memory_info().wset
    protocol = standard_nq_protocol(order=1, n_interactions=3, nq_interval=1,
                                    detection_interval=3, nq_axis="w1", detection_axis="w3")
    axes = {"w1": np.linspace(-2.15, -1.85, GRID), "w3": np.linspace(1.85, 2.15, GRID)}
    t0 = time.perf_counter()
    solver = SpectroscopySolver(backend=backend, eta=ETA)
    solver.feed_model(frenkel_model(n))
    t1 = time.perf_counter()
    result = solver.generate_spectrum(protocol, axes, pathways=REPHASING,
                                      fixed_coordinates={"t2": 10.0})
    t2 = time.perf_counter()
    peak = process.memory_info().peak_wset - base
    np.save(OUT / f"spectrum_{backend}_{n}.npy", result.components["rephasing"])
    print(json.dumps({
        "backend": backend, "n": n, "D": hilbert_dimension(n),
        "build_s": t1 - t0, "spectrum_s": t2 - t1, "peak_MB": peak / 1e6,
        "gmres_calls": len(iterations),
        "gmres_mean_iterations": (sum(iterations) / len(iterations)) if iterations else None,
        "gmres_max_iterations": max(iterations) if iterations else None,
    }))


def driver():
    import numpy as np
    OUT.mkdir(exist_ok=True)
    plan = [("dense", n) for n in range(2, 10)] + [("sparse_sector", n) for n in range(2, 13)]
    rows, skip = [], set()
    for backend, n in plan:
        if backend in skip:
            continue
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        proc = subprocess.run([sys.executable, __file__, "worker", backend, str(n)],
                              cwd=HERE, capture_output=True, text=True, timeout=3600, env=env)
        line = [l for l in proc.stdout.splitlines() if l.startswith("{")]
        if proc.returncode != 0 or not line:
            print(backend, n, "FAILED", proc.stderr[-400:])
            skip.add(backend)
            continue
        row = json.loads(line[-1])
        rows.append(row)
        print(row, flush=True)
        if row["build_s"] + row["spectrum_s"] > 900:
            skip.add(backend)
        (OUT / "scaling.json").write_text(json.dumps(rows, indent=2))
    # dense-sparse agreement where both exist
    for row in rows:
        if row["backend"] == "sparse_sector":
            f = OUT / f"spectrum_dense_{row['n']}.npy"
            if f.exists():
                a, b = np.load(f), np.load(OUT / f"spectrum_sparse_sector_{row['n']}.npy")
                row["relative_difference_to_dense"] = float(np.max(abs(a - b)) / np.max(abs(a)))
    (OUT / "scaling.json").write_text(json.dumps(rows, indent=2))
    print("done")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        worker(sys.argv[2], int(sys.argv[3]))
    else:
        driver()
