"""Time and memory versus Hilbert dimension: Example 2 scaled in chain length.

The XXZ chain of Example 2 at its published settings (magnon cutoff N_max = 4, canonical
state at 4 K, eta = 0.03 meV, rephasing GSB + SE + ESA, t2 = 0) is enlarged from L = 4 to
L = 8 sites, D = sum_{N<=4} C(L, N) = 16, 31, 57, 99, 163. The task is a 3 x 3 subgrid of
the published frequency window. Each (backend, L) runs in its own process; memory is the
peak working set of that process minus its working set after the imports. The dense
backend is run while its memory fits the test machine.

Usage:  python bench_scaling.py            (driver, writes results/scaling.json)
        python bench_scaling.py worker BACKEND L
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
LENGTHS = (4, 5, 6, 7, 8)
DENSE_MAX_L = 6                  # D = 57: 0.17 GB per Liouville matrix


def worker(backend, n_sites):
    import numpy as np
    import psutil
    import qudpy_fdgf.backends.sparse_sector as ss
    from qudpy_fdgf import SpectroscopySolver
    from example_models import (EX2, EX2_PATHWAYS, EX2_PROTOCOL, example2_context,
                                example2_dimension, example2_model)

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
    axes = {"omega_1q": np.linspace(-1.55, -0.55, 3), "omega_emit": np.linspace(0.55, 1.55, 3)}
    options = ({"krylov_tolerance": EX2["krylov_tolerance"]} if backend == "sparse_sector"
               else {"cache_resolvents": False})     # keeps the dense memory at ~3 Liouville matrices
    t0 = time.perf_counter()
    model, _ = example2_model(n_sites=n_sites)
    solver = SpectroscopySolver(backend=backend, eta=EX2["eta"], **options)
    solver.feed_model(model, context=example2_context())
    t1 = time.perf_counter()
    result = solver.generate_spectrum(EX2_PROTOCOL, axes, pathways=EX2_PATHWAYS,
                                      fixed_coordinates={"t2": EX2["t2"]})
    t2 = time.perf_counter()
    peak = process.memory_info().peak_wset - base
    np.save(OUT / f"scaling_{backend}_{n_sites}.npy", result.components["rephasing"])
    print(json.dumps({
        "backend": backend, "L": n_sites, "D": example2_dimension(n_sites, EX2["max_magnons"]),
        "build_s": t1 - t0, "spectrum_s": t2 - t1, "peak_MB": peak / 1e6,
        "gmres_calls": len(iterations),
        "gmres_mean_iterations": (sum(iterations) / len(iterations)) if iterations else None,
        "gmres_max_iterations": max(iterations) if iterations else None,
    }))


def driver():
    import numpy as np
    OUT.mkdir(exist_ok=True)
    plan = [("dense", L) for L in LENGTHS if L <= DENSE_MAX_L] + [("sparse_sector", L) for L in LENGTHS]
    rows = []
    for backend, n_sites in plan:
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        proc = subprocess.run([sys.executable, __file__, "worker", backend, str(n_sites)],
                              cwd=HERE, capture_output=True, text=True, timeout=7200, env=env)
        line = [l for l in proc.stdout.splitlines() if l.startswith("{")]
        if proc.returncode != 0 or not line:
            row = {"backend": backend, "L": n_sites, "failed": True, "stderr": proc.stderr[-400:]}
        else:
            row = json.loads(line[-1])
        rows.append(row)
        print(json.dumps(row), flush=True)
        (OUT / "scaling.json").write_text(json.dumps(rows, indent=2))
    for row in rows:
        if row["backend"] == "sparse_sector" and not row.get("failed"):
            f = OUT / f"scaling_dense_{row['L']}.npy"
            if f.exists():
                a = np.load(f)
                b = np.load(OUT / f"scaling_sparse_sector_{row['L']}.npy")
                row["relative_difference_to_dense"] = float(np.max(abs(a - b)) / np.max(abs(a)))
    (OUT / "scaling.json").write_text(json.dumps(rows, indent=2))
    print("done", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        worker(sys.argv[2], int(sys.argv[3]))
    else:
        driver()
