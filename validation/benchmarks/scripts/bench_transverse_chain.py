"""XXZ ring of L spins in a static field with a transverse component: no conserved magnon number.

The Hamiltonian of Example 1 (J_xy = -0.30, J_z = -0.55 meV, g mu_B B_0 = 0.8724 meV along z) gets a transverse field
h_x sum_j S^x_j, which couples the magnon-number sectors N and N +- 1 exactly as the uniform probe does. The model has no
sector structure and no truncation in N: the full Hilbert space (D = 2^L) is diagonalized and given to EigenbasisKModel
as a single sector, which is what the adapters allow. The ground state is the initial state, the probe is the uniform
magnetization 2 sum_j S^x_j / sqrt(L), the three rephasing sequences GSB, SE and ESA are evaluated on a 9 x 9 grid of the
window of Example 1 with the sparse backend (eta = 0.03 meV).

Usage: python bench_transverse_chain.py L [mode [memory_limit_mb]]        mode = exact (default) | gmres
    exact  the solver as it is: the generator of the closed model is diagonal, every shifted system is a division
    gmres  diagonal_solve=False, preconditioner=None: plain GMRES, to show what the diagonal solve and the preconditioner save
Prints one JSON line (times in s, memory in MB: peak working set above the working set after the imports).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("BENCH_THREADS", "3"))
os.environ.setdefault("OPENBLAS_NUM_THREADS", os.environ["OMP_NUM_THREADS"])
os.environ.setdefault("MKL_NUM_THREADS", os.environ["OMP_NUM_THREADS"])
import json
import sys
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))                 # validation/ (models.py)
import numpy as np
import psutil
import qutip as qt

import models
from qudpy_fdgf import EigenbasisKModel, SpectroscopySolver, pathways_from_signature

warnings.simplefilter("ignore")
L = int(sys.argv[1])
MODE = sys.argv[2] if len(sys.argv) > 2 else "exact"
LIMIT = float(sys.argv[3]) if len(sys.argv) > 3 else None          # memory_limit_mb of generate_spectrum (default 1024)
TRANSVERSE = 0.30                                          # meV
GRID = 9
process = psutil.Process()
baseline = process.memory_info().wset


def peak_mb():
    return (process.memory_info().peak_wset - baseline) / 1e6


report = {"L": L, "D": 2**L, "mode": MODE, "memory_limit_mb": LIMIT, "threads": int(os.environ["OMP_NUM_THREADS"]), "transverse_field_meV": TRANSVERSE}

# ---------------------------------------------------------------- Hamiltonian (sparse, then dense)
start = time.perf_counter()
H, N_op, M = models.example2_chain(L)
sx = [qt.tensor([qt.sigmax() / 2 if k == j else qt.qeye(2) for k in range(L)]) for j in range(L)]
H = H + TRANSVERSE * sum(sx)
mu = (2 / np.sqrt(L)) * sum(sx)
H_dense, mu_dense = H.full(), mu.full()
report["build_s"] = time.perf_counter() - start
report["peak_after_build_MB"] = peak_mb()

# ---------------------------------------------------------------- model: full diagonalization
start = time.perf_counter()
model = EigenbasisKModel(H_dense, mu_dense)
report["model_s"] = time.perf_counter() - start
report["peak_after_model_MB"] = peak_mb()
del H_dense, mu_dense

# ---------------------------------------------------------------- solver and spectrum
options = {} if MODE == "exact" else {"diagonal_solve": False, "preconditioner": None}
solver = SpectroscopySolver(backend="sparse_sector", eta=models.EX2["eta"], krylov_tolerance=1e-10, check_stationarity="off",
                            **options)
start = time.perf_counter()
solver.feed_model(model)
report["feed_s"] = time.perf_counter() - start
pathways = tuple(models.EX2_PATHWAYS)
axes = {"omega_1q": np.linspace(-1.55, -0.55, GRID), "omega_emit": np.linspace(0.55, 1.55, GRID)}
start = time.perf_counter()
extra = {} if LIMIT is None else {"memory_limit_mb": LIMIT}
result = solver.generate_spectrum(models.EX2_PROTOCOL, axes, pathways=pathways, fixed_coordinates={"t2": 0.0}, **extra)
report["spectrum_s"] = time.perf_counter() - start
report["peak_total_MB"] = peak_mb()
statistics = result.diagnostics["solver"]
report["method"] = statistics.get("method")
report["solves"] = statistics.get("solves")
report["iterations_mean"] = statistics.get("iterations_mean")
report["iterations_max"] = statistics.get("iterations_max")
report["max_residual"] = result.diagnostics.get("maximum_resolvent_residual")
report["max_abs_signal"] = float(max(np.abs(v).max() for v in result.pathways.values()))
print(json.dumps(report))
