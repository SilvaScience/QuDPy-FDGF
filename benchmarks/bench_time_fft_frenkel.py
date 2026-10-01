"""Frequency domain versus time sampling + transform for a larger model.

Dissipative Frenkel chain, n = 6 (D = 22, Liouville dimension 484), rephasing
GSB + SE + ESA at t2 = 10 eV^-1, 41 x 41 grid, eta = 5 meV, dense backend.
Same time-domain route as bench_time_fft.py, for accuracy targets 1e-2 and 1e-3.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import time
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from qudpy_fdgf import SpectroscopySolver
from qudpy_fdgf.protocols import standard_nq_protocol
from frenkel_model import REPHASING, frenkel_model, hilbert_dimension

import sys
CLOSED = "closed" in sys.argv          # no collapse channels: linewidth set by eta alone
OUT = Path(__file__).resolve().parent / "results"
N_SITES, ETA, T2 = 6, 0.005, 10.0
W1 = np.linspace(-2.2, -1.8, 41)
W3 = np.linspace(1.8, 2.2, 41)
PROTOCOL = standard_nq_protocol(order=1, n_interactions=3, nq_interval=1, detection_interval=3,
                                nq_axis="w1", detection_axis="w3")

solver = SpectroscopySolver(backend="dense", eta=ETA, max_cache_entries=4096)
solver.feed_model(frenkel_model(N_SITES, dissipative=not CLOSED))
t = time.perf_counter()
ref = solver.generate_spectrum(PROTOCOL, {"w1": W1, "w3": W3}, pathways=REPHASING,
                               fixed_coordinates={"t2": T2}).components["rephasing"]
t_ref = time.perf_counter() - t
scale = np.max(np.abs(ref))
print(f"resolvent (dense, 41 x 41, D = {hilbert_dimension(N_SITES)}): {t_ref:.1f} s", flush=True)


def time_domain(dt, T):
    t_start = time.perf_counter()
    be = solver.backend
    D = be.layout.total_dimension
    detect = be._observable_matrix("polarization").T.reshape(-1, order="F")
    step = expm(be._A_dense * dt)
    U2 = expm(be._A_dense * T2)
    n = int(round(T / dt)) + 1
    tt = dt * np.arange(n)
    w = np.full(n, dt)
    w[0] = w[-1] = dt / 2
    K1 = np.exp(np.outer(1j * W1 - ETA, tt)) * w
    K3 = np.exp(np.outer(tt, 1j * W3 - ETA)) * w[:, None]
    total = np.zeros((W1.size, W3.size), dtype=complex)
    for pathway in REPHASING:
        V = [be._interaction_superoperator(i) for i in pathway.interactions]
        x = V[0] @ be._rho_initial
        X = np.empty((D * D, n), dtype=complex)
        for k in range(n):
            X[:, k] = x
            x = step @ x
        Y = V[2] @ (U2 @ (V[1] @ X))
        acc = np.zeros_like(total)
        for l in range(n):
            acc += np.outer(K1 @ (detect @ Y), K3[l])
            Y = step @ Y
        total += pathway.response_prefactor * acc
    return total, time.perf_counter() - t_start, n


# linewidth of the slowest optical coherence ~ gamma_phi + gamma_1/2 + eta (lower bound)
width = ETA if CLOSED else 0.01 + 0.0025 + ETA
rows = []
for target in ((1e-2,) if CLOSED else (1e-2, 1e-3)):
    best = None
    for dt in ((0.8,) if CLOSED else (0.8, 0.4)):
        for factor in ((1.0, 1.5) if CLOSED else (1.0, 1.5)):
            T = factor * np.log(1 / target) / width
            S, t_time, n = time_domain(dt, T)
            err = float(np.max(np.abs(S - ref)) / scale)
            row = {"target": target, "dt": dt, "T": T, "N_per_axis": n, "error": err, "time_s": t_time}
            rows.append(row)
            print(row, flush=True)
            if err <= target and (best is None or t_time < best["time_s"]):
                best = row
    print(f"target {target}: best {best}", flush=True)
(OUT / ("time_fft_frenkel_closed.json" if CLOSED else "time_fft_frenkel.json")).write_text(json.dumps(
    {"D": hilbert_dimension(N_SITES), "eta": ETA, "reference_time_s": t_ref, "rows": rows}, indent=2))
