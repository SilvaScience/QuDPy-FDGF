"""Direct frequency-domain evaluation versus time propagation + Fourier transform.

Example 1 model (open two-level system), third-order rephasing R1 + R2 at t2 = 10 eV^-1,
requested grid 41 x 41 (omega_1q in [-2.5, -1.5], omega_emit in [1.5, 2.5]) eV.

Reference: QuDPy-FDGF dense backend, both coherence intervals in the frequency domain.
Time-domain route: the same Liouville chain (same generator, interaction superoperators,
detection, and c_p, taken from the dense backend) is sampled on t1, t3 grids with step dt
up to T, using a one-step propagator exp(L dt), and transformed with the same kernel
exp[(i omega - eta) t] (trapezoidal weights) directly on the requested grid; the cost is
dominated by the N1 x N3 propagated samples, as for a zero-padded FFT.

For each linewidth scenario and accuracy target, the cheapest (dt, T) that reaches the
target (max error / max signal) is reported with its wall-clock time.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import time
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from qudpy_fdgf import EigenbasisKModel, FrequencyPathway, SpectroscopySolver
from qudpy_fdgf.protocols import standard_nq_protocol

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
ETA, T2 = 0.002, 10.0
W1 = np.linspace(-2.5, -1.5, 41)
W3 = np.linspace(1.5, 2.5, 41)
PATHWAYS = (FrequencyPathway(name="R1", interactions=("Bu", "Ku", "Bd"), component="rephasing"),
            FrequencyPathway(name="R2", interactions=("Bu", "Bd", "Ku"), component="rephasing"))
PROTOCOL = standard_nq_protocol(order=1, n_interactions=3, nq_interval=1, detection_interval=3,
                                nq_axis="w1", detection_axis="w3")

H = np.diag([0.0, 2.0]).astype(complex)
MU = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
L_RAD = np.array([[0, 1], [0, 0]], dtype=complex)
SZ = np.diag([-1.0, 1.0]).astype(complex)


def solver_for(gamma_1, gamma_phi):
    c_ops = []
    if gamma_1 > 0:
        c_ops.append((L_RAD, gamma_1))
    if gamma_phi > 0:
        c_ops.append((SZ, gamma_phi / 2))
    solver = SpectroscopySolver(backend="dense", eta=ETA)
    solver.feed_model(EigenbasisKModel(H, MU, c_ops_raw=tuple(c_ops)))
    return solver


def reference(solver):
    t = time.perf_counter()
    res = solver.generate_spectrum(PROTOCOL, {"w1": W1, "w3": W3}, pathways=PATHWAYS,
                                   fixed_coordinates={"t2": T2})
    return res.components["rephasing"], time.perf_counter() - t


def time_domain(solver, dt, T):
    """Sampled chain + discrete Fourier-Laplace transform on the requested grid."""
    t_start = time.perf_counter()
    be = solver.backend
    L = be._A_dense
    D = be.layout.total_dimension
    M = be._observable_matrix("polarization")
    detect = M.T.reshape(-1, order="F")           # Tr(M rho) = detect . vec(rho)
    step = expm(L * dt)
    U2 = expm(L * T2)
    n = int(round(T / dt)) + 1
    t = dt * np.arange(n)
    w = np.full(n, dt)
    w[0] = w[-1] = dt / 2                           # trapezoidal weights
    K1 = np.exp(np.outer(1j * W1 - ETA, t)) * w     # (n1, N)
    K3 = np.exp(np.outer(t, 1j * W3 - ETA)) * w[:, None]   # (N, n3)
    total = np.zeros((W1.size, W3.size), dtype=complex)
    for pathway in PATHWAYS:
        V = [be._interaction_superoperator(i) for i in pathway.interactions]
        x = V[0] @ be._rho_initial
        X = np.empty((D * D, n), dtype=complex)     # states along t1
        for k in range(n):
            X[:, k] = x
            x = step @ x
        Y = V[2] @ (U2 @ (V[1] @ X))                # after interactions 2, 3
        acc = np.zeros((W1.size, W3.size), dtype=complex)
        for l in range(n):                           # step along t3
            row = detect @ Y                         # S(t1_k, t3_l), k = 0..N-1
            acc += np.outer(K1 @ row, K3[l])
            Y = step @ Y
        total += pathway.response_prefactor * acc
    return total, time.perf_counter() - t_start, n


SCENARIOS = {   # name: (gamma_1, gamma_phi)
    "example1": (0.04, 0.03),
    "weak_damping": (0.004, 0.003),
    "closed": (0.0, 0.0),
}
TARGETS = (1e-2, 1e-3, 1e-4)
DTS = (0.8, 0.4, 0.2, 0.1)

report = {}
for name, (g1, gphi) in SCENARIOS.items():
    solver = solver_for(g1, gphi)
    ref, t_ref = reference(solver)
    width = g1 / 2 + gphi + ETA
    scale = np.max(np.abs(ref))
    rows = []
    for dt in DTS:
        for factor in (1.0, 1.5, 2.0):
            for target in TARGETS:
                T = factor * np.log(1.0 / target) / width
                n = int(round(T / dt)) + 1
                if n > 8000:
                    continue
                S, t_time, n = time_domain(solver, dt, T)
                err = float(np.max(np.abs(S - ref)) / scale)
                rows.append({"dt": dt, "T": T, "N_per_axis": n, "error": err, "time_s": t_time})
    best = {}
    for target in TARGETS:
        ok = [r for r in rows if r["error"] <= target]
        best[str(target)] = min(ok, key=lambda r: r["time_s"]) if ok else None
    report[name] = {"gamma_1": g1, "gamma_phi": gphi, "eta": ETA, "coherence_width": width,
                    "reference_time_s": t_ref, "best": best, "all": rows}
    print(f"\n{name}: Gamma_2 + eta = {width:.4f} eV; resolvent (41 x 41): {t_ref:.2f} s")
    for target, r in best.items():
        if r:
            print(f"  error <= {target}: dt = {r['dt']}, T = {r['T']:.0f}, N = {r['N_per_axis']} per axis, "
                  f"{r['N_per_axis']**2:.1e} samples, {r['time_s']:.2f} s (error {r['error']:.1e})")
        else:
            print(f"  error <= {target}: not reached within the scanned grid")
    (OUT / "time_fft.json").write_text(json.dumps(report, indent=2))
