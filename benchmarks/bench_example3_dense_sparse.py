"""Example 3: dense versus sparse backend for the fifth-order 2Q response.

Model, UFSS pathways, and protocol copied from examples/chromophore_chi5_2q.ipynb
(N_max = 3, D = 10, eta = 8 meV); grid reduced to 31 x 31 over the same window.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import time
from pathlib import Path

import numpy as np
import ufss
from qudpy_fdgf import (EigenbasisKModel, SpectroscopySolver, standard_nq_protocol,
                        translate_ufss_diagrams)

omega_h, omega_l, coupling = 1.53, 1.58, 0.010
mu_h, mu_l = 1.0, 0.30
U_hh, U_ll, U_hl = -0.020, -0.020, 0.030
fixed_delays = {"t1": 2.0, "t3": 2.0, "t4": 2.0}
phase_discrimination = [(0, 1), (0, 1), (1, 0), (1, 0), (1, 0)]
omega_2q = np.linspace(-3.20, -2.99, 31)
omega_emit = np.linspace(1.42, 1.66, 31)
N_MAX, ETA = 3, 0.008


def operators(maximum_manifold):
    basis = tuple((n_h, total - n_h) for total in range(maximum_manifold + 1)
                  for n_h in range(total, -1, -1))
    index = {state: k for k, state in enumerate(basis)}
    d = len(basis)
    b_h = np.zeros((d, d), dtype=complex)
    b_l = np.zeros_like(b_h)
    for k, (n_h, n_l) in enumerate(basis):
        if n_h > 0:
            b_h[index[(n_h - 1, n_l)], k] = np.sqrt(n_h)
        if n_l > 0:
            b_l[index[(n_h, n_l - 1)], k] = np.sqrt(n_l)
    return basis, b_h, b_l


def calculate(backend):
    basis, b_h, b_l = operators(N_MAX)
    I = np.eye(len(basis), dtype=complex)
    n_h, n_l = b_h.conj().T @ b_h, b_l.conj().T @ b_l
    H = (omega_h * n_h + omega_l * n_l + coupling * (b_h.conj().T @ b_l + b_l.conj().T @ b_h)
         + 0.5 * U_hh * n_h @ (n_h - I) + 0.5 * U_ll * n_l @ (n_l - I) + U_hl * n_h @ n_l)
    J_plus = mu_h * b_h.conj().T + mu_l * b_l.conj().T
    J_minus = J_plus.conj().T
    mu = J_plus + J_minus
    model = EigenbasisKModel(H, mu, j_plus_array=J_plus, j_minus_array=J_minus,
                             detection_op_array=mu,
                             observable_op_arrays={"polarization": mu,
                                                   "polarization_h": mu_h * (b_h + b_h.conj().T),
                                                   "polarization_l": mu_l * (b_l + b_l.conj().T)})
    solver = SpectroscopySolver(backend=backend, eta=ETA, max_cache_entries=4096)
    solver.feed_model(model)
    gen = ufss.DiagramGenerator(detection_type="polarization")
    gen.set_phase_discrimination(phase_discrimination)
    gen.maximum_manifold = N_MAX
    gen.efield_times = [np.array([0.0, 0.0])] * 5
    pathways = translate_ufss_diagrams(gen.get_diagrams(np.arange(5, dtype=float)),
                                       component="chi5_2q")
    solver.set_pathways(pathways)
    protocol = standard_nq_protocol(order=2, nq_interval=2, detection_interval=5,
                                    n_interactions=5, nq_axis="omega_2q", detection_axis="omega_emit")
    start = time.perf_counter()
    result = solver.generate_nq_spectrum(2, protocol,
                                         axes={"omega_2q": omega_2q, "omega_emit": omega_emit},
                                         fixed_coordinates=fixed_delays, pathways=pathways,
                                         observables=("polarization_h", "polarization_l"))
    return result, time.perf_counter() - start, len(basis), len(pathways)


dense, t_d, D, n_path = calculate("dense")
sparse, t_s, _, _ = calculate("sparse_sector")
report = {"D": D, "pathways": n_path, "grid": [31, 31], "eta": ETA,
          "time_dense_s": t_d, "time_sparse_s": t_s, "relative_difference": {}}
for name in ("polarization_h", "polarization_l"):
    a = sum(dense.observables[name].values())
    b = sum(sparse.observables[name].values())
    rel = float(np.max(abs(a - b)) / np.max(abs(a)))
    report["relative_difference"][name] = rel
    print(f"{name}: max|dense - sparse| / max|dense| = {rel:.2e}")
print(f"D = {D}, {n_path} pathways; time dense = {t_d:.1f} s, sparse = {t_s:.1f} s")
out = Path(__file__).with_name("results")
out.mkdir(exist_ok=True)
(out / "example3_dense_sparse.json").write_text(json.dumps(report, indent=2))
