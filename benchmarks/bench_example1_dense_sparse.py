"""Example 1: dense versus sparse backend for a nonlinear, dissipative calculation.

Same model, pathways, protocol, grid, and observables as Listing 1 of the manuscript
(polarization, action-detected population, integrated fluorescence), computed with
backend="dense" and backend="sparse_sector". Reports the maximum difference relative
to the maximum of each map, and the wall-clock time of each backend.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import time
from pathlib import Path

import numpy as np
from qudpy_fdgf import (EigenbasisKModel, FrequencyPathway, ObservableSpec,
                        SpectroscopySolver)
from qudpy_fdgf.protocols import standard_nq_protocol

H = np.diag([0.0, 2.0]).astype(complex)
mu = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
P_e = np.diag([0.0, 1.0]).astype(complex)
L_rad = np.array([[0, 1], [0, 0]], dtype=complex)
sigma_z = np.diag([-1.0, 1.0]).astype(complex)
gamma_1, gamma_phi = 0.04, 0.03

R1 = FrequencyPathway(name="R1", interactions=("Bu", "Ku", "Bd"),
                      component="rephasing", detection="polarization")
R2 = FrequencyPathway(name="R2", interactions=("Bu", "Bd", "Ku"),
                      component="rephasing", detection="polarization")
protocol = standard_nq_protocol(order=1, n_interactions=3, nq_interval=1,
                                detection_interval=3, nq_axis="omega_1q",
                                detection_axis="omega_emit")
axes = {"omega_1q": np.linspace(-2.5, -1.5, 11),
        "omega_emit": np.linspace(1.5, 2.5, 11)}


def run(backend):
    model = EigenbasisKModel(
        H, mu, c_ops_raw=((L_rad, gamma_1), (sigma_z, gamma_phi / 2)),
        observable_op_arrays={"polarization": mu, "excited_population": P_e})
    solver = SpectroscopySolver(backend=backend, eta=0.002)
    solver.feed_model(model)
    radiative = solver.jump_channel_names()[0]
    observables = {
        "polarization": "polarization",
        "population": ObservableSpec.action(
            "population", fourth_interaction="Bu", operator="excited_population"),
        "fluorescence": ObservableSpec.mean_jump(
            "fluorescence", radiative, time_window=(0.0, 5 / gamma_1),
            fourth_interaction="Bu"),
    }
    start = time.perf_counter()
    result = solver.generate_spectrum(protocol, axes, pathways=(R1, R2),
                                      fixed_coordinates={"t2": 10.0},
                                      observables=observables)
    return result, time.perf_counter() - start


dense, t_dense = run("dense")
sparse, t_sparse = run("sparse_sector")
report = {"grid": [11, 11], "t2": 10.0, "eta": 0.002,
          "time_dense_s": t_dense, "time_sparse_s": t_sparse, "relative_difference": {}}
worst = 0.0
for name in ("polarization", "population", "fluorescence"):
    for path in ("R1", "R2"):
        a = dense.observables[name][path]
        b = sparse.observables[name][path]
        rel = float(np.max(np.abs(a - b)) / np.max(np.abs(a)))
        report["relative_difference"][f"{name}/{path}"] = rel
        worst = max(worst, rel)
        print(f"{name:13s} {path}: max|dense - sparse| / max|dense| = {rel:.2e}")
report["worst"] = worst
print(f"worst = {worst:.2e};  time dense = {t_dense:.1f} s, sparse = {t_sparse:.1f} s")
out = Path(__file__).with_name("results")
out.mkdir(exist_ok=True)
(out / "example1_dense_sparse.json").write_text(json.dumps(report, indent=2))
