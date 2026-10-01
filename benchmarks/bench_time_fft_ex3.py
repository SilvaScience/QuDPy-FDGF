"""Example 3 (fifth-order 2Q response): direct frequency domain versus time sampling + transform.

Published calculation: two-mode bosonic-exciton model (D = 10), seven UFSS pathways,
t1 = t3 = t4 = 2 eV^-1, 151 x 151 grid, eta = 8 meV and 2 meV (Fig. 6), no dissipation.

Reference: QuDPy-FDGF dense backend, both resolved intervals in the frequency domain.
Time route: the same Liouville chain (dense-backend generator, interaction superoperators,
detection, c_p) sampled on the 2Q and emission intervals with a one-step propagator and
transformed with exp[(i omega - eta) t] and trapezoidal weights on the 151 x 151 grid.
The cheapest (dt, T) reaching each target error is searched adaptively.
Original QuDPy (v1.1.0, System.coherence2d): a pilot run gives the time and memory per
stored state; the full run is attempted only if its estimated memory fits (QUDPY_MAX_GB).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import psutil
from scipy.linalg import expm
from qudpy_fdgf import SpectroscopySolver
from example_models import (EX3_AXES, EX3_FIXED, EX3_PROTOCOL, example3_model,
                            example3_operators, example3_pathways)

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
W2, W5 = EX3_AXES["omega_2q"], EX3_AXES["omega_emit"]
PATHWAYS = example3_pathways()
MAX_GB = float(os.environ.get("QUDPY_MAX_GB", "1.5"))


def reference(eta):
    solver = SpectroscopySolver(backend="dense", eta=eta, max_cache_entries=4096)
    solver.feed_model(example3_model())
    start = time.perf_counter()
    res = solver.generate_nq_spectrum(2, EX3_PROTOCOL, axes=EX3_AXES, fixed_coordinates=EX3_FIXED,
                                      pathways=PATHWAYS)
    return solver, sum(res.pathways.values()), time.perf_counter() - start


def trapezoid_kernel(omegas, eta, t, dt):
    w = np.full(t.size, dt)
    w[0] = w[-1] = dt / 2
    return np.exp(np.outer(1j * omegas - eta, t)) * w


def time_route(solver, eta, dt, T):
    start = time.perf_counter()
    be = solver.backend
    L = be._A_dense
    detect = be._observable_matrix("polarization").T.reshape(-1, order="F")
    step = expm(L * dt)
    U = {name: expm(L * value) for name, value in EX3_FIXED.items()}
    n = int(round(T / dt)) + 1
    t = dt * np.arange(n)
    K2 = trapezoid_kernel(W2, eta, t, dt)                  # (151, N)
    K5 = trapezoid_kernel(W5, eta, t, dt).T                # (N, 151)
    total = np.zeros((W2.size, W5.size), dtype=complex)
    for pathway in PATHWAYS:
        V = [be._interaction_superoperator(i) for i in pathway.interactions]
        x = V[1] @ (U["t1"] @ (V[0] @ be._rho_initial))
        X = np.empty((x.size, n), dtype=complex)
        for k in range(n):                                  # 2Q interval
            X[:, k] = x
            x = step @ x
        Y = V[4] @ (U["t4"] @ (V[3] @ (U["t3"] @ (V[2] @ X))))
        acc = np.zeros_like(total)
        for l in range(n):                                  # emission interval
            acc += np.outer(K2 @ (detect @ Y), K5[l])
            Y = step @ Y
        total += pathway.response_prefactor * acc
    return total, time.perf_counter() - start, n


def qudpy_run(dt, T):
    """Original QuDPy coherence2d for the seven pathways; returns (signal, time, peak MB, N)."""
    import qutip
    repo = os.environ.get("QUDPY_REPO") or str(Path(__file__).resolve().parents[2] / "QuDPy")
    sys.path.insert(0, repo)
    import qudpy.Classes as classes
    from qudpy.Classes import System
    if not getattr(classes.mesolve, "_qutip5_shim", False):
        # QuDPy v1.1.0 passes e_ops positionally after c_ops (QuTiP 4 signature), which
        # QuTiP 5 rejects; forward it as a keyword. The QuDPy code itself is not modified.
        _mesolve = qutip.mesolve

        def shim(H, rho0, tlist, c_ops=None, e_ops=None, *args, **kwargs):
            return _mesolve(H, rho0, tlist, c_ops=c_ops, e_ops=e_ops, *args, **kwargs)
        shim._qutip5_shim = True
        classes.mesolve = shim
    basis, H, J_plus, _, _ = example3_operators()
    mu = J_plus + J_plus.conj().T
    rho0 = np.zeros((len(basis), len(basis)), dtype=complex)
    rho0[0, 0] = 1.0
    proc = psutil.Process()
    base = proc.memory_info().wset
    with contextlib.redirect_stdout(io.StringIO()):
        system = System(n=len(basis) - 1, H=qutip.Qobj(H), rho=qutip.Qobj(rho0),
                        a=qutip.Qobj(J_plus.conj().T), u=qutip.Qobj(mu), c_ops=[])
    start = time.perf_counter()
    total, n = None, None
    for pathway in PATHWAYS:
        diagram = [(i.label, k) for k, i in enumerate(pathway.interactions)]
        delays = [EX3_FIXED["t1"], T, EX3_FIXED["t3"], EX3_FIXED["t4"], T]
        with contextlib.redirect_stdout(io.StringIO()):
            t2q, temit, dipole = system.coherence2d(time_delays=delays, diagram=diagram,
                                                     scan_id=[1, 4], r=1.0 / dt)
        n = len(t2q)
        h2, h5 = t2q[1] - t2q[0], temit[1] - temit[0]
        S = pathway.response_prefactor * (trapezoid_kernel(W2, ETA, t2q, h2) @ np.asarray(dipole)
                                          @ trapezoid_kernel(W5, ETA, temit, h5).T)
        total = S if total is None else total + S
    return total, time.perf_counter() - start, (proc.memory_info().peak_wset - base) / 1e6, n


report = {}
for ETA in (0.008, 0.002):
    solver, ref, t_ref = reference(ETA)
    scale = np.max(np.abs(ref))
    entry = {"eta_eV": ETA, "reference_time_s": t_ref, "time_route": {}, "qudpy": {}}
    print(f"\neta = {1000 * ETA:.0f} meV: resolvent on 151 x 151 in {t_ref:.1f} s", flush=True)
    for target in (1e-2, 1e-3):
        best = None
        for dt, factor in ((0.8, 1.0), (0.8, 1.5), (0.4, 1.5), (0.4, 2.0)):
            T = factor * np.log(1.0 / target) / ETA
            S, t_time, n = time_route(solver, ETA, dt, T)
            err = float(np.max(np.abs(S - ref)) / scale)
            print(f"  time route target {target:.0e}: dt = {dt}, T = {T:.0f}, N = {n}, "
                  f"error = {err:.1e}, {t_time:.1f} s", flush=True)
            if err <= target:
                best = {"dt": dt, "T": T, "N_per_axis": n, "error": err, "time_s": t_time}
                break
        entry["time_route"][str(target)] = best
        if best is None:
            continue
        # original QuDPy: pilot run, then the full run if its memory estimate fits
        pilot_T = 60.0
        _, t_pilot, mb_pilot, n_pilot = qudpy_run(best["dt"], pilot_T)
        scale_n = (best["N_per_axis"] / n_pilot) ** 2
        estimate = {"pilot_N": n_pilot, "pilot_time_s": t_pilot, "pilot_peak_MB": mb_pilot,
                    "estimated_time_s": t_pilot * scale_n, "estimated_peak_GB": mb_pilot * scale_n / 1e3}
        print(f"  QuDPy pilot N = {n_pilot}: {t_pilot:.1f} s, {mb_pilot:.0f} MB -> estimate for N = "
              f"{best['N_per_axis']}: {estimate['estimated_time_s']:.0f} s, "
              f"{estimate['estimated_peak_GB']:.1f} GB", flush=True)
        if estimate["estimated_peak_GB"] <= MAX_GB:
            S, t_q, mb_q, n_q = qudpy_run(best["dt"], best["T"])
            estimate.update({"ran": True, "time_s": t_q, "peak_MB": mb_q, "N_per_axis": n_q,
                             "error": float(np.max(np.abs(S - ref)) / scale)})
            print(f"  QuDPy full run: {t_q:.0f} s, {mb_q:.0f} MB, error {estimate['error']:.1e}", flush=True)
        else:
            estimate["ran"] = False
        entry["qudpy"][str(target)] = estimate
    report[f"{1000 * ETA:.0f}meV"] = entry
    (OUT / "time_fft_ex3.json").write_text(json.dumps(report, indent=2))
print("done", flush=True)
