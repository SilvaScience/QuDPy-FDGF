"""Cost of a *resolved* map: direct evaluation, sum over states, and a factorized time-domain route.

A map is resolved when its grid spacing is a fraction of the line width, so the number of points per axis
is set by the lines, not chosen freely: N_omega = window / (FWHM / 4), with FWHM = 2 sqrt(3) (Gamma_2 + eta) of the
modulus of a Lorentzian pole. Four routes give the same map, all taking advantage of the factorization
S_ij = <G_j | mid F_i> (N + N propagations instead of N x N):

    direct_dense, direct_sparse   resolvent solves of the library (dense LU; sparse GMRES with the defaults)
    sum_over_states               eigendecomposition of the dense generator, resolvents by division
    time_matrix, time_fft         the same chain sampled in time with the one-step propagator exp(L dt) up to a
                                  length T, then a transform with exp[(i omega - eta) t] and trapezoidal weights:
                                  matrix product on the window (N_omega N_t^2) or zero-padded 2D FFT (N_fft^2 log)

For each accuracy target the cheapest (dt, T, transform) that reaches it is searched; dt goes up to 3.2 eV^-1,
beyond the Nyquist step of the carrier (pi / 2.5 eV): the signal lies in a band of 1 eV, so sampling at the bandwidth
folds the band onto itself without overlap (bandpass sampling), and the error measured includes any aliasing. Problems: the open two-level
system of Example 1 at three line widths (R1 + R2, t2 = 10 eV^-1, window of 1 eV per axis, eta = 2 meV), and the
fifth-order 2Q response of Example 3 (D = 10, seven pathways, 151 x 151 grid, eta = 8 and 2 meV).
Results: results/data/resolved_map.json.   Usage: python bench_resolved_map.py [example1 | example3]
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
import warnings

import numpy as np
from scipy.linalg import expm
from qudpy_fdgf import EigenbasisKModel, FrequencyPathway, SpectroscopySolver
from qudpy_fdgf.protocols import standard_nq_protocol

warnings.simplefilter("ignore")
OUT = DATA / "resolved_map.json"
TARGETS = (1e-2, 1e-3, 1e-4)


# ------------------------------------------------------------------------------------------ routes
def best_of(function, repeats=3):
    """Result of the fastest of ``repeats`` calls and its time."""
    best = None
    for _ in range(repeats):
        start = time.perf_counter()
        value = function()
        elapsed = time.perf_counter() - start
        if best is None or elapsed < best[1]:
            best = (value, elapsed)
    return best


def pieces(backend, pathway, protocol, scanned, fixed):
    """Chain of a pathway -> (matrices before the first scanned interval, matrices between the two)."""
    before, between, stage = [], [], 0
    for interaction, interval in zip(pathway.interactions, protocol.intervals):
        target = before if stage == 0 else between
        target.append(backend._interaction_superoperator(interaction))
        if interval.name in scanned:
            stage += 1
        else:
            target.append(backend._time_propagator(fixed[interval.name]))
    assert stage == 2 and protocol.intervals[-1].name in scanned, "two scanned intervals, the last one ends the chain"
    return before, between


def apply(matrices, vectors):
    for matrix in matrices:
        vectors = matrix @ vectors
    return vectors


def detection_vector(backend):
    return np.conj(backend._observable_matrix("polarization").T.reshape(-1, order="F"))


def direct(kind, eta, model, protocol, axes, pathways, fixed):
    solver = SpectroscopySolver(backend=kind, eta=eta)
    solver.feed_model(model)

    def run():
        result = solver.generate_spectrum(protocol, axes, pathways=pathways, fixed_coordinates=fixed, warn_null=False)
        return sum(result.pathways.values()), result.diagnostics["resolvent_solves"]

    (value, solves), elapsed = best_of(run)
    return value, elapsed, solves, solver


def sum_over_states(solver, eta, protocol, scanned, fixed, pathways, grid1, grid2):
    backend = solver.backend

    def run():
        eigenvalues, vectors = np.linalg.eig(backend._A_dense)
        inverse = np.linalg.inv(vectors)
        d = vectors.conj().T @ detection_vector(backend)
        total = np.zeros((grid1.size, grid2.size), dtype=complex)
        for pathway in pathways:
            before, between = pieces(backend, pathway, protocol, scanned, fixed)
            c = inverse @ apply(before, backend._rho_initial)
            forward = apply(between, vectors @ (c[:, None] / (eta - 1j * grid1[None, :] - eigenvalues[:, None])))
            backward = inverse.conj().T @ (d[:, None] / np.conj(eta - 1j * grid2[None, :] - eigenvalues[:, None]))
            total += pathway.response_prefactor * (backward.conj().T @ forward).T
        return total

    return best_of(run)


def kernel(omegas, eta, t, dt):
    weights = np.full(t.size, dt)
    weights[0] = weights[-1] = dt / 2
    return np.exp(np.outer(1j * omegas - eta, t)) * weights


def time_route(solver, eta, protocol, scanned, fixed, pathways, dt, T, grid1, grid2, transform):
    """Sampled factorized chain, then the transform on the requested grids (matrix) or on the FFT grid (fft)."""
    start = time.perf_counter()
    backend = solver.backend
    step = expm(backend._A_dense * dt)
    adjoint_step = step.conj().T
    d = detection_vector(backend)
    n = int(round(T / dt)) + 1
    t = dt * np.arange(n)
    samples = np.zeros((n, n), dtype=complex)            # S(t_k, t_l), the first scanned interval along rows
    for pathway in pathways:
        before, between = pieces(backend, pathway, protocol, scanned, fixed)
        x = apply(before, backend._rho_initial)
        right = np.empty((x.size, n), dtype=complex)
        for k in range(n):
            right[:, k] = x
            x = step @ x
        right = apply(between, right)
        left = np.empty_like(right)
        y = d
        for l in range(n):
            left[:, l] = y
            y = adjoint_step @ y
        samples += pathway.response_prefactor * (right.T @ left.conj())
    if transform == "matrix":
        value = kernel(grid1, eta, t, dt) @ samples @ kernel(grid2, eta, t, dt).T
    else:
        size = int(round(2 * np.pi / (dt * (grid1[1] - grid1[0]))))
        weights = np.full(n, dt)
        weights[0] = weights[-1] = dt / 2
        factor = weights * np.exp(-eta * t)
        samples *= factor[:, None]
        samples *= factor[None, :]
        padded = np.zeros((size, size), dtype=complex)
        padded[:n, :n] = samples
        del samples
        spectrum = np.fft.ifft2(padded)
        spectrum *= size * size
        del padded
        index1 = np.round(grid1 * size * dt / (2 * np.pi)).astype(int) % size
        index2 = np.round(grid2 * size * dt / (2 * np.pi)).astype(int) % size
        value = spectrum[np.ix_(index1, index2)]
    return value, time.perf_counter() - start, n


def fft_grid(low, high, spacing, dt):
    """FFT bins omega_m = 2 pi m / (N dt) inside [low, high], N the smallest size whose spacing is <= spacing."""
    size = int(np.ceil(2 * np.pi / (dt * spacing)))
    step = 2 * np.pi / (size * dt)
    return step * np.arange(int(np.ceil(low / step)), int(np.floor(high / step)) + 1), size


def search(solver, eta, model, protocol, scanned, fixed, pathways, windows, spacing, width, dts, factors, cap=9000):
    """Cheapest time route (dt, T, transform) for each target; the direct map is recomputed on each FFT grid."""
    rows = []
    for dt in dts:
        grid1, size = fft_grid(*windows[0], spacing, dt)
        grid2, _ = fft_grid(*windows[1], spacing, dt)
        axes = {scanned[0]: grid1, scanned[1]: grid2}
        reference, _, _, _ = direct("dense", eta, model, protocol, axes, pathways, fixed)
        scale = np.max(np.abs(reference))
        for factor in factors:
            for target in TARGETS:
                T = factor * np.log(1.0 / target) / width
                n = int(round(T / dt)) + 1
                if n > cap:
                    continue
                for transform in ("matrix", "fft"):
                    if transform == "fft" and n > size:         # the padded transform must hold the samples
                        continue
                    value, elapsed, n = time_route(solver, eta, protocol, scanned, fixed, pathways, dt, T, grid1, grid2, transform)
                    rows.append({"dt": dt, "T": T, "N_t": n, "N_fft": size, "N_grid": [grid1.size, grid2.size],
                                 "transform": transform, "target": target,
                                 "error": float(np.max(np.abs(value - reference)) / scale), "time_s": elapsed})
    best = {}
    for target in TARGETS:
        ok = [r for r in rows if r["error"] <= target]
        best[str(target)] = min(ok, key=lambda r: r["time_s"]) if ok else None
    return best, rows


def report_routes(name, eta, model, protocol, scanned, fixed, pathways, windows, spacing, width, dts, factors):
    grid1 = np.linspace(*windows[0], int(round((windows[0][1] - windows[0][0]) / spacing)) + 1)
    grid2 = np.linspace(*windows[1], int(round((windows[1][1] - windows[1][0]) / spacing)) + 1)
    axes = {scanned[0]: grid1, scanned[1]: grid2}
    dense_map, t_dense, solves, solver = direct("dense", eta, model, protocol, axes, pathways, fixed)
    sparse_map, t_sparse, solves_sparse, _ = direct("sparse_sector", eta, model, protocol, axes, pathways, fixed)
    states_map, t_states = sum_over_states(solver, eta, protocol, scanned, fixed, pathways, grid1, grid2)
    scale = np.max(np.abs(dense_map))
    entry = {"eta": eta, "width": width, "N_omega": [grid1.size, grid2.size], "spacing": spacing,
             "direct_dense": {"time_s": t_dense, "resolvent_solves": solves},
             "direct_sparse": {"time_s": t_sparse, "resolvent_solves": solves_sparse,
                               "difference_to_dense": float(np.max(np.abs(sparse_map - dense_map)) / scale)},
             "sum_over_states": {"time_s": t_states, "difference_to_dense": float(np.max(np.abs(states_map - dense_map)) / scale)}}
    print(f"{name}: N = {grid1.size} x {grid2.size}; direct dense {t_dense * 1e3:.1f} ms ({solves} solves), sparse {t_sparse * 1e3:.1f} ms, "
          f"sum over states {t_states * 1e3:.1f} ms (differences {entry['sum_over_states']['difference_to_dense']:.1e})", flush=True)
    entry["time_route"], entry["time_route_all"] = search(solver, eta, model, protocol, scanned, fixed, pathways,
                                                          windows, spacing, width, dts, factors)
    for target, row in entry["time_route"].items():
        if row:
            print(f"   target {target}: dt {row['dt']}, T {row['T']:.0f}, N_t {row['N_t']}, {row['transform']}, "
                  f"{row['time_s'] * 1e3:.1f} ms (error {row['error']:.1e})", flush=True)
        else:
            print(f"   target {target}: not reached", flush=True)
    return entry


# ----------------------------------------------------------------------------------------- problems
def example1():
    H = np.diag([0.0, 2.0]).astype(complex)
    MU = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    L_RAD = np.array([[0, 1], [0, 0]], dtype=complex)
    SZ = np.diag([-1.0, 1.0]).astype(complex)
    eta, fixed = 0.002, {"t2": 10.0}
    pathways = (FrequencyPathway(name="R1", interactions=("Bu", "Ku", "Bd"), component="rephasing"),
                FrequencyPathway(name="R2", interactions=("Bu", "Bd", "Ku"), component="rephasing"))
    protocol = standard_nq_protocol(order=1, n_interactions=3, nq_interval=1, detection_interval=3,
                                    nq_axis="w1", detection_axis="w3")
    report = {}
    for name, (gamma_1, gamma_phi) in {"example1": (0.04, 0.03), "weak_damping": (0.004, 0.003), "closed": (0.0, 0.0)}.items():
        channels = ([(L_RAD, gamma_1)] if gamma_1 > 0 else []) + ([(SZ, gamma_phi / 2)] if gamma_phi > 0 else [])
        model = EigenbasisKModel(H, MU, c_ops_raw=tuple(channels))
        width = gamma_1 / 2 + gamma_phi + eta
        spacing = 2 * np.sqrt(3) * width / 4
        report[name] = report_routes(name, eta, model, protocol, ("w1", "w3"), fixed, pathways,
                                     ((-2.5, -1.5), (1.5, 2.5)), spacing, width, (3.2, 1.6, 0.8, 0.4, 0.2), (1.0, 1.5, 2.0))
        report[name].update({"gamma_1": gamma_1, "gamma_phi": gamma_phi})
    return report


def example3():
    from models import EX3_AXES, EX3_FIXED, EX3_PROTOCOL, example3_model, example3_pathways
    pathways = example3_pathways()
    report = {}
    for eta in (0.008, 0.002):
        spacing = EX3_AXES["omega_2q"][1] - EX3_AXES["omega_2q"][0]
        windows = ((EX3_AXES["omega_2q"][0], EX3_AXES["omega_2q"][-1]), (EX3_AXES["omega_emit"][0], EX3_AXES["omega_emit"][-1]))
        report[f"{1000 * eta:.0f}meV"] = report_routes(f"example3 eta = {1000 * eta:.0f} meV", eta, example3_model(), EX3_PROTOCOL,
                                                      ("omega_2q", "omega_emit"), EX3_FIXED, pathways, windows, spacing,
                                                      eta, (1.6, 0.8, 0.4), (1.0, 1.5))
    return report


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    DATA.mkdir(parents=True, exist_ok=True)
    report = json.loads(OUT.read_text()) if OUT.exists() else {}
    if which in ("all", "example1"):
        report["example1"] = example1()
        OUT.write_text(json.dumps(report, indent=2))
    if which in ("all", "example3"):
        report["example3"] = example3()
        OUT.write_text(json.dumps(report, indent=2))
    print("done", flush=True)
