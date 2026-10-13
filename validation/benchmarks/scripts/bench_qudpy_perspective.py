"""Original QuDPy (public System.coherence2d, QuTiP mesolve) versus QuDPy-FDGF on the two coupled modes of Example 2.

Same output for both codes: the maps of the individual Liouville diagrams on a grid of frequencies,
(omega_1, omega_3) in [-1.35, -0.90] x [0.90, 1.35] eV,

    coherent  R1, R2, R3 (UFSS rephasing diagrams, detection operator mu, no radiative channels), and
    action    Q3a, Q4a, Q2b, Q8b (UFSS fluorescence diagrams, detection operator N_exc after a delay of 5 eV^-1,
              radiative channels included),

the diagrams, the model, and the delays being those of the script of the Perspective (Paiva-Ortega et al. 2026):
t2 = 10 eV^-1, Reso = 1.5 / pi samples per eV^-1, system transformed to the eigenbasis, ground state as initial state.
The public coherence2d handles four interactions, so the action diagrams need no modification of QuDPy.

QuDPy: the time signal S(t1, t3) of each diagram on N = int(T Reso) samples per axis, transformed to the grid with the
kernel exp[(i omega - eta) t] and trapezoid weights (the discrete transform at the grid frequencies; on the FFT grid it is
the ifft2 of the Perspective up to the end-point weights). QuDPy-FDGF: direct evaluation at the same frequencies
(dense backend, eta = 1e-5 as in compare_perspective.py). The error of QuDPy is measured against that evaluation, after
the constant factor of the diagram convention (a power of i, checked) between the raw QuDPy signal and the Liouville-space
signal of QuDPy-FDGF; for the prefixes of the window it is also reported (the signal of a shorter window is the prefix of the
signal of the longer one, at the same sampling step).

Usage: python bench_qudpy_perspective.py MODE GRID KIND
    MODE  fdgf | qudpy             (qudpy needs the reference maps written by fdgf)
    GRID  perspective | resolved   (FFT grid of the Perspective: T = 801 eV^-1, 7.8 meV; resolved: T = 1500 eV^-1, 2 meV)
    KIND  coherent | action
Prints one JSON line. Single-thread BLAS; the memory is the peak working set above the working set before the model is built.
The reference maps and the propagated signal of each diagram are kept in validation/results/cache/qudpy_perspective (git-ignored),
so that an interrupted run resumes at the next diagram (the time is the sum of the durations of the diagrams).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # single-thread BLAS, before NumPy is imported
import contextlib
import io
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))                 # validation/ (models.py)
sys.path.insert(0, str(HERE.parents[1] / "analysis"))    # compare_perspective.py
import numpy as np
import psutil
import qutip as qt
import models
from compare_perspective import detector_model, ETA, T2
from qudpy_fdgf import ObservableSpec, SpectroscopySolver

# Original QuDPy (github.com/SilvaScience/QuDPy): QUDPY_REPO, an installed package, or a clone next to this repository.
for _candidate in (os.environ.get("QUDPY_REPO"), HERE.parents[3] / "QuDPy"):
    if _candidate and Path(_candidate, "qudpy", "Classes.py").exists():
        sys.path.insert(0, str(_candidate))
        break
from qudpy.Classes import System
import ufss

RESO = 1.5 / np.pi
BAND = (0.90, 1.35)
GRIDS = {"perspective": {"T": 801.0}, "resolved": {"T": 1500.0, "step": 0.002}}
CACHE = Path(os.environ.get("QUDPY_PERSPECTIVE_CACHE", HERE.parents[1] / "results" / "cache" / "qudpy_perspective"))
LABELS = {"coherent": ("R1", "R2", "R3"), "action": ("Q3a", "Q4a", "Q2b", "Q8b")}
PREFIXES = (300.0, 500.0, 801.0, 1000.0, 1250.0, 1500.0)


def frequency_grid(grid):
    """Positive frequencies w of the grid: omega_3 = w, omega_1 = -w."""
    if grid == "perspective":
        n = int(GRIDS[grid]["T"] * RESO)
        dt = GRIDS[grid]["T"] / (n - 1)
        bins = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(n, dt))
        return bins[(bins >= BAND[0]) & (bins <= BAND[1])]
    return np.arange(BAND[0], BAND[1] + 1e-9, GRIDS[grid]["step"])


def run_fdgf(grid, kind):
    w = frequency_grid(grid)
    axes = {"omega_1q": -w, "omega_emit": w}
    if kind == "coherent":
        model, observables = detector_model({"k_rad": 0.0}), {"polarization": "polarization"}
    else:
        model, observables = detector_model({}), {k: ObservableSpec.action(k, fourth_interaction=k, operator="N_delayed")
                                                  for k in ("Bu", "Kd")}
    solver = SpectroscopySolver(backend="dense", eta=ETA)
    solver.feed_model(model)
    proc = psutil.Process()
    base = proc.memory_info().wset
    start = time.perf_counter()
    maps = []
    if kind == "coherent":
        result = solver.generate_spectrum(models.EX2_PROTOCOL, axes, pathways=models.EX2_PATHWAYS,
                                          fixed_coordinates={"t2": T2}, observables=observables)
        maps = [result.observables["polarization"][p] for p in ("R_SE", "R_GSB", "R_ESA")]
    else:                                                  # Q3a, Q4a, Q2b with Bu; Q8b with Kd, only the four maps asked for
        bu = solver.generate_spectrum(models.EX2_PROTOCOL, axes, pathways=models.EX2_PATHWAYS, fixed_coordinates={"t2": T2},
                                      observables={"Bu": observables["Bu"]})
        kd = solver.generate_spectrum(models.EX2_PROTOCOL, axes, pathways=models.EX2_PATHWAYS[2:], fixed_coordinates={"t2": T2},
                                      observables={"Kd": observables["Kd"]})
        maps = [bu.observables["Bu"][p] for p in ("R_SE", "R_GSB", "R_ESA")] + [kd.observables["Kd"]["R_ESA"]]
    elapsed = time.perf_counter() - start
    peak = (proc.memory_info().peak_wset - base) / 1e6
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(CACHE / f"reference_{grid}_{kind}.npy", np.asarray(maps))
    return {"mode": "fdgf", "grid": grid, "kind": kind, "points": [w.size, w.size], "step_meV": 1e3 * float(np.median(np.diff(w))),
            "time_s": elapsed, "peak_MB": peak, "maps": len(maps)}


def quantum_system(kind):
    """The Perspective's system: eigenbasis, annihilation operator mu_- = a + b, detector mu (coherent) or N_exc (action)."""
    operators = models.example2_operators(**({"k_rad": 0.0} if kind == "coherent" else {}))
    H = operators["H"]
    levels = 3
    a = qt.tensor(qt.destroy(levels), qt.qeye(levels))
    b = qt.tensor(qt.qeye(levels), qt.destroy(levels))
    _, vectors = H.eigenstates()
    transform = lambda op: op.transform(vectors)
    c_ops = [transform(np.sqrt(rate) * op) for op, rate in operators["channels"] if rate > 0]
    detector = operators["detectors"]["polarization"] if kind == "coherent" else a.dag() * a + b.dag() * b
    ground = qt.ket2dm(qt.tensor(qt.basis(levels, 0), qt.basis(levels, 0)))
    with contextlib.redirect_stdout(io.StringIO()):
        return System(H=transform(H), a=transform(a + b), u=transform(detector), c_ops=c_ops, rho=ground, diagonalize=False)


def diagrams(kind):
    generator = ufss.DiagramGenerator
    if kind == "coherent":
        d = generator()
        d.efield_times = [np.array([-1, 1])] * 4
        d.set_phase_discrimination([(0, 1), (1, 0), (1, 0)])
        r3, r1, r2 = d.get_diagrams([0, 100, 200, 300])
        return [r1, r2, r3]
    d = generator(detection_type="fluorescence")
    d.efield_times = [np.array([-1, 1])] * 4
    d.set_phase_discrimination([(0, 1), (1, 0), (1, 0), (0, 1)])
    q8b, q2b, q3a, q4a = d.get_diagrams([0, 100, 200, 300])
    return [q3a, q4a, q2b, q8b]


def trapezoid_weights(n, dt):
    weights = np.full(n, dt)
    weights[0] = weights[-1] = dt / 2
    return weights


def transform(signal, t, w):
    """Discrete transform at the frequencies of the grid: kernel exp[(i omega - eta) t], trapezoid weights."""
    weights = trapezoid_weights(len(t), t[1] - t[0])
    k1 = np.exp(np.outer(-1j * w - ETA, t)) * weights                     # omega_1 = -w
    k3 = np.exp(np.outer(t, 1j * w - ETA)) * weights[:, None]            # omega_3 = +w
    return k1 @ signal @ k3


def nearest_power_of_i(reference, raw):
    """Constant c in {1, i, -1, -i} minimizing |c * raw - reference|, with the fitted complex factor for the record."""
    fitted = np.vdot(raw, reference) / np.vdot(raw, raw)
    return 1j ** int(np.round(np.angle(fitted) / (np.pi / 2))), fitted


def run_qudpy(grid, kind):
    CACHE.mkdir(parents=True, exist_ok=True)
    reference = np.load(CACHE / f"reference_{grid}_{kind}.npy")
    w = frequency_grid(grid)
    T = GRIDS[grid]["T"]
    system = quantum_system(kind)
    delays = [T, T2, T] + ([5.0] if kind == "action" else [])
    listed = diagrams(kind)
    proc = psutil.Process()
    raw, durations, peaks = [], [], []
    for index, diagram in enumerate(listed):
        cached = CACHE / f"raw_{grid}_{kind}_{index}.npz"
        if cached.exists():                       # resumed run: the propagation of this diagram was done and kept
            stored = np.load(cached)
            signal, t1 = stored["signal"], stored["t"]
            elapsed, peak_one = float(stored["time_s"]), float(stored["peak_MB"])
        else:
            base = proc.memory_info().wset
            begin = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                t1, t3, signal = system.coherence2d(time_delays=delays, diagram=diagram, scan_id=[0, 2], r=RESO)
            elapsed = time.perf_counter() - begin
            peak_one = (proc.memory_info().peak_wset - base) / 1e6
            np.savez(cached, signal=np.asarray(signal), t=np.asarray(t1), time_s=elapsed, peak_MB=peak_one)
        raw.append(np.asarray(signal))
        durations.append(elapsed)
        peaks.append(peak_one)
    propagation = float(sum(durations))           # the propagations of the diagrams, one after the other
    peak = float(max(peaks))                      # the largest peak working set of a diagram
    start = time.perf_counter() - propagation     # so that transform_time below is the time spent after the propagation
    n = len(t1)
    scale = np.max(np.abs(reference))
    factors, deviations, errors, prefix_errors = [], [], [], {}
    for label, signal, ref in zip(LABELS[kind], raw, reference):
        c, fitted = nearest_power_of_i(ref, transform(signal, t1, w))
        factors.append(c)
        deviations.append(float(abs(abs(fitted) - 1)))
        errors.append(float(np.max(np.abs(c * transform(signal, t1, w) - ref)) / np.max(np.abs(ref))))
        for prefix in PREFIXES:
            m = int(round(prefix / (t1[1] - t1[0]))) + 1
            if m <= n:
                err = np.max(np.abs(c * transform(signal[:m, :m], t1[:m], w) - ref)) / np.max(np.abs(ref))
                prefix_errors.setdefault(f"{prefix:g}", {})[label] = float(err)
    transform_time = time.perf_counter() - start - propagation
    return {"mode": "qudpy", "grid": grid, "kind": kind, "tight": False, "T": T, "N_per_axis": n, "dt": float(t1[1] - t1[0]),
            "points": [w.size, w.size], "time_s": propagation, "transform_s": transform_time, "peak_MB": peak,
            "error": float(max(errors)), "error_per_diagram": dict(zip(LABELS[kind], errors)),
            "convention_factor": [str(c) for c in factors], "modulus_deviation": float(max(deviations)),
            "error_by_window": {k: float(max(v.values())) for k, v in prefix_errors.items()}}


if __name__ == "__main__":
    mode, grid, kind = sys.argv[1:4]
    report = run_fdgf(grid, kind) if mode == "fdgf" else run_qudpy(grid, kind)
    print(json.dumps(report))
