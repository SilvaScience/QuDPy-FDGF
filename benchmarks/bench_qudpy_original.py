"""Time-domain calculation with the original QuDPy (v1.1.0, System.coherence2d, QuTiP mesolve)
versus the direct frequency-domain evaluation of QuDPy-FDGF, at equal accuracy.

The time-domain signal S(t1, t2, t3) of each rephasing pathway is computed by QuDPy exactly as
published: interactions applied with the lowering operator a, propagation with qutip.mesolve,
detection <u>. With a = mu_- and u = mu, QuDPy's dipole equals the QuDPy-FDGF chain without its
prefactor c_p, which is applied here. The transform uses the kernel exp[(i omega - eta) t] with
trapezoidal weights on the requested grid, i.e. the definition of R_eta; QuDPy's own spectra()
applies an unwindowed ifft2 on its native grid, which is not the same quantity.

Usage: python bench_qudpy_original.py CASE DT T     (CASE = example1 | weak | closed | frenkel)
Prints one JSON line.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import contextlib
import io
import json
import sys
import time

import numpy as np
import psutil
import qutip
# Original QuDPy (github.com/SilvaScience/QuDPy, v1.1.0): QUDPY_REPO, an installed package,
# or a clone next to this repository (../QuDPy).
from pathlib import Path as _Path
for _candidate in (os.environ.get("QUDPY_REPO"), _Path(__file__).resolve().parents[2] / "QuDPy"):
    if _candidate and _Path(_candidate, "qudpy", "Classes.py").exists():
        sys.path.insert(0, str(_candidate))
        break
from qudpy.Classes import System
import qudpy.Classes as _qudpy_classes

if os.environ.get("QUDPY_TIGHT"):                    # tighter mesolve tolerances, QuDPy code untouched
    _mesolve = qutip.mesolve

    def _tight_mesolve(*args, **kwargs):
        options = dict(kwargs.pop("options", None) or {})
        options.setdefault("atol", 1e-10)
        options.setdefault("rtol", 1e-8)
        return _mesolve(*args, options=options, **kwargs)
    _qudpy_classes.mesolve = _tight_mesolve
from qudpy_fdgf import EigenbasisKModel, FrequencyPathway, SpectroscopySolver
from qudpy_fdgf.protocols import standard_nq_protocol

case, DT, T = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
T2 = 10.0
PROTOCOL = standard_nq_protocol(order=1, n_interactions=3, nq_interval=1, detection_interval=3,
                                nq_axis="w1", detection_axis="w3")

if case == "frenkel":
    from frenkel_model import REPHASING, frenkel_blocks, frenkel_model
    n_sites, ETA = 6, 0.005
    W1, W3 = np.linspace(-2.2, -1.8, 41), np.linspace(1.8, 2.2, 41)
    H_b, up_b, number_b, low_b = frenkel_blocks(n_sites)
    dims = {k: v.shape[0] for k, v in H_b.items()}
    off = {"0": 0, "1": 1, "2": 1 + dims["1"]}
    D = sum(dims.values())

    def full(blocks):
        M = np.zeros((D, D), dtype=complex)
        for (t, s), B in blocks.items():
            M[off[t]:off[t] + B.shape[0], off[s]:off[s] + B.shape[1]] = B
        return M
    H = full({(k, k): v for k, v in H_b.items()})
    a = full(low_b)                                  # lowering part of the dipole
    mu = a + a.conj().T
    c_ops = [np.sqrt(2 * 0.01) * full(b) for b in number_b] + [np.sqrt(0.005) * a]
    pathways = REPHASING
    fdgf_model = frenkel_model(n_sites)
else:
    rates = {"example1": (0.04, 0.03), "weak": (0.004, 0.003), "closed": (0.0, 0.0)}[case]
    g1, gphi = rates
    ETA = 0.002
    W1, W3 = np.linspace(-2.5, -1.5, 41), np.linspace(1.5, 2.5, 41)
    H = np.diag([0.0, 2.0]).astype(complex)
    a = 0.5 * np.array([[0, 1], [0, 0]], dtype=complex)      # mu_- = d |g><e|
    mu = a + a.conj().T
    L_rad = np.array([[0, 1], [0, 0]], dtype=complex)
    sz = np.diag([-1.0, 1.0]).astype(complex)
    c_ops = ([np.sqrt(g1) * L_rad] if g1 else []) + ([np.sqrt(gphi / 2) * sz] if gphi else [])
    pathways = (FrequencyPathway(name="R1", interactions=("Bu", "Ku", "Bd"), component="rephasing"),
                FrequencyPathway(name="R2", interactions=("Bu", "Bd", "Ku"), component="rephasing"))
    ops = ((L_rad, g1), (sz, gphi / 2))
    fdgf_model = EigenbasisKModel(H, mu, c_ops_raw=tuple(o for o in ops if o[1] > 0))
    D = 2

# ---------------------------------------------------------------- reference (QuDPy-FDGF)
solver = SpectroscopySolver(backend="dense", eta=ETA, max_cache_entries=4096)
solver.feed_model(fdgf_model)
t0 = time.perf_counter()
ref = solver.generate_spectrum(PROTOCOL, {"w1": W1, "w3": W3}, pathways=pathways,
                               fixed_coordinates={"t2": T2}).components["rephasing"]
t_ref = time.perf_counter() - t0

# ---------------------------------------------------------------- QuDPy time domain
proc = psutil.Process()
base = proc.memory_info().wset
with contextlib.redirect_stdout(io.StringIO()):
    system = System(n=D - 1, H=qutip.Qobj(H), rho=qutip.Qobj(np.diag([1.0] + [0.0] * (D - 1)).astype(complex)),
                    a=qutip.Qobj(a), u=qutip.Qobj(mu), c_ops=[qutip.Qobj(c) for c in c_ops])
r = 1.0 / DT                                          # QuDPy: int(T r) samples on [0, T]
t_start = time.perf_counter()
total = np.zeros((W1.size, W3.size), dtype=complex)
N = None
for pathway in pathways:
    diagram = [(i.label, k) for k, i in enumerate(pathway.interactions)]
    with contextlib.redirect_stdout(io.StringIO()):
        t1, t3, dipole = system.coherence2d(time_delays=[T, T2, T], diagram=diagram, scan_id=[0, 2], r=r)
    N = len(t1)
    w1 = np.gradient(t1) if False else np.full(len(t1), t1[1] - t1[0])
    w1[0] = w1[-1] = (t1[1] - t1[0]) / 2
    w3 = np.full(len(t3), t3[1] - t3[0])
    w3[0] = w3[-1] = (t3[1] - t3[0]) / 2
    K1 = np.exp(np.outer(1j * W1 - ETA, t1)) * w1
    K3 = np.exp(np.outer(t3, 1j * W3 - ETA)) * w3[:, None]
    total += pathway.response_prefactor * (K1 @ np.asarray(dipole) @ K3)
t_qudpy = time.perf_counter() - t_start
peak = (proc.memory_info().peak_wset - base) / 1e6
err = float(np.max(np.abs(total - ref)) / np.max(np.abs(ref)))
print(json.dumps({"case": case, "tight": bool(os.environ.get("QUDPY_TIGHT")), "D": D, "eta": ETA, "dt": float(t1[1] - t1[0]), "T": T, "N_per_axis": N,
                  "error": err, "time_qudpy_s": t_qudpy, "peak_MB_qudpy": peak,
                  "time_fdgf_s": t_ref}))
