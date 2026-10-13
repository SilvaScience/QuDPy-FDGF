"""Error of the impulsive limit and of the rotating-wave approximation for the THz chain of Example 1 (script).

The third-order rephasing map of the six-site XXZ chain at 4 K (N_max = 4, D = 57, eta = 0.03 meV, t2 = 0, the 60 x 60
grid of the example) is computed in the time domain for Gaussian pulses of standard deviation sigma and carrier equal to
the bright magnon energy, with the rotating-wave field (``rwa``) and with the real field (``full``), and compared with the
impulsive, rotating-wave response of QuDPy-FDGF. The time-domain engine is validation/pulses.py. The maps are written
to results/data/example1_pulses.npz; the notebook example1_pulses.ipynb reads them and draws the figure.

    python example1_pulses.py [workers]

The calculation takes about an hour of processor time (a few minutes of the largest pulse durations per map); the jobs
run in parallel. Each map is kept in results/cache/example1_pulses as soon as it is done: relaunching the script after an
interruption computes only the maps that are missing.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                     # validation/ (models.py, pulses.py)
import numpy as np

import models
import pulses

SIGMAS = (0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 3.0)      # pulse standard deviations (meV^-1)
ETA = models.EX1["eta"]
AXES = models.EX1_AXES
OUT = models.DATA / "example1_pulses.npz"
CACHE = models.RESULTS / "cache" / "example1_pulses"          # one file per (sigma, field): an interrupted run resumes here


def cache_file(sigma, rwa):
    return CACHE / f"{'rwa' if rwa else 'full'}_{sigma:g}.npz"


def time_step(sigma):
    """RK4 step: at most 0.02 and a fifth of the pulse duration, with 0.05 an integer number of steps."""
    for step in (0.02, 0.01, 0.005, 0.0025):
        if step <= sigma / 4 + 1e-12:
            return step
    return 0.0025


def job(arguments):
    sigma, rwa = arguments
    if cache_file(sigma, rwa).exists():
        stored = np.load(cache_file(sigma, rwa))
        return sigma, rwa, stored["signal"], float(stored["seconds"])
    energies, mu, mu_plus, rho0 = pulses.chain_system(models.EX1["max_magnons"], models.EX1["temperature"])
    carrier = float(np.sort(np.linalg.eigvalsh(models.example1_model()[1][1]))[0])      # the bright k = 0 magnon energy
    start = time.perf_counter()
    signal = pulses.rephasing_map(energies, mu, mu_plus, rho0, sigma, carrier, rwa, AXES["omega_1q"], AXES["omega_emit"],
                                  ETA, t2=models.EX1["t2"], dt=time_step(sigma))
    elapsed = time.perf_counter() - start
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(cache_file(sigma, rwa), signal=signal, seconds=elapsed)
    return sigma, rwa, signal, elapsed


def reference():
    from qudpy_fdgf import SpectroscopySolver
    model, _ = models.example1_model()
    solver = SpectroscopySolver(backend="sparse_sector", eta=ETA, krylov_tolerance=1e-11)
    solver.feed_model(model, context=models.example1_context())
    result = solver.generate_spectrum(models.EX1_PROTOCOL, AXES, pathways=models.EX1_PATHWAYS_ALL,
                                      fixed_coordinates={"t2": models.EX1["t2"]})
    return result.components["rephasing"]


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    models.results_dirs()
    jobs = [(sigma, rwa) for sigma in sorted(SIGMAS, reverse=True) for rwa in (True, False)]      # longest first
    jobs = [j for j in jobs if not cache_file(*j).exists()] + [j for j in jobs if cache_file(*j).exists()]
    maps, seconds = {}, {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for sigma, rwa, signal, elapsed in pool.map(job, jobs):
            key = f"{'rwa' if rwa else 'full'}_{sigma:g}"
            maps[key], seconds[key] = signal, elapsed
            print(f"sigma {sigma:g} {'rwa ' if rwa else 'full'}: {elapsed:.0f} s", flush=True)
    np.savez(OUT, sigmas=np.array(SIGMAS), omega_1q=AXES["omega_1q"], omega_emit=AXES["omega_emit"], eta=ETA,
             fdgf=reference(), seconds=np.array([seconds[k] for k in sorted(seconds)]),
             keys=np.array(sorted(seconds)), **maps)
    print("saved", OUT)
