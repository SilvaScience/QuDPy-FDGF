"""Run the original-QuDPy comparison series sequentially and collect the results.

Each case runs bench_qudpy_original.py in its own process (one at a time, because the
original coherence2d keeps all N x N propagated states in memory). The (dt, T) pairs are
those that reach 1e-2, 1e-3, and 1e-4 in the reference time-domain run (bench_time_fft.py).
The last two cases are repeated with tighter mesolve tolerances (QUDPY_TIGHT=1).

Writes results/qudpy_original.json.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # single-thread BLAS, before NumPy is imported
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))   # validation/ (models.py)
from models import DATA
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = [
    ("example1", 0.8, 133, False),
    ("example1", 0.4, 177, False),
    ("example1", 0.2, 199, False),
    ("weak", 0.8, 987, False),
    ("weak", 0.8, 1316, False),
    ("example1", 0.2, 199, True),
    ("weak", 0.8, 1316, True),
]

rows = []
for case, dt, T, tight in RUNS:
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    if tight:
        env["QUDPY_TIGHT"] = "1"
    proc = subprocess.run([sys.executable, "bench_qudpy_original.py", case, str(dt), str(T)],
                          cwd=HERE, env=env, capture_output=True, text=True, timeout=7200)
    line = [l for l in proc.stdout.splitlines() if l.startswith("{")]
    row = json.loads(line[-1]) if line else {"case": case, "dt": dt, "T": T, "tight": tight,
                                               "failed": True, "stderr": proc.stderr[-300:]}
    rows.append(row)
    print(json.dumps(row), flush=True)
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "qudpy_original.json").write_text(json.dumps(rows, indent=2))
