"""Run the QuDPy / QuDPy-FDGF comparison on the Perspective model (bench_qudpy_perspective.py) for the four cases.

QuDPy-FDGF first (it writes the reference maps), then the original QuDPy, each in its own process, one at a time (the
original coherence2d keeps all N x N propagated states in memory, and the timings must not overlap).

Writes results/data/qudpy_perspective.json, and resumes from it: the cases already in the file are skipped, and an
interrupted case restarts at its next diagram (propagated signals are kept in results/cache/qudpy_perspective).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # single-thread BLAS, before NumPy is imported
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))   # validation/ (models.py)
from models import DATA
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASES = [("perspective", "coherent"), ("perspective", "action"), ("resolved", "coherent"), ("resolved", "action")]

out = DATA / "qudpy_perspective.json"
rows = json.loads(out.read_text()) if out.exists() else []          # resume: the cases already measured are kept
done = {(r["mode"], r["grid"], r["kind"]) for r in rows if not r.get("failed")}
rows = [r for r in rows if not r.get("failed")]
CACHE = HERE.parents[1] / "results" / "cache" / "qudpy_perspective"
for mode in ("fdgf", "qudpy"):
    for grid, kind in CASES:
        if (mode, grid, kind) in done:
            continue
        if mode == "qudpy" and not (CACHE / f"reference_{grid}_{kind}.npy").exists():      # reference maps lost: recompute (fast)
            subprocess.run([sys.executable, "-W", "ignore", "bench_qudpy_perspective.py", "fdgf", grid, kind], cwd=HERE,
                           capture_output=True, text=True, timeout=3600)
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        proc = subprocess.run([sys.executable, "-W", "ignore", "bench_qudpy_perspective.py", mode, grid, kind],
                              cwd=HERE, env=env, capture_output=True, text=True, timeout=14400)
        line = [l for l in proc.stdout.splitlines() if l.startswith("{")]
        row = json.loads(line[-1]) if line else {"mode": mode, "grid": grid, "kind": kind, "failed": True,
                                                   "stderr": proc.stderr[-400:]}
        rows.append(row)
        print(json.dumps(row), flush=True)
        DATA.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=2))
