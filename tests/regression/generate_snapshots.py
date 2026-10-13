"""Write tests/regression/data/snapshots.npz from the current code.

Run it only on purpose, when a change is meant to alter the numerical results, and explain the
change in the commit message.

    python tests/regression/generate_snapshots.py
"""
import os
import sys
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parent))      # tests/ (helpers.py)
sys.path.insert(0, str(here))

import numpy as np                        # noqa: E402

from regression_cases import CASES        # noqa: E402

data = {}
for case, run in CASES.items():
    for key, values in run().items():
        data[f"{case}::{key}"] = np.asarray(values)
    print(f"{case}: {sum(k.startswith(case + '::') for k in data)} arrays")
target = here / "data" / "snapshots.npz"
target.parent.mkdir(exist_ok=True)
np.savez_compressed(target, **data)
print("written", target, f"({target.stat().st_size / 1024:.0f} kB)")
