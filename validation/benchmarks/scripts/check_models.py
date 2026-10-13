"""Check that models.py reproduces the published numbers of Example 1 (XXZ chain) and of the closed limit of Example 2."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # single-thread BLAS, before NumPy is imported
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))   # validation/ (models.py)
from models import DATA
import json
import time
from pathlib import Path

import numpy as np
from qudpy_fdgf import SpectroscopySolver
from models import (EX1, EX2_CHI5_AXES, EX2_CHI5_ETA, EX2_CHI5_FIXED, EX2_CHI5_PROTOCOL, example1_context,
                    example1_dimension, example1_model, example2_chi5_model, example2_chi5_pathways)

report = {}

# ----------------------------------------------------------------- Example 1 (published: Sec. 5.1)
model, H = example1_model()
D = sum(model.dimension(s) for s in model.sectors())
e1 = np.linalg.eigvalsh(H[1]).min()
e2 = np.linalg.eigvalsh(H[2]).min()
full, _ = example1_model(max_magnons=EX1["n_sites"])
state = full.equilibrium_state(example1_context())
pops = {int(a): float(np.trace(block.as_matrix()).real)
        for (a, b), block in sorted(state.blocks.items()) if a == b}
report["example1"] = {"D": D, "D_formula": example1_dimension(6, 4), "bright_one_magnon_meV": e1,
                      "two_magnon_min_meV": e2, "bimagnon_transition_meV": e2 - e1,
                      "P0": pops[0], "P1": pops[1], "P2": pops[2],
                      "published": {"D": 57, "bright": 1.1224, "two_magnon_min": 2.1108,
                                    "transition": 0.9884, "P0": 0.885331, "P1": 0.102734, "P2": 0.010908}}
print(json.dumps(report["example1"], indent=1))

# ----------------------------------------------------------------- closed limit of Example 2 (published: Sec. 5.2, fifth order)
pathways = example2_chi5_pathways()
solver = SpectroscopySolver(backend="dense", eta=EX2_CHI5_ETA, max_cache_entries=4096)
solver.feed_model(example2_chi5_model())
small = {name: np.linspace(values[0], values[-1], 31) for name, values in EX2_CHI5_AXES.items()}
start = time.perf_counter()
result = solver.generate_spectrum(EX2_CHI5_PROTOCOL, small, fixed_coordinates=EX2_CHI5_FIXED, pathways=pathways, warn_null=False)
elapsed = time.perf_counter() - start
total = sum(result.pathways.values())
report["example2_chi5"] = {"pathways": len(pathways), "D": sum(solver.backend.layout.dimensions.values()),
                           "max_abs_31x31": float(np.abs(total).max()), "time_31x31_s": elapsed,
                           "published": {"pathways": 7, "D": 10}}
print(json.dumps(report["example2_chi5"], indent=1))
out = DATA
out.mkdir(parents=True, exist_ok=True)
(out / "models_check.json").write_text(json.dumps(report, indent=2))
