"""Check that example_models.py reproduces the published numbers of Examples 2 and 3."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import time
from pathlib import Path

import numpy as np
from qudpy_fdgf import SpectroscopySolver
from example_models import (EX2, EX3, EX3_AXES, EX3_FIXED, EX3_PROTOCOL, example2_context,
                            example2_dimension, example2_model, example3_model, example3_pathways)

report = {}

# ----------------------------------------------------------------- Example 2 (published: Sec. 5.2)
model, H = example2_model()
D = sum(model.dimension(s) for s in model.sectors())
e1 = np.linalg.eigvalsh(H[1]).min()
e2 = np.linalg.eigvalsh(H[2]).min()
full, _ = example2_model(max_magnons=EX2["n_sites"])
state = full.equilibrium_state(example2_context())
pops = {int(a): float(np.trace(block.as_matrix()).real)
        for (a, b), block in sorted(state.blocks.items()) if a == b}
report["example2"] = {"D": D, "D_formula": example2_dimension(6, 4), "bright_one_magnon_meV": e1,
                      "two_magnon_min_meV": e2, "bimagnon_transition_meV": e2 - e1,
                      "P0": pops[0], "P1": pops[1], "P2": pops[2],
                      "published": {"D": 57, "bright": 1.1224, "two_magnon_min": 2.1108,
                                    "transition": 0.9884, "P0": 0.885331, "P1": 0.102734, "P2": 0.010908}}
print(json.dumps(report["example2"], indent=1))

# ----------------------------------------------------------------- Example 3 (published: Sec. 5.3, Fig. 6)
pathways = example3_pathways()
solver = SpectroscopySolver(backend="dense", eta=0.008, max_cache_entries=4096)
solver.feed_model(example3_model())
start = time.perf_counter()
result = solver.generate_nq_spectrum(2, EX3_PROTOCOL, axes=EX3_AXES, fixed_coordinates=EX3_FIXED,
                                     pathways=pathways)
elapsed = time.perf_counter() - start
total = sum(result.pathways.values())
i, j = np.unravel_index(np.argmax(np.abs(total)), total.shape)
report["example3"] = {"pathways": len(pathways), "D": sum(solver.backend.layout.dimensions.values()),
                      "peak": [float(EX3_AXES["omega_2q"][i]), float(EX3_AXES["omega_emit"][j])],
                      "max_abs": float(np.abs(total[i, j])), "time_151x151_s": elapsed,
                      "published": {"pathways": 7, "peak": [-3.038, 1.510], "max_abs": 1.26e5}}
print(json.dumps(report["example3"], indent=1))
out = Path(__file__).with_name("results")
out.mkdir(exist_ok=True)
(out / "example_models_check.json").write_text(json.dumps(report, indent=2))
