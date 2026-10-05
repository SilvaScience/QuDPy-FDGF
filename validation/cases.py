"""Extra Example 3 calculations used by the article figures, computed once and cached.

The example notebook (examples/example3_chi5_2q.ipynb) saves the reference case, N_max = 3 and
eta = 8 meV, in results/data/example3.npz. Two figures of the article also need the cases that
differ from it: N_max = 3 at eta = 2 meV, N_max = 2 at eta = 2 meV, and the pathway amplitudes at
N_max = 2 and eta = 8 meV. They are computed here (about 1 minute) and stored in
results/data/example3_extra_cases.npz, so that the figure scripts redraw instantly afterwards.
"""
import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import numpy as np

from qudpy_fdgf import SpectroscopySolver
from models import DATA, EX3_AXES, EX3_FIXED, EX3_PROTOCOL, example3_model, example3_pathways

CACHE = DATA / "example3_extra_cases.npz"


def example3_case(max_manifold, eta):
    """Fifth-order 2Q spectrum on the grid of the example: (spectrum, {pathway name: map}, labels)."""
    pathways = example3_pathways(max_manifold)
    solver = SpectroscopySolver(backend="dense", eta=eta, max_cache_entries=4096)
    solver.feed_model(example3_model(max_manifold))
    solver.set_pathways(pathways)
    result = solver.generate_nq_spectrum(2, EX3_PROTOCOL, axes=EX3_AXES, fixed_coordinates=EX3_FIXED,
                                         pathways=pathways)
    labels = {p.name: " ".join(i.label for i in p.interactions) for p in pathways}
    return result.components["chi5_2q"], {p.name: result.pathways[p.name] for p in pathways}, labels


def example3_extra_cases(recompute=False):
    """Return a dict with the spectra of the three extra cases and the N_max = 2 pathway maps."""
    if CACHE.exists() and not recompute:
        saved = np.load(CACHE)
        return {key: saved[key] for key in saved.files}
    DATA.mkdir(parents=True, exist_ok=True)
    out = {}
    out["spectrum_n3_eta2"], _, _ = example3_case(3, 0.002)
    out["spectrum_n2_eta2"], _, _ = example3_case(2, 0.002)
    spectrum, pathway_maps, labels = example3_case(2, 0.008)
    out["spectrum_n2_eta8"] = spectrum
    out["pathway_names_n2_eta8"] = np.array(list(pathway_maps))
    out["pathway_labels_n2_eta8"] = np.array([labels[name] for name in pathway_maps])
    out["pathway_maps_n2_eta8"] = np.array([pathway_maps[name] for name in pathway_maps])
    np.savez_compressed(CACHE, **out)
    return out
