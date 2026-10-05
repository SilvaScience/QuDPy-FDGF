"""Small deterministic calculations whose outputs are stored in ``data/snapshots.npz``.

The snapshots are the reference for refactors of the numerical core (vectorized generator action,
factorized map evaluation, ...): a change that must not alter any result is accepted only if every
case below still reproduces its stored arrays. They are not physical benchmarks: the physics is
checked by ``tests/physics``. Regenerate the file only on purpose, with
``python tests/regression/generate_snapshots.py``, and say why in the commit message.

The cases are self-contained (NumPy only) and cheap (about 3 s in total).
"""
import itertools

import numpy as np

from helpers import (
    H,
    L_RAD,
    MU,
    P_E,
    PROTOCOL,
    REPHASING,
    SIGMA_Z,
    GAMMA_1,
    GAMMA_PHI,
    three_site_chain,
)
from qudpy_fdgf import (
    EigenbasisKModel,
    FrequencyPathway,
    ObservableSpec,
    PropagationInterval,
    SpectroscopyProtocol,
    SpectroscopySolver,
    ThermodynamicContext,
)
from qudpy_fdgf.protocols import standard_nq_protocol

TIGHT = {"krylov_tolerance": 1e-12}


def _arrays(result):
    """Spectrum result -> flat mapping of every pathway and observable array."""
    out = {}
    for name, values in result.pathways.items():
        out[f"pathway__{name}"] = values
    for observable, per_pathway in result.observables.items():
        for name, values in per_pathway.items():
            out[f"observable__{observable}__{name}"] = values
    return out


# --------------------------------------------------------------------------------------- two-level

def _two_level(backend):
    options = TIGHT if backend == "sparse_sector" else {}
    solver = SpectroscopySolver(backend=backend, eta=0.002, **options)
    solver.feed_model(EigenbasisKModel(
        H, MU, c_ops_raw=((L_RAD, GAMMA_1), (SIGMA_Z, GAMMA_PHI / 2)),
        observable_op_arrays={"polarization": MU, "excited_population": P_E}))
    pathways = (
        FrequencyPathway(name="R1", interactions=("Bu", "Ku", "Bd"), component="rephasing"),
        FrequencyPathway(name="R2", interactions=("Bu", "Bd", "Ku"), component="rephasing"),
        FrequencyPathway(name="R3", interactions=("Bu", "Ku", "Ku"), component="rephasing"),
    )
    observables = {
        "polarization": "polarization",
        "population": ObservableSpec.action("population", fourth_interaction="Bu",
                                            operator="excited_population"),
        "fluorescence": ObservableSpec.mean_jump(
            "fluorescence", solver.jump_channel_names()[0], time_window=(0.0, 125.0),
            fourth_interaction="Bu"),
    }
    axes = {"w1": np.linspace(-2.1, -1.9, 5), "w3": np.linspace(1.9, 2.1, 5)}
    return _arrays(solver.generate_spectrum(PROTOCOL, axes, fixed_coordinates={"t2": 10.0},
                                            pathways=pathways, observables=observables))


# ------------------------------------------------------------------------------ three-site chain
CHAIN_AXES = {"w1": np.linspace(-2.1, -1.9, 3), "w3": np.linspace(1.9, 2.1, 3)}


def _chain_dissipative(backend):
    options = TIGHT if backend == "sparse_sector" else {}
    solver = SpectroscopySolver(backend=backend, eta=0.01, **options)
    solver.feed_model(three_site_chain(dissipative=True))
    return _arrays(solver.generate_spectrum(PROTOCOL, CHAIN_AXES, fixed_coordinates={"t2": 20.0},
                                            pathways=REPHASING))


def _chain_thermal_diagonal():
    solver = SpectroscopySolver(backend="sparse_sector", eta=0.01, preconditioner="diagonal", **TIGHT)
    solver.feed_model(three_site_chain(dissipative=False), context=ThermodynamicContext(temperature=600.0))
    return _arrays(solver.generate_spectrum(PROTOCOL, CHAIN_AXES, fixed_coordinates={"t2": 0.0},
                                            pathways=REPHASING))


def _time_only(backend):
    """Closed model, pure state, time-domain intervals only: rank-one dyad route of the sparse backend."""
    solver = SpectroscopySolver(backend=backend, eta=0.01, **(TIGHT if backend == "sparse_sector" else {}))
    solver.feed_model(three_site_chain(dissipative=False))
    protocol = SpectroscopyProtocol(intervals=(
        PropagationInterval("t1", "time", 1),
        PropagationInterval("t2", "time"),
        PropagationInterval("t3", "time", 1),
    ))
    axes = {"t1": np.array([0.0, 5.0, 10.0]), "t3": np.array([0.0, 5.0, 10.0])}
    return _arrays(solver.generate_spectrum(protocol, axes, fixed_coordinates={"t2": 2.0},
                                            pathways=REPHASING))


# ----------------------------------------------------------------------- two-mode bosons, order 5
def _bosons_chi5(backend="dense"):
    levels, n_max = 4, 3
    destroy = np.diag(np.sqrt(np.arange(1, levels)), 1).astype(complex)
    eye = np.eye(levels, dtype=complex)
    b_h, b_l = np.kron(destroy, eye), np.kron(eye, destroy)
    n_h, n_l = b_h.conj().T @ b_h, b_l.conj().T @ b_l
    one = np.eye(levels * levels, dtype=complex)
    omega_h, omega_l, coupling, u_hh, u_ll, u_hl = 1.53, 1.58, 0.010, -0.020, -0.020, 0.030
    full = (omega_h * n_h + omega_l * n_l + coupling * (b_h.conj().T @ b_l + b_l.conj().T @ b_h)
            + 0.5 * u_hh * n_h @ (n_h - one) + 0.5 * u_ll * n_l @ (n_l - one) + u_hl * n_h @ n_l)
    states = sorted(((a, b) for a in range(levels) for b in range(levels) if a + b <= n_max),
                    key=lambda s: (sum(s), -s[0]))
    keep = [a * levels + b for a, b in states]
    truncate = lambda operator: operator[np.ix_(keep, keep)]
    j_plus = truncate(1.0 * b_h.conj().T + 0.30 * b_l.conj().T)
    mu = j_plus + j_plus.conj().T
    model = EigenbasisKModel(truncate(full), mu, j_plus_array=j_plus, j_minus_array=j_plus.conj().T,
                             detection_op_array=mu)
    pathways = [
        FrequencyPathway(name="P" + "".join(t[0] for t in tail), interactions=("Bu", "Bu") + tail,
                         component="chi5_2q")
        for tail in itertools.product(("Ku", "Bd"), repeat=3) if tail != ("Bd", "Bd", "Bd")
    ]
    protocol = standard_nq_protocol(order=2, nq_interval=2, detection_interval=5, n_interactions=5,
                                    nq_axis="w2q", detection_axis="w5")
    solver = SpectroscopySolver(backend=backend, eta=0.008)
    solver.feed_model(model)
    axes = {"w2q": np.linspace(-3.20, -3.00, 3), "w5": np.linspace(1.45, 1.57, 3)}
    fixed = {"t1": 2.0, "t3": 2.0, "t4": 2.0}
    return _arrays(solver.generate_spectrum(protocol, axes, fixed_coordinates=fixed, pathways=pathways))


CASES = {
    "two_level_dense": lambda: _two_level("dense"),
    "two_level_sparse": lambda: _two_level("sparse_sector"),
    "chain_dissipative_dense": lambda: _chain_dissipative("dense"),
    "chain_dissipative_sparse": lambda: _chain_dissipative("sparse_sector"),
    "chain_thermal_diagonal": _chain_thermal_diagonal,
    "time_only_dense": lambda: _time_only("dense"),
    "time_only_sparse": lambda: _time_only("sparse_sector"),
    "bosons_chi5_dense": _bosons_chi5,
}
