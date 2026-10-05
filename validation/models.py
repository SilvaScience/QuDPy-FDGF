"""The three models of the QuDPy-FDGF article, built with QuTiP.

Shared by the analysis notebooks, the benchmark scripts and the check scripts of this folder.
The example notebooks in ``examples/`` write the same models out in full, so that each can be
read on its own; ``tests/test_validation_models.py`` checks that both give the same spectra.

- Example 1: open two-level system (examples/example1_two_level_open.ipynb);
- Example 2: thermal XXZ spin chain with L sites and a magnon cutoff N_max
  (examples/example2_xxz_chain_thermal.ipynb);
- Example 3: two coupled anharmonic bosonic modes, fifth-order 2Q response
  (examples/example3_chi5_2q.ipynb).

Units: hbar = 1; Examples 1 and 3 in eV, Example 2 in meV (field in tesla, temperature in K).
"""
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np
import qutip as qt
from qudpy_fdgf import (EigenbasisKModel, ExcitationSectorModel, FrequencyPathway,
                        ThermodynamicContext, standard_nq_protocol)

# ============================================================================= locations
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
DATA = RESULTS / "data"                      # numerical results (.npz, .json)
FIGURES = RESULTS / "figure"                 # spectra and figures of the examples
ANALYSIS_FIGURES = RESULTS / "figure_analysis"


def results_dirs():
    """Create the results folders if needed and return (DATA, FIGURES, ANALYSIS_FIGURES)."""
    for folder in (DATA, FIGURES, ANALYSIS_FIGURES):
        folder.mkdir(parents=True, exist_ok=True)
    return DATA, FIGURES, ANALYSIS_FIGURES


# ============================================================================= Example 1
EX1 = dict(omega_eg=2.0, d=0.5, gamma_1=0.04, gamma_phi=0.03, eta=0.002, t2=10.0)
EX1_AXES = {"omega_1q": np.linspace(-2.5, -1.5, 41), "omega_emit": np.linspace(1.5, 2.5, 41)}
EX1_PATHWAYS = (
    FrequencyPathway(name="R1", interactions=("Bu", "Ku", "Bd"), component="rephasing"),
    FrequencyPathway(name="R2", interactions=("Bu", "Bd", "Ku"), component="rephasing"),
)
EX1_PROTOCOL = standard_nq_protocol(order=1, n_interactions=3, nq_interval=1, detection_interval=3,
                                    nq_axis="omega_1q", detection_axis="omega_emit")


def example1_model(gamma_1=None, gamma_phi=None, omega_eg=None, d=None):
    """Open two-level system. The dephasing channel is sigma_z with rate gamma_phi / 2."""
    g1 = EX1["gamma_1"] if gamma_1 is None else gamma_1
    gphi = EX1["gamma_phi"] if gamma_phi is None else gamma_phi
    w = EX1["omega_eg"] if omega_eg is None else omega_eg
    dipole = EX1["d"] if d is None else d
    ground, excited = qt.basis(2, 0), qt.basis(2, 1)
    H = w * excited * excited.dag()
    mu = dipole * (excited * ground.dag() + ground * excited.dag())
    channels = tuple(c for c in ((ground * excited.dag(), g1), (qt.sigmaz(), gphi / 2)) if c[1] > 0)
    return EigenbasisKModel(
        H, mu, c_ops_raw=channels,
        observable_op_arrays={"polarization": mu, "excited_population": excited * excited.dag(),
                              "ground_population": ground * ground.dag()})


# ============================================================================= Example 2
MU_B_MEV_PER_T = 0.0578838
BOLTZMANN_MEV_PER_K = 0.08617333262
EX2 = dict(J_xy=-0.30, J_z=-0.55, g_factor=2.10, static_field=7.1773, eta=0.03,
           temperature=4.0, n_sites=6, max_magnons=4, t2=0.0, krylov_tolerance=1e-11)
EX2_AXES = {"omega_1q": np.linspace(-1.55, -0.55, 60), "omega_emit": np.linspace(0.55, 1.55, 60)}
EX2_PATHWAYS = (
    FrequencyPathway(name="GSB", interactions=("Bu", "Bd", "Ku"), component="rephasing"),
    FrequencyPathway(name="SE", interactions=("Bu", "Ku", "Bd"), component="rephasing"),
    FrequencyPathway(name="ESA", interactions=("Bu", "Ku", "Ku"), component="rephasing"),
)
EX2_PROTOCOL = EX1_PROTOCOL


def example2_chain(n_sites=None, **parameters):
    """XXZ ring as QuTiP objects: (H, magnon-number operator, uniform raising operator).

    H is measured from the fully polarized state. ``parameters`` override J_xy, J_z, g_factor,
    static_field.
    """
    p = {**EX2, **parameters}
    L = EX2["n_sites"] if n_sites is None else n_sites

    def site(op, j):
        return qt.tensor([op if k == j else qt.qeye(2) for k in range(L)])

    Sx = [site(qt.sigmax() / 2, j) for j in range(L)]
    Sy = [site(qt.sigmay() / 2, j) for j in range(L)]
    Sz = [site(qt.sigmaz() / 2, j) for j in range(L)]
    H = 0
    for j in range(L):
        k = (j + 1) % L
        H += (p["J_xy"] * (Sx[j] * Sx[k] + Sy[j] * Sy[k]) + p["J_z"] * Sz[j] * Sz[k]
              - p["g_factor"] * MU_B_MEV_PER_T * p["static_field"] * Sz[j])
    H = H - qt.expect(H, qt.tensor([qt.basis(2, 0)] * L))
    N_op = sum(0.5 - Sz[j] for j in range(L))
    M_minus = sum(site(qt.sigmam(), j) for j in range(L)) / np.sqrt(L)
    return H, N_op, M_minus


def example2_model(n_sites=None, max_magnons=None, **parameters):
    """Sector model of the XXZ chain truncated at max_magnons; returns (model, hamiltonian blocks)."""
    n_max = EX2["max_magnons"] if max_magnons is None else max_magnons
    H, N_op, M_minus = example2_chain(n_sites, **parameters)
    magnons = np.rint(N_op.diag().real).astype(int)
    members = {N: np.flatnonzero(magnons == N) for N in range(n_max + 1)}
    H_full, M_full = H.full(), M_minus.full()
    hamiltonians = {N: H_full[np.ix_(i, i)] for N, i in members.items()}
    raising = {(N + 1, N): M_full[np.ix_(members[N + 1], members[N])] for N in range(n_max)}
    model = ExcitationSectorModel(hamiltonians, raising, initial_sector=0,
                                  boltzmann_constant=BOLTZMANN_MEV_PER_K)
    return model, hamiltonians


def example2_context(temperature=None):
    return ThermodynamicContext(temperature=EX2["temperature"] if temperature is None else temperature)


def example2_dimension(n_sites, max_magnons):
    return sum(comb(n_sites, n) for n in range(max_magnons + 1))


# ============================================================================= Example 3
EX3 = dict(omega_h=1.53, omega_l=1.58, coupling=0.010, mu_h=1.0, mu_l=0.30,
           U_hh=-0.020, U_ll=-0.020, U_hl=0.030, max_manifold=3)
EX3_FIXED = {"t1": 2.0, "t3": 2.0, "t4": 2.0}
EX3_AXES = {"omega_2q": np.linspace(-3.20, -2.99, 151), "omega_emit": np.linspace(1.42, 1.66, 151)}
EX3_PHASE_DISCRIMINATION = [(0, 1), (0, 1), (1, 0), (1, 0), (1, 0)]
EX3_PROTOCOL = standard_nq_protocol(order=2, nq_interval=2, detection_interval=5, n_interactions=5,
                                    nq_axis="omega_2q", detection_axis="omega_emit")


def example3_qobj(max_manifold=None, anharmonicities=None):
    """Two-mode model restricted to n_h + n_l <= max_manifold, as dense Qobj in manifold order.

    Returns (states, H, J_plus, polarization_h, polarization_l). ``anharmonicities`` is
    (U_hh, U_ll, U_hl); the default is the one of the example.
    """
    n_max = EX3["max_manifold"] if max_manifold is None else max_manifold
    U_hh, U_ll, U_hl = ((EX3["U_hh"], EX3["U_ll"], EX3["U_hl"]) if anharmonicities is None
                        else anharmonicities)
    levels = n_max + 1
    b_h = qt.tensor(qt.destroy(levels), qt.qeye(levels))
    b_l = qt.tensor(qt.qeye(levels), qt.destroy(levels))
    n_h, n_l = b_h.dag() * b_h, b_l.dag() * b_l
    one = qt.qeye([levels, levels])
    H_full = (EX3["omega_h"] * n_h + EX3["omega_l"] * n_l
              + EX3["coupling"] * (b_h.dag() * b_l + b_l.dag() * b_h)
              + 0.5 * U_hh * n_h * (n_h - one) + 0.5 * U_ll * n_l * (n_l - one) + U_hl * n_h * n_l)
    states = sorted(((a, b) for a in range(levels) for b in range(levels) if a + b <= n_max),
                    key=lambda s: (sum(s), -s[0]))
    keep = [a * levels + b for a, b in states]

    def truncate(operator):
        return qt.Qobj(operator.full()[np.ix_(keep, keep)])

    return (tuple(states), truncate(H_full),
            truncate(EX3["mu_h"] * b_h.dag() + EX3["mu_l"] * b_l.dag()),
            truncate(EX3["mu_h"] * (b_h + b_h.dag())), truncate(EX3["mu_l"] * (b_l + b_l.dag())))


def example3_operators(max_manifold=None, anharmonicities=None):
    """NumPy version of example3_qobj: (states, H, J_plus, polarization_h, polarization_l)."""
    states, H, J_plus, P_h, P_l = example3_qobj(max_manifold, anharmonicities)
    return states, H.full(), J_plus.full(), P_h.full(), P_l.full()


def example3_model(max_manifold=None, anharmonicities=None):
    _, H, J_plus, P_h, P_l = example3_qobj(max_manifold, anharmonicities)
    mu = J_plus + J_plus.dag()
    return EigenbasisKModel(H, mu, j_plus_array=J_plus, j_minus_array=J_plus.dag(), detection_op_array=mu,
                            observable_op_arrays={"polarization": mu, "polarization_h": P_h,
                                                  "polarization_l": P_l})


def example3_pathways(max_manifold=None):
    import ufss
    from qudpy_fdgf import translate_ufss_diagrams
    generator = ufss.DiagramGenerator(detection_type="polarization")
    generator.set_phase_discrimination(EX3_PHASE_DISCRIMINATION)
    generator.maximum_manifold = EX3["max_manifold"] if max_manifold is None else max_manifold
    generator.efield_times = [np.array([0.0, 0.0])] * 5
    return translate_ufss_diagrams(generator.get_diagrams(np.arange(5, dtype=float)), component="chi5_2q")
