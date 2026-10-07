"""The models of the QuDPy-FDGF examples, built with QuTiP.

Shared by the analysis notebooks, the benchmark scripts and the check scripts of this folder.
The example notebooks in ``examples/`` write the same models out in full, so that each can be
read on its own; ``tests/consistency/test_validation_models.py`` (a local folder) checks that both give the same spectra.

- quickstart: open two-level system (examples/quickstart_two_level_model.ipynb), constants ``QS``;
- Example 1: thermal XXZ spin chain with L sites and a magnon cutoff N_max
  (examples/example1_xxz_chain_thermal.ipynb), constants ``EX1``;
- Example 2: two coupled dissipative bosonic modes, coherent versus action detection
  (examples/example2_coupled_modes_detection.ipynb), constants ``EX2``, and its closed limit at fifth order,
  constants ``EX2_CHI5``.

The former Example 3 (two coupled anharmonic bosonic modes, fifth-order 2Q response) is in old_files/models_example3.py.

Units: hbar = 1; the quickstart and Example 2 in eV, Example 1 in meV (field in tesla, temperature in K).
"""
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np
import qutip as qt
from qudpy_fdgf import (EigenbasisKModel, ExcitationSectorModel, FrequencyPathway,
                        ThermodynamicContext, pathways_from_signature, standard_nq_protocol)

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


# ============================================================================= quickstart (open two-level system)
QS = dict(omega_eg=2.0, d=0.5, gamma_1=0.04, gamma_phi=0.03, eta=0.002, t2=10.0)
QS_AXES = {"omega_1q": np.linspace(-2.5, -1.5, 41), "omega_emit": np.linspace(1.5, 2.5, 41)}
# R1 (stimulated emission) and R2 (ground-state bleach) exist without population transfer. R3 is the
# excited-state-absorption sequence of a three-level model; in this two-level model it is non-zero only
# because the radiative channel moves population from |e> to |g> during t2, which refills the bleach.
QS_PATHWAYS = (
    FrequencyPathway(name="R1", interactions=("Bu", "Ku", "Bd"), component="rephasing"),
    FrequencyPathway(name="R2", interactions=("Bu", "Bd", "Ku"), component="rephasing"),
    FrequencyPathway(name="R3", interactions=("Bu", "Ku", "Ku"), component="rephasing"),
)
QS_PROTOCOL = standard_nq_protocol(order=1, n_interactions=3, nq_interval=1, detection_interval=3,
                                    nq_axis="omega_1q", detection_axis="omega_emit")


def quickstart_model(gamma_1=None, gamma_phi=None, omega_eg=None, d=None):
    """Open two-level system. The dephasing channel is sigma_z with rate gamma_phi / 2."""
    g1 = QS["gamma_1"] if gamma_1 is None else gamma_1
    gphi = QS["gamma_phi"] if gamma_phi is None else gamma_phi
    w = QS["omega_eg"] if omega_eg is None else omega_eg
    dipole = QS["d"] if d is None else d
    ground, excited = qt.basis(2, 0), qt.basis(2, 1)
    H = w * excited * excited.dag()
    mu = dipole * (excited * ground.dag() + ground * excited.dag())
    channels = tuple(c for c in ((ground * excited.dag(), g1), (qt.sigmaz(), gphi / 2)) if c[1] > 0)
    return EigenbasisKModel(
        H, mu, c_ops_raw=channels,
        observable_op_arrays={"polarization": mu, "excited_population": excited * excited.dag(),
                              "ground_population": ground * ground.dag()})


# ============================================================================= Example 1 (thermal XXZ chain)
MU_B_MEV_PER_T = 0.0578838
BOLTZMANN_MEV_PER_K = 0.08617333262
EX1 = dict(J_xy=-0.30, J_z=-0.55, g_factor=2.10, static_field=7.1773, eta=0.03,
           temperature=4.0, n_sites=6, max_magnons=4, t2=0.0, krylov_tolerance=1e-11)
EX1_AXES = {"omega_1q": np.linspace(-1.55, -0.55, 60), "omega_emit": np.linspace(0.55, 1.55, 60)}
EX1_PATHWAYS = (
    FrequencyPathway(name="GSB", interactions=("Bu", "Bd", "Ku"), component="rephasing"),
    FrequencyPathway(name="SE", interactions=("Bu", "Ku", "Bd"), component="rephasing"),
    FrequencyPathway(name="ESA", interactions=("Bu", "Ku", "Ku"), component="rephasing"),
)
EX1_PROTOCOL = QS_PROTOCOL
# The complete rephasing response has eight sequences (first interaction Bu or Kd, then two of Ku and Bd). Three are non-zero
# for the ground state (GSB, SE, ESA, EX1_PATHWAYS); the thermally populated sectors activate the five others.
EX1_PATHWAYS_ALL = tuple(pathways_from_signature("-++", component="rephasing"))


def example1_chain(n_sites=None, **parameters):
    """XXZ ring as QuTiP objects: (H, magnon-number operator, uniform raising operator).

    H is measured from the fully polarized state. ``parameters`` override J_xy, J_z, g_factor,
    static_field.
    """
    p = {**EX1, **parameters}
    L = EX1["n_sites"] if n_sites is None else n_sites

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


def example1_model(n_sites=None, max_magnons=None, **parameters):
    """Sector model of the XXZ chain truncated at max_magnons; returns (model, hamiltonian blocks)."""
    n_max = EX1["max_magnons"] if max_magnons is None else max_magnons
    H, N_op, M_minus = example1_chain(n_sites, **parameters)
    magnons = np.rint(N_op.diag().real).astype(int)
    members = {N: np.flatnonzero(magnons == N) for N in range(n_max + 1)}
    H_full, M_full = H.full(), M_minus.full()
    hamiltonians = {N: H_full[np.ix_(i, i)] for N, i in members.items()}
    raising = {(N + 1, N): M_full[np.ix_(members[N + 1], members[N])] for N in range(n_max)}
    model = ExcitationSectorModel(hamiltonians, raising, initial_sector=0,
                                  boltzmann_constant=BOLTZMANN_MEV_PER_K)
    return model, hamiltonians


def example1_context(temperature=None):
    return ThermodynamicContext(temperature=EX1["temperature"] if temperature is None else temperature)


def example1_dimension(n_sites, max_magnons):
    return sum(comb(n_sites, n) for n in range(max_magnons + 1))


def example1_bimagnon(n_sites=None, **parameters):
    """Two-magnon levels of the ring by total momentum K = 2 pi m / L: {m: (energies, nearest-neighbour probabilities)}.

    P_nn = <sum_j n_j n_(j+1)> is the probability that two magnons are on neighbouring sites (0.4 for the uncorrelated pair
    of a six-site ring). The levels of each sector are sorted; the lowest level of K = 0 is the bimagnon seen by the uniform
    probe, the next one is the first scattering level of that sector.
    """
    L = EX1["n_sites"] if n_sites is None else n_sites
    H, N_op, _ = example1_chain(L, **parameters)
    magnons = np.rint(N_op.diag().real).astype(int)
    two = np.flatnonzero(magnons == 2)
    H2 = H.full()[np.ix_(two, two)]

    def translated(index):
        bits = [(index >> (L - 1 - j)) & 1 for j in range(L)]
        bits = bits[-1:] + bits[:-1]
        return sum(bit << (L - 1 - j) for j, bit in enumerate(bits))

    translation = np.zeros((2**L, 2**L))
    for index in range(2**L):
        translation[translated(index), index] = 1.0
    T2 = translation[np.ix_(two, two)]
    occupation = [np.array([(index >> (L - 1 - j)) & 1 for index in range(2**L)], dtype=float) for j in range(L)]
    P_nn = np.diag(sum(occupation[j] * occupation[(j + 1) % L] for j in range(L)))[np.ix_(two, two)]
    levels = {}
    for m in range(L):
        phase = np.exp(2j * np.pi * m / L)
        projector = sum(np.linalg.matrix_power(T2, k) * phase ** (-k) for k in range(L)) / L
        weights, vectors = np.linalg.eigh((projector + projector.conj().T) / 2)
        basis = vectors[:, weights > 0.5]
        energies, states = np.linalg.eigh(basis.conj().T @ H2 @ basis)
        pairs = [np.real(states[:, i].conj() @ basis.conj().T @ P_nn @ basis @ states[:, i]) for i in range(energies.size)]
        levels[m] = (energies, np.array(pairs))
    return levels


# ============================================================================= Example 2 (coupled modes)
# Two coupled, weakly anharmonic bosonic modes (three levels each, D = 9) with five kinds of dissipation,
# probed by coherent emission and by action (photoluminescence-type) detection of the same rephasing pathways.
EX2 = dict(omega_a=1.0, omega_b=1.2, coupling=0.001, anharmonicity=0.010, mu_a=1.0, mu_b=1.0,
           gamma_phi=0.010, k_relax=0.001, k_transfer=0.005, k_rad=0.0001, eta=1e-5, t2=10.0)
EX2_AXES = {"omega_1q": np.linspace(-1.30, -0.90, 201), "omega_emit": np.linspace(0.90, 1.30, 201)}
EX2_PATHWAYS = (
    FrequencyPathway(name="R_GSB", interactions=("Bu", "Bd", "Ku"), component="rephasing", detection="polarization"),
    FrequencyPathway(name="R_SE", interactions=("Bu", "Ku", "Bd"), component="rephasing", detection="polarization"),
    FrequencyPathway(name="R_ESA", interactions=("Bu", "Ku", "Ku"), component="rephasing", detection="polarization"),
)
EX2_PROTOCOL = QS_PROTOCOL


def example2_operators(**changes):
    """Operators of Example 2: dict with H, mu_plus, channels [(operator, rate)] and the detector operators.

    ``changes`` replaces entries of EX2 (for instance ``k_transfer=0.010``). The detectors are the transition
    operator mu (coherent emission), the projector A_PL = |e_a><e_a| + |e_b><e_b| on the two one-quantum
    eigenstates, the unbalanced projector 2|e_a><e_a| + |e_b><e_b|, and the excitation number N_exc.
    """
    p = {**EX2, **changes}
    levels = 3
    a = qt.tensor(qt.destroy(levels), qt.qeye(levels))
    b = qt.tensor(qt.qeye(levels), qt.destroy(levels))
    n_a, n_b = a.dag() * a, b.dag() * b
    one = qt.qeye([levels, levels])
    H = (p["omega_a"] * n_a + p["omega_b"] * n_b
         + 0.5 * p["anharmonicity"] * (n_a * (n_a - one) + n_b * (n_b - one))
         + p["coupling"] * (a.dag() * b + b.dag() * a))
    mu_plus = p["mu_a"] * a.dag() + p["mu_b"] * b.dag()
    mu = mu_plus + mu_plus.dag()
    drop = qt.Qobj(np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]))        # |0><1| on one mode
    a_drop, b_drop = qt.tensor(drop, qt.qeye(levels)), qt.tensor(qt.qeye(levels), drop)
    channels = [(n_a, p["gamma_phi"]), (n_b, p["gamma_phi"]),
                (a, p["k_relax"]), (b, p["k_relax"]),
                (a_drop.dag() * b_drop, p["k_transfer"]),
                (a_drop, p["k_rad"]), (b_drop, p["k_rad"])]
    energies, eigenstates = H.eigenstates()
    e_a, e_b = eigenstates[1], eigenstates[2]                          # the two one-quantum eigenstates
    detectors = {"polarization": mu,
                 "A_PL": e_a * e_a.dag() + e_b * e_b.dag(),
                 "A_PL_unbalanced": 2 * e_a * e_a.dag() + e_b * e_b.dag(),
                 "N_exc": n_a + n_b}
    return dict(H=H, mu_plus=mu_plus, channels=channels, detectors=detectors)


def example2_model(**changes):
    """EigenbasisKModel of Example 2 (see example2_operators for ``changes``)."""
    operators = example2_operators(**changes)
    mu_plus = operators["mu_plus"]
    return EigenbasisKModel(operators["H"], operators["detectors"]["polarization"], c_ops_raw=operators["channels"],
                            j_plus_array=mu_plus, j_minus_array=mu_plus.dag(),
                            detection_op_array=operators["detectors"]["polarization"],
                            observable_op_arrays=operators["detectors"])


# ----------------------------------------------------------------------------- closed limit of Example 2, fifth order
# The Hamiltonian of Example 2 without dissipation, truncated by total excitation number n_a + n_b <= N_max (D = 10 for
# N_max = 3): the fifth-order double-quantum response needs the three-excitation manifold, which the three levels per mode
# of the third-order model do not contain completely.
EX2_CHI5_AXES = {"omega_2q": np.linspace(-2.46, -1.96, 251), "omega_emit": np.linspace(0.96, 1.26, 301)}
EX2_CHI5_ETA = 0.002
EX2_CHI5_FIXED = {"t1": 2.0, "t3": 2.0, "t4": 2.0}
EX2_CHI5_PHASE_DISCRIMINATION = [(0, 1), (0, 1), (1, 0), (1, 0), (1, 0)]          # signature --+++
EX2_CHI5_PROTOCOL = standard_nq_protocol(order=2, nq_interval=2, detection_interval=5, n_interactions=5,
                                         nq_axis="omega_2q", detection_axis="omega_emit")


def example2_chi5_qobj(max_manifold=3, per_mode=None, **changes):
    """(states, H, J_plus) of the closed Hamiltonian of Example 2, with the basis states (n_a, n_b) ordered by manifold.

    By default the space is truncated by total excitation number, n_a + n_b <= max_manifold. ``per_mode=n`` keeps instead
    the n levels 0, ..., n-1 of each mode (the truncation of the third-order model for n = 3). ``changes`` replaces entries
    of EX2, for instance ``coupling=0.0`` or ``anharmonicity=0.0``.
    """
    p = {**EX2, **changes}
    levels = max_manifold + 1 if per_mode is None else per_mode
    a = qt.tensor(qt.destroy(levels), qt.qeye(levels))
    b = qt.tensor(qt.qeye(levels), qt.destroy(levels))
    n_a, n_b = a.dag() * a, b.dag() * b
    one = qt.qeye([levels, levels])
    H_full = (p["omega_a"] * n_a + p["omega_b"] * n_b
              + 0.5 * p["anharmonicity"] * (n_a * (n_a - one) + n_b * (n_b - one))
              + p["coupling"] * (a.dag() * b + b.dag() * a))
    states = sorted(((i, j) for i in range(levels) for j in range(levels) if per_mode is not None or i + j <= max_manifold),
                    key=lambda s: (sum(s), -s[0]))
    keep = [i * levels + j for i, j in states]

    def truncate(operator):
        return qt.Qobj(operator.full()[np.ix_(keep, keep)])

    return tuple(states), truncate(H_full), truncate(p["mu_a"] * a.dag() + p["mu_b"] * b.dag())


def example2_chi5_model(max_manifold=3, per_mode=None, **changes):
    """EigenbasisKModel of the closed limit (see example2_chi5_qobj)."""
    _, H, J_plus = example2_chi5_qobj(max_manifold, per_mode, **changes)
    mu = J_plus + J_plus.dag()
    return EigenbasisKModel(H, mu, j_plus_array=J_plus, j_minus_array=J_plus.dag(), detection_op_array=mu,
                            observable_op_arrays={"polarization": mu})


def example2_chi5_pathways(max_manifold=3):
    """The fifth-order 2Q pathways (phase discrimination --+++), generated by UFSS up to ``max_manifold`` excitations."""
    import ufss
    from qudpy_fdgf import translate_ufss_diagrams
    generator = ufss.DiagramGenerator(detection_type="polarization")
    generator.set_phase_discrimination(EX2_CHI5_PHASE_DISCRIMINATION)
    generator.maximum_manifold = max_manifold
    generator.efield_times = [np.array([0.0, 0.0])] * 5
    return translate_ufss_diagrams(generator.get_diagrams(np.arange(5, dtype=float)), component="chi5_2q")
