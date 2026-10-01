"""The three examples of the manuscript, as importable models for validation and benchmarks.

Each builder reproduces the corresponding notebook exactly (parameters, pathways,
protocol, frequency grids, eta):
- Example 1: open two-level system (examples/two_level_open_observables.ipynb);
- Example 2: thermally initialized XXZ spin chain (examples/spin_chain_k_space.ipynb),
  parameterized by the number of sites L and the magnon cutoff N_max;
- Example 3: fifth-order 2Q response of the two-mode bosonic-exciton model
  (examples/chromophore_chi5_2q.ipynb), parameterized by the excitation cutoff.
"""
from itertools import combinations

import numpy as np
from qudpy_fdgf import (EigenbasisKModel, ExcitationSectorModel, FrequencyPathway,
                        ThermodynamicContext, standard_nq_protocol)

# =============================================================================== Example 1
EX1 = dict(omega_eg=2.0, d=0.5, gamma_1=0.04, gamma_phi=0.03, eta=0.002, t2=10.0)
EX1_AXES = {"omega_1q": np.linspace(-2.5, -1.5, 41), "omega_emit": np.linspace(1.5, 2.5, 41)}
EX1_PATHWAYS = (
    FrequencyPathway(name="R1", interactions=("Bu", "Ku", "Bd"), component="rephasing"),
    FrequencyPathway(name="R2", interactions=("Bu", "Bd", "Ku"), component="rephasing"),
)
EX1_PROTOCOL = standard_nq_protocol(order=1, n_interactions=3, nq_interval=1, detection_interval=3,
                                    nq_axis="omega_1q", detection_axis="omega_emit")


def example1_operators():
    H = np.diag([0.0, EX1["omega_eg"]]).astype(complex)
    mu = EX1["d"] * np.array([[0, 1], [1, 0]], dtype=complex)
    lowering = EX1["d"] * np.array([[0, 1], [0, 0]], dtype=complex)        # mu_-
    L_rad = np.array([[0, 1], [0, 0]], dtype=complex)
    sigma_z = np.diag([-1.0, 1.0]).astype(complex)
    return H, mu, lowering, L_rad, sigma_z


def example1_model(gamma_1=None, gamma_phi=None):
    H, mu, _, L_rad, sigma_z = example1_operators()
    g1 = EX1["gamma_1"] if gamma_1 is None else gamma_1
    gphi = EX1["gamma_phi"] if gamma_phi is None else gamma_phi
    channels = tuple(c for c in ((L_rad, g1), (sigma_z, gphi / 2)) if c[1] > 0)
    return EigenbasisKModel(H, mu, c_ops_raw=channels,
                            observable_op_arrays={"polarization": mu,
                                                  "excited_population": np.diag([0.0, 1.0]).astype(complex)})


# =============================================================================== Example 2
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


def _xxz_sector_hamiltonian(n_sites, n_magnons, J_xy, J_z, g_factor, static_field):
    basis = tuple(combinations(range(n_sites), n_magnons))
    index = {state: k for k, state in enumerate(basis)}
    H = np.zeros((len(basis), len(basis)), dtype=complex)
    one_magnon_cost = g_factor * MU_B_MEV_PER_T * static_field - J_z
    hopping = 0.5 * J_xy
    for column, state in enumerate(basis):
        occupied = set(state)
        adjacent = sum((site + 1) % n_sites in occupied for site in occupied)
        H[column, column] = n_magnons * one_magnon_cost + J_z * adjacent
        for site in state:
            for neighbor in ((site - 1) % n_sites, (site + 1) % n_sites):
                if neighbor in occupied:
                    continue
                target = tuple(sorted((occupied - {site}) | {neighbor}))
                H[index[target], column] += hopping
    return basis, H


def _magnon_creation_block(source_basis, target_basis, profile):
    target_index = {state: k for k, state in enumerate(target_basis)}
    block = np.zeros((len(target_basis), len(source_basis)), dtype=complex)
    for column, state in enumerate(source_basis):
        occupied = set(state)
        for site, amplitude in enumerate(profile):
            if site in occupied or amplitude == 0:
                continue
            block[target_index[tuple(sorted((*state, site)))], column] += amplitude
    return block


def example2_model(n_sites=None, max_magnons=None):
    """XXZ chain of n_sites spins truncated at max_magnons; returns (model, hamiltonian blocks)."""
    L = EX2["n_sites"] if n_sites is None else n_sites
    n_max = EX2["max_magnons"] if max_magnons is None else max_magnons
    bases, hamiltonians = {}, {}
    for n in range(n_max + 1):
        bases[n], hamiltonians[n] = _xxz_sector_hamiltonian(
            L, n, EX2["J_xy"], EX2["J_z"], EX2["g_factor"], EX2["static_field"])
    profile = np.ones(L, dtype=complex) / np.sqrt(L)                 # uniform THz field
    raising = {(n + 1, n): _magnon_creation_block(bases[n], bases[n + 1], profile)
               for n in range(n_max)}
    model = ExcitationSectorModel(hamiltonians, raising, initial_sector=0,
                                  boltzmann_constant=BOLTZMANN_MEV_PER_K)
    return model, hamiltonians


def example2_context():
    return ThermodynamicContext(temperature=EX2["temperature"])


def example2_dimension(n_sites, max_magnons):
    from math import comb
    return sum(comb(n_sites, n) for n in range(max_magnons + 1))


# =============================================================================== Example 3
EX3 = dict(omega_h=1.53, omega_l=1.58, coupling=0.010, mu_h=1.0, mu_l=0.30,
           U_hh=-0.020, U_ll=-0.020, U_hl=0.030, max_manifold=3)
EX3_FIXED = {"t1": 2.0, "t3": 2.0, "t4": 2.0}
EX3_AXES = {"omega_2q": np.linspace(-3.20, -2.99, 151), "omega_emit": np.linspace(1.42, 1.66, 151)}
EX3_PHASE_DISCRIMINATION = [(0, 1), (0, 1), (1, 0), (1, 0), (1, 0)]
EX3_PROTOCOL = standard_nq_protocol(order=2, nq_interval=2, detection_interval=5, n_interactions=5,
                                    nq_axis="omega_2q", detection_axis="omega_emit")


def example3_operators(max_manifold=None):
    n_max = EX3["max_manifold"] if max_manifold is None else max_manifold
    basis = tuple((n_h, total - n_h) for total in range(n_max + 1) for n_h in range(total, -1, -1))
    index = {state: k for k, state in enumerate(basis)}
    d = len(basis)
    b_h = np.zeros((d, d), dtype=complex)
    b_l = np.zeros_like(b_h)
    for k, (n_h, n_l) in enumerate(basis):
        if n_h > 0:
            b_h[index[(n_h - 1, n_l)], k] = np.sqrt(n_h)
        if n_l > 0:
            b_l[index[(n_h, n_l - 1)], k] = np.sqrt(n_l)
    I = np.eye(d, dtype=complex)
    n_h_op, n_l_op = b_h.conj().T @ b_h, b_l.conj().T @ b_l
    H = (EX3["omega_h"] * n_h_op + EX3["omega_l"] * n_l_op
         + EX3["coupling"] * (b_h.conj().T @ b_l + b_l.conj().T @ b_h)
         + 0.5 * EX3["U_hh"] * n_h_op @ (n_h_op - I) + 0.5 * EX3["U_ll"] * n_l_op @ (n_l_op - I)
         + EX3["U_hl"] * n_h_op @ n_l_op)
    J_plus = EX3["mu_h"] * b_h.conj().T + EX3["mu_l"] * b_l.conj().T
    return basis, H, J_plus, b_h, b_l


def example3_model(max_manifold=None):
    basis, H, J_plus, b_h, b_l = example3_operators(max_manifold)
    J_minus = J_plus.conj().T
    mu = J_plus + J_minus
    return EigenbasisKModel(H, mu, j_plus_array=J_plus, j_minus_array=J_minus, detection_op_array=mu,
                            observable_op_arrays={
                                "polarization": mu,
                                "polarization_h": EX3["mu_h"] * (b_h + b_h.conj().T),
                                "polarization_l": EX3["mu_l"] * (b_l + b_l.conj().T)})


def example3_pathways(max_manifold=None):
    import ufss
    from qudpy_fdgf import translate_ufss_diagrams
    generator = ufss.DiagramGenerator(detection_type="polarization")
    generator.set_phase_discrimination(EX3_PHASE_DISCRIMINATION)
    generator.maximum_manifold = EX3["max_manifold"] if max_manifold is None else max_manifold
    generator.efield_times = [np.array([0.0, 0.0])] * 5
    return translate_ufss_diagrams(generator.get_diagrams(np.arange(5, dtype=float)), component="chi5_2q")
