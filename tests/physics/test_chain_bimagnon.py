"""The bimagnon of the six-site XXZ ring of Example 1 is bound at fixed total momentum, not against 2 omega(0).

The two-magnon sector of the ring splits by total momentum K = 2 pi m / L. At K = 0, the sector reached from the bright
magnon by the uniform probe, the lowest level (2.1108 meV) has a high probability P_nn that two magnons are neighbours
(0.67, against 0.4 for uncorrelated magnons) and lies 0.447 meV below the first scattering level (2.5583 meV) of the same
sector, 0.434 meV below the free pair 2 omega(2 pi / L). The distance to 2 omega(0), -0.134 meV, is not a binding energy:
two magnons cannot both be at k = 0. It is the displacement of the excited-state-absorption line from the bright line,
which the third-order spectrum shows. In the display convention of the article (the real part of i S), the bleach and the
stimulated emission are positive and the excited-state absorption, which ends on that bimagnon line, is negative.

Built with NumPy only (Pauli matrices); the same numbers, with QuTiP, are in examples/example1_xxz_chain_thermal.ipynb.
"""
from math import comb

import numpy as np

from qudpy_fdgf import ExcitationSectorModel, FrequencyPathway, SpectroscopySolver, standard_nq_protocol

L = 6
J_XY, J_Z, G, MU_B, FIELD = -0.30, -0.55, 2.10, 0.0578838, 7.1773


def ring():
    sx, sy, sz = (np.array(m, dtype=complex) / 2 for m in ([[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]))

    def site(op, j):
        out = np.eye(1)
        for k in range(L):
            out = np.kron(out, op if k == j else np.eye(2))
        return out

    H = sum(J_XY * (site(sx, j) @ site(sx, (j + 1) % L) + site(sy, j) @ site(sy, (j + 1) % L))
            + J_Z * site(sz, j) @ site(sz, (j + 1) % L) - G * MU_B * FIELD * site(sz, j) for j in range(L))
    number = sum(0.5 * np.eye(2**L) - site(sz, j) for j in range(L))
    occupations = [0.5 * np.eye(2**L) - site(sz, j) for j in range(L)]
    neighbours = sum(occupations[j] @ occupations[(j + 1) % L] for j in range(L))
    H = H.real - H[0, 0].real * np.eye(2**L)                                  # measured from the polarized state
    return H, np.rint(np.diag(number.real)).astype(int), neighbours.real


def translation():
    matrix = np.zeros((2**L, 2**L))
    for index in range(2**L):
        bits = [(index >> (L - 1 - j)) & 1 for j in range(L)]
        bits = bits[-1:] + bits[:-1]
        matrix[sum(b << (L - 1 - j) for j, b in enumerate(bits)), index] = 1.0
    return matrix


def sector_levels(momentum):
    H, magnons, neighbours = ring()
    two = np.flatnonzero(magnons == 2)
    H2, T2, P2 = H[np.ix_(two, two)], translation()[np.ix_(two, two)], neighbours[np.ix_(two, two)]
    phase = np.exp(2j * np.pi * momentum / L)
    projector = sum(np.linalg.matrix_power(T2, m) * phase ** (-m) for m in range(L)) / L
    weights, vectors = np.linalg.eigh((projector + projector.conj().T) / 2)
    basis = vectors[:, weights > 0.5]
    energies, states = np.linalg.eigh(basis.conj().T @ H2 @ basis)
    pairs = np.array([np.real(states[:, i].conj() @ basis.conj().T @ P2 @ basis @ states[:, i]) for i in range(energies.size)])
    return energies, pairs


def omega_one(k):
    return G * MU_B * FIELD - J_Z + J_XY * np.cos(k)


def test_bound_state_and_scattering_levels_at_zero_momentum():
    energies, pairs = sector_levels(0)
    assert energies.size == 3
    np.testing.assert_allclose(energies, [2.1108, 2.5583, 3.3155], atol=5e-5)
    uncorrelated = L / comb(L, 2)
    assert pairs[0] > 0.6 and pairs[1] < uncorrelated < pairs[0]


def test_binding_energy_is_taken_in_the_sector_of_the_bright_state():
    energies, _ = sector_levels(0)
    assert abs((energies[1] - energies[0]) - 0.4475) < 1e-3                       # first scattering level of K = 0
    assert abs((2 * omega_one(2 * np.pi / L) - energies[0]) - 0.4341) < 1e-3       # free pair 2 omega(2 pi / L)
    assert abs((energies[0] - 2 * omega_one(0.0)) + 0.1341) < 1e-3                  # the displacement of the ESA line


def test_the_lowest_two_magnon_level_has_zero_momentum():
    lowest = {m: sector_levels(m)[0][0] for m in range(L)}
    assert min(lowest, key=lowest.get) == 0


def test_excited_state_absorption_has_the_sign_opposite_to_bleach_and_emission():
    """At the excitation pole of the bright magnon, Re(i S) is +1/eta^2 for the bleach and the emission (exactly: the matrix
    element of the uniform probe from the polarized state is 1) and negative for the absorption of the bimagnon line."""
    eta = 0.03
    H, magnons, _ = ring()
    members = {N: np.flatnonzero(magnons == N) for N in range(3)}
    lowering = np.array([[0, 0], [1, 0]], dtype=complex)                         # S^- on one site
    M = np.zeros((2**L, 2**L), dtype=complex)
    for j in range(L):
        operator = np.eye(1)
        for k in range(L):
            operator = np.kron(operator, lowering if k == j else np.eye(2))
        M += operator / np.sqrt(L)
    model = ExcitationSectorModel({N: H[np.ix_(i, i)] for N, i in members.items()},
                                  {(N + 1, N): M[np.ix_(members[N + 1], members[N])] for N in range(2)}, initial_sector=0)
    solver = SpectroscopySolver(backend="dense", eta=eta)
    solver.feed_model(model)
    bright = omega_one(0.0)
    bimagnon = sector_levels(0)[0][0] - bright
    sequences = {"GSB": ("Bu", "Bd", "Ku"), "SE": ("Bu", "Ku", "Bd"), "ESA": ("Bu", "Ku", "Ku")}
    pathways = [FrequencyPathway(name=name, interactions=labels, component="rephasing") for name, labels in sequences.items()]
    protocol = standard_nq_protocol(order=1, nq_interval=1, detection_interval=3, n_interactions=3,
                                    nq_axis="omega_1q", detection_axis="omega_emit")
    result = solver.generate_spectrum(protocol, {"omega_1q": np.array([-bright]), "omega_emit": np.array([bright, bimagnon])},
                                      pathways=pathways, fixed_coordinates={"t2": 0.0})
    shown = {name: (1j * np.asarray(result.pathways[name])).real.ravel() for name in sequences}       # [bright, bimagnon]
    np.testing.assert_allclose(shown["GSB"][0], 1 / eta**2, rtol=1e-6)
    np.testing.assert_allclose(shown["SE"][0], shown["GSB"][0], rtol=1e-9)
    assert shown["ESA"][1] < 0
    assert 1.4 < -shown["ESA"][1] / shown["GSB"][0] < 1.7                           # (2 - 2/L) times the share of the bimagnon
