"""The bimagnon of the six-site XXZ ring of Example 1 is bound at fixed total momentum, not against 2 omega(0).

The two-magnon sector of the ring splits by total momentum K = 2 pi m / L. At K = 0, the sector reached from the bright
magnon by the uniform probe, the lowest level (2.1108 meV) has a high probability P_nn that two magnons are neighbours
(0.67, against 0.4 for uncorrelated magnons) and lies 0.447 meV below the first scattering level (2.5583 meV) of the same
sector, 0.434 meV below the free pair 2 omega(2 pi / L). The distance to 2 omega(0), -0.134 meV, is not a binding energy:
two magnons cannot both be at k = 0. It is the displacement of the excited-state-absorption line from the bright line,
which the third-order spectrum shows.

Built with NumPy only (Pauli matrices); the same numbers, with QuTiP, are in examples/example2_xxz_chain_thermal.ipynb.
"""
from math import comb

import numpy as np

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
