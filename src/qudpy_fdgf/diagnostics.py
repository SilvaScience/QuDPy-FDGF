"""Decay rates implied by the GKSL channels of a loaded model.

The rate supplied with a collapse channel is the Lindblad coefficient gamma_c of
gamma_c D_c[rho]; its relation to measured population and coherence decay
times depends on the normalization of the jump operator L_c (for a diagonal
channel, |x><y| decays at gamma_c (l_x - l_y)^2 / 2). ``decay_rates`` reports
what the declared channels actually imply, in the eigenbasis of the
Hamiltonian:

- secular rates, read from the diagonal of the generator in the basis
  |x><y| of eigenstate dyads (any Hilbert dimension);
- optionally, the exact eigenvalues lambda_nu = -kappa_nu - i Omega_nu of the
  generator (small dimensions), which include non-secular couplings.

The secular coherence rates are the pole half-widths of a frequency-resolved
interval when the generator does not couple different dyads.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DecayRates:
    """Rates implied by the GKSL channels, in the Hamiltonian eigenbasis.

    Attributes
    ----------
    energies : (D,) eigenvalues of the Hamiltonian, ascending.
    sectors : (D,) dominant model sector of each eigenstate.
    sector_weights : (D, n_sectors) weight of each eigenstate in each sector.
    channel_names : names of the GKSL channels.
    population_decay : (D,) secular decay rate of each eigenstate population,
        Gamma_x = sum_c gamma_c (<x|L_c^+ L_c|x> - |<x|L_c|x>|^2).
    transition_rates : (D, D) secular rates W[y, x] of the population transfer
        x -> y, sum_c gamma_c |<y|L_c|x>|^2 (zero diagonal).
    coherence_decay : (D, D) secular decay rate gamma_xy of |x><y|,
        sum_c gamma_c [(<x|K_c|x> + <y|K_c|y>)/2 - Re(<x|L_c|x> <y|L_c|y>^*)],
        with K_c = L_c^+ L_c.
    coherence_frequency : (D, D) secular frequency omega_xy of |x><y|, equal
        to E_x - E_y up to dissipative shifts; rho_xy(t) ~ exp[-(gamma_xy +
        i omega_xy) t].
    modes : exact eigenvalues of the generator sorted by decay rate, or None.
    """

    energies: np.ndarray
    sectors: tuple
    sector_weights: np.ndarray
    channel_names: tuple
    population_decay: np.ndarray
    transition_rates: np.ndarray
    coherence_decay: np.ndarray
    coherence_frequency: np.ndarray
    modes: np.ndarray | None = None

    @property
    def dimension(self):
        return self.energies.size

    def coherences(self, between=None):
        """Coherence rows (x, y, omega_xy, gamma_xy) with omega_xy > 0.

        ``between=(sector_a, sector_b)`` keeps |x><y| with x mainly in
        sector_a and y mainly in sector_b (in either order), e.g. the optical
        coherences between the ground and one-excitation manifolds.
        """
        rows = []
        for x in range(self.dimension):
            for y in range(self.dimension):
                if self.coherence_frequency[x, y] <= 0.0:
                    continue
                if between is not None:
                    pair = {self.sectors[x], self.sectors[y]}
                    if pair != set(between) and not (
                        len(set(between)) == 1 and pair == set(between)
                    ):
                        continue
                rows.append((x, y, float(self.coherence_frequency[x, y]),
                             float(self.coherence_decay[x, y])))
        return sorted(rows, key=lambda row: row[2])


def _dense(operator, dimension):
    """Linear operator -> dense matrix (one application per basis column)."""
    return np.asarray(operator @ np.eye(dimension, dtype=np.complex128),
                      dtype=np.complex128)


def _generator_matrix(hamiltonian, jumps, rates):
    """Column-stacked matrix of A = -i[H, .] + sum_c gamma_c D_c (Eqs. 10-12)."""
    dimension = hamiltonian.shape[0]
    identity = np.eye(dimension, dtype=np.complex128)
    generator = -1j * (np.kron(identity, hamiltonian) - np.kron(hamiltonian.T, identity))
    for jump, rate in zip(jumps, rates):
        k = jump.conj().T @ jump
        generator += rate * (np.kron(jump.conj(), jump)
                             - 0.5 * np.kron(identity, k)
                             - 0.5 * np.kron(k.T, identity))
    return generator


def decay_rates(backend, *, modes=False, max_mode_dimension=40):
    """Built backend -> DecayRates of its Hamiltonian and GKSL channels.

    ``modes=True`` also diagonalizes the full generator (a D^2 x D^2 matrix),
    which is allowed only for D <= max_mode_dimension. Within a degenerate
    eigenspace of the Hamiltonian the secular rates depend on the chosen
    eigenvectors; the exact modes do not.
    """
    layout = backend.layout
    dimension = layout.total_dimension
    hamiltonian = _dense(backend._hamiltonian, dimension)
    hamiltonian = 0.5 * (hamiltonian + hamiltonian.conj().T)
    energies, vectors = np.linalg.eigh(hamiltonian)

    weights = np.array([[np.sum(np.abs(vectors[layout.slices[s], k]) ** 2)
                         for s in layout.sectors] for k in range(dimension)])
    sectors = tuple(layout.sectors[int(np.argmax(row))] for row in weights)

    names, rates, jumps_site, jumps_eig = [], [], [], []
    for channel, operator in backend._collapse_operators:
        jump = _dense(operator, dimension)
        names.append(channel.name)
        rates.append(float(channel.rate))
        jumps_site.append(jump)
        jumps_eig.append(vectors.conj().T @ jump @ vectors)

    population = np.zeros(dimension)
    transitions = np.zeros((dimension, dimension))
    dissipative = np.zeros((dimension, dimension), dtype=np.complex128)
    for jump, rate in zip(jumps_eig, rates):
        k_diag = np.real(np.einsum("ix,ix->x", jump.conj(), jump))   # <x|L^+L|x>
        l_diag = np.diag(jump)
        population += rate * (k_diag - np.abs(l_diag) ** 2)
        transitions += rate * np.abs(jump) ** 2
        dissipative += rate * (np.outer(l_diag, l_diag.conj())
                               - 0.5 * k_diag[:, None] - 0.5 * k_diag[None, :])
    np.fill_diagonal(transitions, 0.0)
    diagonal = -1j * (energies[:, None] - energies[None, :]) + dissipative

    exact = None
    if modes:
        if dimension > int(max_mode_dimension):
            raise ValueError(
                f"Exact modes require D <= max_mode_dimension "
                f"({max_mode_dimension}); this model has D = {dimension}."
            )
        generator = _generator_matrix(hamiltonian, jumps_site, rates)
        exact = np.linalg.eigvals(generator)
        exact = exact[np.lexsort((exact.imag, -exact.real))]

    return DecayRates(
        energies=energies,
        sectors=sectors,
        sector_weights=weights,
        channel_names=tuple(names),
        population_decay=population,
        transition_rates=transitions,
        coherence_decay=-diagonal.real,
        coherence_frequency=-diagonal.imag,
        modes=exact,
    )
