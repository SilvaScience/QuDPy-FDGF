"""Dissipative Frenkel-exciton chain truncated to the 0-, 1-, and 2-exciton manifolds.

n sites, hard-core excitons, H = sum_i eps_i n_i + J sum_i (b_i^+ b_{i+1} + h.c.);
Hilbert dimension D = 1 + n + n(n-1)/2. Dissipation: local pure dephasing on every
site (L = n_i, Lindblad coefficient 2 gamma_phi, i.e. coherence dephasing gamma_phi)
and collective radiative decay (L = sum_i b_i, rate gamma_1).
"""
from itertools import combinations

import numpy as np
from qudpy_fdgf import ExcitationSectorModel, FrequencyPathway


def frenkel_blocks(n, eps0=2.0, gradient=0.01, J=-0.05):
    one = list(range(n))
    two = list(combinations(range(n), 2))
    idx2 = {pair: k for k, pair in enumerate(two)}
    eps = eps0 + gradient * np.arange(n)

    H1 = np.diag(eps).astype(complex)
    for i in range(n - 1):
        H1[i, i + 1] = H1[i + 1, i] = J
    H2 = np.zeros((len(two), len(two)), dtype=complex)
    for (i, j), k in idx2.items():
        H2[k, k] = eps[i] + eps[j]
        for a, b in ((i, j), (j, i)):          # move the excitation at a to a +/- 1
            for c in (a - 1, a + 1):
                if 0 <= c < n and c != b:
                    H2[idx2[tuple(sorted((c, b)))], k] += J
    # raising operator sum_i b_i^+ (unit, parallel dipoles)
    up10 = np.ones((n, 1), dtype=complex)
    up21 = np.zeros((len(two), n), dtype=complex)
    for (i, j), k in idx2.items():
        up21[k, i] = up21[k, j] = 1.0
    number = []
    for s in one:
        d1 = np.zeros((n, n), dtype=complex)
        d1[s, s] = 1.0
        d2 = np.diag([1.0 if s in pair else 0.0 for pair in two]).astype(complex)
        number.append({("1", "1"): d1, ("2", "2"): d2})
    lowering = {("0", "1"): up10.conj().T, ("1", "2"): up21.conj().T}
    hamiltonian = {"0": np.zeros((1, 1), dtype=complex), "1": H1, "2": H2}
    raising = {("1", "0"): up10, ("2", "1"): up21}
    return hamiltonian, raising, number, lowering


def frenkel_model(n, gamma_phi=0.01, gamma_1=0.005, dissipative=True):
    hamiltonian, raising, number, lowering = frenkel_blocks(n)
    c_ops = ()
    if dissipative:
        c_ops = tuple((blocks, 2.0 * gamma_phi) for blocks in number) + ((lowering, gamma_1),)
    return ExcitationSectorModel(hamiltonian, raising, c_ops_raw=c_ops, initial_sector="0")


def hilbert_dimension(n):
    return 1 + n + n * (n - 1) // 2


REPHASING = (
    FrequencyPathway(name="GSB", interactions=("Bu", "Bd", "Ku"), component="rephasing"),
    FrequencyPathway(name="SE", interactions=("Bu", "Ku", "Bd"), component="rephasing"),
    FrequencyPathway(name="ESA", interactions=("Bu", "Ku", "Ku"), component="rephasing"),
)
