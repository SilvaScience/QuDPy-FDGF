"""solver.decay_rates(): rates implied by the declared GKSL channels."""
import numpy as np
import pytest

from qudpy_fdgf import EigenbasisKModel, SpectroscopySolver
from test_integrated_jump import three_site_chain

GAMMA_1, GAMMA_PHI = 0.04, 0.03
H = np.diag([0.0, 2.0]).astype(complex)
MU = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
L_RAD = np.array([[0, 1], [0, 0]], dtype=complex)      # |g><e|
SIGMA_Z = np.diag([-1.0, 1.0]).astype(complex)
P_E = np.diag([0.0, 1.0]).astype(complex)


def two_level(c_ops, backend="dense"):
    solver = SpectroscopySolver(backend=backend, eta=0.002)
    solver.feed_model(EigenbasisKModel(H, MU, c_ops_raw=c_ops))
    return solver


@pytest.mark.parametrize("backend", ("dense", "sparse_sector"))
def test_example1_rates(backend):
    rates = two_level(((L_RAD, GAMMA_1), (SIGMA_Z, GAMMA_PHI / 2)), backend).decay_rates()
    g, e = 0, 1
    assert rates.energies == pytest.approx([0.0, 2.0])
    assert rates.population_decay[e] == pytest.approx(GAMMA_1)
    assert rates.population_decay[g] == pytest.approx(0.0, abs=1e-15)
    assert rates.transition_rates[g, e] == pytest.approx(GAMMA_1)          # e -> g
    assert rates.coherence_decay[e, g] == pytest.approx(GAMMA_1 / 2 + GAMMA_PHI)
    assert rates.coherence_frequency[e, g] == pytest.approx(2.0)
    assert rates.coherences() == [(e, g, pytest.approx(2.0), pytest.approx(0.05))]


def test_normalization_of_dephasing_channels():
    """sigma_z needs gamma_phi/2 and a projector 2 gamma_phi for the same coherence rate."""
    target = GAMMA_1 / 2 + GAMMA_PHI
    sigma_z = two_level(((L_RAD, GAMMA_1), (SIGMA_Z, GAMMA_PHI / 2))).decay_rates()
    projector = two_level(((L_RAD, GAMMA_1), (P_E, 2 * GAMMA_PHI))).decay_rates()
    wrong = two_level(((L_RAD, GAMMA_1), (SIGMA_Z, GAMMA_PHI))).decay_rates()
    assert sigma_z.coherence_decay[1, 0] == pytest.approx(target)
    assert projector.coherence_decay[1, 0] == pytest.approx(target)
    assert wrong.coherence_decay[1, 0] == pytest.approx(GAMMA_1 / 2 + 2 * GAMMA_PHI)


def test_exact_modes_of_example1():
    rates = two_level(((L_RAD, GAMMA_1), (SIGMA_Z, GAMMA_PHI / 2))).decay_rates(modes=True)
    expected = np.array([0.0, -GAMMA_1, -0.05 + 2.0j, -0.05 - 2.0j])
    for value in expected:
        assert np.min(np.abs(rates.modes - value)) < 1e-12


def test_modes_match_the_dense_generator_and_backends_agree():
    dense = SpectroscopySolver(backend="dense", eta=0.005)
    dense.feed_model(three_site_chain())
    sparse = SpectroscopySolver(backend="sparse_sector", eta=0.005)
    sparse.feed_model(three_site_chain())
    rates = dense.decay_rates(modes=True)
    reference = np.linalg.eigvals(dense.backend._A_dense)
    for value in reference:
        assert np.min(np.abs(rates.modes - value)) < 1e-10
    other = sparse.decay_rates()
    assert other.coherence_decay == pytest.approx(rates.coherence_decay, abs=1e-12)
    assert other.population_decay == pytest.approx(rates.population_decay, abs=1e-12)
    optical = rates.coherences(between=("0", "1"))
    assert len(optical) == 3
    assert all(1.8 < omega < 2.2 and gamma > 0 for _, _, omega, gamma in optical)


def test_modes_are_limited_to_small_dimensions():
    solver = SpectroscopySolver(backend="dense", eta=0.005)
    solver.feed_model(three_site_chain())                     # D = 7
    with pytest.raises(ValueError):
        solver.decay_rates(modes=True, max_mode_dimension=5)
