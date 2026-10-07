"""Action detection with the excitation number reads the coherent signal (Example 2).

For two coupled, anharmonic and dissipative bosonic modes, the action signal whose detector is the
excitation number N = n_a + n_b equals -i times the polarization, pathway by pathway and to machine
precision, provided the two readout alternatives (Bu and Kd) are summed and N is read right after the
fourth interaction. The reason is algebraic: Tr(N [mu_-, rho]) = Tr([N, mu_-] rho) = -Tr(mu_- rho),
whatever the dynamics before the readout. A readout alternative alone, or a detector that is not the
excitation number (a projector on the one-quantum states), does not have this property.

The same relation is shown, with figures, in examples/example2_coupled_modes_detection.ipynb.
"""
import numpy as np
import pytest

from qudpy_fdgf import EigenbasisKModel, ObservableSpec, SpectroscopySolver

from helpers import PROTOCOL, REPHASING

LEVELS = 3
ETA = 1e-3
AXES = {"w1": np.array([-1.21, -1.19, -1.01, -0.99]), "w3": np.array([0.99, 1.01, 1.19, 1.21])}


def two_mode_model(k_transfer=5e-3):
    """Two bosonic modes (omega 1.0 and 1.2, J = 1 meV, anharmonicity 10 meV) with seven GKSL channels."""
    lower = np.diag(np.sqrt(np.arange(1, LEVELS)), 1).astype(complex)
    identity = np.eye(LEVELS, dtype=complex)
    a, b = np.kron(lower, identity), np.kron(identity, lower)
    n_a, n_b = a.conj().T @ a, b.conj().T @ b
    one = np.eye(LEVELS**2, dtype=complex)
    H = (1.0 * n_a + 1.2 * n_b + 0.005 * (n_a @ (n_a - one) + n_b @ (n_b - one))
         + 1e-3 * (a.conj().T @ b + b.conj().T @ a))
    mu_plus = a.conj().T + b.conj().T
    mu = mu_plus + mu_plus.conj().T
    drop = np.zeros((LEVELS, LEVELS), dtype=complex)
    drop[0, 1] = 1.0
    a_drop, b_drop = np.kron(drop, identity), np.kron(identity, drop)
    channels = [(n_a, 1e-2), (n_b, 1e-2), (a, 1e-3), (b, 1e-3), (a_drop.conj().T @ b_drop, k_transfer),
                (a_drop, 1e-4), (b_drop, 1e-4)]
    _, vectors = np.linalg.eigh(H)
    e_a, e_b = vectors[:, 1], vectors[:, 2]                        # the one-quantum eigenstates
    projector = np.outer(e_a, e_a.conj()) + np.outer(e_b, e_b.conj())
    return EigenbasisKModel(
        H, mu, c_ops_raw=channels, j_plus_array=mu_plus, j_minus_array=mu_plus.conj().T, detection_op_array=mu,
        observable_op_arrays={"polarization": mu, "N_exc": n_a + n_b, "A_PL": projector})


def spectrum(model, observables):
    solver = SpectroscopySolver(backend="dense", eta=ETA)
    solver.feed_model(model)
    return solver.generate_spectrum(PROTOCOL, AXES, pathways=REPHASING, fixed_coordinates={"t2": 10.0},
                                    observables=observables)


def total(result, name):
    return sum(result.observables[name][pathway.name] for pathway in REPHASING)


@pytest.fixture(scope="module")
def result():
    both = ("Bu", "Kd")
    return spectrum(two_mode_model(), {
        "polarization": "polarization",
        "action_N": ObservableSpec.action("action_N", fourth_interaction=both, operator="N_exc"),
        "action_N_bu": ObservableSpec.action("action_N_bu", fourth_interaction="Bu", operator="N_exc"),
        "action_PL": ObservableSpec.action("action_PL", fourth_interaction=both, operator="A_PL"),
    })


def relative_difference(signal, polarization):
    return np.max(np.abs(signal + 1j * polarization)) / np.max(np.abs(polarization))


def test_excitation_number_action_signal_is_minus_i_times_the_polarization_for_each_pathway(result):
    for pathway in REPHASING:
        polarization = result.observables["polarization"][pathway.name]
        assert relative_difference(result.observables["action_N"][pathway.name], polarization) < 1e-11


def test_the_identity_survives_a_change_of_the_dissipation():
    changed = spectrum(two_mode_model(k_transfer=2e-2), {
        "polarization": "polarization",
        "action_N": ObservableSpec.action("action_N", fourth_interaction=("Bu", "Kd"), operator="N_exc"),
    })
    assert relative_difference(total(changed, "action_N"), total(changed, "polarization")) < 1e-11


def test_a_single_readout_alternative_breaks_the_identity(result):
    assert relative_difference(total(result, "action_N_bu"), total(result, "polarization")) > 0.1


def test_a_projector_on_the_one_quantum_states_is_not_the_coherent_signal(result):
    assert relative_difference(total(result, "action_PL"), total(result, "polarization")) > 0.1
