"""Fifth-order double-quantum response of the closed two-mode model of Example 4.

The Hamiltonian of Example 4 without dissipation (omega 1.0 and 1.2, J = 1 meV, anharmonicity 10 meV),
on the states with n_a + n_b <= N_max. Three exact properties, none of which depends on how the spectrum looks:

- with the anharmonicity set to zero the seven UFSS pathways cancel, which needs the complete three-excitation
  manifold (the truncation at two excitations leaves the missing pathway, three levels per mode leave a residue);
- with the anharmonicity the cancellation is lifted;
- for J = 0 the response of the two modes is the sum of the responses of two independent one-mode models.

The same checks, with figures, are in validation/analysis/example4_chi5_analysis.ipynb.
"""
import numpy as np
import pytest

from qudpy_fdgf import EigenbasisKModel, SpectroscopySolver, standard_nq_protocol, translate_ufss_diagrams

ufss = pytest.importorskip("ufss")

OMEGA = (1.0, 1.2)
ETA = 2e-3
AXES = {"omega_2q": np.linspace(-2.46, -1.96, 5), "omega_emit": np.linspace(0.96, 1.26, 5)}
DELAYS = {"t1": 2.0, "t3": 2.0, "t4": 2.0}
PROTOCOL = standard_nq_protocol(order=2, nq_interval=2, detection_interval=5, n_interactions=5,
                                nq_axis="omega_2q", detection_axis="omega_emit")


def fifth_order_pathways(max_manifold):
    generator = ufss.DiagramGenerator(detection_type="polarization")
    generator.set_phase_discrimination([(0, 1), (0, 1), (1, 0), (1, 0), (1, 0)])
    generator.maximum_manifold = max_manifold
    generator.efield_times = [np.array([0.0, 0.0])] * 5
    return translate_ufss_diagrams(generator.get_diagrams(np.arange(5, dtype=float)), component="chi5_2q")


def two_mode_model(states, coupling=1e-3, anharmonicity=1e-2):
    """Closed model on the basis states (n_a, n_b) listed in ``states`` (ordered by manifold)."""
    index = {state: k for k, state in enumerate(states)}
    dimension = len(states)
    a_dagger, b_dagger = np.zeros((dimension, dimension)), np.zeros((dimension, dimension))
    for (i, j), k in index.items():
        if (i + 1, j) in index:
            a_dagger[index[(i + 1, j)], k] = np.sqrt(i + 1)
        if (i, j + 1) in index:
            b_dagger[index[(i, j + 1)], k] = np.sqrt(j + 1)
    n_a = np.array([i for i, _ in states], dtype=float)
    n_b = np.array([j for _, j in states], dtype=float)
    H = np.diag(OMEGA[0] * n_a + OMEGA[1] * n_b + 0.5 * anharmonicity * (n_a * (n_a - 1) + n_b * (n_b - 1)))
    H = H + coupling * (a_dagger @ b_dagger.T + b_dagger @ a_dagger.T)
    J_plus = a_dagger + b_dagger
    mu = J_plus + J_plus.T
    return EigenbasisKModel(H.astype(complex), mu.astype(complex), j_plus_array=J_plus.astype(complex),
                            j_minus_array=J_plus.T.astype(complex), detection_op_array=mu.astype(complex),
                            observable_op_arrays={"polarization": mu.astype(complex)})


def by_total(max_manifold):
    return sorted(((i, j) for i in range(max_manifold + 1) for j in range(max_manifold + 1) if i + j <= max_manifold),
                  key=lambda s: (sum(s), -s[0]))


def spectrum(model, max_manifold):
    solver = SpectroscopySolver(backend="dense", eta=ETA, max_cache_entries=4096)
    solver.feed_model(model)
    pathways = fifth_order_pathways(max_manifold)
    result = solver.generate_nq_spectrum(2, PROTOCOL, axes=AXES, fixed_coordinates=DELAYS, pathways=pathways)
    largest = max(np.abs(values).max() for values in result.pathways.values())
    return result.components["chi5_2q"], largest, len(pathways)


def test_seven_pathways_cancel_for_harmonic_modes():
    total, largest, count = spectrum(two_mode_model(by_total(3), anharmonicity=0.0), 3)
    assert count == 7 and largest > 0.0
    assert np.abs(total).max() / largest < 1e-10


def test_the_cancellation_needs_three_excitations():
    # at two excitations the missing pathway is exactly minus the sum of the six that remain
    total, largest, count = spectrum(two_mode_model(by_total(2), anharmonicity=0.0), 2)
    assert count == 6
    assert np.abs(total).max() / largest > 0.1
    # three levels per mode keep |2,1>, |1,2>, |2,2> but not |3,0> and |0,3>
    per_mode = sorted(((i, j) for i in range(3) for j in range(3)), key=lambda s: (sum(s), -s[0]))
    total, largest, _ = spectrum(two_mode_model(per_mode, anharmonicity=0.0), 3)
    assert np.abs(total).max() / largest > 1e-2


def test_anharmonicity_lifts_the_cancellation():
    total, largest, _ = spectrum(two_mode_model(by_total(3)), 3)
    assert np.abs(total).max() / largest > 1e-2


def test_independent_modes_add():
    two_modes, _, _ = spectrum(two_mode_model(by_total(3), coupling=0.0), 3)
    one_mode_sum = 0
    for index in (0, 1):
        # one mode: the other mode never leaves its ground level (n = 0), so its states are the ladder of mode `index`
        ladder = [(n, 0) if index == 0 else (0, n) for n in range(4)]
        model = two_mode_model(ladder, coupling=0.0)
        one_mode_sum = one_mode_sum + spectrum(model, 3)[0]
    assert np.abs(two_modes - one_mode_sum).max() / np.abs(two_modes).max() < 1e-10
