"""Harmonic-limit cancellation of the fifth-order double-quantum response.

A harmonic oscillator driven linearly has no nonlinear response: the seven UFSS pathways of the
fifth-order 2Q signal must cancel. With an anharmonicity the cancellation is lifted. The
manifold cutoff (three excitations) is the one that closes the pathways, so the test also
checks that the truncation itself creates no spurious response.

The two-mode version of this check is in validation/analysis/example3_analysis.ipynb.
"""
import numpy as np
import pytest

from qudpy_fdgf import EigenbasisKModel, SpectroscopySolver, standard_nq_protocol, translate_ufss_diagrams

ufss = pytest.importorskip("ufss")

OMEGA, MANIFOLD = 1.5, 3
AXES = {"omega_2q": np.linspace(-3.05, -2.95, 5), "omega_emit": np.linspace(1.45, 1.55, 5)}
DELAYS = {"t1": 2.0, "t3": 2.0, "t4": 2.0}


def fifth_order_pathways():
    generator = ufss.DiagramGenerator(detection_type="polarization")
    generator.set_phase_discrimination([(0, 1), (0, 1), (1, 0), (1, 0), (1, 0)])
    generator.maximum_manifold = MANIFOLD
    generator.efield_times = [np.array([0.0, 0.0])] * 5
    diagrams = generator.get_diagrams(np.arange(5, dtype=float))
    return translate_ufss_diagrams(diagrams, component="chi5_2q")


def oscillator_spectrum(anharmonicity):
    """Single oscillator truncated at MANIFOLD excitations; returns (total, largest single pathway)."""
    n = np.arange(MANIFOLD + 1)
    lowering = np.diag(np.sqrt(n[1:]), k=1).astype(complex)
    H = np.diag(OMEGA * n + 0.5 * anharmonicity * n * (n - 1)).astype(complex)
    J_plus = lowering.conj().T
    mu = J_plus + J_plus.conj().T
    model = EigenbasisKModel(H, mu, j_plus_array=J_plus, j_minus_array=J_plus.conj().T,
                             detection_op_array=mu, observable_op_arrays={"polarization": mu})
    solver = SpectroscopySolver(backend="dense", eta=0.008, max_cache_entries=4096)
    solver.feed_model(model)
    pathways = fifth_order_pathways()
    solver.set_pathways(pathways)
    protocol = standard_nq_protocol(order=2, nq_interval=2, detection_interval=5, n_interactions=5,
                                    nq_axis="omega_2q", detection_axis="omega_emit")
    result = solver.generate_nq_spectrum(2, protocol, axes=AXES, fixed_coordinates=DELAYS, pathways=pathways)
    largest = max(np.abs(values).max() for values in result.pathways.values())
    return np.abs(result.components["chi5_2q"]).max(), largest, len(pathways)


def test_seven_pathways_cancel_for_a_harmonic_oscillator():
    total, largest, count = oscillator_spectrum(0.0)
    assert count == 7
    assert largest > 0.0
    assert total / largest < 1e-10


def test_anharmonicity_lifts_the_cancellation():
    total, largest, _ = oscillator_spectrum(-0.02)
    assert total / largest > 1e-2
