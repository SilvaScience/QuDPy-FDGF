"""Time-domain calculation with Gaussian pulses of finite duration (validation/pulses.py).

The engine behind the study of the impulsive limit and of the rotating-wave approximation (validation/analysis/example1_pulses.ipynb)
is checked on a closed three-level ladder (ground state, one excitation at 1.0, two at 2.05):

- the hierarchy of the multilinear terms agrees with a direct, non-perturbative propagation with phase cycling and
  amplitude extrapolation, with the rotating-wave field and with the real field, with overlapping pulses;
- for pulses much shorter than the dynamics, the rotating-wave field reproduces the impulsive response of QuDPy-FDGF
  (the eight rephasing sequences, ground state), up to the area of the pulses cubed.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

from qudpy_fdgf import EigenbasisKModel, SpectroscopySolver, pathways_from_signature, standard_nq_protocol

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "validation"))
import pulses  # noqa: E402

ENERGIES = np.array([0.0, 1.0, 2.05])
RAISING = np.array([[0, 0, 0], [1.0, 0, 0], [0, 0.8, 0]], dtype=complex)           # |e><g| + 0.8 |f><e|
MU = RAISING + RAISING.conj().T
RHO0 = np.diag([1.0, 0.0, 0.0]).astype(complex)


def test_hierarchy_agrees_with_phase_cycled_propagation():
    for rwa in (True, False):
        hierarchy = pulses.Hierarchy(ENERGIES, MU, RAISING, RHO0, sigma=0.5, carrier=1.0, rwa=rwa, dt=0.02)
        times_1, trajectory = hierarchy.first_order()
        times, values, _, _ = hierarchy.polarization(1.0, 0.0, times_1, trajectory)
        cycled_times, cycled = pulses.phase_cycled_polarization(ENERGIES, MU, RAISING, RHO0, 0.5, 1.0, rwa, 1.0,
                                                                amplitude=0.01, dt=0.02)
        common = values[np.isin(np.round(times, 6), np.round(cycled_times, 6))]
        assert common.size == cycled.size
        assert np.max(np.abs(common - cycled)) / np.max(np.abs(common)) < 1e-5


def test_short_rwa_pulses_give_the_impulsive_response():
    eta, sigma = 0.1, 0.02
    omega_1 = np.linspace(-1.2, -0.8, 5)
    omega_3 = np.linspace(0.8, 1.2, 5)
    model = EigenbasisKModel(np.diag(ENERGIES).astype(complex), MU, j_plus_array=RAISING, j_minus_array=RAISING.conj().T,
                             detection_op_array=MU, observable_op_arrays={"polarization": MU})
    solver = SpectroscopySolver(backend="dense", eta=eta)
    solver.feed_model(model)
    protocol = standard_nq_protocol(order=1, n_interactions=3, nq_interval=1, detection_interval=3,
                                    nq_axis="omega_1q", detection_axis="omega_emit")
    reference = solver.generate_spectrum(protocol, {"omega_1q": omega_1, "omega_emit": omega_3},
                                         pathways=pathways_from_signature("-++", component="rephasing"),
                                         fixed_coordinates={"t2": 0.0}).components["rephasing"]
    signal = pulses.rephasing_map(ENERGIES, MU, RAISING, RHO0, sigma, 1.0, True, omega_1, omega_3, eta, dt=0.005)
    scale, error = pulses.scale_and_error(signal, reference)
    area = sigma * np.sqrt(2 * np.pi)
    assert error < 5e-3
    assert abs(abs(scale) / area**3 - 1.0) < 1e-2
