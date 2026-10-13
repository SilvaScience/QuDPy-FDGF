"""Integrated jump observables on an infinite window: ``time_window=(t0, np.inf)``.

The count rate tends to its stationary value ``r_ss = Tr(M rho_ss)`` times the trace of the state, so the
integral over an infinite window converges only above that background. The solver integrates the
excess, ``int_{t0}^inf [Tr(M e^{L s} rho) - r_ss Tr(rho)] ds``; when the stationary state has no
jumps (zero temperature, decay to the ground state) ``r_ss = 0`` and the result is the total count, the
quantum yield of the excitation.

Checks, on dense and sparse backends: the radiative yield of an open two-level system is exactly 1; a
non-counted channel gives the branching ratio gamma_r / (gamma_r + gamma_nr); a window starting at t0
multiplies the yield by exp(-Gamma t0); the limit of a long finite window; the background subtraction
with an uphill channel; and the errors (trapezoid rule, closed model, invalid windows).
"""
import numpy as np
import pytest

from qudpy_fdgf import EigenbasisKModel, ObservableSpec, SpectroscopySolver
from qudpy_fdgf.exceptions import ConvergenceError

from helpers import BACKENDS, GAMMA_1, GAMMA_PHI, H, L_RAD, MU, P_E, PROTOCOL, R1, SIGMA_Z

AXES = {"w1": np.array([-2.0, -1.98]), "w3": np.array([2.0, 2.02])}
FIXED = {"t2": 10.0}
L_UP = L_RAD.conj().T                                  # |e><g|


def solver_for(channels, backend):
    model = EigenbasisKModel(H, MU, c_ops_raw=channels, observable_op_arrays={"polarization": MU, "excited_population": P_E})
    solver = SpectroscopySolver(backend=backend, eta=0.002, check_stationarity="off")
    solver.feed_model(model)
    return solver


def counts(solver, window, channel="c_op_0"):
    observables = {
        "pop": ObservableSpec.action("pop", fourth_interaction="Bu", operator="excited_population"),
        "fl": ObservableSpec.mean_jump("fl", channel, time_window=window, fourth_interaction="Bu"),
    }
    result = solver.generate_spectrum(PROTOCOL, AXES, pathways=(R1,), fixed_coordinates=FIXED, observables=observables,
                                      warn_null=False)
    return result.observables["fl"]["R1"] / result.observables["pop"]["R1"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_radiative_yield_of_an_open_two_level_system_is_one(backend):
    solver = solver_for(((L_RAD, GAMMA_1), (SIGMA_Z, GAMMA_PHI / 2)), backend)
    assert np.max(np.abs(counts(solver, (0.0, np.inf)) - 1.0)) < 1e-9


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_channel_that_is_not_counted_gives_the_branching_ratio(backend):
    nonradiative = 0.015
    solver = solver_for(((L_RAD, GAMMA_1), (L_RAD, nonradiative), (SIGMA_Z, GAMMA_PHI / 2)), backend)
    expected = GAMMA_1 / (GAMMA_1 + nonradiative)
    assert np.max(np.abs(counts(solver, (0.0, np.inf), "c_op_0") - expected)) < 1e-9
    assert np.max(np.abs(counts(solver, (0.0, np.inf), "c_op_1") - (1.0 - expected))) < 1e-9


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_window_that_starts_later_keeps_the_surviving_fraction(backend):
    start = 30.0
    solver = solver_for(((L_RAD, GAMMA_1), (SIGMA_Z, GAMMA_PHI / 2)), backend)
    assert np.max(np.abs(counts(solver, (start, np.inf)) - np.exp(-GAMMA_1 * start))) < 1e-9


@pytest.mark.parametrize("backend", BACKENDS)
def test_infinite_window_is_the_limit_of_long_finite_windows(backend):
    solver = solver_for(((L_RAD, GAMMA_1), (SIGMA_Z, GAMMA_PHI / 2)), backend)
    finite = counts(solver, (5.0, 5.0 + 60.0 / GAMMA_1))
    assert np.max(np.abs(counts(solver, (5.0, np.inf)) - finite)) < 1e-9


@pytest.mark.parametrize("backend", BACKENDS)
def test_stationary_background_is_subtracted(backend):
    """An uphill channel counted at its stationary rate: the long-window excess over the background is the infinite-window value."""
    up_rate = 0.004
    solver = solver_for(((L_RAD, GAMMA_1), (L_UP, up_rate)), backend)
    base = solver.backend
    start, stop = 5.0, 5.0 + 150.0 / (GAMMA_1 + up_rate)
    infinite = ObservableSpec.mean_jump("inf", "c_op_1", time_window=(start, np.inf))
    long = ObservableSpec.mean_jump("long", "c_op_1", time_window=(start, stop))
    rho_ss = base.steady_state()[0].reshape(-1, order="F")
    rate_ss = np.vdot(base.detector_vector(infinite.instantaneous()), rho_ss)          # Tr(M rho_ss)
    assert abs(rate_ss) > 1e-4                                                          # the background is not zero
    identity = np.eye(base.layout.total_dimension, dtype=complex).reshape(-1, order="F")
    rng = np.random.default_rng(11)
    state = rng.standard_normal(identity.size) + 1j * rng.standard_normal(identity.size)
    excess = np.vdot(base.detector_vector(long), state) - rate_ss * np.vdot(identity, state) * (stop - start)
    assert np.vdot(base.detector_vector(infinite), state) == pytest.approx(excess, rel=1e-8, abs=1e-8)


def test_trapezoid_rule_is_rejected_on_an_infinite_window():
    with pytest.raises(ValueError, match="trapezoid"):
        ObservableSpec.mean_jump("fl", "c_op_0", time_window=(0.0, np.inf), integration="trapezoid")


@pytest.mark.parametrize("window", [(np.inf, np.inf), (-1.0, np.inf), (np.nan, np.inf), (5.0, 1.0)])
def test_invalid_windows_are_rejected(window):
    with pytest.raises(ValueError):
        ObservableSpec.mean_jump("fl", "c_op_0", time_window=window)


def test_a_generator_without_a_unique_stationary_state_is_rejected_by_the_dense_backend():
    solver = solver_for(((SIGMA_Z, GAMMA_PHI / 2),), "dense")        # dephasing only: every population is stationary
    observable = ObservableSpec.mean_jump("fl", "c_op_0", time_window=(0.0, np.inf), fourth_interaction="Bu")
    with pytest.raises((ConvergenceError, ValueError), match="stationary"):
        solver.generate_spectrum(PROTOCOL, AXES, pathways=(R1,), fixed_coordinates=FIXED, observables={"fl": observable},
                                 warn_null=False)
