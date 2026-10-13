"""Check of the reference state: ||L rho|| / ||rho|| must vanish, or the solver says so."""
import warnings

import numpy as np
import pytest

from helpers import GAMMA_1, H, L_RAD, MU, SIGMA_Z, three_site_chain
from qudpy_fdgf import (
    DenseDensityBlock,
    DensityState,
    EigenbasisKModel,
    FrequencyPathway,
    PropagationInterval,
    SpectroscopyProtocol,
    SpectroscopySolver,
    StationarityError,
    StationarityWarning,
    ThermodynamicContext,
)

BACKENDS = ("dense", "sparse_sector")
# toy model in meV: transition 2 meV, radiative decay 0.2 meV; at 10 K the Gibbs state has 9 % of excited
# population that the channel empties, so it is not stationary.
H_MEV = np.diag([0.0, 2.0]).astype(complex)
K_BOLTZMANN_MEV = 0.08617333262


def thermal_decaying_model():
    return EigenbasisKModel(H_MEV, 0.5 * np.array([[0, 1], [1, 0]], complex),
                            c_ops_raw=((L_RAD, 0.2),), boltzmann_constant=K_BOLTZMANN_MEV)


def test_ground_state_with_decay_is_stationary_and_silent():
    for backend in BACKENDS:
        solver = SpectroscopySolver(backend=backend, eta=0.002)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            solver.feed_model(EigenbasisKModel(H, MU, c_ops_raw=((L_RAD, GAMMA_1), (SIGMA_Z, 0.015))))
        assert solver.stationarity_residual() < 1e-14


def test_gibbs_state_of_a_closed_model_is_stationary():
    solver = SpectroscopySolver(backend="sparse_sector", eta=0.01)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        solver.feed_model(three_site_chain(dissipative=False), context=ThermodynamicContext(temperature=600.0))
    assert solver.stationarity_residual() < 1e-14


@pytest.mark.parametrize("backend", BACKENDS)
def test_gibbs_state_with_a_decay_channel_warns(backend):
    solver = SpectroscopySolver(backend=backend, eta=0.02)
    with pytest.warns(StationarityWarning, match="not stationary"):
        solver.feed_model(thermal_decaying_model(), context=ThermodynamicContext(temperature=10.0))
    # populations 0.9106 / 0.0894 and rate 0.2: ||L rho|| / ||rho|| = 0.028
    assert solver.stationarity_residual() == pytest.approx(0.028, abs=2e-3)


def test_both_backends_report_the_same_residual():
    residuals = []
    for backend in BACKENDS:
        solver = SpectroscopySolver(backend=backend, eta=0.02, check_stationarity="off")
        solver.feed_model(thermal_decaying_model(), context=ThermodynamicContext(temperature=10.0))
        residuals.append(solver.stationarity_residual())
    assert residuals[0] == pytest.approx(residuals[1], rel=1e-12)


def test_error_mode_raises_and_off_mode_is_silent():
    context = ThermodynamicContext(temperature=10.0)
    with pytest.raises(StationarityError, match="detailed balance"):
        SpectroscopySolver(backend="dense", eta=0.02, check_stationarity="error").feed_model(
            thermal_decaying_model(), context=context)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        SpectroscopySolver(backend="dense", eta=0.02, check_stationarity="off").feed_model(
            thermal_decaying_model(), context=context)


def test_tolerance_is_adjustable():
    context = ThermodynamicContext(temperature=10.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        SpectroscopySolver(backend="dense", eta=0.02, stationarity_tolerance=0.1).feed_model(
            thermal_decaying_model(), context=context)


def test_invalid_mode_is_rejected():
    with pytest.raises(ValueError, match="check_stationarity"):
        SpectroscopySolver(backend="dense", eta=0.02, check_stationarity="sometimes")


class _CoherentStart(EigenbasisKModel):
    """Closed two-level model prepared in the superposition (|g> + |e>) / sqrt 2: it oscillates."""

    def initial_condition(self, context=None):
        return DensityState(blocks={("k0", "k0"): DenseDensityBlock(0.5 * np.ones((2, 2), complex))})


def test_a_coherent_initial_state_of_a_closed_model_warns():
    solver = SpectroscopySolver(backend="dense", eta=0.02)
    with pytest.warns(StationarityWarning):
        solver.feed_model(_CoherentStart(H, MU))
    assert solver.stationarity_residual() > 0.1


def test_summary_reports_the_residual():
    solver = SpectroscopySolver(backend="dense", eta=0.02, check_stationarity="off")
    solver.feed_model(thermal_decaying_model(), context=ThermodynamicContext(temperature=10.0))
    assert solver.summary()["stationarity_residual"] == pytest.approx(solver.stationarity_residual())


def test_calculation_still_runs_after_a_warning():
    solver = SpectroscopySolver(backend="dense", eta=0.02)
    with pytest.warns(StationarityWarning):
        solver.feed_model(thermal_decaying_model(), context=ThermodynamicContext(temperature=10.0))
    protocol = SpectroscopyProtocol(intervals=[PropagationInterval("omega1", "frequency")])
    result = solver.generate_spectrum(protocol, {"omega1": np.array([2.0])},
                                      pathways=[FrequencyPathway("lin", interactions=["Ku"], component="linear")])
    assert np.isfinite(result.pathways["lin"]).all()
