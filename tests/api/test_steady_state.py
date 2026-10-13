"""steady_state(): the stationary state of the generator, and feed_model(initial_state=...)."""
import warnings

import numpy as np
import pytest

from helpers import GAMMA_1, GAMMA_PHI, H, L_RAD, MU, SIGMA_Z
from qudpy_fdgf import (
    DenseDensityBlock,
    DensityState,
    EigenbasisKModel,
    ExcitationSectorModel,
    FrequencyPathway,
    PropagationInterval,
    PureState,
    SpectroscopyProtocol,
    SpectroscopySolver,
    StationarityWarning,
    SteadyStateWarning,
    ThermodynamicContext,
)
from qudpy_fdgf.exceptions import ConvergenceError, ModelContractError

BACKENDS = ("dense", "sparse_sector")
H_MEV = np.diag([0.0, 2.0]).astype(complex)
L_UP = L_RAD.conj().T
K_B_MEV = 0.08617333262
ETA = 0.002


def solver_for(backend, model, **options):
    options = {"krylov_tolerance": 1e-12, **options} if backend == "sparse_sector" else options
    solver = SpectroscopySolver(backend=backend, eta=ETA, check_stationarity="off", **options)
    solver.feed_model(model)
    return solver


def matrix_of(solver, state):
    return solver.backend._assemble_density_state(state)


def two_level(channels):
    return EigenbasisKModel(H, MU, c_ops_raw=channels)


def detailed_balance_model(temperature, rate=0.2, with_dephasing=True):
    """2 meV transition with radiative decay and the thermal excitation that satisfies detailed balance."""
    up = np.exp(-2.0 / (K_B_MEV * temperature))
    channels = [(L_RAD, rate), (L_UP, rate * up)]
    if with_dephasing:
        channels.append((SIGMA_Z, 0.05))
    return EigenbasisKModel(H_MEV, 0.5 * np.array([[0, 1], [1, 0]], complex), c_ops_raw=tuple(channels),
                            energy_unit="meV")


@pytest.mark.parametrize("backend", BACKENDS)
def test_radiative_decay_leaves_the_ground_state(backend):
    solver = solver_for(backend, two_level(((L_RAD, GAMMA_1), (SIGMA_Z, GAMMA_PHI / 2))))
    state = solver.steady_state()
    assert matrix_of(solver, state) == pytest.approx(np.diag([1.0, 0.0]), abs=1e-12)
    info = state.metadata["steady_state"]
    assert info["unique"] is (True if backend == "dense" else None)
    assert info["relative_residual"] < 1e-12 and info["hermiticity_error"] < 1e-12
    assert info["method"] == ("lu" if backend == "dense" else "gmres")


@pytest.mark.parametrize("backend", BACKENDS)
def test_detailed_balance_gives_the_gibbs_state(backend):
    temperature = 10.0
    solver = solver_for(backend, detailed_balance_model(temperature))
    rho = matrix_of(solver, solver.steady_state())
    weight = np.exp(-2.0 / (K_B_MEV * temperature))
    assert rho == pytest.approx(np.diag([1.0, weight]) / (1.0 + weight), abs=1e-10)
    gibbs = SpectroscopySolver(backend="dense", eta=ETA)
    gibbs.feed_model(detailed_balance_model(temperature), context=ThermodynamicContext(temperature=temperature))
    assert rho == pytest.approx(gibbs.backend._initial_density_matrix, abs=1e-10)


@pytest.mark.parametrize("backend", BACKENDS)
def test_populations_balance_the_up_and_down_rates_without_detailed_balance(backend):
    down, up = 0.3, 0.1
    solver = solver_for(backend, two_level(((L_RAD, down), (L_UP, up))))
    rho = matrix_of(solver, solver.steady_state())
    assert rho.real[1, 1] == pytest.approx(up / (up + down), abs=1e-10)
    assert np.trace(rho) == pytest.approx(1.0, abs=1e-12)
    assert rho == pytest.approx(rho.conj().T, abs=1e-12)
    assert np.linalg.eigvalsh(rho)[0] > -1e-12


@pytest.mark.parametrize("backend", BACKENDS)
def test_closed_model_has_no_unique_stationary_state(backend):
    solver = solver_for(backend, two_level(()))
    with pytest.warns(SteadyStateWarning, match="not one-dimensional"):
        state = solver.steady_state(check_unique=True)
    assert state.metadata["steady_state"]["unique"] is False
    assert matrix_of(solver, state) == pytest.approx(np.eye(2) / 2, abs=1e-10)     # the maximally mixed element


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_decoupled_level_without_relaxation_is_detected(backend):
    energies = np.diag([0.0, 2.0, 5.0]).astype(complex)
    dipole = np.array([[0, 0.5, 0], [0.5, 0, 0], [0, 0, 0]], dtype=complex)
    decay = np.zeros((3, 3), dtype=complex)
    decay[0, 1] = 1.0
    solver = solver_for(backend, EigenbasisKModel(energies, dipole, c_ops_raw=((decay, 0.1),)))
    with pytest.warns(SteadyStateWarning, match="not unique"):
        solver.steady_state(check_unique=True)


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_unique_state_is_silent_and_does_not_depend_on_the_reference_state(backend):
    model = detailed_balance_model(10.0)
    plain = solver_for(backend, model)
    thermal = SpectroscopySolver(backend=backend, eta=ETA, check_stationarity="off")
    thermal.feed_model(model, context=ThermodynamicContext(temperature=40.0))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        first, second = plain.steady_state(check_unique=True), thermal.steady_state(check_unique=True)
    assert matrix_of(plain, first) == pytest.approx(matrix_of(thermal, second), abs=1e-10)


@pytest.mark.parametrize("backend", BACKENDS)
def test_sector_blocks_of_the_state_follow_the_model(backend):
    hamiltonians = {"0": np.zeros((1, 1)), "1": np.array([[2.0]])}
    raising = {("1", "0"): np.array([[1.0]])}
    lowering = {("0", "1"): np.array([[1.0]])}
    model = ExcitationSectorModel(hamiltonians, raising, c_ops_raw=((lowering, 0.1),), initial_sector="0")
    state = solver_for(backend, model).steady_state()
    assert set(state.blocks) == {("0", "0")}
    assert state.blocks[("0", "0")].as_matrix() == pytest.approx(np.array([[1.0]]), abs=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_imposing_the_steady_state_cures_the_stationarity_warning(backend):
    """A thermal state with a decay channel that breaks detailed balance is not stationary; the steady state is."""
    model = EigenbasisKModel(H_MEV, 0.5 * np.array([[0, 1], [1, 0]], complex), c_ops_raw=((L_RAD, 0.2),),
                             energy_unit="meV")
    context = ThermodynamicContext(temperature=10.0)
    solver = SpectroscopySolver(backend=backend, eta=ETA)
    with pytest.warns(StationarityWarning):
        solver.feed_model(model, context=context)
    for imposed in ("steady", None):
        cured = SpectroscopySolver(backend=backend, eta=ETA)
        state = solver.steady_state() if imposed is None else "steady"
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cured.feed_model(model, context=context, initial_state=state)
        assert cured.stationarity_residual() < 1e-10
        assert cured.backend._initial_density_matrix == pytest.approx(np.diag([1.0, 0.0]), abs=1e-10)


@pytest.mark.parametrize("backend", BACKENDS)
def test_linear_response_scales_with_the_stationary_ground_population(backend):
    down, up = 0.3, 0.1
    model = two_level(((L_RAD, down), (L_UP, up), (SIGMA_Z, 0.015)))
    protocol = SpectroscopyProtocol(intervals=[PropagationInterval("omega", "frequency")])
    linear = FrequencyPathway("lin", interactions=["Ku"], component="linear")
    axes = {"omega": np.array([2.0])}
    ground = DensityState(blocks={("k0", "k0"): DenseDensityBlock(np.diag([1.0, 0.0]))})
    values = {}
    for label, reference in (("steady", "steady"), ("ground", ground)):
        solver = SpectroscopySolver(backend=backend, eta=ETA, check_stationarity="off")
        solver.feed_model(model, initial_state=reference)               # same channels: only the reference differs
        values[label] = solver.generate_spectrum(protocol, axes, pathways=[linear], warn_null=False).pathways["lin"][0]
    assert abs(values["steady"] / values["ground"]) == pytest.approx(down / (up + down), rel=1e-9)


class StateFreeModel:
    """Model that must not be asked for its reference state."""

    def __init__(self, model):
        self._model = model

    def __getattr__(self, name):
        if name in ("initial_condition", "equilibrium_state", "initial_state"):
            raise AssertionError(f"{name} was requested")
        return getattr(self._model, name)


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_model_is_not_asked_for_a_state_that_is_imposed(backend):
    model = StateFreeModel(two_level(((L_RAD, 0.3), (L_UP, 0.1))))
    pure = PureState("k0", np.array([0.0, 1.0]))
    solver = SpectroscopySolver(backend=backend, eta=ETA, check_stationarity="off")
    solver.feed_model(model, initial_state=pure)
    assert solver.backend._initial_is_pure
    assert solver.backend._initial_density_matrix == pytest.approx(np.diag([0.0, 1.0]))
    solver = SpectroscopySolver(backend=backend, eta=ETA)
    solver.feed_model(model, initial_state="steady")
    assert not solver.backend._initial_is_pure
    assert solver.stationarity_residual() < 1e-10


def test_invalid_initial_states_are_rejected():
    model = two_level(((L_RAD, 0.3),))
    solver = SpectroscopySolver(backend="dense", eta=ETA)
    for bad in (np.eye(2), "ground", 3):
        with pytest.raises(TypeError, match="initial_state"):
            solver.feed_model(model, initial_state=bad)
    with pytest.raises(ModelContractError, match="unit trace"):
        solver.feed_model(model, initial_state=DensityState(blocks={("k0", "k0"): DenseDensityBlock(np.diag([0.5, 0.0]))}))


def test_steady_state_needs_a_loaded_model():
    with pytest.raises(RuntimeError, match="feed_model"):
        SpectroscopySolver(backend="dense", eta=ETA).steady_state()


def test_the_sparse_backend_does_not_test_uniqueness_unless_asked():
    solver = solver_for("sparse_sector", two_level(((L_RAD, 0.3), (L_UP, 0.1))))
    assert solver.steady_state().metadata["steady_state"]["unique"] is None
    assert "second_solve_difference" in solver.steady_state(check_unique=True).metadata["steady_state"]
    dense = solver_for("dense", two_level(((L_RAD, 0.3), (L_UP, 0.1))))
    assert dense.steady_state().metadata["steady_state"]["unique"] is True          # free on the dense backend


def test_a_solve_that_does_not_converge_raises_an_error_that_names_the_remedies():
    """A tolerance below machine precision cannot be reached: the solver gives up after krylov_maxiter cycles."""
    solver = solver_for("sparse_sector", two_level(((L_RAD, 0.3), (L_UP, 0.1))), krylov_maxiter=2, krylov_tolerance=1e-30)
    with pytest.raises(ConvergenceError, match="initial_state") as error:
        solver.steady_state()
    assert "dense backend" in str(error.value) and "krylov_maxiter" in str(error.value)
