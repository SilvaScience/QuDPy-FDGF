"""Consistency checks of the supplied operators: Hermitian Hamiltonian, J_minus = J_plus^dagger."""
import warnings

import numpy as np
import pytest

from helpers import H, MU
from qudpy_fdgf import (
    EigenbasisKModel,
    FrequencyPathway,
    ModelConsistencyWarning,
    ModelContractError,
    PropagationInterval,
    PureState,
    SpectroscopyProtocol,
    SpectroscopySolver,
    ExcitationSectorModel,
)

BACKENDS = ("dense", "sparse_sector")


class MinimalModel:
    """One-sector, two-level SectorModel written directly against the contract."""

    def __init__(self, hamiltonian, lowering=None):
        self.hamiltonian = np.asarray(hamiltonian, dtype=complex)
        self.raising = np.array([[0, 0], [1, 0]], dtype=complex)
        self.lowering = self.raising.conj().T if lowering is None else np.asarray(lowering, dtype=complex)

    def sectors(self):
        return ("s",)

    def dimension(self, sector):
        return 2

    def hamiltonian_blocks(self, source):
        return {"s": self.hamiltonian}

    def transition_blocks(self, operator_name, direction, source):
        return {"s": self.raising if direction == "plus" else self.lowering}

    def observable_blocks(self, observable_name, source):
        return {"s": self.raising + self.lowering}

    def initial_condition(self, context=None):
        return PureState("s", np.array([1.0, 0.0], dtype=complex))


LINEAR = FrequencyPathway("lin", interactions=["Ku"], component="linear")
PROTOCOL = SpectroscopyProtocol(intervals=[PropagationInterval("omega", "frequency")])


def run_linear(model, backend, **options):
    solver = SpectroscopySolver(backend=backend, eta=0.02, check_stationarity="off", **options)
    solver.feed_model(model)
    return solver.generate_spectrum(PROTOCOL, {"omega": np.array([2.0])}, pathways=[LINEAR])


@pytest.mark.parametrize("backend", BACKENDS)
def test_hermitian_hamiltonian_is_accepted_silently(backend):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        run_linear(MinimalModel(np.diag([0.0, 2.0])), backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_non_hermitian_hamiltonian_is_rejected(backend):
    solver = SpectroscopySolver(backend=backend, eta=0.02, check_stationarity="off")
    with pytest.raises(ModelContractError, match="not Hermitian"):
        solver.feed_model(MinimalModel([[0.0, 1.0], [0.0, 2.0]]))


def test_hermiticity_tolerance_is_adjustable():
    nearly = np.array([[0.0, 1e-7], [0.0, 2.0]])           # |H - H^dagger| ~ 1e-7
    with pytest.raises(ModelContractError):
        SpectroscopySolver(backend="dense", eta=0.02, check_stationarity="off").feed_model(MinimalModel(nearly))
    SpectroscopySolver(backend="dense", eta=0.02, check_stationarity="off",
                       hermiticity_tolerance=1e-3).feed_model(MinimalModel(nearly))


@pytest.mark.parametrize("backend", BACKENDS)
def test_mismatched_lowering_blocks_warn_once(backend):
    model = MinimalModel(np.diag([0.0, 2.0]), lowering=2.0 * np.array([[0, 1], [0, 0]]))
    solver = SpectroscopySolver(backend=backend, eta=0.02, check_stationarity="off")
    solver.feed_model(model)
    with pytest.warns(ModelConsistencyWarning, match="not the adjoint") as record:
        for omega in (1.9, 2.0, 2.1):
            solver.generate_spectrum(PROTOCOL, {"omega": np.array([omega])}, pathways=[LINEAR])
    assert len([w for w in record if issubclass(w.category, ModelConsistencyWarning)]) == 1


def test_explicit_blocks_that_are_adjoint_are_silent_and_mismatch_warns():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        solver = SpectroscopySolver(backend="dense", eta=0.02, check_stationarity="off")
        solver.feed_model(EigenbasisKModel(H, MU, j_plus_array=np.array([[0, 0], [1, 0]], complex),
                                           j_minus_array=np.array([[0, 1], [0, 0]], complex)))
        solver.generate_spectrum(PROTOCOL, {"omega": np.array([2.0])}, pathways=[LINEAR])
    solver = SpectroscopySolver(backend="dense", eta=0.02, check_stationarity="off")
    solver.feed_model(EigenbasisKModel(H, MU, j_plus_array=np.array([[0, 0], [1, 0]], complex),
                                       j_minus_array=np.array([[0, 1], [0, 0]], complex) * 0.5))
    with pytest.warns(ModelConsistencyWarning):
        solver.generate_spectrum(PROTOCOL, {"omega": np.array([2.0])}, pathways=[LINEAR])


def test_provided_adapters_pass_both_checks_silently():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for backend in BACKENDS:
            solver = SpectroscopySolver(backend=backend, eta=0.02, check_stationarity="off")
            solver.feed_model(EigenbasisKModel(H, MU))                       # automatic energy decomposition
            solver.generate_spectrum(PROTOCOL, {"omega": np.array([2.0])}, pathways=[LINEAR])
            sectors = ExcitationSectorModel({0: np.zeros((1, 1)), 1: np.array([[1.0]])}, {(1, 0): np.array([[1.0]])})
            solver = SpectroscopySolver(backend=backend, eta=0.02, check_stationarity="off")
            solver.feed_model(sectors)
            solver.generate_spectrum(PROTOCOL, {"omega": np.array([1.0])}, pathways=[LINEAR])
