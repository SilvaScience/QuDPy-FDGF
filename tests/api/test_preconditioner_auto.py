"""preconditioner="auto" (default): diagonal when the Hamiltonian blocks are diagonal (eigenbasis), none otherwise."""
import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import aslinearoperator

from helpers import H, MU, PROTOCOL, REPHASING, SIGMA_Z, three_site_chain
from qudpy_fdgf import EigenbasisKModel, SpectroscopySolver
from qudpy_fdgf.contracts import CollapseChannel, PureState

ETA = 0.002
AXES = {"w1": np.linspace(-2.1, -1.9, 3), "w3": np.linspace(1.9, 2.1, 3)}


def chosen(model, **options):
    solver = SpectroscopySolver(backend="sparse_sector", eta=ETA, krylov_tolerance=1e-12, check_stationarity="off",
                               **options)
    solver.feed_model(model)
    return solver


def spectrum(solver):
    return solver.generate_spectrum(PROTOCOL, AXES, pathways=REPHASING, fixed_coordinates={"t2": 10.0},
                                    warn_null=False)


class WrappedBlocks:
    """Model whose operator blocks go through ``wrap``; everything else is delegated."""

    def __init__(self, model, wrap):
        self._model, self._wrap = model, wrap

    def __getattr__(self, name):
        return getattr(self._model, name)

    def _wrapped(self, blocks):
        return {target: self._wrap(block) for target, block in dict(blocks).items()}

    def hamiltonian_blocks(self, source):
        return self._wrapped(self._model.hamiltonian_blocks(source))

    def transition_blocks(self, operator_name, direction, source):
        return self._wrapped(self._model.transition_blocks(operator_name, direction, source))

    def observable_blocks(self, observable_name, source):
        return self._wrapped(self._model.observable_blocks(observable_name, source))

    def collapse_channels(self, context=None):
        return [CollapseChannel(c.name, c.rate, {s: self._wrapped(b) for s, b in c.operator_blocks.items()}, c.metadata)
                for c in self._model.collapse_channels(context)]


class SiteBasisModel:
    """Two-level model given in a basis that does not diagonalize its Hamiltonian."""

    def __init__(self, with_channel=False):
        angle = 0.4
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=complex)
        self.hamiltonian = rotation @ np.diag([0.0, 2.0]).astype(complex) @ rotation.conj().T
        self.dipole = rotation @ MU @ rotation.conj().T
        self.with_channel = with_channel

    def sectors(self):
        return ("s",)

    def dimension(self, sector):
        return 2

    def hamiltonian_blocks(self, source):
        return {"s": self.hamiltonian}

    def transition_blocks(self, operator_name, direction, source):
        return {"s": self.dipole}

    def observable_blocks(self, observable_name, source):
        return {"s": self.dipole}

    def collapse_channels(self, context=None):
        return [CollapseChannel("dephasing", 0.01, {"s": {"s": np.diag([-1.0, 1.0]).astype(complex)}})] if self.with_channel else []

    def initial_condition(self, context=None):
        return PureState("s", np.linalg.eigh(self.hamiltonian)[1][:, 0])


def test_closed_model_in_its_eigenbasis_gets_the_diagonal_preconditioner():
    solver = chosen(three_site_chain(dissipative=False))
    assert solver.backend.preconditioner == "diagonal"
    assert solver.backend.preconditioner_option == "auto"
    result = spectrum(solver)
    assert result.diagnostics["solver"]["preconditioner"] == "diagonal"
    assert result.diagnostics["solver"]["iterations_max"] == 1


def test_diagonal_collapse_operators_keep_the_generator_diagonal():
    solver = chosen(EigenbasisKModel(H, MU, c_ops_raw=((SIGMA_Z, 0.015),)))            # pure dephasing in the eigenbasis
    assert solver.backend.preconditioner == "diagonal"
    assert spectrum(solver).diagnostics["solver"]["iterations_max"] == 1


def test_collapse_operators_that_couple_dyads_keep_it_on_and_it_helps():
    """With a diagonal Hamiltonian the diagonal is the dominant part of the generator, exact or not."""
    solver = chosen(three_site_chain(dissipative=True))
    assert solver.backend.preconditioner == "diagonal"
    automatic = spectrum(solver).diagnostics["solver"]
    assert automatic["preconditioner"] == "diagonal"
    plain = spectrum(chosen(three_site_chain(dissipative=True), preconditioner=None)).diagnostics["solver"]
    assert automatic["iterations_mean"] * automatic["solves"] < plain["iterations_mean"] * plain["solves"]


def test_a_hamiltonian_that_is_not_diagonal_leaves_it_off():
    assert chosen(SiteBasisModel()).backend.preconditioner is None
    assert chosen(SiteBasisModel(with_channel=True)).backend.preconditioner is None


def test_sparse_blocks_are_inspected_and_matrix_free_blocks_are_not():
    closed = three_site_chain(dissipative=False)
    assert chosen(WrappedBlocks(closed, sparse.csr_matrix)).backend.preconditioner == "diagonal"
    assert chosen(WrappedBlocks(SiteBasisModel(), sparse.csr_matrix)).backend.preconditioner is None
    assert chosen(WrappedBlocks(closed, aslinearoperator)).backend.preconditioner is None       # cannot be inspected


def test_explicit_choices_override_auto_and_invalid_ones_are_rejected():
    assert chosen(three_site_chain(dissipative=False), preconditioner=None).backend.preconditioner is None
    assert chosen(three_site_chain(dissipative=True), preconditioner="diagonal").backend.preconditioner == "diagonal"
    with pytest.raises(ValueError, match="'auto'"):
        SpectroscopySolver(backend="sparse_sector", eta=ETA, preconditioner="ilu").feed_model(three_site_chain())


@pytest.mark.parametrize("dissipative", (False, True))
def test_the_automatic_choice_does_not_change_the_spectrum(dissipative):
    automatic = spectrum(chosen(three_site_chain(dissipative))).components["rephasing"]
    plain = spectrum(chosen(three_site_chain(dissipative), preconditioner=None)).components["rephasing"]
    assert np.abs(automatic - plain).max() <= 1e-9 * np.abs(plain).max()


def test_the_dense_backend_reports_no_preconditioner():
    solver = SpectroscopySolver(backend="dense", eta=ETA)
    solver.feed_model(three_site_chain(dissipative=False))
    assert spectrum(solver).diagnostics["solver"]["preconditioner"] is None
