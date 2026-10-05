"""The generator and the sector operators act on whole matrices, not column by column.

The reference is the column-by-column action that the code used before (``helpers``) and, independently,
the Kronecker form of the GKSL generator built from the materialized operators.
"""
import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, aslinearoperator

from helpers import BACKENDS, reference_generator_action, three_site_chain, two_level_solver
from qudpy_fdgf import SpectroscopySolver
from qudpy_fdgf.backends.base import apply_adjoint_operator_matrix, apply_operator_matrix
from qudpy_fdgf.contracts import CollapseChannel
from qudpy_fdgf.exceptions import SectorError


class VectorOnly:
    """Operator that knows only ``matvec`` and ``rmatvec`` (not a SciPy operator): the fallback path."""

    def __init__(self, block):
        self._block = np.asarray(block.toarray() if sparse.issparse(block) else block)
        self.shape, self.dtype = self._block.shape, self._block.dtype
        self.matvec_calls = 0

    def matvec(self, vector):
        self.matvec_calls += 1
        return self._block @ vector

    def rmatvec(self, vector):
        self.matvec_calls += 1
        return self._block.conj().T @ vector


class CountingOperator(VectorOnly):
    """Operator with ``matmat``/``rmatmat``: the generator must use them and never call ``matvec``."""

    def __init__(self, block):
        super().__init__(block)
        self.matmat_calls = 0

    def matmat(self, matrix):
        self.matmat_calls += 1
        return self._block @ matrix

    def rmatmat(self, matrix):
        self.matmat_calls += 1
        return self._block.conj().T @ matrix


class WrappedModel:
    """Sector model whose operator blocks are replaced by ``wrap(block)``."""

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
        return [
            CollapseChannel(c.name, c.rate, {s: self._wrapped(b) for s, b in c.operator_blocks.items()}, c.metadata)
            for c in self._model.collapse_channels(context)
        ]


WRAPPERS = {
    "arrays": lambda block: block,
    "csr": sparse.csr_matrix,
    "scipy_operator": aslinearoperator,
    "scipy_matvec_only": lambda block: LinearOperator(
        block.shape, dtype=complex, matvec=lambda v, b=block: b @ v, rmatvec=lambda v, b=block: b.conj().T @ v),
    "vector_only": VectorOnly,
    "counting": CountingOperator,
}


def chain_solver(wrapper="arrays", backend="sparse_sector"):
    solver = SpectroscopySolver(backend=backend, eta=0.01, check_stationarity="off")
    solver.feed_model(WrappedModel(three_site_chain(dissipative=True), WRAPPERS[wrapper]))
    return solver


def random_matrix(dimension, columns=None, seed=7):
    rng = np.random.default_rng(seed)
    shape = (dimension, dimension) if columns is None else (dimension, columns)
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


@pytest.mark.parametrize("wrapper", sorted(WRAPPERS))
def test_generator_matches_the_column_by_column_action(wrapper):
    generator = chain_solver(wrapper).backend.generator
    dimension = generator.dimension
    for seed in (1, 2):
        rho = random_matrix(dimension, seed=seed)
        vector = rho.reshape(-1, order="F")
        forward = generator.matvec(vector).reshape((dimension, dimension), order="F")
        backward = generator.rmatvec(vector).reshape((dimension, dimension), order="F")
        assert forward == pytest.approx(reference_generator_action(generator, rho), rel=1e-12, abs=1e-12)
        assert backward == pytest.approx(reference_generator_action(generator, rho, adjoint=True), rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_generator_matches_the_gksl_generator_in_kronecker_form(backend):
    solver = two_level_solver(backend)
    base = solver.backend
    generator, size = base.generator, base.layout.total_dimension
    h = base.materialize(base._hamiltonian)
    eye = np.eye(size)
    expected = -1j * (np.kron(eye, h) - np.kron(h.T, eye))                # column stacking: vec(A rho B) = (B^T kron A) vec rho
    assert generator.collapse_operators
    for channel, operator in generator.collapse_operators:
        c = base.materialize(operator)
        k = c.conj().T @ c
        expected += channel.rate * (np.kron(c.conj(), c) - 0.5 * np.kron(eye, k) - 0.5 * np.kron(k.T, eye))
    forward = base.materialize(generator.linear_operator)
    assert forward == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert base.materialize(LinearOperator(forward.shape, dtype=complex, matvec=generator.rmatvec)) == pytest.approx(
        expected.conj().T, rel=1e-12, abs=1e-12)                          # the adjoint action is the adjoint matrix
    assert generator.diagonal() == pytest.approx(np.diag(expected), rel=1e-12, abs=1e-12)


def test_operator_diagonal_matches_the_dense_operator_for_several_chunks():
    base = chain_solver().backend
    dense = base.materialize(base._hamiltonian)
    diagonal, column_norms = base.generator.operator_diagonal(base._hamiltonian)
    assert diagonal == pytest.approx(np.diag(dense), abs=1e-13)
    assert column_norms == pytest.approx(np.sum(abs(dense) ** 2, axis=0), abs=1e-13)


@pytest.mark.parametrize("wrapper", ["counting", "vector_only"])
def test_matmat_is_used_when_available_and_the_fallback_loops_over_columns_otherwise(wrapper):
    base = chain_solver(wrapper).backend
    size = base.layout.total_dimension
    hamiltonian_blocks = [block for blocks in base._hamiltonian_blocks.values() for block in blocks.values()]
    for block in hamiltonian_blocks:
        block.matvec_calls = 0
        block.matmat_calls = 0
    base.generator.matvec(random_matrix(size).reshape(-1, order="F"))
    if wrapper == "counting":
        # H rho and rho H: one matrix product each, no call per column
        assert [(b.matvec_calls, b.matmat_calls) for b in hamiltonian_blocks] == [(0, 2)] * len(hamiltonian_blocks)
    else:
        assert [b.matvec_calls for b in hamiltonian_blocks] == [2 * size] * len(hamiltonian_blocks)


@pytest.mark.parametrize("wrapper", sorted(WRAPPERS))
def test_transition_and_observable_actions_match_dense_blocks(wrapper):
    solver = chain_solver(wrapper)
    base = solver.backend
    reference_model = three_site_chain(dissipative=True)
    size = base.layout.total_dimension
    matrix = random_matrix(size, columns=5)

    def dense(blocks_of):
        full = np.zeros((size, size), dtype=complex)
        for source in base.layout.sectors:
            for target, block in blocks_of(source).items():
                full[base.layout.slices[target], base.layout.slices[source]] += block
        return full

    for direction in ("plus", "minus"):
        full = dense(lambda source: reference_model.transition_blocks("light_matter", direction, source))
        got = base.apply_transition_matrix(matrix, operator_name="light_matter", direction=direction)
        assert got == pytest.approx(full @ matrix, rel=1e-12, abs=1e-12)
        column = base.apply_transition(matrix[:, 2], operator_name="light_matter", direction=direction)
        assert column == pytest.approx(got[:, 2], rel=1e-12, abs=1e-12)
    full = dense(lambda source: reference_model.observable_blocks("polarization", source))
    assert base.apply_observable_matrix(matrix, "polarization") == pytest.approx(full @ matrix, rel=1e-12, abs=1e-12)


def test_constrained_transition_keeps_its_errors_and_size_checks():
    base = chain_solver().backend
    size = base.layout.total_dimension
    sectors = base.layout.sectors
    with pytest.raises(ValueError, match="incompatible"):
        base.apply_transition_matrix(np.ones((size + 1, 2)), operator_name="light_matter", direction="plus")
    with pytest.raises(ValueError, match="incompatible"):
        base.apply_transition(np.ones(size + 1), operator_name="light_matter", direction="plus")
    with pytest.raises(SectorError, match="matches no model block"):
        base.apply_transition_matrix(np.ones((size, 2)), operator_name="light_matter", direction="plus",
                                     source_sector=sectors[0], target_sector=sectors[0])


def test_materialize_is_independent_of_the_chunk_size():
    base = chain_solver().backend
    reference = base.materialize(base._hamiltonian, chunk=1000)
    for chunk in (1, 3, 64):
        assert base.materialize(base._hamiltonian, chunk=chunk) == pytest.approx(reference, abs=1e-14)


@pytest.mark.parametrize("operator", [np.arange(12.0).reshape(3, 4) * (1 + 1j), sparse.csr_matrix(np.arange(12.0).reshape(3, 4) * (1 - 2j))])
def test_matrix_helpers_for_arrays_and_sparse_matrices(operator):
    dense = operator.toarray() if sparse.issparse(operator) else operator
    rng = np.random.default_rng(3)
    right = rng.standard_normal((4, 5)) + 1j * rng.standard_normal((4, 5))
    left = rng.standard_normal((3, 5)) + 1j * rng.standard_normal((3, 5))
    assert apply_operator_matrix(operator, right) == pytest.approx(dense @ right, abs=1e-13)
    assert apply_adjoint_operator_matrix(operator, left) == pytest.approx(dense.conj().T @ left, abs=1e-13)
