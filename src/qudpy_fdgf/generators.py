"""Shared generator for time-domain Liouville-space dynamics.

``EvolutionGenerator`` combines a Hamiltonian with optional GKSL collapse
channels and exposes matrix-free forward and adjoint actions. Backends use the
same object for time propagation and shifted frequency-domain resolvents,
which keeps sign and damping conventions consistent.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import LinearOperator

_DIAGONAL_CHUNK = 256


def _stack_columns(action, matrix):
    """Vector action and matrix -> action on every column (the output may have another length)."""
    columns = [np.asarray(action(matrix[:, column]), dtype=np.complex128).reshape(-1)
               for column in range(matrix.shape[1])]
    if not columns:
        return np.empty((0, 0), dtype=np.complex128)
    return np.column_stack(columns)


def operator_matmat(operator, matrix):
    """Operator and matrix -> ``O @ matrix``, column by column only if ``O`` has no ``matmat``."""
    matmat = getattr(operator, "matmat", None)
    if matmat is not None:
        return np.asarray(matmat(matrix), dtype=np.complex128)
    return _stack_columns(operator.matvec, matrix)


def operator_rmatmat(operator, matrix):
    """Operator and matrix -> ``O^dagger @ matrix``, column by column only if ``O`` has no ``rmatmat``."""
    rmatmat = getattr(operator, "rmatmat", None)
    if rmatmat is not None:
        return np.asarray(rmatmat(matrix), dtype=np.complex128)
    return _stack_columns(operator.rmatvec, matrix)


class EvolutionGenerator:
    """Hamiltonian and collapse operators -> matrix-free ``A rho``.

    Frequency intervals apply ``[(eta-i*omega)I - A]**-1``. The operators act on the whole
    density matrix at once (``matmat``/``rmatmat`` when the operator provides them), so that one
    application of the generator costs a few matrix products instead of a Python loop over the
    columns of ``rho``.
    """

    # Exclude the legacy Green-function ``-1j`` factor to avoid rotating the
    # pathway perturbative coefficient twice.
    frequency_prefactor = 1.0

    def __init__(self, hamiltonian, collapse_operators=()):
        self.hamiltonian = hamiltonian
        self.collapse_operators = tuple(collapse_operators)
        self.dimension = int(hamiltonian.shape[0])
        self.liouville_dimension = self.dimension * self.dimension
        self.linear_operator = LinearOperator(
            (self.liouville_dimension, self.liouville_dimension),
            dtype=np.complex128,
            matvec=self.matvec,
            rmatvec=self.rmatvec,
        )

    @property
    def is_unitary(self):
        return not self.collapse_operators

    @property
    def kind(self):
        return "unitary" if self.is_unitary else "lindblad"

    def _as_matrix(self, value):
        return np.asarray(value, dtype=np.complex128).reshape(
            (self.dimension, self.dimension), order="F"
        )

    @staticmethod
    def _left(matrix, operator, *, dagger=False):
        """Matrix M and operator O -> O M (``dagger=True``: O^dagger M)."""
        return operator_rmatmat(operator, matrix) if dagger else operator_matmat(operator, matrix)

    @staticmethod
    def _right(matrix, operator, *, dagger=False):
        """Matrix M and operator O -> M O (``dagger=True``: M O^dagger).

        M O = (O^dagger M^dagger)^dagger and M O^dagger = (O M^dagger)^dagger.
        """
        conjugate = matrix.conj().T
        acted = operator_matmat(operator, conjugate) if dagger else operator_rmatmat(operator, conjugate)
        return acted.conj().T

    def apply_matrix(self, density):
        """rho -> -i[H, rho] + sum_c gamma_c (C rho C^+ - (K rho + rho K)/2), K = C^+ C."""
        density = self._as_matrix(density)
        result = -1j * (self._left(density, self.hamiltonian) - self._right(density, self.hamiltonian))

        for channel, operator in self.collapse_operators:
            c_rho = self._left(density, operator)                       # C rho
            rho_c_dagger = self._right(density, operator, dagger=True)  # rho C^+
            jump = self._left(rho_c_dagger, operator)                   # C rho C^+
            k_left = self._left(c_rho, operator, dagger=True)           # C^+ C rho
            k_right = self._right(rho_c_dagger, operator)               # rho C^+ C
            result += channel.rate * (jump - 0.5 * k_left - 0.5 * k_right)
        return result

    def apply_adjoint_matrix(self, observable):
        """O -> i[H, O] + sum_c gamma_c (C^+ O C - (K O + O K)/2)."""
        observable = self._as_matrix(observable)
        result = 1j * (
            self._left(observable, self.hamiltonian, dagger=True)
            - self._right(observable, self.hamiltonian, dagger=True)
        )

        for channel, operator in self.collapse_operators:
            c_dagger_o = self._left(observable, operator, dagger=True)      # C^+ O
            jump = self._right(c_dagger_o, operator)                        # C^+ O C
            c_o = self._left(observable, operator)                          # C O
            k_left = self._left(c_o, operator, dagger=True)                 # C^+ C O
            o_c_dagger = self._right(observable, operator, dagger=True)     # O C^+
            k_right = self._right(o_c_dagger, operator)                     # O C^+ C
            result += channel.rate * (jump - 0.5 * k_left - 0.5 * k_right)
        return result

    def operator_diagonal(self, operator):
        """Hilbert operator -> diagonal <x|O|x> and column norms ||O|x>||^2."""
        diagonal = np.empty(self.dimension, dtype=np.complex128)
        column_norms = np.empty(self.dimension)
        for first in range(0, self.dimension, _DIAGONAL_CHUNK):
            last = min(first + _DIAGONAL_CHUNK, self.dimension)
            basis = np.zeros((self.dimension, last - first), dtype=np.complex128)
            basis[np.arange(first, last), np.arange(last - first)] = 1.0
            columns = operator_matmat(operator, basis)
            diagonal[first:last] = columns[np.arange(first, last), np.arange(last - first)]
            column_norms[first:last] = np.sum(columns.real ** 2 + columns.imag ** 2, axis=0)
        return diagonal, column_norms

    def diagonal(self):
        """Generator -> its diagonal in the dyad basis |x><y| (column-stacked).

        A|x><y| has the diagonal coefficient
        -i(H_xx - H_yy) + sum_c gamma_c [L_xx L_yy^* - (K_xx + K_yy)/2],
        with K = L^+ L. The real part is never positive, so the shifted
        diagonal (eta - i omega) - A_xy,xy does not vanish for eta > 0. In the
        eigenbasis of a closed generator it is the generator itself.
        """
        h_diagonal, _ = self.operator_diagonal(self.hamiltonian)
        result = -1j * (h_diagonal[:, None] - h_diagonal[None, :])
        for channel, operator in self.collapse_operators:
            l_diagonal, k_diagonal = self.operator_diagonal(operator)
            result += channel.rate * (
                np.outer(l_diagonal, l_diagonal.conj())
                - 0.5 * k_diagonal[:, None]
                - 0.5 * k_diagonal[None, :]
            )
        return result.reshape(-1, order="F")

    def matvec(self, density_vector):
        return self.apply_matrix(density_vector).reshape(-1, order="F")

    def rmatvec(self, observable_vector):
        return self.apply_adjoint_matrix(observable_vector).reshape(
            -1, order="F"
        )

    @staticmethod
    def frequency_shift(omega, eta):
        """Frequency and damping -> direct-resolvent shift."""
        return complex(float(eta), -float(omega))
