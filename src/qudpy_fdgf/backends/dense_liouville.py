"""Dense Liouville-space backend for small reference systems.

The backend materializes the full Liouvillian and evaluates propagation,
resolvents, pathways, and observables with dense linear algebra. It provides
an exact baseline for validation and compact models where the squared Hilbert
dimension remains affordable.
"""

from collections import OrderedDict
import warnings

import numpy as np
from scipy.linalg import LinAlgWarning, expm, lstsq, lu_factor, lu_solve
from scipy.linalg.lapack import zgecon

from .base import BackendBase
from ..capabilities import BackendCapabilities, Capabilities
from ..exceptions import ConvergenceError
from ..observables import ObservableSpec
from ..results import PathwayResult


class DenseLiouvilleBackend(BackendBase):
    """Exact backend that materializes the full Liouville space."""

    name = "dense_liouville"
    supports_factorized = True
    capabilities = Capabilities(
        pure_state=True,
        mixed_state=True,
        time_domain=True,
        frequency_domain=True,
        hilbert_resolvent=True,
        lindblad=True,
        finite_temperature=True,
        coupled_sectors=True,
        matrix_free=False,
    )
    backend_capabilities = BackendCapabilities(
        state_kinds=("pure", "mixed"),
        supported_pairs=(
            ("unitary", "time"),
            ("unitary", "frequency"),
            ("lindblad", "time"),
            ("lindblad", "frequency"),
        ),
        exactness="exact",
        matrix_free=False,
    )

    def __init__(
        self,
        *,
        eta=None,
        cache_resolvents=True,
        max_cache_entries=32,
        **options,
    ):
        super().__init__(eta=eta, **options)
        self.cache_resolvents = bool(cache_resolvents)
        self.max_cache_entries = int(max_cache_entries)
        self._resolvent_cache = OrderedDict()
        self._time_cache = OrderedDict()
        self._integrated_detector_cache = {}
        self._H_dense = None
        self._A_dense = None
        self._I_liouville = None
        self._rho_initial = None
        self._transition_matrix_cache = {}
        self._observable_matrix_cache = {}

    def build(self, model, context=None, initial_state=None):
        super().build(model, context=context, initial_state=initial_state)
        dimension = self.layout.total_dimension
        self._H_dense = self.materialize(self._hamiltonian)
        self._A_dense = self.materialize(self.generator.linear_operator)
        self._I_liouville = np.eye(
            dimension * dimension, dtype=np.complex128
        )
        self._rho_initial = self._initial_density_matrix.reshape(
            -1, order="F"
        )
        return self

    def clear_caches(self):
        self._resolvent_cache.clear()
        self._time_cache.clear()
        self._integrated_detector_cache.clear()

    def _bounded_cache_put(self, cache, key, value):
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > self.max_cache_entries:
            cache.popitem(last=False)

    def _resolvent(self, omega, eta):
        key = (round(float(omega), 14), round(float(eta), 14))
        cached = self._resolvent_cache.get(key)
        if cached is not None:
            self._resolvent_cache.move_to_end(key)
            return cached
        shift = self.generator.frequency_shift(omega, eta)
        matrix = shift * self._I_liouville - self._A_dense
        inverse = self.generator.frequency_prefactor * np.linalg.inv(matrix)
        self._counters["resolvent_factorizations"] += 1
        if self.cache_resolvents:
            self._bounded_cache_put(self._resolvent_cache, key, inverse)
        return inverse

    def _time_propagator(self, delay):
        key = round(float(delay), 14)
        cached = self._time_cache.get(key)
        if cached is not None:
            self._time_cache.move_to_end(key)
            return cached
        propagator = expm(self._A_dense * float(delay))
        self._bounded_cache_put(self._time_cache, key, propagator)
        return propagator

    def _resolvent_block(self, states, omega, eta, adjoint):
        """Vectors and frequency -> resolvent (or its adjoint) applied to each column.

        A resolvent already in the cache is applied as a matrix product; otherwise the shifted
        matrix is factorized once and solved for all the columns, without storing its inverse.
        """
        key = (round(float(omega), 14), round(float(eta), 14))
        cached = self._resolvent_cache.get(key)
        prefactor = self.generator.frequency_prefactor
        if cached is not None:
            self._resolvent_cache.move_to_end(key)
            result = (
                np.conj(cached.T @ states.conj()) if adjoint else cached @ states
            )
        else:
            shift = self.generator.frequency_shift(omega, eta)
            matrix = shift * self._I_liouville - self._A_dense
            if adjoint:
                matrix = matrix.conj().T
            raw = np.linalg.solve(matrix, states)
            result = prefactor * raw
            self._counters["resolvent_factorizations"] += 1
            norms = np.linalg.norm(states, axis=0)
            residuals = np.linalg.norm(matrix @ raw - states, axis=0)
            for norm, residual in zip(norms, residuals):
                self._record_solve(
                    relative_residual=residual / norm if norm > 0 else 0.0,
                    omega=float(omega),
                    eta=float(eta),
                    adjoint=bool(adjoint),
                )
        self._counters["resolvent_solves"] += states.shape[1]
        return result

    def interval_block(self, states, domain, values, eta=None, *, adjoint=False):
        """Liouville vectors -> propagator or resolvent (or adjoint) at several coordinates."""
        states = np.asarray(states, dtype=np.complex128)
        eta = self.eta if eta is None else float(eta)
        out = np.empty((states.shape[0], states.shape[1], len(values)), dtype=np.complex128)
        for index, value in enumerate(values):
            if domain == "time":
                propagator = self._time_propagator(value)
                out[:, :, index] = (
                    np.conj(propagator.T @ states.conj()) if adjoint else propagator @ states
                )
                self._counters["propagations"] += states.shape[1]
            else:
                out[:, :, index] = self._resolvent_block(states, value, eta, adjoint)
        return out

    def _reference_state_changed(self):
        self._rho_initial = self._initial_density_matrix.reshape(-1, order="F")

    def _solve_stationary(self, reference, weights, **_):
        """Dense LU of ``L + |reference>><<weights|``; its reciprocal condition number detects a degenerate kernel.

        The matrix is singular when the kernel of ``L`` has dimension above one; the error estimate
        ``eps / rcond`` then exceeds 1e-6 and the minimum-norm solution of the (consistent) system
        is returned instead, with ``unique`` false.
        """
        matrix = self._A_dense.copy()
        matrix += np.outer(reference, np.conj(weights))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", LinAlgWarning)       # an exactly singular matrix is handled below
            factors = lu_factor(matrix, check_finite=False)
        norm = float(np.linalg.norm(matrix, 1))
        reciprocal = float(zgecon(factors[0], norm, norm="1")[0])
        epsilon = float(np.finfo(float).eps)
        error_estimate = epsilon / max(reciprocal, np.finfo(float).tiny)
        unique = error_estimate < 1e-6
        if unique:
            solution = lu_solve(factors, reference, check_finite=False)
        else:
            solution = lstsq(matrix, reference, lapack_driver="gelsy", check_finite=False)[0]
        return solution, {
            "method": "lu" if unique else "lstsq",
            "unique": bool(unique),
            "reciprocal_condition": reciprocal,
            "error_estimate": error_estimate,
        }

    def solve_hilbert_resolvent(self, vector, omega, *, eta=None, **_):
        """Vector and frequency -> exact dense Hilbert resolvent."""
        vector = np.asarray(vector, dtype=np.complex128).reshape(-1)
        dimension = self.layout.total_dimension
        if vector.size != dimension:
            raise ValueError("Size is incompatible with the Hilbert space.")
        eta = self.eta if eta is None else float(eta)
        matrix = (
            complex(float(omega), eta)
            * np.eye(dimension, dtype=np.complex128)
            - self._H_dense
        )
        solution = np.linalg.solve(matrix, vector)
        residual = float(np.linalg.norm(matrix @ solution - vector))
        return solution, {
            "method": "dense_solve",
            "residual_norm": residual,
            "omega": float(omega),
            "eta": eta,
        }

    def _transition_matrix(self, interaction):
        key = (
            interaction.operator,
            interaction.direction,
            interaction.source_sector,
            interaction.target_sector,
        )
        if key in self._transition_matrix_cache:
            return self._transition_matrix_cache[key]
        matrix = self.apply_transition_matrix(
            np.eye(self.layout.total_dimension, dtype=np.complex128),
            operator_name=interaction.operator,
            direction=interaction.direction,
            source_sector=interaction.source_sector,
            target_sector=interaction.target_sector,
        )
        self._transition_matrix_cache[key] = matrix
        return matrix

    def _observable_matrix(self, observable):
        spec = (
            observable
            if isinstance(observable, ObservableSpec)
            else ObservableSpec(name=str(observable))
        )
        if spec.kind == "integrated_jump":
            raise ValueError(
                "An integrated_jump observable has no final matrix; "
                "use time-domain evaluation."
            )
        key = (spec.kind, spec.operator, spec.channel)
        if key in self._observable_matrix_cache:
            return self._observable_matrix_cache[key]
        identity = np.eye(self.layout.total_dimension, dtype=np.complex128)
        if spec.kind == "jump_rate":
            matrix = self.apply_jump_effect_matrix(identity, spec.channel)
        else:
            matrix = self.apply_observable_matrix(identity, spec.operator)
        self._observable_matrix_cache[key] = matrix
        return matrix

    def _interaction_superoperator(self, interaction):
        operator = self._transition_matrix(interaction)
        identity = np.eye(self.layout.total_dimension, dtype=np.complex128)
        if interaction.label in {"Ku", "Kd"}:
            return np.kron(identity, operator)

        # For bra interactions, direction describes the operator acting on the
        # bra vector; right multiplication of rho therefore uses its adjoint.
        right_operator = operator.conj().T
        return np.kron(right_operator.T, identity)

    def _propagate_pathway(self, pathway, protocol, coordinates):
        protocol.validate_pathway(pathway)
        protocol.validate_coordinates(coordinates)
        self._check_transition_operators(pathway)
        response = self._rho_initial.copy()
        for interaction, interval in zip(
            pathway.interactions, protocol.intervals
        ):
            response = self._interaction_superoperator(interaction) @ response
            if interval.domain == "time":
                response = self._time_propagator(
                    coordinates[interval.name]
                ) @ response
                self._counters["propagations"] += 1
            elif interval.domain == "frequency":
                eta = self.eta if interval.eta is None else interval.eta
                response = self._resolvent(
                    coordinates[interval.name], eta
                ) @ response
                self._counters["resolvent_solves"] += 1

        return response

    def _detect_response(self, response, observable):
        dimension = self.layout.total_dimension
        rho = response.reshape((dimension, dimension), order="F")
        matrix = self._observable_matrix(observable)
        value = np.trace(matrix @ rho)
        if observable.kind == "jump_rate":
            value *= observable.efficiency
        return value

    def _projection_branches(self, response, observable):
        projections = observable.projection_interactions
        if not projections:
            return ((response, 1.0 + 0.0j),)
        return tuple(
            (
                self._interaction_superoperator(interaction) @ response,
                interaction.perturbative_prefactor,
            )
            for interaction in projections
        )

    def _detect_observable(self, response, observable):
        detector = observable.without_projection()
        return sum(
            prefactor * self._detect_response(projected, detector)
            for projected, prefactor in self._projection_branches(
                response, observable
            )
        )

    def _integrated_detector(self, spec):
        """Window and channel -> integral of the Heisenberg-evolved detector.

        Returns X = int_{t0}^{t1} exp(s A^dagger) vec(M^dagger) ds, so that the
        integrated count of any state is <<X|rho>>. The integral is exact: it
        is the last column of exp[(t1 - t0) B] with the augmented generator
        B = [[A^dagger, vec(M^dagger)], [0, 0]]. X does not depend on the
        pathway state and is cached.
        """
        start, stop = spec.time_window
        key = (spec.channel, start, stop)
        cached = self._integrated_detector_cache.get(key)
        if cached is not None:
            return cached
        if np.isinf(stop):
            integrated = self._infinite_window_detector(spec)
            self._integrated_detector_cache[key] = integrated
            return integrated
        matrix = self._observable_matrix(spec.instantaneous().without_projection())
        vector = matrix.conj().T.reshape(-1, order="F")
        size = vector.size
        heisenberg = self._A_dense.conj().T
        augmented = np.zeros((size + 1, size + 1), dtype=np.complex128)
        augmented[:size, :size] = heisenberg
        augmented[:size, size] = vector
        integrated = expm((stop - start) * augmented)[:size, size]
        if start > 0.0:
            integrated = expm(start * heisenberg) @ integrated
        integrated = np.asarray(integrated, dtype=np.complex128)
        self._integrated_detector_cache[key] = integrated
        return integrated

    def _solve_excess_integral(self, right_hand_side, rho_ss, identity):
        """Dense LU of ``L^dagger - |1>><<rho_ss|``; an ill-conditioned matrix means a degenerate kernel."""
        matrix = self._A_dense.conj().T - np.outer(identity, np.conj(rho_ss))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", LinAlgWarning)
            factors = lu_factor(matrix, check_finite=False)
        norm = float(np.linalg.norm(matrix, 1))
        reciprocal = float(zgecon(factors[0], norm, norm="1")[0])
        if np.finfo(float).eps / max(reciprocal, np.finfo(float).tiny) > 1e-6:
            raise ConvergenceError(
                "The stationary state of the generator is not unique or the generator is singular on the "
                "complement of its stationary state: the infinite-window count is not defined."
            )
        return lu_solve(factors, right_hand_side, check_finite=False)

    def _evolve_adjoint(self, vector, time):
        return expm(time * self._A_dense.conj().T) @ vector

    def _integrated_jump_trapezoid(self, response, spec):
        """Legacy quadrature over n_steps samples of the window."""
        start, stop = spec.time_window
        times = np.linspace(start, stop, spec.n_steps)
        detector = spec.instantaneous().without_projection()
        samples = []
        for time in times:
            value = 0.0j
            for projected, prefactor in self._projection_branches(
                response, spec
            ):
                propagated = self._time_propagator(time) @ projected
                value += prefactor * self._detect_response(
                    propagated, detector
                )
            samples.append(value)
        return np.trapezoid(np.asarray(samples, dtype=np.complex128), times)

    def _integrated_jump_values(self, response, observables):
        values = {}
        for spec in observables:
            if spec.integration == "trapezoid":
                values[spec.name] = self._integrated_jump_trapezoid(
                    response, spec
                )
                continue
            detector = self._integrated_detector(spec)
            value = sum(
                prefactor * np.vdot(detector, projected)
                for projected, prefactor in self._projection_branches(
                    response, spec
                )
            )
            values[spec.name] = spec.efficiency * value
        return values

    def calc_pathway_observables(
        self, pathway, protocol, coordinates, observables
    ):
        """One pathway propagation -> responses for multiple observables."""
        response = self._propagate_pathway(pathway, protocol, coordinates)
        specs = tuple(observables)
        values = {}
        integrated = []
        for spec in specs:
            if spec.kind == "integrated_jump":
                integrated.append(spec)
            else:
                values[spec.name] = self._detect_observable(response, spec)
        values.update(self._integrated_jump_values(response, integrated))
        values = {
            name: pathway.response_prefactor * value
            for name, value in values.items()
        }
        diagnostics = {
            "backend": self.name,
            "hilbert_dimension": self.layout.total_dimension,
            "liouville_dimension": self.layout.total_dimension**2,
            "representation": "dense_liouville",
            "generator_kind": self.generator.kind,
            "generator_convention": "A=-i[H,.]+D",
            "observable_names": tuple(spec.name for spec in specs),
        }
        return PathwayResult(
            value=complex(values[pathway.detection]),
            pathway=pathway.name,
            diagnostics=diagnostics,
            observables={name: complex(value) for name, value in values.items()},
        )

    def calc_pathway(self, pathway, protocol, coordinates):
        """Pathway inputs -> response for its default observable."""
        result = self.calc_pathway_observables(
            pathway,
            protocol,
            coordinates,
            (ObservableSpec(name=pathway.detection),),
        )
        return result
