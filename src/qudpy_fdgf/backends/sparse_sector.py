"""Sparse, sector-based exact-diagonalization backend.

The backend propagates ket and bra branches without materializing dense
Hilbert operators. It supports matrix-free time evolution and Hilbert-space
resolvents, while retaining a Liouville-vector route for frequency-domain
pathways and small-system checks.
"""

from collections import OrderedDict
import hashlib

import numpy as np
from scipy.sparse import issparse
from scipy.sparse.linalg import LinearOperator, expm_multiply, gmres

from .base import BackendBase
from ..capabilities import BackendCapabilities, Capabilities
from ..exceptions import ConvergenceError
from ..observables import ObservableSpec
from ..results import PathwayResult


def _is_diagonal(operator, tolerance=1e-12):
    """Whether an array or sparse block is diagonal; anything else (matrix-free) is not known to be."""
    if issparse(operator):
        coordinates = operator.tocoo()
        data = np.abs(coordinates.data)
        scale = float(data.max()) if data.size else 0.0
        off = data[coordinates.row != coordinates.col]
        return bool(off.size == 0 or off.max() <= tolerance * max(scale, np.finfo(float).tiny))
    if isinstance(operator, np.ndarray) and operator.ndim == 2 and operator.shape[0] == operator.shape[1]:
        scale = float(np.abs(operator).max()) if operator.size else 0.0
        off = np.abs(operator - np.diag(np.diag(operator)))
        return bool(off.max(initial=0.0) <= tolerance * max(scale, np.finfo(float).tiny))
    return False


class SparseSectorBackend(BackendBase):
    """Pathway inputs -> sparse ket/bra branch propagation.

    Time evolution and Hilbert resolvents remain matrix-free.
    ``preconditioner="diagonal"`` preconditions every GMRES resolvent with the
    inverse of the shifted diagonal of the generator (or of the Hamiltonian).
    The diagonal is exact when the generator is diagonal in the dyad basis, that
    is for a model supplied in its eigenbasis (diagonal Hamiltonian blocks) whose
    collapse operators, if any, are diagonal too: each solve then converges in one
    iteration. With collapse channels that couple dyads it remains an approximation
    that still divides the iteration count by a large factor, because the coupling
    is weak next to the energy differences on the diagonal.
    The default ``"auto"`` selects ``"diagonal"`` when the Hamiltonian blocks are
    diagonal blocks of single sectors (the eigenbasis the model adapters provide) and
    ``None`` otherwise; ``self.preconditioner`` holds the choice made.
    """

    preconditioners = (None, "diagonal", "auto")

    name = "sparse_sector"
    supports_factorized = True
    _solve_method = "gmres"
    capabilities = Capabilities(
        pure_state=True,
        mixed_state=True,
        time_domain=True,
        frequency_domain=True,
        hilbert_resolvent=True,
        lindblad=True,
        finite_temperature=True,
        coupled_sectors=True,
        matrix_free=True,
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
        matrix_free=True,
    )

    def __init__(
        self,
        *,
        eta=None,
        krylov_tolerance=1e-10,
        krylov_maxiter=None,
        resolvent_restart=None,
        preconditioner="auto",
        cache_propagations=True,
        max_cache_entries=32,
        **options,
    ):
        super().__init__(eta=eta, **options)
        if preconditioner not in self.preconditioners:
            raise ValueError(
                "preconditioner must be None, 'diagonal' or 'auto'; "
                f"got {preconditioner!r}."
            )
        self.preconditioner_option = preconditioner
        self.preconditioner = None if preconditioner == "auto" else preconditioner
        self.krylov_tolerance = float(krylov_tolerance)
        self.krylov_maxiter = (
            None if krylov_maxiter is None else int(krylov_maxiter)
        )
        self.resolvent_restart = (
            None if resolvent_restart is None else int(resolvent_restart)
        )
        self.cache_propagations = bool(cache_propagations)
        self.max_cache_entries = int(max_cache_entries)
        self._time_cache = OrderedDict()
        self._liouville_time_cache = OrderedDict()
        self._integrated_detector_cache = {}
        self._liouville = None
        self._liouville_diagonal = None
        self._hamiltonian_diagonal = None
        self._initial_density_vector = None

    def _hamiltonian_is_diagonal(self):
        """Whether every Hamiltonian block is a diagonal block of one sector (a model in its eigenbasis).

        Then the diagonal of the generator in the dyad basis is its unperturbed part, the energy
        differences, and collapse operators only perturb it. Blocks that are neither arrays nor sparse
        matrices cannot be inspected and count as not diagonal.
        """
        for source, blocks in self._hamiltonian_blocks.items():
            for target, operator in blocks.items():
                if target != source or not _is_diagonal(operator):
                    return False
        return True

    def build(self, model, context=None, initial_state=None):
        super().build(model, context=context, initial_state=initial_state)
        if self.preconditioner_option == "auto":
            self.preconditioner = "diagonal" if self._hamiltonian_is_diagonal() else None
        self._liouville = self.generator.linear_operator
        self._liouville_diagonal = None
        self._hamiltonian_diagonal = None
        if self.preconditioner == "diagonal":
            self._liouville_diagonal = self.generator.diagonal()
            self._hamiltonian_diagonal, _ = self.generator.operator_diagonal(
                self._hamiltonian
            )
        self._initial_density_vector = self._initial_density_matrix.reshape(
            -1, order="F"
        )
        self._liouville_adjoint = LinearOperator(
            self._liouville.shape,
            dtype=np.complex128,
            matvec=self._liouville.rmatvec,
            rmatvec=self._liouville.matvec,
        )
        return self

    def factorized_applicable(self, protocol):
        """Frequency axes, a dissipative generator or a mixed state take the Liouville route.

        A closed model with a pure state and time intervals only is evaluated as a rank-one dyad,
        which is cheaper than any Liouville vector; the factorized evaluation does not apply.
        """
        return bool(
            protocol.frequency_axis_names
            or not self.generator.is_unitary
            or not self._initial_is_pure
        )

    @staticmethod
    def _diagonal_preconditioner(shifted_diagonal):
        """Shifted diagonal d -> LinearOperator applying v / d."""
        size = shifted_diagonal.size
        return LinearOperator(
            (size, size),
            dtype=np.complex128,
            matvec=lambda value: np.asarray(value).reshape(-1)
            / shifted_diagonal,
        )

    def clear_caches(self):
        self._time_cache.clear()
        self._liouville_time_cache.clear()
        self._integrated_detector_cache.clear()

    def _density_matrix(self, vector):
        """Column-stacked density vector -> square matrix."""
        dimension = self.layout.total_dimension
        return np.asarray(vector, dtype=np.complex128).reshape(
            (dimension, dimension), order="F"
        )

    def _liouville_matvec(self, density_vector):
        """Density vector -> matrix-free generator action."""
        return self.generator.matvec(density_vector)

    def _transition_on_matrix(self, matrix, interaction):
        return self.apply_transition_matrix(
            matrix,
            operator_name=interaction.operator,
            direction=interaction.direction,
            source_sector=interaction.source_sector,
            target_sector=interaction.target_sector,
        )

    def _apply_transition_left(self, density_vector, interaction):
        """Density vector and interaction -> ``J rho``."""
        return self._transition_on_matrix(
            self._density_matrix(density_vector), interaction
        ).reshape(-1, order="F")

    def _apply_transition_right(self, density_vector, interaction):
        """Density vector and interaction -> right bra-adjoint action ``rho J^dagger``."""
        rho = self._density_matrix(density_vector)
        acted_dagger = self._transition_on_matrix(rho.conj().T, interaction)
        return acted_dagger.conj().T.reshape(-1, order="F")

    def _propagate_liouville(self, density_vector, delay):
        delay = float(delay)
        density_vector = np.asarray(
            density_vector, dtype=np.complex128
        ).reshape(-1)
        if delay == 0:
            return density_vector.copy()
        fingerprint = hashlib.blake2b(
            density_vector.view(np.uint8), digest_size=16
        ).digest()
        key = (round(delay, 14), fingerprint)
        cached = self._liouville_time_cache.get(key)
        if cached is not None:
            self._liouville_time_cache.move_to_end(key)
            return cached.copy()
        propagated = np.asarray(
            expm_multiply(
                delay * self._liouville,
                density_vector,
                traceA=0.0,
            ),
            dtype=np.complex128,
        )
        self._counters["propagations"] += 1
        self._liouville_time_cache[key] = propagated.copy()
        self._liouville_time_cache.move_to_end(key)
        while len(self._liouville_time_cache) > self.max_cache_entries:
            self._liouville_time_cache.popitem(last=False)
        return propagated

    def _reference_state_changed(self):
        self._initial_density_vector = self._initial_density_matrix.reshape(-1, order="F")

    def _solve_stationary(
        self, reference, weights, *, restart=100, check_unique=False, check_cycles=20
    ):
        """GMRES on ``L + |r>><<weights|`` with ``r = reference``; a second solve with another ``r`` tests the kernel.

        If the kernel of ``L`` is one-dimensional the solution is its trace-one element whatever
        the trace-one vector ``r``. If the kernel is larger the system is singular and consistent, and
        GMRES, which works in the Krylov space of ``r``, returns the projection of ``r`` on the kernel
        (normalized), which changes with ``r``: that is what the comparison detects. The first ``r``
        is the maximally mixed state, which is itself stationary for a unital generator (a closed
        model), so that the answer is then the maximally mixed state rather than an arbitrary element.

        A random ``r`` excites every slow mode, so the second solve is as stiff as the relaxation gap
        of the generator: it is limited to ``check_cycles`` restart cycles of ``restart`` iterations
        (2000 iterations by default). If it has not converged by then the uniqueness is reported as
        unverified (``unique`` is ``None``); the first solve is unaffected. The test is off by default:
        it costs seconds to tens of seconds from D = 57 and is conclusive only for small or well
        conditioned generators (D = 16 needs 1900 iterations, D = 31 needs 5700 for a decay rate 1/1000
        of the energy scale), where the dense backend tests uniqueness for free.
        """
        size = reference.size
        dimension = self.layout.total_dimension

        def solve(right_hand_side, maxiter):
            operator = LinearOperator(
                (size, size),
                dtype=np.complex128,
                matvec=lambda value: self.generator.matvec(value)
                + right_hand_side * np.vdot(weights, value),
            )
            iterations = []
            solution, status = gmres(
                operator,
                right_hand_side,
                rtol=self.krylov_tolerance,
                atol=0.0,
                restart=min(int(restart), size),
                maxiter=maxiter,
                callback=iterations.append,
                callback_type="pr_norm",
            )
            if status != 0:
                residual = float(np.linalg.norm(operator @ solution - right_hand_side))
                return None, len(iterations), residual
            return solution, len(iterations), None

        def trace(vector):
            return np.sum(vector[:: dimension + 1])

        # a stiff or degenerate generator makes GMRES crawl: 100 restart cycles unless krylov_maxiter says otherwise
        first, iterations, residual = solve(reference, 100 if self.krylov_maxiter is None else self.krylov_maxiter)
        if first is None:
            raise ConvergenceError(
                "GMRES did not converge for the stationary state "
                f"(residual={residual:.3e} after {iterations} iterations): the relaxation gap of the "
                "generator is probably too small for this solver (or the kernel is degenerate). Raise "
                "krylov_maxiter, use the dense backend for a small model, or supply the state with "
                "initial_state."
            )
        info = {"method": "gmres", "unique": None, "iterations": iterations, "second_solve_difference": None}
        if check_unique:
            generator = np.random.default_rng(97)
            other = generator.standard_normal(size) + 1j * generator.standard_normal(size)
            second, second_iterations, _ = solve(other / trace(other), check_cycles)
            info["second_solve_iterations"] = second_iterations
            if second is not None:
                second = second / trace(second)
                difference = float(np.linalg.norm(first - second) / np.linalg.norm(first))
                info["second_solve_difference"] = difference
                info["unique"] = bool(difference < max(1e-6, 1e3 * self.krylov_tolerance))
        return first, info

    def _gmres(self, operator, right_hand_side, preconditioner, *, initial_guess=None):
        """GMRES run -> (solution, SciPy info, number of iterations)."""
        iterations = []
        solution, info = gmres(
            operator,
            right_hand_side,
            M=preconditioner,
            x0=initial_guess,
            rtol=self.krylov_tolerance,
            atol=0.0,
            restart=self.resolvent_restart,
            maxiter=self.krylov_maxiter,
            callback=iterations.append,
            callback_type="pr_norm",
        )
        return solution, info, len(iterations)

    def _solve_liouville_resolvent(self, density_vector, omega, eta, *, adjoint=False):
        """Vector and frequency -> GMRES solution of the shifted system (or of its adjoint)."""
        shift = self.generator.frequency_shift(omega, eta)
        dimension = self.layout.total_dimension**2
        forward = LinearOperator(
            (dimension, dimension),
            dtype=np.complex128,
            matvec=lambda value: shift * value
            - self._liouville.matvec(value),
            rmatvec=lambda value: np.conj(shift) * value
            - self._liouville.rmatvec(value),
        )
        operator = forward.H if adjoint else forward
        diagonal = (
            None
            if self._liouville_diagonal is None
            else shift - self._liouville_diagonal
        )
        if adjoint and diagonal is not None:
            diagonal = np.conj(diagonal)
        preconditioner = (
            None if diagonal is None else self._diagonal_preconditioner(diagonal)
        )
        density_vector = np.asarray(density_vector, dtype=np.complex128).reshape(-1)
        right_hand_side = float(np.linalg.norm(density_vector))
        raw_solution, info, iterations = self._gmres(operator, density_vector, preconditioner)
        residual = float(
            np.linalg.norm(operator @ raw_solution - density_vector)
        )
        if info != 0:
            raise ConvergenceError(
                "Liouville GMRES resolvent did not converge "
                f"(info={info}, residual={residual:.3e})."
            )
        self._counters["resolvent_solves"] += 1
        self._max_residual = max(self._max_residual, residual)
        self._record_solve(
            iterations=iterations,
            relative_residual=residual / right_hand_side if right_hand_side > 0 else 0.0,
            omega=float(omega),
            eta=float(eta),
            adjoint=bool(adjoint),
        )
        prefactor = self.generator.frequency_prefactor
        solution = (np.conj(prefactor) if adjoint else prefactor) * raw_solution
        return np.asarray(solution, dtype=np.complex128), residual

    def _propagate_liouville_block(self, states, delay, adjoint):
        """Vectors and delay -> ``exp(L t)`` (or its adjoint) applied to each column at once."""
        delay = float(delay)
        if delay == 0:
            return states.copy()
        generator = self._liouville_adjoint if adjoint else self._liouville
        propagated = np.asarray(
            expm_multiply(delay * generator, states, traceA=0.0),
            dtype=np.complex128,
        )
        self._counters["propagations"] += states.shape[1]
        return propagated

    def interval_block(self, states, domain, values, eta=None, *, adjoint=False):
        """Liouville vectors -> propagator or resolvent (or adjoint) at several coordinates."""
        states = np.asarray(states, dtype=np.complex128)
        eta = self.eta if eta is None else float(eta)
        out = np.empty((states.shape[0], states.shape[1], len(values)), dtype=np.complex128)
        for index, value in enumerate(values):
            if domain == "time":
                out[:, :, index] = self._propagate_liouville_block(states, value, adjoint)
            else:
                for column in range(states.shape[1]):
                    out[:, column, index], _ = self._solve_liouville_resolvent(
                        states[:, column], value, eta, adjoint=adjoint
                    )
        return out

    def _detect_density(self, density_vector, observable):
        observable_rho = self.apply_detection_operator_matrix(
            self._density_matrix(density_vector), observable
        )
        value = np.trace(observable_rho)
        if observable.kind == "jump_rate":
            value *= observable.efficiency
        return value

    def _projection_branches(self, density_vector, observable):
        projections = observable.projection_interactions
        if not projections:
            return ((density_vector, 1.0 + 0.0j),)
        branches = []
        for interaction in projections:
            if interaction.side == "ket":
                projected = self._apply_transition_left(
                    density_vector, interaction
                )
            else:
                projected = self._apply_transition_right(
                    density_vector, interaction
                )
            branches.append(
                (projected, interaction.perturbative_prefactor)
            )
        return tuple(branches)

    def _detect_density_observable(self, density_vector, observable):
        detector = observable.without_projection()
        return sum(
            prefactor * self._detect_density(projected, detector)
            for projected, prefactor in self._projection_branches(
                density_vector, observable
            )
        )

    def _propagate_liouville_pathway(self, pathway, protocol, coordinates):
        response = self._initial_density_vector.copy()
        max_residual = 0.0
        for interaction, interval in zip(
            pathway.interactions, protocol.intervals
        ):
            if interaction.side == "ket":
                response = self._apply_transition_left(
                    response, interaction
                )
            else:
                response = self._apply_transition_right(
                    response, interaction
                )
            if interval.domain == "time":
                response = self._propagate_liouville(
                    response, coordinates[interval.name]
                )
            elif interval.domain == "frequency":
                eta = self.eta if interval.eta is None else interval.eta
                response, residual = self._solve_liouville_resolvent(
                    response,
                    coordinates[interval.name],
                    eta,
                )
                max_residual = max(max_residual, residual)

        return response, max_residual

    def _detect_rank_one(self, ket, bra, observable):
        detected_ket = self.apply_detection_operator(ket, observable)
        return np.vdot(bra, detected_ket)

    def _detect_rank_one_observable(self, ket, bra, observable):
        detector = observable.without_projection()
        projections = observable.projection_interactions
        if not projections:
            return self._detect_rank_one(ket, bra, detector)
        value = 0.0j
        for interaction in projections:
            projected_ket = ket
            projected_bra = bra
            branch = ket if interaction.side == "ket" else bra
            branch = self.apply_transition(
                branch,
                operator_name=interaction.operator,
                direction=interaction.direction,
                source_sector=interaction.source_sector,
                target_sector=interaction.target_sector,
            )
            if interaction.side == "ket":
                projected_ket = branch
            else:
                projected_bra = branch
            value += interaction.perturbative_prefactor * self._detect_rank_one(
                projected_ket, projected_bra, detector
            )
        return value

    def _detector_vector(self, detector):
        """Detector -> vec(M^dagger), so that Tr(M rho) = <<M^dagger|rho>>."""
        matrix = self.apply_detection_operator_matrix(
            np.eye(self.layout.total_dimension, dtype=np.complex128), detector
        )
        return matrix.conj().T.reshape(-1, order="F")

    def _integrated_detector(self, spec):
        """Window and channel -> integral of the Heisenberg-evolved detector.

        Returns X = int_{t0}^{t1} exp(s A^dagger) vec(M^dagger) ds, so that the
        integrated count of any state is <<X|rho>>. The integral is exact: it
        is the last column of exp[(t1 - t0) B] with the augmented generator
        B = [[A^dagger, vec(M^dagger)], [0, 0]], applied matrix-free. X does
        not depend on the pathway state and is cached.
        """
        start, stop = spec.time_window
        key = (spec.channel, start, stop)
        cached = self._integrated_detector_cache.get(key)
        if cached is not None:
            return cached
        vector = self._detector_vector(spec.instantaneous().without_projection())
        size = vector.size
        generator = self._liouville            # matvec: A, rmatvec: A^dagger

        def augmented_matvec(value):
            value = np.asarray(value, dtype=np.complex128).reshape(-1)
            result = np.zeros(size + 1, dtype=np.complex128)
            result[:size] = generator.rmatvec(value[:size]) + value[size] * vector
            return result

        def augmented_rmatvec(value):
            value = np.asarray(value, dtype=np.complex128).reshape(-1)
            result = np.empty(size + 1, dtype=np.complex128)
            result[:size] = generator.matvec(value[:size])
            result[size] = np.vdot(vector, value[:size])
            return result

        augmented = LinearOperator(
            (size + 1, size + 1),
            dtype=np.complex128,
            matvec=augmented_matvec,
            rmatvec=augmented_rmatvec,
        )
        seed = np.zeros(size + 1, dtype=np.complex128)
        seed[size] = 1.0
        integrated = np.asarray(
            expm_multiply((stop - start) * augmented, seed, traceA=0.0),
            dtype=np.complex128,
        )[:size]
        if start > 0.0:
            heisenberg = LinearOperator(
                (size, size),
                dtype=np.complex128,
                matvec=generator.rmatvec,
                rmatvec=generator.matvec,
            )
            integrated = np.asarray(
                expm_multiply(start * heisenberg, integrated, traceA=0.0),
                dtype=np.complex128,
            )
        self._integrated_detector_cache[key] = integrated
        return integrated

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
                propagated = self._propagate_liouville(projected, time)
                value += prefactor * self._detect_density(
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
        protocol.validate_pathway(pathway)
        protocol.validate_coordinates(coordinates)
        self._check_transition_operators(pathway)
        specs = tuple(observables)
        integrated = [
            spec for spec in specs if spec.kind == "integrated_jump"
        ]
        if (
            protocol.frequency_axis_names
            or not self.generator.is_unitary
            or not self._initial_is_pure
        ):
            response, max_residual = self._propagate_liouville_pathway(
                pathway, protocol, coordinates
            )
            values = {
                spec.name: self._detect_density_observable(response, spec)
                for spec in specs
                if spec.kind != "integrated_jump"
            }
            values.update(self._integrated_jump_values(response, integrated))
            representation = "matrix_free_liouville"
        else:
            ket = self._initial_vector.copy()
            bra = self._initial_vector.copy()
            for interaction, interval in zip(
                pathway.interactions, protocol.intervals
            ):
                branch = ket if interaction.side == "ket" else bra
                branch = self.apply_transition(
                    branch,
                    operator_name=interaction.operator,
                    direction=interaction.direction,
                    source_sector=interaction.source_sector,
                    target_sector=interaction.target_sector,
                )
                if interaction.side == "ket":
                    ket = branch
                else:
                    bra = branch

                if interval.domain == "time":
                    delay = float(coordinates[interval.name])
                    ket = self.propagate(ket, delay)
                    bra = self.propagate(bra, delay)

            values = {
                spec.name: self._detect_rank_one_observable(ket, bra, spec)
                for spec in specs
                if spec.kind != "integrated_jump"
            }
            # Unitary models have no GKSL channel and therefore zero jump
            # counts. An actual integrated jump is evaluated by the Liouville
            # branch above whenever collapse channels are present.
            for spec in integrated:
                if spec.channel not in self._collapse_channel_map:
                    self.apply_jump_effect(
                        np.zeros(self.layout.total_dimension), spec.channel
                    )
            values.update(
                {spec.name: 0.0j for spec in integrated}
            )
            max_residual = 0.0
            representation = "rank_one_dyad"

        values = {
            name: pathway.response_prefactor * value
            for name, value in values.items()
        }
        diagnostics = {
            "backend": self.name,
            "hilbert_dimension": self.layout.total_dimension,
            "liouville_dimension": self.layout.total_dimension**2,
            "representation": representation,
            "generator_kind": self.generator.kind,
            "generator_convention": "A=-i[H,.]+D",
            "maximum_resolvent_residual": max_residual,
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
        return self.calc_pathway_observables(
            pathway,
            protocol,
            coordinates,
            (ObservableSpec(name=pathway.detection),),
        )

    def propagate(self, vector, delay):
        """Vector and delay -> Krylov action ``exp(-i H delay) vector``."""
        delay = float(delay)
        vector = np.asarray(vector, dtype=np.complex128).reshape(-1)
        if delay == 0:
            return vector.copy()

        # Cache branches shared by pathways. Keys include vector bytes, so
        # max_cache_entries intentionally bounds memory use.
        key = None
        if self.cache_propagations:
            fingerprint = hashlib.blake2b(
                vector.view(np.uint8), digest_size=16
            ).digest()
            key = (round(delay, 14), fingerprint)
            cached = self._time_cache.get(key)
            if cached is not None:
                self._time_cache.move_to_end(key)
                return cached.copy()

        propagated = np.asarray(
            expm_multiply(
                (-1j * delay) * self._hamiltonian,
                vector,
                traceA=0.0,
            ),
            dtype=np.complex128,
        )
        if key is not None:
            self._time_cache[key] = propagated.copy()
            self._time_cache.move_to_end(key)
            while len(self._time_cache) > self.max_cache_entries:
                self._time_cache.popitem(last=False)
        return propagated

    def solve_hilbert_resolvent(
        self,
        vector,
        omega,
        *,
        eta=None,
        initial_guess=None,
    ):
        """Vector and frequency -> GMRES Hilbert resolvent."""
        vector = np.asarray(vector, dtype=np.complex128).reshape(-1)
        eta = self.eta if eta is None else float(eta)
        shift = complex(float(omega), eta)
        dimension = self.layout.total_dimension

        operator = LinearOperator(
            (dimension, dimension),
            dtype=np.complex128,
            matvec=lambda value: shift * value
            - self._hamiltonian.matvec(value),
            rmatvec=lambda value: np.conj(shift) * value
            - self._hamiltonian.rmatvec(value),
        )
        preconditioner = (
            None
            if self._hamiltonian_diagonal is None
            else self._diagonal_preconditioner(
                shift - self._hamiltonian_diagonal
            )
        )
        solution, info, iterations = self._gmres(
            operator, vector, preconditioner, initial_guess=initial_guess
        )
        residual = float(np.linalg.norm(operator @ solution - vector))
        norm = float(np.linalg.norm(vector))
        diagnostics = {
            "gmres_info": int(info),
            "iterations": iterations,
            "residual_norm": residual,
            "relative_residual": residual / norm if norm > 0 else 0.0,
            "omega": float(omega),
            "eta": eta,
        }
        if info != 0:
            raise ConvergenceError(
                f"GMRES did not converge (info={info}, residual={residual:.3e})."
            )
        return np.asarray(solution, dtype=np.complex128), diagnostics
