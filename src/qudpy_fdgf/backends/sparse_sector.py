"""Sparse, sector-based exact-diagonalization backend.

The backend propagates ket and bra branches without materializing dense
Hilbert operators. It supports matrix-free time evolution and Hilbert-space
resolvents, while retaining a Liouville-vector route for frequency-domain
pathways and small-system checks.
"""

from collections import OrderedDict
import hashlib

import numpy as np
from scipy.sparse.linalg import LinearOperator, expm_multiply, gmres

from .base import BackendBase
from ..capabilities import BackendCapabilities, Capabilities
from ..exceptions import ConvergenceError
from ..observables import ObservableSpec
from ..results import PathwayResult


class SparseSectorBackend(BackendBase):
    """Pathway inputs -> sparse ket/bra branch propagation.

    Time evolution and Hilbert resolvents remain matrix-free.
    ``preconditioner="diagonal"`` preconditions every GMRES resolvent with the
    inverse of the shifted diagonal of the generator (or of the Hamiltonian).
    The diagonal is exact for a closed model supplied in its eigenbasis, so
    each solve then converges in one iteration; with collapse channels that
    couple dyads it remains an approximation.
    """

    preconditioners = (None, "diagonal")

    name = "sparse_sector"
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
        eta=0.05,
        krylov_tolerance=1e-10,
        krylov_maxiter=None,
        resolvent_restart=None,
        preconditioner=None,
        cache_propagations=True,
        max_cache_entries=32,
        **options,
    ):
        super().__init__(eta=eta, **options)
        if preconditioner not in self.preconditioners:
            raise ValueError(
                "preconditioner must be None or 'diagonal'; "
                f"got {preconditioner!r}."
            )
        self.preconditioner = preconditioner
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

    def build(self, model, context=None):
        super().build(model, context=context)
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
        return self

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

    def _apply_to_columns(self, vector, action):
        """Matrix and Hilbert action -> column-wise action."""
        dimension = self.layout.total_dimension
        matrix = np.asarray(vector, dtype=np.complex128).reshape(
            (dimension, dimension), order="F"
        )
        result = np.empty_like(matrix)
        for column in range(dimension):
            result[:, column] = action(matrix[:, column])
        return result

    def _liouville_matvec(self, density_vector):
        """Density vector -> matrix-free generator action."""
        return self.generator.matvec(density_vector)

    def _apply_transition_left(self, density_vector, interaction):
        return self._apply_to_columns(
            density_vector,
            lambda vector: self.apply_transition(
                vector,
                operator_name=interaction.operator,
                direction=interaction.direction,
                source_sector=interaction.source_sector,
                target_sector=interaction.target_sector,
            ),
        ).reshape(-1, order="F")

    def _apply_transition_right(self, density_vector, interaction):
        """Density vector and interaction -> right bra-adjoint action."""
        dimension = self.layout.total_dimension
        rho = np.asarray(density_vector, dtype=np.complex128).reshape(
            (dimension, dimension), order="F"
        )
        acted_dagger = self._apply_to_columns(
            rho.conj().T.reshape(-1, order="F"),
            lambda vector: self.apply_transition(
                vector,
                operator_name=interaction.operator,
                direction=interaction.direction,
                source_sector=interaction.source_sector,
                target_sector=interaction.target_sector,
            ),
        )
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
        self._liouville_time_cache[key] = propagated.copy()
        self._liouville_time_cache.move_to_end(key)
        while len(self._liouville_time_cache) > self.max_cache_entries:
            self._liouville_time_cache.popitem(last=False)
        return propagated

    def _solve_liouville_resolvent(self, density_vector, omega, eta):
        shift = self.generator.frequency_shift(omega, eta)
        dimension = self.layout.total_dimension**2
        operator = LinearOperator(
            (dimension, dimension),
            dtype=np.complex128,
            matvec=lambda value: shift * value
            - self._liouville.matvec(value),
            rmatvec=lambda value: np.conj(shift) * value
            - self._liouville.rmatvec(value),
        )
        preconditioner = (
            None
            if self._liouville_diagonal is None
            else self._diagonal_preconditioner(
                shift - self._liouville_diagonal
            )
        )
        raw_solution, info = gmres(
            operator,
            density_vector,
            M=preconditioner,
            rtol=self.krylov_tolerance,
            atol=0.0,
            restart=self.resolvent_restart,
            maxiter=self.krylov_maxiter,
        )
        residual = float(
            np.linalg.norm(operator @ raw_solution - density_vector)
        )
        if info != 0:
            raise ConvergenceError(
                "Liouville GMRES resolvent did not converge "
                f"(info={info}, residual={residual:.3e})."
            )
        solution = self.generator.frequency_prefactor * raw_solution
        return np.asarray(solution, dtype=np.complex128), residual

    def _detect_density(self, density_vector, observable):
        dimension = self.layout.total_dimension
        observable_rho = self._apply_to_columns(
            density_vector,
            lambda vector: self.apply_detection_operator(vector, observable),
        )
        value = np.trace(observable_rho.reshape((dimension, dimension)))
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
        dimension = self.layout.total_dimension
        matrix = np.empty((dimension, dimension), dtype=np.complex128)
        basis = np.zeros(dimension, dtype=np.complex128)
        for index in range(dimension):
            basis[index] = 1.0
            matrix[:, index] = self.apply_detection_operator(basis, detector)
            basis[index] = 0.0
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
        solution, info = gmres(
            operator,
            vector,
            M=preconditioner,
            x0=initial_guess,
            rtol=self.krylov_tolerance,
            atol=0.0,
            restart=self.resolvent_restart,
            maxiter=self.krylov_maxiter,
        )
        residual = float(np.linalg.norm(operator @ solution - vector))
        diagnostics = {
            "gmres_info": int(info),
            "residual_norm": residual,
            "omega": float(omega),
            "eta": eta,
        }
        if info != 0:
            raise ConvergenceError(
                f"GMRES did not converge (info={info}, residual={residual:.3e})."
            )
        return np.asarray(solution, dtype=np.complex128), diagnostics
