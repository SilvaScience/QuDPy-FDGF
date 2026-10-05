"""Shared infrastructure for QuDPy-FDGF numerical backends.

This module validates external models, arranges Hilbert-space sectors, and
normalizes operator actions across dense, sparse, and matrix-free inputs. It
also implements common observable and transition handling so concrete
backends can focus on propagation and resolvent algorithms.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import inspect
import warnings

import numpy as np
from scipy.sparse import issparse
from scipy.sparse.linalg import LinearOperator

from ..capabilities import BackendCapabilities, Capabilities
from ..contracts import (
    CollapseChannel,
    DenseDensityBlock,
    DensityState,
    LowRankDensityBlock,
    PureState,
    ThermodynamicContext,
)
from ..exceptions import ModelConsistencyWarning, ModelContractError, SectorError
from ..generators import EvolutionGenerator, operator_matmat, operator_rmatmat
from ..observables import ObservableSpec


def apply_operator(operator, vector):
    """Array, sparse matrix, or operator and vector -> operator action."""
    if hasattr(operator, "matvec"):
        return np.asarray(operator.matvec(vector), dtype=np.complex128)
    return np.asarray(operator @ vector, dtype=np.complex128)


def apply_adjoint_operator(operator, vector):
    """Array, sparse matrix, or operator and vector -> adjoint action."""
    if hasattr(operator, "rmatvec"):
        return np.asarray(operator.rmatvec(vector), dtype=np.complex128)
    if hasattr(operator, "getH"):
        return np.asarray(operator.getH() @ vector, dtype=np.complex128)
    return np.asarray(operator.conj().T @ vector, dtype=np.complex128)


def _is_array_like(operator):
    return isinstance(operator, np.ndarray) or issparse(operator)


def apply_operator_matrix(operator, matrix):
    """Array, sparse matrix, or operator and matrix -> operator action on every column."""
    matrix = np.asarray(matrix, dtype=np.complex128)
    if _is_array_like(operator):
        return np.asarray(operator @ matrix, dtype=np.complex128)
    if hasattr(operator, "matvec") or hasattr(operator, "matmat"):
        return operator_matmat(operator, matrix)
    return np.asarray(operator @ matrix, dtype=np.complex128)


def apply_adjoint_operator_matrix(operator, matrix):
    """Array, sparse matrix, or operator and matrix -> adjoint action on every column.

    For arrays, O^dagger M = conj(O^T conj(M)): the transpose of a dense or CSR operator is a
    view, so no adjoint is built at each call.
    """
    matrix = np.asarray(matrix, dtype=np.complex128)
    if _is_array_like(operator):
        return np.asarray((operator.T @ matrix.conj()).conj(), dtype=np.complex128)
    if hasattr(operator, "rmatvec") or hasattr(operator, "rmatmat"):
        return operator_rmatmat(operator, matrix)
    if hasattr(operator, "getH"):
        return np.asarray(operator.getH() @ matrix, dtype=np.complex128)
    return np.asarray(operator.conj().T @ matrix, dtype=np.complex128)


def operator_shape(operator):
    """External operator -> validated shape."""
    shape = getattr(operator, "shape", None)
    if shape is None or len(shape) != 2:
        raise ModelContractError("Every operator must expose shape=(m, n).")
    return tuple(int(item) for item in shape)


@dataclass(frozen=True)
class SectorLayout:
    """Deterministic sector layout within a concatenated vector."""

    sectors: tuple
    dimensions: dict
    slices: dict
    total_dimension: int

    @classmethod
    def from_model(cls, model):
        sectors = tuple(model.sectors())
        if not sectors:
            raise ModelContractError("Model does not provide any sector.")
        if len(set(sectors)) != len(sectors):
            raise ModelContractError("Model sectors must be unique.")

        dimensions = {}
        slices = {}
        offset = 0
        for sector in sectors:
            dimension = int(model.dimension(sector))
            if dimension <= 0:
                raise ModelContractError(
                    f"Invalid dimension {dimension} for sector {sector!r}."
                )
            dimensions[sector] = dimension
            slices[sector] = slice(offset, offset + dimension)
            offset += dimension
        return cls(sectors, dimensions, slices, offset)

    def empty(self):
        return np.zeros(self.total_dimension, dtype=np.complex128)

    def sector_view(self, vector, sector):
        return vector[self.slices[sector]]

    def embed(self, sector, vector):
        if sector not in self.slices:
            raise SectorError(f"Unknown initial sector: {sector!r}")
        vector = np.asarray(vector, dtype=np.complex128).reshape(-1)
        expected = self.dimensions[sector]
        if vector.size != expected:
            raise ModelContractError(
                f"State for sector {sector!r} has size {vector.size}; "
                f"expected {expected}."
            )
        result = self.empty()
        result[self.slices[sector]] = vector
        return result


class BackendBase(ABC):
    """Base class for numerical backends."""

    name = "base"
    capabilities = Capabilities()
    backend_capabilities = BackendCapabilities(
        state_kinds=("pure",),
        supported_pairs=(("unitary", "time"),),
        exactness="exact",
        matrix_free=True,
    )

    def __init__(self, *, eta=0.05, **options):
        eta = float(eta)
        if eta < 0:
            raise ValueError("eta must be non-negative.")
        self.eta = eta
        self.options = dict(options)
        self.model = None
        self.model_capabilities = Capabilities()
        self.layout = None
        self.initial_state = None
        self.initial_condition = None
        self._initial_vector = None
        self._initial_density_matrix = None
        self._initial_is_pure = False
        self._hamiltonian_blocks = {}
        self._collapse_channels = ()
        self._collapse_operators = ()
        self._collapse_channel_map = {}
        self._transition_blocks = {}
        self._checked_transition_operators = set()
        self._observable_blocks = {}
        self._hamiltonian = None
        self.generator = None
        self.reset_counters()

    # -------------------------------------------------------------------------- solve counters
    def reset_counters(self):
        """Zero the counters of linear solves and propagations (see ``counters``)."""
        self._counters = {
            "resolvent_solves": 0,
            "resolvent_factorizations": 0,
            "propagations": 0,
        }
        self._max_residual = 0.0

    def counters(self):
        """Work done since ``reset_counters``.

        ``resolvent_solves``: resolvents applied to a vector (one GMRES run, or one solve or product
        with a dense inverse); ``resolvent_factorizations``: dense matrices factorized or inverted
        (dense backend only); ``propagations``: time propagators applied to a vector, cache hits
        excluded; ``maximum_resolvent_residual``: largest absolute residual of the GMRES solves.
        """
        result = dict(self._counters)
        result["maximum_resolvent_residual"] = self._max_residual
        return result

    def build(self, model, context=None):
        """Model and context -> initialized sector operators and state."""
        required = (
            "sectors",
            "dimension",
            "hamiltonian_blocks",
            "transition_blocks",
            "observable_blocks",
        )
        missing = [
            name for name in required if not callable(getattr(model, name, None))
        ]
        if missing:
            raise ModelContractError(
                "Model violates the contract; missing methods: "
                + ", ".join(missing)
            )

        self.model = model
        self.context = ThermodynamicContext.from_value(context)
        capabilities = getattr(model, "capabilities", None)
        capabilities = capabilities() if callable(capabilities) else capabilities
        self.model_capabilities = Capabilities.from_value(capabilities)
        self.layout = SectorLayout.from_model(model)
        self._cache_and_validate_hamiltonian()
        self._hamiltonian = LinearOperator(
            shape=(
                self.layout.total_dimension,
                self.layout.total_dimension,
            ),
            dtype=np.complex128,
            matvec=self._hamiltonian_matvec,
            rmatvec=self._hamiltonian_rmatvec,
            matmat=self._hamiltonian_matmat,
            rmatmat=self._hamiltonian_rmatmat,
        )
        self._check_hamiltonian_hermiticity()
        self._cache_and_validate_collapse_channels()
        self.generator = EvolutionGenerator(
            self._hamiltonian,
            self._collapse_operators,
        )
        self._load_initial_condition()
        return self

    def _check_hamiltonian_hermiticity(self):
        """Hamiltonian blocks -> error if ``H x != H^dagger x`` for random vectors.

        Tolerance: backend option ``hermiticity_tolerance`` (default 1e-10, relative).
        """
        tolerance = float(self.options.get("hermiticity_tolerance", 1e-10))
        generator = np.random.default_rng(12345)
        size = self.layout.total_dimension
        for _ in range(2):
            vector = generator.standard_normal(size) + 1j * generator.standard_normal(size)
            try:
                forward = self._hamiltonian.matvec(vector)
                adjoint = self._hamiltonian.rmatvec(vector)
            except (AttributeError, NotImplementedError, TypeError):
                return        # matrix-free blocks without an adjoint action: nothing to compare
            scale = max(
                float(np.linalg.norm(forward)),
                float(np.linalg.norm(adjoint)),
                float(np.finfo(float).tiny),
            )
            deviation = float(np.linalg.norm(forward - adjoint)) / scale
            if deviation > tolerance:
                raise ModelContractError(
                    "The Hamiltonian blocks are not Hermitian: "
                    f"||H x - H^dagger x|| / ||H x|| = {deviation:.2e} > {tolerance:.0e}."
                )

    def _check_transition_operators(self, pathway):
        """Pathway -> one-time check of every transition operator it uses."""
        for interaction in pathway.interactions:
            name = str(interaction.operator)
            if name not in self._checked_transition_operators:
                self._checked_transition_operators.add(name)
                self._check_transition_pair(name)

    def _check_transition_pair(self, name):
        """Raising and lowering blocks -> warning if ``J_minus`` is not ``J_plus^dagger``.

        A Hermitian light-matter operator ``mu = J_plus + J_minus`` requires it. Tolerance:
        backend option ``transition_tolerance`` (default 1e-8, relative).
        """
        tolerance = float(self.options.get("transition_tolerance", 1e-8))
        plus = {
            source: self._get_transition_blocks(name, "plus", source)
            for source in self.layout.sectors
        }
        minus = {
            source: self._get_transition_blocks(name, "minus", source)
            for source in self.layout.sectors
        }
        generator = np.random.default_rng(54321)
        vector = generator.standard_normal(
            self.layout.total_dimension
        ) + 1j * generator.standard_normal(self.layout.total_dimension)
        try:
            lowering = self._block_operator_matvec(minus, vector)
            raising_adjoint = self._block_operator_rmatvec(plus, vector)
        except (AttributeError, NotImplementedError, TypeError):
            return
        scale = max(
            float(np.linalg.norm(lowering)),
            float(np.linalg.norm(raising_adjoint)),
            float(np.finfo(float).tiny),
        )
        deviation = float(np.linalg.norm(lowering - raising_adjoint)) / scale
        if deviation > tolerance:
            warnings.warn(
                f"The lowering blocks of operator {name!r} are not the adjoint of its raising "
                f"blocks (relative deviation {deviation:.2e}); mu = J_plus + J_minus is then not "
                "Hermitian and the detected signal is not that of a physical observable.",
                ModelConsistencyWarning,
                stacklevel=4,
            )

    def _call_with_optional_context(self, method):
        try:
            parameters = inspect.signature(method).parameters
        except (TypeError, ValueError):
            parameters = None
        if parameters is not None and not parameters:
            return method()
        return method(self.context)

    def _resolve_initial_condition(self):
        if self.context is not None:
            equilibrium = getattr(self.model, "equilibrium_state", None)
            if callable(equilibrium):
                return equilibrium(self.context)

        initial_condition = getattr(self.model, "initial_condition", None)
        if callable(initial_condition):
            return self._call_with_optional_context(initial_condition)

        initial_state = getattr(self.model, "initial_state", None)
        if callable(initial_state):
            return initial_state()
        raise ModelContractError(
            "Model must expose initial_condition(context), "
            "equilibrium_state(context), or initial_state()."
        )

    def _density_block_matrix(self, block):
        if isinstance(block, DenseDensityBlock):
            return block.as_matrix()
        if isinstance(block, LowRankDensityBlock):
            X, S, Y = block.factors()
            return X @ S @ Y.conj().T
        if hasattr(block, "toarray"):
            return np.asarray(block.toarray(), dtype=np.complex128)
        matrix = np.asarray(block, dtype=np.complex128)
        if matrix.ndim != 2:
            raise ModelContractError(
                "Every DensityState block must be a matrix."
            )
        return matrix

    def _assemble_density_state(self, state):
        dimension = self.layout.total_dimension
        density = np.zeros((dimension, dimension), dtype=np.complex128)
        for (ket_sector, bra_sector), block in state.blocks.items():
            if ket_sector not in self.layout.dimensions:
                raise SectorError(
                    f"Unknown initial ket sector: {ket_sector!r}"
                )
            if bra_sector not in self.layout.dimensions:
                raise SectorError(
                    f"Unknown initial bra sector: {bra_sector!r}"
                )
            matrix = self._density_block_matrix(block)
            expected = (
                self.layout.dimensions[ket_sector],
                self.layout.dimensions[bra_sector],
            )
            if matrix.shape != expected:
                raise ModelContractError(
                    f"Density block {(ket_sector, bra_sector)!r} has shape "
                    f"{matrix.shape}; expected {expected}."
                )
            density[
                self.layout.slices[ket_sector],
                self.layout.slices[bra_sector],
            ] = matrix
        return density

    def _validate_physical_density(self, density):
        tolerance = float(
            self.options.get("density_matrix_tolerance", 1e-10)
        )
        if not np.allclose(
            density, density.conj().T, atol=tolerance, rtol=0.0
        ):
            raise ModelContractError(
                "Initial density matrix is not Hermitian."
            )
        trace = np.trace(density)
        if abs(trace.imag) > tolerance or not np.isclose(
            trace.real, 1.0, atol=tolerance, rtol=0.0
        ):
            raise ModelContractError(
                "Initial density matrix must have unit trace; "
                f"got {trace}."
            )
        eigenvalues = np.linalg.eigvalsh(
            0.5 * (density + density.conj().T)
        )
        if float(np.min(eigenvalues)) < -tolerance:
            raise ModelContractError(
                "Initial density matrix is not positive "
                f"(smallest eigenvalue={np.min(eigenvalues):.3e})."
            )

    def stationarity_residual(self):
        """Reference state -> ``||L rho|| / ||rho||`` (Frobenius norms; 0 for a stationary state).

        The perturbative response assumes that the state before the first pulse does not evolve
        under the field-free generator.
        """
        vector = np.asarray(self._initial_density_matrix, dtype=np.complex128).reshape(-1, order="F")
        norm = float(np.linalg.norm(vector))
        return float(np.linalg.norm(self.generator.matvec(vector))) / norm

    def _load_initial_condition(self):
        condition = self._resolve_initial_condition()
        if not isinstance(condition, (PureState, DensityState)):
            if isinstance(condition, tuple) and len(condition) in {2, 3}:
                condition = PureState(*condition)
            else:
                raise ModelContractError(
                    "Initial condition must be PureState, DensityState, "
                    "or (sector, vector[, energy])."
                )

        self.initial_condition = condition
        if isinstance(condition, PureState):
            vector = condition.normalized_vector()
            self.initial_state = PureState(
                condition.sector,
                vector,
                condition.energy,
                dict(condition.metadata),
            )
            self._initial_vector = self.layout.embed(
                condition.sector, vector
            )
            self._initial_density_matrix = np.outer(
                self._initial_vector, self._initial_vector.conj()
            )
            self._initial_is_pure = True
            return

        density = self._assemble_density_state(condition)
        self._validate_physical_density(density)
        self.initial_state = condition
        self._initial_vector = None
        self._initial_density_matrix = density
        self._initial_is_pure = False

    def _normalize_blocks(self, blocks, *, source, family):
        if blocks is None:
            return {}
        try:
            blocks = dict(blocks)
        except (TypeError, ValueError) as exc:
            raise ModelContractError(
                f"{family}({source!r}) must return a mapping."
            ) from exc
        normalized = {}
        source_dimension = self.layout.dimensions[source]
        for target, operator in blocks.items():
            if target not in self.layout.dimensions:
                raise SectorError(
                    f"{family} connects {source!r} to unknown sector {target!r}."
                )
            expected = (
                self.layout.dimensions[target],
                source_dimension,
            )
            actual = operator_shape(operator)
            if actual != expected:
                raise ModelContractError(
                    f"{family} block {source!r}->{target!r} has shape {actual}; "
                    f"expected shape {expected}."
                )
            normalized[target] = operator
        return normalized

    def _cache_and_validate_hamiltonian(self):
        for source in self.layout.sectors:
            blocks = self.model.hamiltonian_blocks(source)
            self._hamiltonian_blocks[source] = self._normalize_blocks(
                blocks,
                source=source,
                family="hamiltonian_blocks",
            )

    def _matrix_argument(self, matrix):
        matrix = np.asarray(matrix, dtype=np.complex128)
        if matrix.ndim != 2 or matrix.shape[0] != self.layout.total_dimension:
            raise ValueError("Size is incompatible with the Hilbert space.")
        return matrix

    def _hamiltonian_matmat(self, matrix):
        matrix = self._matrix_argument(matrix)
        result = np.zeros(matrix.shape, dtype=np.complex128)
        for source, blocks in self._hamiltonian_blocks.items():
            source_matrix = matrix[self.layout.slices[source]]
            for target, operator in blocks.items():
                result[self.layout.slices[target]] += apply_operator_matrix(
                    operator, source_matrix
                )
        return result

    def _hamiltonian_rmatmat(self, matrix):
        matrix = self._matrix_argument(matrix)
        result = np.zeros(matrix.shape, dtype=np.complex128)
        for source, blocks in self._hamiltonian_blocks.items():
            for target, operator in blocks.items():
                result[self.layout.slices[source]] += apply_adjoint_operator_matrix(
                    operator, matrix[self.layout.slices[target]]
                )
        return result

    def _hamiltonian_matvec(self, vector):
        vector = np.asarray(vector, dtype=np.complex128).reshape(-1)
        return self._hamiltonian_matmat(vector[:, None])[:, 0]

    def _hamiltonian_rmatvec(self, vector):
        vector = np.asarray(vector, dtype=np.complex128).reshape(-1)
        return self._hamiltonian_rmatmat(vector[:, None])[:, 0]

    def _block_operator_matmat(self, blocks_by_source, matrix):
        matrix = self._matrix_argument(matrix)
        result = np.zeros(matrix.shape, dtype=np.complex128)
        for source, blocks in blocks_by_source.items():
            source_matrix = matrix[self.layout.slices[source]]
            for target, operator in blocks.items():
                result[self.layout.slices[target]] += apply_operator_matrix(
                    operator, source_matrix
                )
        return result

    def _block_operator_rmatmat(self, blocks_by_source, matrix):
        matrix = self._matrix_argument(matrix)
        result = np.zeros(matrix.shape, dtype=np.complex128)
        for source, blocks in blocks_by_source.items():
            for target, operator in blocks.items():
                result[self.layout.slices[source]] += apply_adjoint_operator_matrix(
                    operator, matrix[self.layout.slices[target]]
                )
        return result

    def _block_operator_matvec(self, blocks_by_source, vector):
        vector = np.asarray(vector, dtype=np.complex128).reshape(-1)
        return self._block_operator_matmat(blocks_by_source, vector[:, None])[:, 0]

    def _block_operator_rmatvec(self, blocks_by_source, vector):
        vector = np.asarray(vector, dtype=np.complex128).reshape(-1)
        return self._block_operator_rmatmat(blocks_by_source, vector[:, None])[:, 0]

    def _cache_and_validate_collapse_channels(self):
        method = getattr(self.model, "collapse_channels", None)
        raw_channels = (
            self._call_with_optional_context(method)
            if callable(method)
            else ()
        )
        channels = []
        operators = []
        for raw_channel in raw_channels or ():
            channel = (
                raw_channel
                if isinstance(raw_channel, CollapseChannel)
                else CollapseChannel(**raw_channel)
                if isinstance(raw_channel, dict)
                else None
            )
            if channel is None:
                raise ModelContractError(
                    "collapse_channels() must return CollapseChannel objects "
                    "or compatible mappings."
                )
            normalized = {}
            for source, blocks in channel.operator_blocks.items():
                if source not in self.layout.dimensions:
                    raise SectorError(
                        f"Channel {channel.name!r}: unknown source sector "
                        f"{source!r}."
                    )
                normalized[source] = self._normalize_blocks(
                    blocks,
                    source=source,
                    family=f"collapse_channel[{channel.name}]",
                )
            dimension = self.layout.total_dimension
            operator = LinearOperator(
                (dimension, dimension),
                dtype=np.complex128,
                matvec=lambda value, blocks=normalized: (
                    self._block_operator_matvec(blocks, value)
                ),
                rmatvec=lambda value, blocks=normalized: (
                    self._block_operator_rmatvec(blocks, value)
                ),
                matmat=lambda value, blocks=normalized: (
                    self._block_operator_matmat(blocks, value)
                ),
                rmatmat=lambda value, blocks=normalized: (
                    self._block_operator_rmatmat(blocks, value)
                ),
            )
            channels.append(channel)
            operators.append((channel, operator))
        self._collapse_channels = tuple(channels)
        self._collapse_operators = tuple(operators)
        self._collapse_channel_map = {
            channel.name: (channel, operator)
            for channel, operator in self._collapse_operators
        }

    def jump_channel_names(self):
        """Return GKSL channel names available for jump counting."""
        return tuple(channel.name for channel in self._collapse_channels)

    def apply_jump_effect_matrix(self, matrix, channel_name):
        """Channel and Hilbert matrix -> ``gamma L^dagger L`` action on every column."""
        try:
            channel, operator = self._collapse_channel_map[str(channel_name)]
        except KeyError as exc:
            available = ", ".join(self.jump_channel_names()) or "none"
            raise KeyError(
                f"Unknown GKSL channel {channel_name!r}; available: {available}."
            ) from exc
        matrix = np.asarray(matrix, dtype=np.complex128)
        return channel.rate * apply_adjoint_operator_matrix(
            operator,
            apply_operator_matrix(operator, matrix),
        )

    def apply_jump_effect(self, vector, channel_name):
        """Channel and Hilbert vector -> ``gamma L^dagger L`` action."""
        vector = np.asarray(vector, dtype=np.complex128).reshape(-1)
        return self.apply_jump_effect_matrix(vector[:, None], channel_name)[:, 0]

    def apply_detection_operator_matrix(self, matrix, observable):
        """Matrix and observable -> final contraction or jump rate on every column."""
        spec = (
            observable
            if isinstance(observable, ObservableSpec)
            else ObservableSpec(name=str(observable))
        )
        if spec.kind == "operator":
            return self.apply_observable_matrix(matrix, spec.operator)
        if spec.kind == "jump_rate":
            return self.apply_jump_effect_matrix(matrix, spec.channel)
        raise ValueError(
            "An integrated_jump observable requires time-window evaluation, "
            "not a final contraction."
        )

    def apply_detection_operator(self, vector, observable):
        """State and observable -> final contraction or jump rate."""
        vector = np.asarray(vector, dtype=np.complex128).reshape(-1)
        return self.apply_detection_operator_matrix(vector[:, None], observable)[:, 0]

    def _get_transition_blocks(self, operator_name, direction, source):
        direction = str(direction).lower()
        if direction not in {"plus", "minus"}:
            raise ValueError("direction must be 'plus' or 'minus'.")
        key = (str(operator_name), direction, source)
        if key not in self._transition_blocks:
            blocks = self.model.transition_blocks(*key)
            self._transition_blocks[key] = self._normalize_blocks(
                blocks,
                source=source,
                family=f"transition_blocks[{operator_name},{direction}]",
            )
        return self._transition_blocks[key]

    def _selected_transition_blocks(
        self, operator_name, direction, source_sector, target_sector
    ):
        """Transition constraints -> list of (source, target, block) that match them."""
        selected = []
        sources = (
            (source_sector,)
            if source_sector is not None
            else self.layout.sectors
        )
        for source in sources:
            if source not in self.layout.dimensions:
                raise SectorError(f"Unknown source sector: {source!r}")
            blocks = self._get_transition_blocks(
                operator_name, direction, source
            )
            for target, operator in blocks.items():
                if target_sector is not None and target != target_sector:
                    continue
                selected.append((source, target, operator))
        if (
            target_sector is not None
            and target_sector not in self.layout.dimensions
        ):
            raise SectorError(f"Unknown target sector: {target_sector!r}")
        if (
            source_sector is not None or target_sector is not None
        ) and not selected:
            raise SectorError(
                "Constrained transition matches no model block: "
                f"operator={operator_name!r}, direction={direction!r}, "
                f"source={source_sector!r}, target={target_sector!r}."
            )
        return selected

    def apply_transition_matrix(
        self,
        matrix,
        *,
        operator_name,
        direction,
        source_sector=None,
        target_sector=None,
    ):
        """Matrix and transition -> transition applied to every column."""
        matrix = self._matrix_argument(matrix)
        result = np.zeros(matrix.shape, dtype=np.complex128)
        for source, target, operator in self._selected_transition_blocks(
            operator_name, direction, source_sector, target_sector
        ):
            result[self.layout.slices[target]] += apply_operator_matrix(
                operator, matrix[self.layout.slices[source]]
            )
        return result

    def apply_transition_adjoint_matrix(
        self,
        matrix,
        *,
        operator_name,
        direction,
        source_sector=None,
        target_sector=None,
    ):
        """Matrix and transition T -> T^dagger applied to every column."""
        matrix = self._matrix_argument(matrix)
        result = np.zeros(matrix.shape, dtype=np.complex128)
        for source, target, operator in self._selected_transition_blocks(
            operator_name, direction, source_sector, target_sector
        ):
            result[self.layout.slices[source]] += apply_adjoint_operator_matrix(
                operator, matrix[self.layout.slices[target]]
            )
        return result

    def apply_transition(
        self,
        vector,
        *,
        operator_name,
        direction,
        source_sector=None,
        target_sector=None,
    ):
        """Sector vector and transition -> concatenated target vector."""
        vector = np.asarray(vector, dtype=np.complex128).reshape(-1)
        return self.apply_transition_matrix(
            vector[:, None],
            operator_name=operator_name,
            direction=direction,
            source_sector=source_sector,
            target_sector=target_sector,
        )[:, 0]

    def _get_observable_blocks(self, observable_name, source):
        key = (str(observable_name), source)
        if key not in self._observable_blocks:
            blocks = self.model.observable_blocks(*key)
            self._observable_blocks[key] = self._normalize_blocks(
                blocks,
                source=source,
                family=f"observable_blocks[{observable_name}]",
            )
        return self._observable_blocks[key]

    def apply_observable_matrix(self, matrix, observable_name):
        """Matrix and observable -> observable applied to every column."""
        matrix = self._matrix_argument(matrix)
        result = np.zeros(matrix.shape, dtype=np.complex128)
        for source in self.layout.sectors:
            source_matrix = matrix[self.layout.slices[source]]
            for target, operator in self._get_observable_blocks(
                observable_name, source
            ).items():
                result[self.layout.slices[target]] += apply_operator_matrix(
                    operator, source_matrix
                )
        return result

    def apply_observable(self, vector, observable_name):
        """Concatenated vector and observable -> observable action."""
        vector = np.asarray(vector, dtype=np.complex128).reshape(-1)
        return self.apply_observable_matrix(vector[:, None], observable_name)[:, 0]

    def materialize(self, operator, *, chunk=64):
        """Operator action -> dense matrix for small spaces only (columns evaluated by blocks)."""
        rows, columns = operator_shape(operator)
        matrix = np.empty((rows, columns), dtype=np.complex128)
        for first in range(0, columns, chunk):
            last = min(first + chunk, columns)
            basis = np.zeros((columns, last - first), dtype=np.complex128)
            basis[np.arange(first, last), np.arange(last - first)] = 1.0
            matrix[:, first:last] = apply_operator_matrix(operator, basis)
        return matrix

    # ------------------------------------------------------- primitives of the factorized evaluation
    supports_factorized = False

    def factorized_applicable(self, protocol):
        """Protocol -> whether the factorized evaluation is available for it."""
        return bool(self.supports_factorized)

    def liouville_initial_vector(self):
        """Reference state -> column-stacked density vector."""
        return np.asarray(
            self._initial_density_matrix, dtype=np.complex128
        ).reshape(-1, order="F")

    def interaction_block(self, states, interaction, *, adjoint=False):
        """Liouville vectors (columns) -> interaction superoperator, or its adjoint, on each.

        A ket interaction is ``rho -> T rho`` (adjoint ``Y -> T^+ Y``), a bra interaction is
        ``rho -> rho T^+`` (adjoint ``Y -> Y T``); ``T`` is the transition operator of the
        interaction. Both are evaluated on the matrices of all the columns at once.
        """
        size = self.layout.total_dimension
        states = np.asarray(states, dtype=np.complex128)
        count = states.shape[1]
        cube = states.reshape((size, size, count), order="F")        # cube[x, y, c] = rho_c[x, y]
        action = (
            self.apply_transition_adjoint_matrix
            if adjoint
            else self.apply_transition_matrix
        )
        constraints = {
            "operator_name": interaction.operator,
            "direction": interaction.direction,
            "source_sector": interaction.source_sector,
            "target_sector": interaction.target_sector,
        }
        if interaction.side == "ket":
            result = action(cube.reshape(size, size * count), **constraints)
            result = result.reshape(size, size, count)
        else:
            flipped = np.conj(cube).transpose(1, 0, 2).reshape(size, size * count)
            acted = action(flipped, **constraints).reshape(size, size, count)
            result = np.conj(acted).transpose(1, 0, 2)
        return np.ascontiguousarray(result).reshape((size * size, count), order="F")

    def interval_block(self, states, domain, values, eta=None, *, adjoint=False):
        """Liouville vectors -> propagator or resolvent (or its adjoint) at several coordinates.

        ``states`` has shape ``(D**2, k)``; the result has shape ``(D**2, k, len(values))``.
        ``domain`` is ``"time"`` (``exp(L t)``) or ``"frequency"`` (the resolvent of broadening
        ``eta``). Implemented by the backends that support the factorized evaluation.
        """
        raise NotImplementedError(
            f"Backend {self.name!r} does not implement the factorized evaluation."
        )

    def detector_vector(self, spec):
        """Observable -> vector ``d`` with value ``<<d|rho>>`` for the column-stacked ``rho``.

        Follows the pointwise evaluation: the operator or jump-rate contraction (efficiency only
        for a jump rate), the projection interactions of an action-detected observable and the
        integrated detector (exact, or ``n_steps`` trapezoid samples) of an integrated jump.
        """
        scale = 1.0
        if spec.kind == "integrated_jump" and spec.integration == "exact":
            base = self._integrated_detector(spec)
            scale = spec.efficiency
        else:
            matrix = self.apply_detection_operator_matrix(
                np.eye(self.layout.total_dimension, dtype=np.complex128),
                spec.instantaneous().without_projection(),
            )
            base = matrix.conj().T.reshape(-1, order="F")
            if spec.kind == "jump_rate":
                scale = spec.efficiency
            elif spec.kind == "integrated_jump":             # trapezoid quadrature of the window
                times = np.linspace(*spec.time_window, spec.n_steps)
                weights = np.trapezoid(np.eye(spec.n_steps), times)
                block = self.interval_block(
                    base[:, None], "time", times, adjoint=True
                )[:, 0, :]
                base = block @ weights
                scale = spec.efficiency
        projections = spec.projection_interactions
        if not projections:
            return scale * np.asarray(base, dtype=np.complex128)
        total = np.zeros_like(base, dtype=np.complex128)
        for interaction in projections:
            acted = self.interaction_block(base[:, None], interaction, adjoint=True)[:, 0]
            total += np.conj(interaction.perturbative_prefactor) * acted
        return scale * total

    def clear_caches(self):
        """Clear coordinate-dependent caches."""

    @abstractmethod
    def calc_pathway(self, pathway, protocol, coordinates):
        """Pathway, protocol, and coordinates -> pathway response."""
