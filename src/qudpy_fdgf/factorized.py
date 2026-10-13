"""Factorized evaluation of pathway responses on a grid.

A pathway response is the linear functional ``<<d| O_K ... O_1 |rho_0>>`` of a chain of operations: the
interaction superoperators, the propagators ``exp(L t)`` of the time intervals and the resolvents of the
frequency intervals. When the coordinates of two or more intervals are scanned, the pointwise evaluation
repeats the same solves at every grid point: for two axes of ``N`` points it solves ``2 N**2`` shifted
systems per pathway. Cutting the chain between two scanned intervals separates it into a forward part,
evaluated once per point of its own axes, and an adjoint part, evaluated once per point of the others:

    S[i, j] = <<G_j | F_i>>,    F_i = O_{c-1} ... O_1 rho_0,    G_j = (O_K ... O_c)^+ d,

which needs ``N_left + N_right`` solves instead of ``N_left * N_right`` and one matrix product to
contract. Chains that begin the same way (the three rephasing pathways of a two-level system all begin
with the same bra interaction and the same resolvent) share their common prefix, and the observables
share the forward part, since each is a different detection vector ``d``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math

import numpy as np

_COMPLEX_BYTES = 16
_OBSERVABLE = "_observable"       # label of the axis that indexes the detection vectors


@dataclass(frozen=True)
class Operation:
    """One step of a pathway chain, in the order of application."""

    kind: str                         # "interaction", "time" or "frequency"
    key: tuple                        # identity of the operator, without its coordinate
    axis: str | None = None           # name of the scanned axis; None for a fixed coordinate
    value: float | None = None        # fixed coordinate
    eta: float | None = None          # broadening of a frequency interval
    interaction: object = None

    @property
    def tag(self):
        """Hashable identity used to recognize shared prefixes."""
        if self.axis is not None:
            return self.key + ("axis", self.axis)
        if self.value is None:
            return self.key
        return self.key + ("value", round(float(self.value), 14))


def pathway_operations(backend, pathway, protocol, scanned, fixed):
    """Pathway and protocol -> ordered operations; ``scanned`` names the axes, ``fixed`` the other values."""
    operations = []
    for interaction, interval in zip(pathway.interactions, protocol.intervals):
        operations.append(
            Operation(
                "interaction",
                (
                    "V",
                    interaction.operator,
                    interaction.direction,
                    interaction.source_sector,
                    interaction.target_sector,
                    interaction.side,
                ),
                interaction=interaction,
            )
        )
        if interval.domain == "identity":
            continue
        axis = interval.name if interval.name in scanned else None
        value = None if axis is not None else fixed[interval.name]
        if interval.domain == "time":
            operations.append(Operation("time", ("U",), axis=axis, value=value))
        else:
            eta = backend.eta if interval.eta is None else float(interval.eta)
            operations.append(
                Operation("frequency", ("R", round(eta, 14)), axis=axis, value=value, eta=eta)
            )
    return operations


@dataclass(frozen=True)
class Plan:
    """Where the chain is cut and what the cut is expected to cost."""

    cut: int                  # operations[:cut] are applied forward, operations[cut:] as adjoints
    left_axes: tuple          # scanned axes of the forward part, in chain order
    right_axes: tuple         # scanned axes of the adjoint part, in chain order
    solves: int               # scanned-operation applications per chain, without sharing
    pointwise_solves: int     # the same for the pointwise evaluation
    left_points: int
    right_points: int

    def describe(self):
        return {
            "cut": self.cut,
            "left_axes": self.left_axes,
            "right_axes": self.right_axes,
            "estimated_solves_per_pathway_and_observable": self.solves,
            "pointwise_solves_per_pathway": self.pointwise_solves,
        }


def plan_factorization(operations, sizes, n_observables=1):
    """Operations and axis sizes -> best cut, or ``None`` when fewer than two axes are scanned.

    The cost counts applications of a scanned operation to a vector for one pathway and ``n_observables``
    detection vectors (the forward part is shared by the observables), without the sharing between pathways.
    """
    scanned = [index for index, operation in enumerate(operations) if operation.axis is not None]
    if len(scanned) < 2:
        return None
    axes = [operations[index].axis for index in scanned]
    counts = [int(sizes[axis]) for axis in axes]
    total = math.prod(counts)
    best = None
    for split in range(1, len(scanned)):
        left, right = counts[:split], counts[split:]
        cost_left = sum(math.prod(left[: q + 1]) for q in range(split))
        cost_right = sum(math.prod(right[q:]) for q in range(len(right)))
        cost = cost_left + n_observables * cost_right
        candidate = (cost, max(math.prod(left), math.prod(right)), split)
        if best is None or candidate < best:
            best = candidate
    cost, _, split = best
    left_points, right_points = math.prod(counts[:split]), math.prod(counts[split:])
    # operations between the two scanned intervals go to the side that has fewer vectors
    cut = scanned[split] if left_points <= right_points * n_observables else scanned[split - 1] + 1
    return Plan(
        cut=cut,
        left_axes=tuple(axes[:split]),
        right_axes=tuple(axes[split:]),
        solves=cost,
        pointwise_solves=len(scanned) * total,
        left_points=left_points,
        right_points=right_points,
    )


class _Chain:
    """Start state and operations of one side of one pathway, with the keys of its intermediate states.

    The start is an array with one column per start vector (one for ``rho_0``; one per observable for
    the detection vectors, labelled ``_observable``).
    """

    def __init__(self, side, start, labels, dims, operations, adjoint):
        self.start = (np.asarray(start, dtype=np.complex128), list(labels), list(dims))
        self.operations = operations
        self.adjoint = adjoint
        keys, prefix = [], (side,)
        for operation in operations:
            prefix = prefix + (operation.tag,)
            keys.append(prefix)
        self.keys = keys


class _ChainCache:
    """Applies chains to their start vectors, sharing common prefixes.

    A state is kept only while another chain still needs it (reference counts), so the memory in use
    is that of the shared prefixes plus the chain being evaluated.
    """

    def __init__(self, backend, grids, chains):
        self.backend = backend
        self.grids = grids
        self.memo = {}
        self.references = Counter()
        for chain in chains:
            for key in chain.keys:
                self.references[key] += 1

    def _release(self, key):
        self.references[key] -= 1
        if self.references[key] <= 0:
            self.memo.pop(key, None)

    def _apply(self, state, operation, adjoint):
        array, labels, dims = state
        backend = self.backend
        if operation.kind == "interaction":
            return backend.interaction_block(array, operation.interaction, adjoint=adjoint), labels, dims
        if operation.axis is None:
            block = backend.interval_block(
                array, operation.kind, [operation.value], operation.eta, adjoint=adjoint
            )
            return block[:, :, 0], labels, dims
        values = self.grids[operation.axis]
        block = backend.interval_block(array, operation.kind, values, operation.eta, adjoint=adjoint)
        rows, columns, points = block.shape
        return (
            block.reshape(rows, columns * points),
            labels + [operation.axis],
            dims + [points],
        )

    def leaf(self, chain):
        """Chain -> state after its last operation. Call ``release_leaf`` once it has been used."""
        keys = chain.keys
        hit = -1
        for position in range(len(keys) - 1, -1, -1):
            if keys[position] in self.memo:
                hit = position
                break
        for position in range(hit):          # skipped: this chain no longer needs them
            self._release(keys[position])
        if hit >= 0:
            state = self.memo[keys[hit]]
        else:
            state = chain.start
        for position in range(hit + 1, len(keys)):
            new = self._apply(state, chain.operations[position], chain.adjoint)
            if position > 0:                 # the previous state has been consumed by this chain
                self._release(keys[position - 1])
            if self.references[keys[position]] > 1:
                self.memo[keys[position]] = new
            state = new
        return state

    def release_leaf(self, chain):
        self._release(chain.keys[-1])


@dataclass
class FactorizedResult:
    pathway_values: dict
    observable_values: dict
    info: dict


def evaluate_factorized(
    backend,
    protocol,
    axis_names,
    axis_values,
    fixed,
    pathways,
    specs,
    plan,
    *,
    memory_limit_bytes=None,
):
    """Evaluate every pathway and observable on the grid by cutting the chains at ``plan.cut``.

    Returns ``FactorizedResult`` with values of shape ``tuple(len(v) for v in axis_values)``.
    """
    sizes = {name: len(values) for name, values in zip(axis_names, axis_values)}
    full_grids = dict(zip(axis_names, axis_values))
    shape = tuple(sizes[name] for name in axis_names)
    scanned = set(axis_names)
    vector_bytes = backend.layout.total_dimension ** 2 * _COMPLEX_BYTES
    n_specs = len(specs)

    for pathway in pathways:
        protocol.validate_pathway(pathway)
        backend._check_transition_operators(pathway)
    operations = [pathway_operations(backend, p, protocol, scanned, fixed) for p in pathways]
    detectors = np.column_stack([backend.detector_vector(spec) for spec in specs])
    start = backend.liouville_initial_vector()[:, None]
    cut = plan.cut

    left = [_Chain("L", start, [], [], ops[:cut], adjoint=False) for ops in operations]
    right = [
        _Chain("R", detectors, [_OBSERVABLE], [n_specs], list(reversed(ops[cut:])), adjoint=True)
        for ops in operations
    ]

    left_vectors = len({chain.keys[-1] for chain in left}) * plan.left_points
    right_vectors = len({chain.keys[-1] for chain in right}) * plan.right_points * n_specs
    stored_side = "left" if left_vectors <= right_vectors else "right"
    stored_chains, stream_chains = (left, right) if stored_side == "left" else (right, left)
    stored_vectors = min(left_vectors, right_vectors)
    stream_leaves = len({chain.keys[-1] for chain in stream_chains})
    stream_axes = plan.right_axes if stored_side == "left" else plan.left_axes
    stream_width = n_specs if stored_side == "left" else 1
    # axis of the streamed side that its chains meet first; vectors held per index of that axis
    first_axis = stream_axes[-1] if stored_side == "left" else stream_axes[0]
    per_index = (
        stream_leaves * stream_width * math.prod(sizes[a] for a in stream_axes if a != first_axis)
    )

    slab = sizes[first_axis]
    if memory_limit_bytes is not None:
        budget = max(int(memory_limit_bytes // vector_bytes) - stored_vectors, 0)
        slab = int(min(max(budget // (2 * per_index), 1), sizes[first_axis]))
    slabs = math.ceil(sizes[first_axis] / slab)

    # stored side: whole grid, kept until the end
    cache = _ChainCache(backend, full_grids, stored_chains)
    stored = []
    for chain in stored_chains:
        stored.append(cache.leaf(chain))
        cache.release_leaf(chain)
    del cache

    pathway_values = {pathway.name: np.zeros(shape, dtype=np.complex128) for pathway in pathways}
    observable_values = {
        spec.name: {pathway.name: np.zeros(shape, dtype=np.complex128) for pathway in pathways}
        for spec in specs
    }
    position = axis_names.index(first_axis)

    def contract(pathway_index, forward, adjoint, selection):
        f_array, f_labels, f_dims = forward
        g_array, g_labels, g_dims = adjoint
        block = (np.conj(g_array).T @ f_array).reshape(g_dims + f_dims)
        labels = g_labels + f_labels
        block = block.transpose(
            [labels.index(_OBSERVABLE)] + [labels.index(name) for name in axis_names]
        )
        pathway = pathways[pathway_index]
        for spec_index, spec in enumerate(specs):
            contribution = pathway.response_prefactor * block[spec_index]
            observable_values[spec.name][pathway.name][selection] = contribution
            if spec.name == pathway.detection:
                pathway_values[pathway.name][selection] = contribution

    for lo in range(0, sizes[first_axis], slab):
        hi = min(lo + slab, sizes[first_axis])
        grids = dict(full_grids)
        grids[first_axis] = full_grids[first_axis][lo:hi]
        selection = [slice(None)] * len(axis_names)
        selection[position] = slice(lo, hi)
        selection = tuple(selection)
        cache = _ChainCache(backend, grids, stream_chains)
        for pathway_index, chain in enumerate(stream_chains):
            streamed = cache.leaf(chain)
            if stored_side == "left":
                contract(pathway_index, stored[pathway_index], streamed, selection)
            else:
                contract(pathway_index, streamed, stored[pathway_index], selection)
            cache.release_leaf(chain)
        del cache

    info = plan.describe()
    info.update(
        {
            "stored_side": stored_side,
            "stored_vectors": int(stored_vectors),
            "slabs": int(slabs),
            "detectors": len(specs),
        }
    )
    return FactorizedResult(pathway_values, observable_values, info)
