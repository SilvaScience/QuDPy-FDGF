"""Factorized evaluation of a grid: same numbers as the pointwise evaluation, far fewer solves."""
import sys
from pathlib import Path

import numpy as np
import pytest

from helpers import (
    BACKENDS,
    GAMMA_1,
    GAMMA_PHI,
    H,
    L_RAD,
    MU,
    PROTOCOL,
    REPHASING,
    SIGMA_Z,
    three_site_chain,
    two_level_solver,
)
from qudpy_fdgf import (
    EigenbasisKModel,
    FrequencyPathway,
    Interaction,
    ObservableSpec,
    PropagationInterval,
    SpectroscopyProtocol,
    SpectroscopySolver,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "regression"))
import regression_cases as cases  # noqa: E402

TIGHT = {"krylov_tolerance": 1e-12}
AXES = {"w1": np.linspace(-2.1, -1.9, 7), "w3": np.linspace(1.9, 2.1, 6)}
FIXED = {"t2": 10.0}


def solver_for(backend, **options):
    options = {**TIGHT, **options} if backend == "sparse_sector" else options
    solver = SpectroscopySolver(backend=backend, eta=0.002, **options)
    solver.feed_model(EigenbasisKModel(
        H, MU, c_ops_raw=((L_RAD, GAMMA_1), (SIGMA_Z, GAMMA_PHI / 2)),
        observable_op_arrays={"polarization": MU, "excited_population": np.diag([0.0, 1.0]).astype(complex)}))
    return solver


def observables(solver):
    return {
        "polarization": "polarization",
        "population": ObservableSpec.action("population", fourth_interaction="Bu", operator="excited_population"),
        "fluorescence": ObservableSpec.mean_jump("fluorescence", solver.jump_channel_names()[0],
                                                 time_window=(0.0, 125.0), fourth_interaction="Bu"),
        "fluorescence_late": ObservableSpec.mean_jump("fluorescence_late", solver.jump_channel_names()[0],
                                                      time_window=(5.0, 50.0), efficiency=0.5),
        "fluorescence_trapezoid": ObservableSpec.mean_jump(
            "fluorescence_trapezoid", solver.jump_channel_names()[0], time_window=(0.0, 60.0),
            n_steps=9, integration="trapezoid", fourth_interaction="Bu"),
    }


def assert_same(first, second, tolerance):
    assert set(first.pathways) == set(second.pathways)
    for name in first.pathways:
        scale = np.abs(first.pathways[name]).max()
        assert np.abs(first.pathways[name] - second.pathways[name]).max() <= tolerance * scale, name
    for observable, per_pathway in first.observables.items():
        for name, values in per_pathway.items():
            reference = second.observables[observable][name]
            scale = max(np.abs(values).max(), 1e-300)
            assert np.abs(values - reference).max() <= tolerance * scale, (observable, name)


@pytest.mark.parametrize("backend", BACKENDS)
def test_factorized_equals_pointwise_for_every_kind_of_observable(backend):
    solver = solver_for(backend)
    axes = {"w1": AXES["w1"][:4], "w3": AXES["w3"][:3]}              # the pointwise trapezoid detector is slow
    kwargs = dict(fixed_coordinates=FIXED, pathways=REPHASING, observables=observables(solver), warn_null=False)
    pointwise = solver.generate_spectrum(PROTOCOL, axes, strategy="pointwise", **kwargs)
    factorized = solver.generate_spectrum(PROTOCOL, axes, strategy="factorized", **kwargs)
    assert pointwise.diagnostics["strategy"] == "pointwise"
    assert factorized.diagnostics["strategy"] == "factorized"
    assert_same(pointwise, factorized, 1e-10)
    assert_same(pointwise, solver.generate_spectrum(PROTOCOL, axes, **kwargs), 1e-10)       # auto


def test_solve_counts_of_the_two_evaluations():
    """Pointwise: 2 N1 N2 solves per pathway. Factorized: N1 (forward, shared) + one adjoint chain per observable."""
    solver = solver_for("sparse_sector")
    n1, n2, pathways = len(AXES["w1"]), len(AXES["w3"]), len(REPHASING)
    kwargs = dict(fixed_coordinates=FIXED, pathways=REPHASING, warn_null=False)
    pointwise = solver.generate_spectrum(PROTOCOL, AXES, strategy="pointwise", **kwargs)
    assert pointwise.diagnostics["resolvent_solves"] == 2 * n1 * n2 * pathways
    one = solver.generate_spectrum(PROTOCOL, AXES, strategy="factorized", **kwargs)
    assert one.diagnostics["resolvent_solves"] == n1 + n2                    # all pathways share both resolvents
    several = solver.generate_spectrum(PROTOCOL, AXES, strategy="factorized", observables=observables(solver), **kwargs)
    assert several.diagnostics["resolvent_solves"] == n1 + 5 * n2             # five detection vectors


def test_dense_backend_factorizes_each_matrix_once_for_all_detection_vectors():
    solver = solver_for("dense")
    result = solver.generate_spectrum(PROTOCOL, AXES, fixed_coordinates=FIXED, pathways=REPHASING,
                                      observables=observables(solver), strategy="factorized", warn_null=False)
    assert result.diagnostics["resolvent_factorizations"] == len(AXES["w1"]) + len(AXES["w3"])


def test_auto_takes_the_factorized_evaluation_when_it_applies():
    solver = solver_for("sparse_sector")
    result = solver.generate_spectrum(PROTOCOL, AXES, fixed_coordinates=FIXED, pathways=REPHASING, warn_null=False)
    assert result.diagnostics["strategy"] == "factorized"
    info = result.diagnostics["factorization"]
    assert info["left_axes"] == ("w1",) and info["right_axes"] == ("w3",)
    assert info["slabs"] == 1
    assert result.diagnostics["pathways"]["GSB"] == {}                       # no per-point record


def test_one_axis_is_pointwise_and_cannot_be_forced():
    solver = solver_for("sparse_sector")
    axes = {"w1": AXES["w1"]}
    fixed = {"w3": 2.0, **FIXED}
    auto = solver.generate_spectrum(PROTOCOL, axes, fixed_coordinates=fixed, pathways=REPHASING, warn_null=False)
    assert auto.diagnostics["strategy"] == "pointwise"
    with pytest.raises(ValueError, match="two scanned axes"):
        solver.generate_spectrum(PROTOCOL, axes, fixed_coordinates=fixed, pathways=REPHASING, strategy="factorized")


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError, match="strategy"):
        solver_for("dense").generate_spectrum(PROTOCOL, AXES, fixed_coordinates=FIXED, pathways=REPHASING,
                                              strategy="fast")


@pytest.mark.parametrize("backend", BACKENDS)
def test_axes_in_any_order_and_three_axes(backend):
    solver = solver_for(backend)
    reordered = {"w3": AXES["w3"], "w1": AXES["w1"]}
    kwargs = dict(fixed_coordinates=FIXED, pathways=REPHASING, warn_null=False)
    first = solver.generate_spectrum(PROTOCOL, reordered, strategy="pointwise", **kwargs)
    second = solver.generate_spectrum(PROTOCOL, reordered, strategy="factorized", **kwargs)
    assert second.shape == (len(AXES["w3"]), len(AXES["w1"]))
    assert_same(first, second, 1e-10)
    # w1, t2 and w3 all scanned
    axes = {"w1": AXES["w1"][:4], "t2": np.array([0.0, 3.0, 10.0]), "w3": AXES["w3"][:3]}
    kwargs = dict(pathways=REPHASING, warn_null=False)
    first = solver.generate_spectrum(PROTOCOL, axes, strategy="pointwise", **kwargs)
    second = solver.generate_spectrum(PROTOCOL, axes, strategy="factorized", **kwargs)
    assert_same(first, second, 1e-10)
    assert second.diagnostics["factorization"]["left_axes"] + second.diagnostics["factorization"]["right_axes"] \
        == ("w1", "t2", "w3")


@pytest.mark.parametrize("memory_limit_mb, left_first", [(1e-6, True), (1e-6, False)])
@pytest.mark.parametrize("backend", BACKENDS)
def test_memory_limit_processes_the_grid_in_slabs(backend, memory_limit_mb, left_first):
    solver = solver_for(backend)
    axes = {"w1": np.linspace(-2.1, -1.9, 9), "w3": np.linspace(1.9, 2.1, 4)}
    if not left_first:
        axes = {"w1": np.linspace(-2.1, -1.9, 4), "w3": np.linspace(1.9, 2.1, 9)}
    kwargs = dict(fixed_coordinates=FIXED, pathways=REPHASING, observables=observables(solver), warn_null=False)
    whole = solver.generate_spectrum(PROTOCOL, axes, strategy="factorized", **kwargs)
    slabbed = solver.generate_spectrum(PROTOCOL, axes, strategy="factorized", memory_limit_mb=memory_limit_mb, **kwargs)
    assert whole.diagnostics["factorization"]["slabs"] == 1
    assert slabbed.diagnostics["factorization"]["slabs"] > 1
    assert slabbed.diagnostics["resolvent_solves"] == whole.diagnostics["resolvent_solves"]
    assert_same(whole, slabbed, 1e-12)


def test_auto_falls_back_to_pointwise_when_even_one_slab_does_not_fit():
    solver = solver_for("sparse_sector")
    result = solver.generate_spectrum(PROTOCOL, AXES, fixed_coordinates=FIXED, pathways=REPHASING,
                                      warn_null=False, memory_limit_mb=1e-6)
    assert result.diagnostics["strategy"] == "pointwise"


@pytest.mark.parametrize("backend", BACKENDS)
def test_regression_cases_agree_between_the_two_evaluations(backend):
    for name, case in (
        ("two_level", lambda **o: cases._two_level(backend, **o)),
        ("chain_dissipative", lambda **o: cases._chain_dissipative(backend, **o)),
        ("bosons_chi5", lambda **o: cases._bosons_chi5(backend, **o)),
    ):
        pointwise, factorized = case(strategy="pointwise"), case(strategy="factorized")
        assert set(pointwise) == set(factorized)
        for key, values in pointwise.items():
            scale = max(np.abs(values).max(), 1e-300)
            assert np.abs(values - factorized[key]).max() <= 1e-9 * scale, (name, key)


def test_thermal_mixed_state_with_the_diagonal_preconditioner():
    pointwise = cases._chain_thermal_diagonal(strategy="pointwise")
    factorized = cases._chain_thermal_diagonal(strategy="factorized")
    for key, values in pointwise.items():
        assert np.abs(values - factorized[key]).max() <= 1e-9 * np.abs(values).max(), key


def time_time_protocol():
    return SpectroscopyProtocol(intervals=(
        PropagationInterval("t1", "time", 1),
        PropagationInterval("w2", "frequency"),
        PropagationInterval("t3", "time", 1),
    ))


@pytest.mark.parametrize("backend", BACKENDS)
def test_two_time_axes_with_a_fixed_frequency_interval(backend):
    options = TIGHT if backend == "sparse_sector" else {}
    solver = SpectroscopySolver(backend=backend, eta=0.01, **options)
    solver.feed_model(three_site_chain(dissipative=True))
    axes = {"t1": np.array([0.0, 4.0, 9.0]), "t3": np.array([0.0, 2.0, 5.0, 11.0])}
    kwargs = dict(fixed_coordinates={"w2": 0.05}, pathways=REPHASING, warn_null=False)
    first = solver.generate_spectrum(time_time_protocol(), axes, strategy="pointwise", **kwargs)
    second = solver.generate_spectrum(time_time_protocol(), axes, strategy="factorized", **kwargs)
    assert_same(first, second, 1e-9)
    assert second.diagnostics["propagations"] > 0


def test_closed_pure_time_domain_problem_keeps_the_rank_one_route():
    solver = SpectroscopySolver(backend="sparse_sector", eta=0.01)
    solver.feed_model(three_site_chain(dissipative=False))
    protocol = SpectroscopyProtocol(intervals=(
        PropagationInterval("t1", "time", 1), PropagationInterval("t2", "time"), PropagationInterval("t3", "time", 1)))
    axes = {"t1": np.array([0.0, 5.0]), "t3": np.array([0.0, 5.0])}
    kwargs = dict(fixed_coordinates={"t2": 2.0}, pathways=REPHASING, warn_null=False)
    assert solver.generate_spectrum(protocol, axes, **kwargs).diagnostics["strategy"] == "pointwise"
    with pytest.raises(ValueError, match="rank-one"):
        solver.generate_spectrum(protocol, axes, strategy="factorized", **kwargs)
    # the dense backend has no such route and factorizes it
    dense = SpectroscopySolver(backend="dense", eta=0.01)
    dense.feed_model(three_site_chain(dissipative=False))
    assert dense.generate_spectrum(protocol, axes, **kwargs).diagnostics["strategy"] == "factorized"


def test_nq_spectrum_accepts_the_strategy():
    solver = solver_for("dense")
    first = solver.generate_nq_spectrum(1, PROTOCOL, AXES, fixed_coordinates=FIXED, pathways=REPHASING,
                                        strategy="pointwise", warn_null=False)
    second = solver.generate_nq_spectrum(1, PROTOCOL, AXES, fixed_coordinates=FIXED, pathways=REPHASING,
                                         strategy="factorized", warn_null=False)
    assert second.diagnostics["strategy"] == "factorized"
    assert np.abs(first.components["1Q"] - second.components["1Q"]).max() <= 1e-10 * np.abs(first.components["1Q"]).max()


def test_pathways_that_share_a_prefix_share_the_solves():
    """Two copies of a pathway, and a third that differs after the first resolvent, add no solve for the first part."""
    solver = solver_for("sparse_sector")
    twin = FrequencyPathway(name="GSB_copy", interactions=("Bu", "Bd", "Ku"), component="rephasing")
    result = solver.generate_spectrum(PROTOCOL, AXES, fixed_coordinates=FIXED, pathways=list(REPHASING) + [twin],
                                      strategy="factorized", warn_null=False)
    assert result.diagnostics["resolvent_solves"] == len(AXES["w1"]) + len(AXES["w3"])
    assert np.array_equal(result.pathways["GSB"], result.pathways["GSB_copy"])


def test_a_pathway_that_vanishes_is_still_reported_by_the_factorized_evaluation():
    solver = solver_for("dense")
    dead = FrequencyPathway(name="dead", interactions=("Kd", "Bd", "Ku"), component="rephasing")      # lowers the ground state
    result = solver.generate_spectrum(PROTOCOL, AXES, fixed_coordinates=FIXED, pathways=[REPHASING[0], dead],
                                      strategy="factorized", warn_null=False)
    assert result.null_pathways() == ("dead",)


# ---------------------------------------------------------------------- primitives of the adjoint chain
@pytest.mark.parametrize("backend", BACKENDS)
def test_adjoint_primitives_satisfy_the_adjoint_identity(backend):
    base = solver_for(backend).backend
    size = base.layout.total_dimension ** 2
    rng = np.random.default_rng(5)
    x = rng.standard_normal((size, 3)) + 1j * rng.standard_normal((size, 3))
    y = rng.standard_normal((size, 3)) + 1j * rng.standard_normal((size, 3))
    for label in ("Ku", "Kd", "Bu", "Bd"):
        interaction = Interaction(label)
        forward = base.interaction_block(x, interaction)
        adjoint = base.interaction_block(y, interaction, adjoint=True)
        assert np.vdot(y[:, 1], forward[:, 2]) == pytest.approx(np.vdot(adjoint[:, 1], x[:, 2]), abs=1e-12)
    for domain, values in (("frequency", [1.9, -2.0]), ("time", [0.0, 3.0])):
        forward = base.interval_block(x, domain, values)
        adjoint = base.interval_block(y, domain, values, adjoint=True)
        for index in range(len(values)):
            assert np.vdot(y[:, 0], forward[:, 2, index]) == pytest.approx(
                np.vdot(adjoint[:, 0, index], x[:, 2]), abs=1e-10)


@pytest.mark.parametrize("backend", BACKENDS)
def test_interaction_block_is_the_column_by_column_pointwise_action(backend):
    base = solver_for(backend).backend
    size = base.layout.total_dimension
    rng = np.random.default_rng(6)
    states = rng.standard_normal((size * size, 4)) + 1j * rng.standard_normal((size * size, 4))
    for label in ("Ku", "Kd", "Bu", "Bd"):
        interaction = Interaction(label)
        block = base.interaction_block(states, interaction)
        for column in range(4):
            if backend == "dense":
                expected = base._interaction_superoperator(interaction) @ states[:, column]
            else:
                action = base._apply_transition_left if interaction.side == "ket" else base._apply_transition_right
                expected = action(states[:, column], interaction)
            assert block[:, column] == pytest.approx(expected, abs=1e-13)


def test_counters_report_the_work_of_a_pointwise_run():
    solver = solver_for("dense")
    solver.generate_spectrum(PROTOCOL, AXES, fixed_coordinates=FIXED, pathways=REPHASING,
                             strategy="pointwise", warn_null=False)
    counters = solver.backend.counters()
    assert counters["resolvent_solves"] == 2 * len(AXES["w1"]) * len(AXES["w3"]) * len(REPHASING)
    assert counters["propagations"] == len(AXES["w1"]) * len(AXES["w3"]) * len(REPHASING)
