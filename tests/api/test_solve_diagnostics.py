"""Diagnostics of the linear solves: iterations, true relative residual, conditioning of the shifted systems."""
import numpy as np
import pytest

from helpers import PROTOCOL, REPHASING, three_site_chain, two_level_solver
from qudpy_fdgf import Interaction, SpectroscopyProtocol, SpectroscopySolver, PropagationInterval

AXES = {"w1": np.linspace(-2.15, -1.9, 4), "w3": np.linspace(1.9, 2.15, 4)}
ETA = 0.002
FIXED = {"t2": 10.0}


def run(backend, *, model=None, strategy="auto", **options):
    solver = SpectroscopySolver(backend=backend, eta=ETA, **options)
    solver.feed_model(model if model is not None else three_site_chain(dissipative=False))
    result = solver.generate_spectrum(PROTOCOL, AXES, pathways=REPHASING, fixed_coordinates=FIXED,
                                      strategy=strategy, warn_null=False)
    return solver, result


def test_sparse_result_reports_iterations_and_true_relative_residuals():
    solver, result = run("sparse_sector", krylov_tolerance=1e-10, preconditioner=None, diagonal_solve=False)
    info = result.diagnostics["solver"]
    assert info["method"] == "gmres"
    assert info["solves"] == result.diagnostics["resolvent_solves"] > 0
    assert 1 <= info["iterations_mean"] <= info["iterations_max"]
    assert 0 < info["relative_residual_mean"] <= info["relative_residual_max"] <= 1e-9
    assert "records" not in info


def test_pointwise_and_factorized_report_the_same_kind_of_statistics():
    _, pointwise = run("sparse_sector", strategy="pointwise", preconditioner=None, diagonal_solve=False)
    _, factorized = run("sparse_sector", strategy="factorized", preconditioner=None, diagonal_solve=False)
    assert pointwise.diagnostics["solver"]["solves"] > factorized.diagnostics["solver"]["solves"]
    for result in (pointwise, factorized):
        assert result.diagnostics["solver"]["relative_residual_max"] <= 1e-9


def test_relative_residual_is_the_true_residual_not_the_preconditioned_one():
    solver, _ = run("sparse_sector", krylov_tolerance=1e-6, preconditioner="diagonal",
                    model=three_site_chain(dissipative=True))
    base = solver.backend
    rng = np.random.default_rng(3)
    size = base.layout.total_dimension ** 2
    vector = rng.standard_normal(size) + 1j * rng.standard_normal(size)
    base.reset_counters()
    solution, _ = base._solve_liouville_resolvent(vector, 2.0, ETA)
    shift = base.generator.frequency_shift(2.0, ETA)
    true = np.linalg.norm(shift * solution - base.generator.matvec(solution) - vector) / np.linalg.norm(vector)
    assert base.solve_statistics()["relative_residual_max"] == pytest.approx(true, rel=1e-8, abs=1e-14)


def test_exact_diagonal_preconditioner_needs_one_iteration_and_leaves_no_residual():
    _, result = run("sparse_sector", krylov_tolerance=1e-12, preconditioner="diagonal", diagonal_solve=False)
    info = result.diagnostics["solver"]
    assert info["iterations_max"] == 1
    assert info["relative_residual_max"] < 1e-11


def test_records_give_one_entry_per_solve():
    solver, result = run("sparse_sector", krylov_tolerance=1e-10, record_solves=True, strategy="factorized",
                          preconditioner=None, diagonal_solve=False)
    info = result.diagnostics["solver"]
    records = info["records"]
    assert len(records) == info["solves"]
    assert sum(item["iterations"] for item in records) == pytest.approx(info["iterations_mean"] * info["solves"])
    assert {item["adjoint"] for item in records} == {False, True}          # forward and adjoint resolvents
    assert max(item["relative_residual"] for item in records) == pytest.approx(info["relative_residual_max"])
    assert all(item["eta"] == ETA for item in records)


def test_dense_backend_reports_the_residual_of_its_direct_solves():
    _, factorized = run("dense", strategy="factorized")
    info = factorized.diagnostics["solver"]
    assert info["method"] == "direct"
    assert "iterations_mean" not in info
    assert 0 <= info["relative_residual_max"] < 1e-12
    _, pointwise = run("dense", strategy="pointwise")
    assert pointwise.diagnostics["solver"]["relative_residual_max"] is None       # inverse products: no residual


def test_generator_norm_is_a_close_lower_estimate_of_the_spectral_norm():
    solver, _ = run("dense", model=three_site_chain(dissipative=True))
    exact = np.linalg.norm(solver.backend._A_dense, 2)
    estimate = solver.backend.generator_norm()
    assert 0.95 * exact <= estimate <= exact * (1 + 1e-12)


@pytest.mark.parametrize("dissipative", (False, True))
def test_condition_estimate_bounds_the_shifted_systems_of_a_normal_generator(dissipative):
    """Closed generators are normal; the weak dissipation of the chain keeps the same bound in practice."""
    solver, result = run("dense", model=three_site_chain(dissipative), strategy="factorized")
    info = result.diagnostics["solver"]
    base = solver.backend
    identity = np.eye(base.layout.total_dimension ** 2)
    omegas = np.concatenate([AXES["w1"], AXES["w3"]])
    worst = max(np.linalg.cond(base.generator.frequency_shift(w, ETA) * identity - base._A_dense) for w in omegas)
    assert worst <= info["condition_estimate"]
    assert info["forward_error_estimate"] == pytest.approx(info["condition_estimate"] * info["relative_residual_max"])


def test_forward_error_of_a_loose_solve_is_below_the_condition_estimate_times_the_residual():
    solver, _ = run("sparse_sector", krylov_tolerance=1e-6, preconditioner=None, diagonal_solve=False)
    base = solver.backend
    dense = SpectroscopySolver(backend="dense", eta=ETA)
    dense.feed_model(three_site_chain(dissipative=False))
    rho = dense.backend._initial_density_matrix.reshape(-1, order="F")
    start = dense.backend.interaction_block(rho[:, None], Interaction("Bu"))[:, 0]
    for omega in (1.95, 2.0, 2.01):
        shift = base.generator.frequency_shift(omega, ETA)
        exact = np.linalg.solve(shift * np.eye(start.size) - dense.backend._A_dense, start)
        base.reset_counters()
        solution, _ = base._solve_liouville_resolvent(start, omega, ETA)
        residual = base.solve_statistics()["relative_residual_max"]
        condition = (ETA + abs(omega) + base.generator_norm()) / ETA
        assert np.linalg.norm(solution - exact) / np.linalg.norm(exact) <= condition * residual


def test_no_frequency_interval_means_no_condition_estimate():
    solver = SpectroscopySolver(backend="dense", eta=ETA)
    solver.feed_model(three_site_chain(dissipative=False))
    protocol = SpectroscopyProtocol(intervals=(PropagationInterval("t1", "time", 1), PropagationInterval("t2", "time"),
                                               PropagationInterval("t3", "time", 1)))
    result = solver.generate_spectrum(protocol, {"t1": np.array([0.0, 2.0]), "t3": np.array([0.0, 2.0])},
                                      fixed_coordinates={"t2": 1.0}, pathways=REPHASING, warn_null=False)
    assert result.diagnostics["solver"]["condition_estimate"] is None
    assert result.diagnostics["solver"]["solves"] == 0


def test_hilbert_resolvent_reports_iterations_and_relative_residual():
    solver = SpectroscopySolver(backend="sparse_sector", eta=ETA, krylov_tolerance=1e-10, preconditioner=None,
                              diagonal_solve=False)
    solver.feed_model(three_site_chain(dissipative=False))
    vector = np.random.default_rng(1).normal(size=solver.backend.layout.total_dimension)
    _, info = solver.backend.solve_hilbert_resolvent(vector, 2.01)
    assert info["iterations"] >= 1
    assert info["relative_residual"] == pytest.approx(info["residual_norm"] / np.linalg.norm(vector))
    assert info["relative_residual"] < 1e-9
