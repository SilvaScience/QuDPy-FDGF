"""Exact diagonal solve of the sparse backend: a generator that is diagonal in the dyad basis is inverted by division.

A closed model in its eigenbasis, or one whose collapse operators are diagonal (pure dephasing), has
``L|x><y| = L_xy |x><y|``; ``diagonal_solve=True`` (default) then solves every shifted system by
``b / [(eta - i omega) - L_xy]`` instead of GMRES. The result is compared with the dense backend
(explicit inverse, independent) and with GMRES.
"""
import numpy as np
import pytest

from helpers import H, L_RAD, MU, PROTOCOL, REPHASING, SIGMA_Z, three_site_chain
from qudpy_fdgf import ConvergenceError, EigenbasisKModel, SpectroscopySolver

ETA = 0.002
AXES = {"w1": np.linspace(-2.15, -1.9, 4), "w3": np.linspace(1.9, 2.15, 4)}


def solver_for(model, backend="sparse_sector", **options):
    solver = SpectroscopySolver(backend=backend, eta=ETA, check_stationarity="off", **options)
    solver.feed_model(model)
    return solver


def spectrum(solver, **kwargs):
    return solver.generate_spectrum(PROTOCOL, AXES, pathways=REPHASING, fixed_coordinates={"t2": 10.0},
                                    warn_null=False, **kwargs)


def difference(a, b):
    return np.max(np.abs(a - b)) / np.max(np.abs(b))


def dephasing_only():
    """Two-level system with a complex diagonal collapse operator: L_xx L_yy^* is not real."""
    return EigenbasisKModel(H, MU, c_ops_raw=((np.diag([0.3, 1.0 + 0.5j]).astype(complex), 0.02),))


def closed():
    return three_site_chain(dissipative=False)


@pytest.mark.parametrize("model", (closed, dephasing_only), ids=("closed", "dephasing"))
def test_diagonal_generator_is_solved_by_division_and_matches_the_dense_inverse(model):
    solver = solver_for(model())
    assert solver.backend.generator_is_diagonal
    result = spectrum(solver)
    info = result.diagnostics["solver"]
    assert info["method"] == "diagonal" and info["exact"] is True
    assert "iterations_mean" not in info and info["solves"] > 0
    reference = spectrum(solver_for(model(), backend="dense")).components["rephasing"]
    assert difference(result.components["rephasing"], reference) < 1e-12


@pytest.mark.parametrize("model", (closed, dephasing_only), ids=("closed", "dephasing"))
def test_gmres_gives_the_same_spectrum_and_the_same_number_of_solves(model):
    exact = spectrum(solver_for(model()))
    gmres = spectrum(solver_for(model(), diagonal_solve=False, preconditioner=None, krylov_tolerance=1e-12))
    assert gmres.diagnostics["solver"]["method"] == "gmres"
    assert exact.diagnostics["solver"]["solves"] == gmres.diagnostics["solver"]["solves"]
    assert difference(exact.components["rephasing"], gmres.components["rephasing"]) < 1e-9


def test_pointwise_and_factorized_evaluations_agree_with_the_diagonal_solve():
    solver = solver_for(closed())
    pointwise = spectrum(solver, strategy="pointwise").components["rephasing"]
    factorized = spectrum(solver, strategy="factorized").components["rephasing"]
    assert difference(pointwise, factorized) < 1e-12


def test_a_collapse_operator_that_moves_population_is_not_diagonal():
    model = EigenbasisKModel(H, MU, c_ops_raw=((L_RAD, 0.04),))
    solver = solver_for(model)
    assert not solver.backend.generator_is_diagonal
    result = spectrum(solver)
    assert result.diagnostics["solver"]["method"] == "gmres"
    reference = spectrum(solver_for(model, backend="dense")).components["rephasing"]
    assert difference(result.components["rephasing"], reference) < 1e-8


def test_the_option_turns_it_off():
    solver = solver_for(closed(), diagonal_solve=False)
    assert solver.backend.generator_is_diagonal and not solver.backend.diagonal_solve
    assert spectrum(solver).diagnostics["solver"]["method"] == "gmres"


def test_a_dissipative_chain_in_its_eigenbasis_keeps_gmres_with_the_preconditioner():
    solver = solver_for(three_site_chain(dissipative=True))
    assert not solver.backend.generator_is_diagonal
    info = spectrum(solver).diagnostics["solver"]
    assert info["method"] == "gmres" and info["preconditioner"] == "diagonal"


def test_hilbert_resolvent_of_a_diagonal_hamiltonian_is_exact():
    solver = solver_for(closed())
    base = solver.backend
    vector = np.random.default_rng(5).normal(size=base.layout.total_dimension) + 0j
    dense = solver_for(closed(), backend="dense").backend
    for omega in (1.95, 2.01, 4.05):
        solution, info = base.solve_hilbert_resolvent(vector, omega)
        assert info["method"] == "diagonal" and info["iterations"] == 0
        h_matrix = np.diag(base._hamiltonian_diagonal)
        exact = np.linalg.solve((omega + 1j * ETA) * np.eye(vector.size) - h_matrix, vector)
        assert np.max(np.abs(solution - exact)) < 1e-12 * np.max(np.abs(exact))


def test_eta_zero_on_a_pole_is_reported_not_divided_by_zero():
    solver = SpectroscopySolver(backend="sparse_sector", eta=0.0, check_stationarity="off")
    solver.feed_model(EigenbasisKModel(H, MU))
    with pytest.raises(ConvergenceError, match="singular"):
        solver.generate_spectrum(PROTOCOL, {"w1": np.array([-2.0]), "w3": np.array([2.0])}, pathways=REPHASING,
                                 fixed_coordinates={"t2": 10.0}, warn_null=False)
