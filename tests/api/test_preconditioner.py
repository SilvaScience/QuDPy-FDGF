"""Diagonal GMRES preconditioner of the sparse backend.

``preconditioner="diagonal"`` divides by the shifted diagonal of the generator
(Liouville resolvents) or of the Hamiltonian (Hilbert resolvents). Checks: the
generator diagonal against the dense generator, unchanged spectra with and
without collapse channels, one iteration per solve for a closed model in its
eigenbasis, fewer iterations with dissipation, and input validation.
"""
import numpy as np
import pytest

from qudpy_fdgf import SpectroscopySolver
from helpers import PROTOCOL, REPHASING, three_site_chain

AXES = {"w1": np.linspace(-2.15, -1.9, 4), "w3": np.linspace(1.9, 2.15, 4)}
ETA = 0.002


def spectrum(model, backend, **options):
    solver = SpectroscopySolver(backend=backend, eta=ETA, **options)
    solver.feed_model(model)
    result = solver.generate_spectrum(PROTOCOL, AXES, pathways=REPHASING,
                                      fixed_coordinates={"t2": 10.0})
    return result.components["rephasing"]


def test_generator_diagonal_matches_the_dense_generator():
    dense = SpectroscopySolver(backend="dense", eta=ETA)
    dense.feed_model(three_site_chain())
    sparse = SpectroscopySolver(backend="sparse_sector", eta=ETA, preconditioner="diagonal")
    sparse.feed_model(three_site_chain())
    reference = np.diag(dense.backend._A_dense)
    assert np.max(np.abs(sparse.backend.generator.diagonal() - reference)) < 1e-14
    assert np.all(reference.real <= 1e-15)


@pytest.mark.parametrize("dissipative", (False, True))
def test_spectra_are_unchanged(dissipative):
    reference = spectrum(three_site_chain(dissipative), "dense")
    plain = spectrum(three_site_chain(dissipative), "sparse_sector", krylov_tolerance=1e-12)
    preconditioned = spectrum(three_site_chain(dissipative), "sparse_sector",
                              krylov_tolerance=1e-12, preconditioner="diagonal")
    scale = np.max(np.abs(reference))
    assert np.max(np.abs(plain - reference)) / scale < 1e-9
    assert np.max(np.abs(preconditioned - reference)) / scale < 1e-9


def test_closed_model_in_its_eigenbasis_needs_one_iteration(gmres_iterations):
    spectrum(three_site_chain(dissipative=False), "sparse_sector",
             krylov_tolerance=1e-12, preconditioner="diagonal")
    assert gmres_iterations and max(gmres_iterations) == 1


def test_dissipation_needs_fewer_iterations(gmres_iterations):
    spectrum(three_site_chain(), "sparse_sector", krylov_tolerance=1e-12)
    plain = sum(gmres_iterations)
    gmres_iterations.clear()
    spectrum(three_site_chain(), "sparse_sector", krylov_tolerance=1e-12,
             preconditioner="diagonal")
    assert sum(gmres_iterations) < plain


def test_hilbert_resolvent_is_unchanged():
    solvers = {}
    for name, option in (("plain", None), ("diagonal", "diagonal")):
        solvers[name] = SpectroscopySolver(backend="sparse_sector", eta=ETA,
                                           krylov_tolerance=1e-12, preconditioner=option)
        solvers[name].feed_model(three_site_chain(dissipative=False))
    vector = np.random.default_rng(1).normal(size=solvers["plain"].backend.layout.total_dimension)
    for omega in (1.95, 2.01, 4.05):
        plain, _ = solvers["plain"].backend.solve_hilbert_resolvent(vector, omega)
        diagonal, _ = solvers["diagonal"].backend.solve_hilbert_resolvent(vector, omega)
        assert np.max(np.abs(diagonal - plain)) / np.max(np.abs(plain)) < 1e-9


def test_unknown_preconditioner_is_rejected():
    with pytest.raises(ValueError, match="preconditioner"):
        SpectroscopySolver(backend="sparse_sector", preconditioner="ilu").feed_model(
            three_site_chain())
