"""The broadening eta has no default: forgetting it is an error that explains what it is."""
import numpy as np
import pytest

from helpers import H, MU
from qudpy_fdgf import EigenbasisKModel, SpectroscopySolver
from qudpy_fdgf.backends import DenseLiouvilleBackend, SparseSectorBackend


@pytest.mark.parametrize("backend", ("dense", "sparse_sector"))
def test_solver_without_eta_is_an_error_that_explains_it(backend):
    with pytest.raises(ValueError, match="eta is required") as error:
        SpectroscopySolver(backend=backend)
    message = str(error.value)
    assert "homogeneous broadening" in message and "same unit as the Hamiltonian" in message


@pytest.mark.parametrize("backend_class", (DenseLiouvilleBackend, SparseSectorBackend))
def test_backends_without_eta_are_an_error_too(backend_class):
    with pytest.raises(ValueError, match="eta is required"):
        backend_class()


@pytest.mark.parametrize("eta", (-0.1, float("nan"), float("inf")))
def test_invalid_eta_is_rejected(eta):
    with pytest.raises(ValueError, match="eta"):
        SpectroscopySolver(backend="dense", eta=eta)


def test_explicit_eta_is_used_and_zero_is_allowed():
    solver = SpectroscopySolver(backend="sparse_sector", eta=0.003).feed_model(EigenbasisKModel(H, MU))
    assert solver.eta == 0.003 and solver.backend.eta == 0.003
    assert SpectroscopySolver(backend="dense", eta=0.0).eta == 0.0
