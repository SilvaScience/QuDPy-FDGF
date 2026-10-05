"""Shared test configuration."""
import os

# Single-thread BLAS: the matrices are tiny and multithreading only adds overhead.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import pytest

import qudpy_fdgf.backends.sparse_sector as sparse_sector


@pytest.fixture
def gmres_iterations(monkeypatch):
    """Record the inner GMRES iterations of every sparse-backend solve."""
    iterations = []
    original = sparse_sector.gmres

    def counting(*args, **kwargs):
        count = [0]

        def callback(_):
            count[0] += 1
        out = original(*args, callback=callback, callback_type="pr_norm", **kwargs)
        iterations.append(count[0])
        return out
    monkeypatch.setattr(sparse_sector, "gmres", counting)
    return iterations
