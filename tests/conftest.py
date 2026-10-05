"""Shared test configuration."""
import os

# Single-thread BLAS: the matrices are tiny and multithreading only adds overhead.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")
