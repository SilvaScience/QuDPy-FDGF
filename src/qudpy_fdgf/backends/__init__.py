"""Numerical backends included in the standard QuDPy-FDGF release.

This package exports the shared backend interface and the two production
implementations: a dense Liouville backend for small reference systems and a
sparse sector backend for matrix-free calculations. Prototype backends are kept
outside the package, in ``prototypes/`` at the repository root.
"""

from .base import BackendBase, SectorLayout
from .dense_liouville import DenseLiouvilleBackend
from .sparse_sector import SparseSectorBackend

__all__ = [
    "BackendBase",
    "DenseLiouvilleBackend",
    "SectorLayout",
    "SparseSectorBackend",
]
