# Prototypes

Code that was developed along with QuDPy-FDGF but is **not part of the release**: it is not installed with the package
(`pip install` does not copy this folder), it has no tests and it is not maintained. It is kept for continued development.

| Module | Content |
|---|---|
| `low_rank_liouville.py` | `LowRankLiouvilleBackend`, a backend that stores the density matrix in low-rank form |
| `many_body.py`, `many_body_dynamics.py` | contracts and settings for many-body ground-state and dynamics calculations |
| `tenpy_dmrg.py`, `tenpy_tdvp.py` | TeNPy DMRG and TDVP engines (need `physics-tenpy`) |

Use them from the repository root, after installing the package (`pip install -e .`):

```python
from prototypes.low_rank_liouville import LowRankLiouvilleBackend
```

They used to be the package `qudpy_fdgf.experimental`; their imports of `qudpy_fdgf` are now absolute.
