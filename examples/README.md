# Examples

The three notebooks below are the examples of the QuDPy-FDGF article. They share one structure:
physical model (built with QuTiP), model and solver, pathways, protocol and spectrum, figures,
a verification that does not depend on how the spectrum looks, and a summary. Every code cell is
preceded by a markdown cell that explains it.

| Notebook | System | Order | What it shows | Run time |
|---|---|---|---|---|
| `example1_two_level_open.ipynb` | Open two-level system | 3rd | Collapse channels, hand-written and UFSS pathways, polarization, action population and integrated fluorescence from one propagation, analytical identities | ~10 s |
| `example2_xxz_chain_thermal.ipynb` | Thermal XXZ spin chain, 6 sites | 3rd | Many-body model from `qt.tensor`, magnon-number sectors, canonical initial state at 4 K, sparse backend with diagonal preconditioner, bimagnon ESA | ~2 min |
| `example3_chi5_2q.ipynb` | Two coupled anharmonic bosonic modes | 5th | Bosonic operators from QuTiP, UFSS pathway generation, double-quantum map, mode-resolved detection | ~1 min |

Each notebook ends by saving its results in `../validation/results/data/` and its figures in
`../validation/results/figure/`. Convergence studies, analytical checks and benchmarks of the three
examples are in `../validation/` (see its README); run an example before its analysis notebook.

Other files in this folder:

- `quickstart.py`: short script (the Quick start of the API documentation);
- `old_scripts/`: earlier versions of the examples (`two_level_open_observables*.ipynb`,
  `spin_chain_k_space.ipynb`, `chromophore_chi5_2q.ipynb`, `fock_space_solver_tutorial.ipynb`,
  `two_level_minimal.py`, `example1_usage.py`), replaced by the
  notebooks above and kept until the review is finished.
