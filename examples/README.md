# Examples

The three notebooks below are the examples of QuDPy-FDGF. They share one structure:
physical model (built with QuTiP), model and solver, pathways, protocol and spectrum, figures,
a verification that does not depend on how the spectrum looks, and a summary. Every code cell is
preceded by a markdown cell that explains it.

| Notebook | System | Order | What it shows | Run time |
|---|---|---|---|---|
| `quickstart_two_level_model.ipynb` | Open two-level system | 3rd | Collapse channels, hand-written and UFSS pathways, polarization, action population and integrated fluorescence from one propagation, analytical identities | ~10 s |
| `example1_xxz_chain_thermal.ipynb` | Thermal XXZ spin chain, 6 sites | 3rd | Many-body model from `qt.tensor`, magnon-number sectors, canonical initial state at 4 K, sparse backend with exact diagonal solve, the eight rephasing sequences, bimagnon by total momentum | ~10 s |
| `example2_coupled_modes_detection.ipynb` | Two coupled dissipative bosonic modes, 9 states (closed limit, 10 states, in section 8) | 3rd, and 5th in the closed limit | Seven GKSL channels (dephasing, relaxation cascade, population transfer, radiative decay), `decay_rates()` checked against the QuTiP Liouvillian, coherent emission versus photoluminescence-type action detection from one propagation, antidiagonal linewidths, exact identity S_N = -i S_pol for the excitation-number readout | ~5 s |

Each notebook ends by saving its results in `../validation/results/data/` and its figures in
`../validation/results/figure/`. Convergence studies, analytical checks and benchmarks of the
examples are in `../validation/` (see its README); run an example before its analysis notebook.
Example 2 is compared with the data of the Perspective it is taken from in `validation/analysis/example2_analysis.ipynb`.
The notebook of the open two-level system, `quickstart_two_level_model.ipynb`, is the tutorial of the article (Sec. 4) and
the source of its listing; Examples 1 and 2 are those of Sec. 5.

Other files in this folder:

- `old_scripts/quickstart.py`: short script (the Quick start of the API documentation and the reference test run of the README; it stays in `old_scripts/`);
- `old_scripts/`: earlier versions of the examples (`two_level_open_observables*.ipynb`,
  `spin_chain_k_space.ipynb`, `chromophore_chi5_2q.ipynb`, `fock_space_solver_tutorial.ipynb`,
  `two_level_minimal.py`, `example1_usage.py`) and `example3_chi5_2q.ipynb`, the fifth-order
  double-quantum response of two coupled anharmonic bosonic modes (UFSS pathway generation,
  mode-resolved detection) that the article no longer uses: its fifth-order double-quantum
  response is now computed in the closed limit of the coupled-mode model, in
  `../validation/analysis/example2_chi5_analysis.ipynb`. They are replaced by the notebooks
  above and kept until the review is finished. The analysis and benchmark files of the former
  Example 3 are in `../validation/old_files/`.
