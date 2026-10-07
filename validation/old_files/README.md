# old_files

Files that are no longer current, kept with the paths they had under `validation/`. They keep the names and the
numbering of the examples as they were before the renumbering (Example 1 = open two-level system, Example 2 = thermal
XXZ chain, Example 3 = fifth-order response of two coupled anharmonic bosonic modes, Example 4 = two coupled dissipative
modes), and they run with the repository as it was at commit `ae47e8f`. They are not maintained, and no test depends on them.

| Content | Files |
|---|---|
| Former Example 3 | `analysis/example3_analysis.ipynb`, `analysis/fig_example3_*.py`, `benchmarks/example3_benchmark.ipynb`, `benchmarks/scripts/bench_example3_dense_sparse.py`, `benchmarks/scripts/bench_time_fft_ex3.py`, `cases.py`, `models_example3.py` (its model, taken out of `models.py`), results (`example3*.npz`, `example3_dense_sparse.json`, `time_fft_ex3.json`, `resolved_map_example3.json`, figures) |
| Original QuDPy against the two-level system | `benchmarks/scripts/bench_qudpy_original.py`, `benchmarks/scripts/bench_qudpy_series.py`, `results/data/qudpy_original.json` |
| Two-level system against a time-domain route | `benchmarks/scripts/bench_time_fft.py`, `benchmarks/example1_benchmark.ipynb`, `results/data/time_fft.json`, `results/figure_analysis/example1_benchmark_*.pdf` (the comparison is now made by `bench_resolved_map.py`) |

The notebook of the former Example 3 is in `../../examples/old_scripts/example3_chi5_2q.ipynb`.
