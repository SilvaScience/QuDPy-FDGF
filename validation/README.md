# Validation

Everything that checks and measures the examples of QuDPy-FDGF, apart from the
examples themselves (`../examples/`). The examples are numbered as in the article: the
`quickstart` (open two-level system, Sec. 4), Example 1 (thermal XXZ spin chain) and Example 2
(two coupled dissipative modes, with its closed limit at fifth order). What the article no longer
uses, among them the former Example 3 (fifth-order response of two coupled anharmonic bosonic
modes), is in `old_files/`. The folder separates two kinds of questions:

- **analysis**: is the result right and converged? (convergence, analytical limits, backend agreement)
- **benchmark**: how much does it cost? (time, memory, iterations)

```text
validation/
  models.py             The models, built with QuTiP, with their pathways, protocols and grids
                        (constants QS, EX1, EX2, EX2_CHI5). Used by the analysis notebooks and the benchmark scripts.
  pulses.py             Time-domain engine with Gaussian pulses of finite duration (independent check of the impulsive limit)
  analysis/             {quickstart,example1,example2}_analysis.ipynb, plus the scripts that draw the article figures
  benchmarks/           example1_benchmark.ipynb, and scripts/ (the measurements)
  results/
    data/               numerical results: the .npz saved by the examples, the .json of the benchmarks
    figure/             spectra and figures of the examples
    figure_analysis/    figures of the analysis and benchmark notebooks
  old_files/            what is no longer current (see the last section)
```

## Order of use

1. Run an example notebook in `../examples/`. Its last section saves `results/data/<name>.npz`
   (`quickstart.npz`, `example1.npz`, `example2.npz`) and its figures in `results/figure/`.
2. The analysis notebook of that example loads the `.npz` as its reference and recomputes only the
   cases that differ (other cutoff, other broadening, other backend). It fails with a clear message
   if the file is missing.
3. The benchmark notebook only reads the `.json` files in `results/data/`, so it opens instantly.
   Set `rerun = True` in its first cell to run the scripts in `benchmarks/scripts/` again.

The example notebooks write their models out in full so that each can be read alone. A local
folder `tests/consistency/` (not distributed) checks that they give the same spectra as `models.py`.

## Notebooks

| Notebook | Contents |
|---|---|
| `analysis/quickstart_analysis.ipynb` | Linewidth, waiting time, fluorescence window, selection rule of the fourth interaction, exact versus trapezoidal jump integral |
| `analysis/example1_analysis.ipynb` | Convergence in the magnon cutoff with the eight rephasing sequences, dense versus sparse agreement, GMRES iterations with and without the preconditioner |
| `analysis/example1_pulses.ipynb` | Error of the impulsive limit and of the rotating-wave approximation for the THz chain of Example 1, against a time-domain calculation with Gaussian pulses of finite duration (`pulses.py`, sweep `example1_pulses.py`) |
| `analysis/example2_analysis.ipynb` | Comparison of Example 2 with the stored data of the Perspective repository (six scenarios): coherent and action half widths and amplitudes, sign convention of the action diagrams, excitation-number identity, unbalanced scenario, PL projector, estimator. Reads `results/data/perspective_comparison.json` |
| `analysis/example2_chi5_analysis.ipynb` | Closed limit of Example 2, fifth order: truncation independence of the third-order response, lines of the 2Q map, pathway orderings with and without the three-excitation manifold, truncation at two excitations, harmonic-limit cancellation, additivity of independent modes |
| `benchmarks/example1_benchmark.ipynb` | Dense versus sparse, scaling with chain length, GMRES iterations versus η |

The scripts in `analysis/` redraw the figures of the article into `results/figure/`:

| Script | Figure | Data |
|---|---|---|
| `fig_quickstart_projection.py` | Quickstart, selection rule of the readout interaction (nine maps) | recomputed (241 x 241, ~20 s) |
| `fig_quickstart_validation.py` | Quickstart, waiting time, fluorescence window, quadrature | recomputed (~5 s) |
| `fig_example1_manifolds_thermal.py` | Example 1, dispersion, bimagnon transitions, thermal populations | `example1.npz` |
| `fig_example1_thermal_spectroscopy.py` | Example 1, complete thermal rephasing response | recomputed (~15 min; `plot` redraws from the stored `.npz`) |

`analysis/compare_perspective.py` evaluates QuDPy-FDGF at the ten points of each stored cut of the Perspective repository and applies its estimator (`python compare_perspective.py "<Perspective repo>/Data/Borealis Data"`, ~5 s); it writes `results/data/perspective_comparison.json`. The Perspective data themselves are not part of this repository (they are deposited with the article, doi 10.5683/SP4/UKSLI2).

## Benchmark scripts

Each script in `benchmarks/scripts/` writes a JSON file to `results/data/`; the files present are
the reference results used in the article. All scripts restrict BLAS to one thread
(`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`): for the small matrices used
here, multithreaded BLAS is slower and makes timings irreproducible (`bench_transverse_chain.py` takes the number of threads
from `BENCH_THREADS`, three by default). Reference timings: one core of a laptop processor, Python 3.13, NumPy 2.5, SciPy 1.18,
QuTiP 5.3. They indicate scaling, not optimized performance.

Run a script from anywhere, for example `python validation/benchmarks/scripts/bench_scaling.py`.

| Script | What it measures | Output | Time |
|---|---|---|---|
| `bench_quickstart_dense_sparse.py` | Dense versus sparse backend, pointwise versus factorized, the quickstart with its three observables, 41 x 41 grid | `quickstart_dense_sparse.json` | ~10 s |
| `bench_example1_dense_sparse.py` | Dense versus sparse (plain GMRES and preconditioned) backend, pointwise versus factorized, Example 1 (D = 57), 5 x 5 grid; `skip-dense-pointwise` leaves out the slow dense run | `example1_dense_sparse.json` | ~40 min (~3 min without the dense pointwise run) |
| `bench_modes_dense_sparse.py` | Dense versus sparse backend, pointwise versus factorized, Example 2 (third order, D = 9) and its closed limit (fifth order, D = 10, seven pathways): 31 x 31 grid and the full grid of the example | `modes_dense_sparse.json` | ~1 min |
| `bench_grid_size.py` | Time and resolvent solves versus the number of grid points per axis N, pointwise versus factorized: quickstart, Example 1, Example 2, closed limit of Example 2 | `grid_size.json` | ~10 min |
| `bench_scaling.py` | Time and peak memory versus Hilbert dimension D, Example 1 enlarged from L = 4 to 10 sites, 9 x 9 grid; four configurations (`dense`, `sparse_plain`, `sparse`, `sparse_pointwise`; one argument runs one of them), one process per case | `scaling.json` | ~10 min |
| `bench_transverse_chain.py` | Example 1 with a transverse field (no sector structure, D = 2^L), L = 5 to 11, exact diagonal solve or plain GMRES, memory limit | `transverse_chain.json` | seconds to ~11 min (L = 11) |
| `bench_generator_action.py` | One application of the Liouvillian, matrix products versus one call per column (D = 57 and 163), and a 2 x 2 spectrum at D = 57 | `generator_action.json` | ~1 min |
| `bench_preconditioner_dissipative.py` | Diagonal preconditioner versus plain GMRES on dissipative models with a diagonal H (three-site chain, six-site XXZ chain with collective decay or local dephasing, rates from weak to far above the energy scale); the basis of the `auto` choice | `preconditioner_dissipative.json` | ~8 min |
| `bench_gmres_eta.py` | GMRES iterations per resolvent solve versus η, D = 57, 3 x 3 grid (`diagonal` argument: with the preconditioner) | `gmres_eta.json`, `gmres_eta_diagonal.json` | ~3 min |
| `bench_resolved_map.py` | Resolved maps at equal resolution (grid spacing a quarter of the line width): direct evaluation, sum over states, and the cheapest factorized time route (matrix transform or zero-padded FFT, bandpass sampling) for targets 1e-2 to 1e-4; the quickstart at three linewidths, Example 2 at two sets of rates (`modes`) and its closed limit at η = 2 and 1 meV (`chi5`) | `resolved_map.json` | tens of minutes per problem |
| `bench_qudpy_perspective.py`, `bench_qudpy_perspective_series.py` | Original QuDPy (`System.coherence2d`, QuTiP `mesolve`) against QuDPy-FDGF on the diagrams of Example 2: coherent (three) and action (four) diagrams, on the grid of the Perspective and on a grid resolved at 2 meV; the series script runs the cases one after another and resumes | `qudpy_perspective.json` | minutes to hours (the longest case, 4 action diagrams on the resolved grid, takes about 2.5 h) |
| `check_models.py` | Checks that `models.py` reproduces the published numbers of Example 1 and the dimension and pathway count of the closed limit of Example 2 | `models_check.json` | ~1 min |

## Requirements

- `qudpy_fdgf` and its dependencies, QuTiP, Matplotlib;
- `ufss` (pathways of the closed limit of Example 2);
- `psutil` (memory measurement in `bench_scaling.py`, `bench_transverse_chain.py` and `bench_qudpy_perspective.py`);
- for the original-QuDPy comparison, a clone of github.com/SilvaScience/QuDPy (v1.1.0): set
  `QUDPY_REPO`, or place the clone next to this repository (`../QuDPy`).
- To run the notebooks from the command line: `nbformat` and `nbclient`.

## Notes on the comparisons

- The iterations of the sparse backend come from the solver itself (`result.diagnostics["solver"]`:
  number of solves, mean and maximum iterations, true relative residual); the scripts no longer
  wrap `gmres`. The grid is evaluated with `strategy="auto"` (factorized for two or more scanned axes)
  unless a script says `pointwise`; `preconditioner="auto"` is the default of the sparse backend, and
  the scripts that need plain GMRES pass `preconditioner=None`.
- In the time-domain comparisons, the time signal is transformed with the kernel
  `exp[(i omega - eta) t]` and trapezoidal weights directly on the requested frequency grid, so
  the transform adds no error of its own.
- The original QuDPy `coherence2d` fails with QuTiP 5 when a fixed delay precedes the first scan,
  which is why the fifth-order response is not run with it.

## old_files

What the article no longer uses, kept with the paths it had under `validation/`: the former Example 3
(analysis notebook, figure scripts, benchmark notebook and scripts, results, and `cases.py`, with its model
`models_example3.py`), the comparison of the original QuDPy with the open two-level system
(`bench_qudpy_original.py`, `bench_qudpy_series.py`, `qudpy_original.json`), and the benchmarks of the
two-level system against a time-domain route (`bench_time_fft.py`, `example1_benchmark.ipynb`; the same
comparison is now made by `bench_resolved_map.py`). These files keep the names and the numbering they had before the
renumbering of the examples, and they run with the repository as it was at commit `ae47e8f`; they are
not maintained and no test depends on them.
