# Validation

Everything that checks and measures the examples of QuDPy-FDGF, apart from the
examples themselves (`../examples/`). The former Example 3 (fifth-order response of two coupled
anharmonic bosonic modes) is no longer an example of the article: its notebook is in
`../examples/old_scripts/`, and its analysis and benchmark notebooks and scripts stay here. The
folder separates two kinds of questions:

- **analysis**: is the result right and converged? (convergence, analytical limits, backend agreement)
- **benchmark**: how much does it cost? (time, memory, iterations)

```text
validation/
  models.py             The four models, built with QuTiP, with their pathways, protocols and grids.
                        Used by the analysis notebooks and the benchmark scripts.
  cases.py              Extra Example 3 calculations for the figures, cached in results/data/
  analysis/             example{1,2,3}_analysis.ipynb, plus the scripts that draw the article figures
  benchmarks/           example{1,2,3}_benchmark.ipynb, and scripts/ (the measurements)
  results/
    data/               numerical results: the .npz saved by the examples, the .json of the benchmarks
    figure/             spectra and figures of the examples
    figure_analysis/    figures of the analysis and benchmark notebooks
```

## Order of use

1. Run an example notebook in `../examples/`. Its last section saves `results/data/exampleN.npz`
   and its figures in `results/figure/`.
2. The analysis notebook of that example loads the `.npz` as its reference and recomputes only the
   cases that differ (other cutoff, other broadening, other backend). It fails with a clear message
   if the file is missing.
3. The benchmark notebooks only read the `.json` files in `results/data/`, so they open instantly.
   Set `rerun = True` in their first cell to run the scripts in `benchmarks/scripts/` again.

The example notebooks write their models out in full so that each can be read alone.
`tests/test_validation_models.py` checks that they give the same spectra as `models.py`.

## Notebooks

| Notebook | Contents |
|---|---|
| `analysis/example1_analysis.ipynb` | Linewidth, waiting time, fluorescence window, selection rule of the fourth interaction, exact versus trapezoidal jump integral |
| `analysis/example2_analysis.ipynb` | Convergence in the magnon cutoff, dense versus sparse agreement, GMRES iterations with and without the preconditioner |
| `analysis/example3_analysis.ipynb` | Spectrum at η = 2 meV, pathway completeness, truncation at two excitations, harmonic-limit cancellation |
| `analysis/example2_pulses.ipynb` | Error of the impulsive limit and of the rotating-wave approximation for the THz chain of Example 1, against a time-domain calculation with Gaussian pulses of finite duration (`pulses.py`, sweep `example2_pulses.py`) |
| `analysis/example4_chi5_analysis.ipynb` | Closed limit of Example 4, fifth order: truncation independence of the third-order response, lines of the 2Q map, pathway orderings with and without the three-excitation manifold, truncation at two excitations, harmonic-limit cancellation, additivity of independent modes |
| `analysis/example4_analysis.ipynb` | Comparison of Example 4 with the stored data of the Perspective repository (six scenarios): coherent and action half widths and amplitudes, sign convention of the action diagrams, excitation-number identity, unbalanced scenario, PL projector, estimator. Reads `results/data/perspective_comparison.json` |
| `benchmarks/example1_benchmark.ipynb` | Dense versus sparse, frequency domain versus time sampling, original QuDPy |
| `benchmarks/example2_benchmark.ipynb` | Dense versus sparse, scaling with chain length, GMRES iterations versus η |
| `benchmarks/example3_benchmark.ipynb` | Dense versus sparse, frequency domain versus time sampling, estimate for the original QuDPy |

The scripts in `analysis/` redraw the figures of the article into `results/figure/`:

| Script | Figure | Data |
|---|---|---|
| `fig_example1_projection.py` | Example 1, selection rule of the readout interaction (nine maps) | recomputed (241 x 241, ~20 s) |
| `fig_example1_validation.py` | Example 1, waiting time, fluorescence window, quadrature | recomputed (~5 s) |
| `fig_example2_manifolds_thermal.py` | Example 2, dispersion, bimagnon transitions, thermal populations | `example2.npz` |
| `fig_example2_thermal_spectroscopy.py` | Example 2, complete thermal rephasing response | recomputed (~15 min; `plot` redraws from the stored `.npz`) |
| `fig_example3_2q_map.py` | Example 3, 2Q map at eta = 8 and 2 meV | `example3.npz` + `example3_extra_cases.npz` |
| `fig_example3_validation.py` | Example 3, pathway orderings, truncation, mode-resolved detection | `example3.npz` + `example3_extra_cases.npz` |

`analysis/compare_perspective.py` evaluates QuDPy-FDGF at the ten points of each stored cut of the Perspective repository and applies its estimator (`python compare_perspective.py "<Perspective repo>/Data/Borealis Data"`, ~5 s); it writes `results/data/perspective_comparison.json`. The Perspective data themselves are not part of this repository.

`cases.py` computes the extra Example 3 cases (eta = 2 meV, N_max = 2) once and stores them in
`results/data/example3_extra_cases.npz`.

## Benchmark scripts

Each script in `benchmarks/scripts/` writes a JSON file to `results/data/`; the files present are
the reference results used in the article. All scripts restrict BLAS to one thread
(`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`): for the small matrices used
here, multithreaded BLAS is slower and makes timings irreproducible. Reference timings: one core
of a laptop processor, Python 3.13, NumPy 2.5, SciPy 1.18, QuTiP 5.3. They indicate scaling, not
optimized performance.

Run a script from anywhere, for example `python validation/benchmarks/scripts/bench_scaling.py`.

| Script | What it measures | Output | Time |
|---|---|---|---|
| `bench_example1_dense_sparse.py` | Dense versus sparse backend, pointwise versus factorized, Example 1 with its three observables, 41 x 41 grid | `example1_dense_sparse.json` | ~10 s |
| `bench_example2_dense_sparse.py` | Dense versus sparse (plain GMRES and preconditioned) backend, pointwise versus factorized, Example 2 (D = 57), 5 x 5 grid; `skip-dense-pointwise` leaves out the slow dense run | `example2_dense_sparse.json` | ~40 min (~3 min without the dense pointwise run) |
| `bench_example3_dense_sparse.py` | Dense versus sparse backend, pointwise versus factorized, Example 3 (fifth order, 7 pathways), 31 x 31 grid, and the factorized 151 x 151 map | `example3_dense_sparse.json` | ~20 s |
| `bench_grid_size.py` | Time and resolvent solves versus the number of grid points per axis N, pointwise versus factorized, Examples 1 to 3 | `grid_size.json` | ~10 min |
| `bench_scaling.py` | Time and peak memory versus Hilbert dimension D, Example 2 enlarged from L = 4 to 10 sites, 9 x 9 grid; four configurations (`dense`, `sparse_plain`, `sparse`, `sparse_pointwise`; one argument runs one of them), one process per case | `scaling.json` | ~10 min |
| `bench_generator_action.py` | One application of the Liouvillian, matrix products versus one call per column (D = 57 and 163), and a 2 x 2 spectrum at D = 57 | `generator_action.json` | ~1 min |
| `bench_preconditioner_dissipative.py` | Diagonal preconditioner versus plain GMRES on dissipative models with a diagonal H (three-site chain, six-site XXZ chain with collective decay or local dephasing, rates from weak to far above the energy scale); the basis of the `auto` choice | `preconditioner_dissipative.json` | ~8 min |
| `bench_gmres_eta.py` | GMRES iterations per resolvent solve versus η, D = 57, 3 x 3 grid (`diagonal` argument: with the preconditioner) | `gmres_eta.json`, `gmres_eta_diagonal.json` | ~3 min |
| `bench_time_fft.py` | Frequency domain versus time sampling + transform, two-level system, three linewidths, targets 1e-2 to 1e-4 | `time_fft.json` | ~5 min |
| `bench_time_fft_ex3.py` | Same comparison for Example 3, at η = 8 and 2 meV, with an extrapolation for the original QuDPy | `time_fft_ex3.json` | ~5 min |
| `bench_resolved_map.py` | Resolved maps at equal resolution (grid spacing a quarter of the line width): direct evaluation, sum over states, and the cheapest factorized time route (matrix transform or zero-padded FFT, bandpass sampling) for targets 1e-2 to 1e-4; Example 1 at three linewidths and Example 3 at η = 8 and 2 meV (`example1` or `example3` argument) | `resolved_map.json` | ~25 min |
| `bench_qudpy_original.py` | One case of the comparison with the original QuDPy (`System.coherence2d`, QuTiP `mesolve`) at given (dt, T) | one JSON line | seconds to minutes |
| `bench_qudpy_series.py` | Runs the original-QuDPy cases one after another, including tighter `mesolve` tolerances | `qudpy_original.json` | ~15 min |
| `check_models.py` | Checks that `models.py` reproduces the published numbers of Examples 2 and 3 | `models_check.json` | ~1 min |

## Requirements

- `qudpy_fdgf` and its dependencies, QuTiP, Matplotlib;
- `ufss` (Example 3 pathways);
- `psutil` (memory measurement in `bench_scaling.py` and `bench_qudpy_original.py`);
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
  which is why Example 3 is not run with it: its cost is extrapolated from a small pilot.
