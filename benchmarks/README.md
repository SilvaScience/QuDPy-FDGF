# Benchmarks and validation

Scripts that produce the numbers of the section "Validation and performance" of the
QuDPy-FDGF manuscript. Each script writes a JSON file to `results/`; the files present in
`results/` are the reference results used in the manuscript.

All scripts restrict BLAS to one thread (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`). For the small matrices used here, multithreaded BLAS is slower and
makes timings irreproducible. Reference timings: one core of a laptop processor, Python 3.13,
NumPy 2.5, SciPy 1.18, QuTiP 5.3. They indicate scaling, not optimized performance.

Run the scripts from this directory: `python bench_scaling.py`, etc.

| Script | What it measures | Output | Time |
|---|---|---|---|
| `bench_example1_dense_sparse.py` | Dense versus sparse backend, Example 1 with its three observables (polarization, action population, integrated fluorescence), 11 x 11 grid | `example1_dense_sparse.json` | ~25 min (sparse integrated fluorescence) |
| `bench_example3_dense_sparse.py` | Dense versus sparse backend, Example 3 (fifth-order 2Q, 7 UFSS pathways), 31 x 31 grid | `example3_dense_sparse.json` | ~20 s |
| `bench_scaling.py` | Time and peak memory versus Hilbert dimension D, dense and sparse, dissipative Frenkel chain (`frenkel_model.py`), one process per case | `scaling.json` | ~15 min |
| `bench_gmres_eta.py` | GMRES iterations per shifted Liouville solve versus eta, D = 22, with and without dissipation | `gmres_eta.json` | ~3 min |
| `bench_time_fft.py` | Frequency domain versus time sampling + transform (same Liouville chain), two-level system, three linewidths, targets 1e-2 to 1e-4 | `time_fft.json` | ~5 min |
| `bench_time_fft_frenkel.py` | Same comparison for the Frenkel chain, D = 22 (`closed` argument: no dissipation) | `time_fft_frenkel.json`, `time_fft_frenkel_closed.json` | ~10 min each |
| `bench_qudpy_original.py` | One case of the comparison with the original QuDPy (`System.coherence2d`, QuTiP `mesolve`) at given (dt, T) | one JSON line | seconds to minutes |
| `bench_qudpy_series.py` | Runs the original-QuDPy cases one after another (including tighter `mesolve` tolerances) | `qudpy_original.json` | ~15 min |

## Requirements

- `qudpy_fdgf` and its dependencies;
- `psutil` (memory measurement in `bench_scaling.py` and `bench_qudpy_original.py`);
- `ufss` (Example 3 pathways);
- QuTiP and the original QuDPy (github.com/SilvaScience/QuDPy, v1.1.0) for the comparison
  with the original code: set `QUDPY_REPO` to a clone, install it, or place the clone next to
  this repository (`../QuDPy`).

## Notes on the comparisons

- The number of GMRES iterations is not recorded by the solver. `bench_gmres_eta.py` and
  `bench_scaling.py` wrap the `gmres` call of the sparse backend with a counting callback;
  the solver code itself is not modified.
- In the time-domain comparisons, the time signal is transformed with the kernel
  `exp[(i omega - eta) t]` and trapezoidal weights directly on the requested frequency grid,
  which is the definition of the resolvent. The original QuDPy `spectra()` method applies an
  unwindowed inverse FFT on its native grid, which is not the same quantity; only its
  time-domain propagation (`coherence2d`) is used.
- `coherence2d` stores all N x N propagated states, so its memory grows as N^2; the longest
  windows needed for narrow lines exceed the memory of a laptop.
