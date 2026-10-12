# Changelog

Changes to the code of QuDPy-FDGF, newest first. The article and the API reference manual
(`docs/`) are updated with each entry. Releases are numbered major.minor.patch, starting from 1.0.0;
the compatibility and deprecation policy is the section "Versions and compatibility" of the manual.

## Unreleased

### Breaking changes
- Minimum versions raised to the tested ones: Python >= 3.13, NumPy >= 2.5, SciPy >= 1.18, Matplotlib >= 3.11 (they were
  3.10, 1.24, 1.10 and 3.7, never tested; `gmres(rtol=...)` needs SciPy >= 1.12 and `np.trapezoid` NumPy >= 2.0). The optional
  extras carry the tested lower bounds as well. Tested on Windows 11 only.
- `eta` has no default: `SpectroscopySolver(...)`, `DenseLiouvilleBackend` and `SparseSectorBackend`
  raise a `ValueError` that explains the parameter when it is missing (the former `0.05` has no
  meaning without a unit). Negative, infinite and NaN values are rejected.
- `SparseSectorBackend`: the default is `preconditioner="auto"` instead of `None`. The diagonal
  preconditioner is applied when every Hamiltonian block is a diagonal block of one sector (a model in its
  eigenbasis, as the adapters provide), whatever the collapse operators, and left off otherwise (site basis,
  couplings between sectors, matrix-free blocks that cannot be inspected). It is exact for closed models and
  diagonal collapse operators (one iteration per solve instead of hundreds: Example 2, D = 57, 1 082
  iterations and 1.1 s become 1 iteration and 0.01 s) and approximate, but far better than none, with
  operators that couple coherences. Measured against plain GMRES on 30 cases (three-site chain with
  dissipation x0.1 to x1000, six-site XXZ chain with collective decay or local dephasing from 0.001 to 10 meV,
  two broadenings, three kinds of start vector, `validation/benchmarks/scripts/bench_preconditioner_dissipative.py`):
  the ratio of iterations is 3e-4 to 0.76 in 28 cases (54 000 iterations in one plain solve against 1 114
  preconditioned), 1.20 and 1.23 at a local dephasing of 10 meV, an order of magnitude above the energy scale of the
  chain; true residuals comparable (all <= 1e-10). The numbers do not change beyond the solver tolerance.
  `preconditioner=None` gives the former plain GMRES (the dense-versus-sparse benchmark scripts of Examples 2 and 3
  pass it); `"diagonal"` forces the preconditioner. `diagnostics["solver"]["preconditioner"]` and
  `backend.preconditioner` report the choice.
- `EigenbasisKModel` and `ExcitationSectorModel`: `boltzmann_constant` no longer defaults to the
  value in eV/K. New argument `energy_unit` (`"eV"`, `"meV"`, `"cm-1"`, `"THz"`; the last is the
  ordinary frequency E/h, k_B/h = 0.0208 THz/K) sets it; `boltzmann_constant` explicit is still
  accepted (in the unit of the Hamiltonian per kelvin); both together, an unknown unit or a
  non-positive constant raise `ValueError`. A temperature above zero without either now raises a
  `ValueError` at `feed_model` instead of silently using eV/K; zero temperature needs neither.
  A model that ran in eV with a thermal state must now say `energy_unit="eV"`.

### Examples and validation renumbered (as in the article)
- The examples follow the numbering of the article: `examples/quickstart_two_level_model.ipynb` (was
  `example1_two_level_open.ipynb`, the tutorial of Sec. 4), `examples/example1_xxz_chain_thermal.ipynb` (was
  `example2_xxz_chain_thermal.ipynb`) and `examples/example2_coupled_modes_detection.ipynb` (was
  `example4_coupled_modes_detection.ipynb`). The same renaming applies to the files of `validation/` (analysis
  and benchmark notebooks and scripts, results, figures) and to the constants and functions of
  `validation/models.py`: `QS`/`quickstart_model` (was `EX1`/`example1_model`), `EX1`/`example1_*` (was `EX2`/`example2_*`,
  the XXZ chain), `EX2`/`example2_*` and `EX2_CHI5_*` (was `EX4`/`example4_*`, the coupled modes and their closed limit).
- The former Example 3 (fifth-order response of two coupled anharmonic bosonic modes) leaves the examples:
  its notebook is in `examples/old_scripts/`, and its analysis, benchmark, results and model (`models_example3.py`),
  together with the comparison of the original QuDPy with the open two-level system and its time-domain benchmark,
  are in `validation/old_files/` (see its README). The fifth-order response is now that of the closed limit of
  Example 2 (`validation/analysis/example2_chi5_analysis.ipynb`).
- `README.md`: the test suite has 247 tests (250 with the local `tests/consistency/` folder) and takes about 30 s.

### Changed
- Documentation: the stationarity residual `||L rho|| / ||rho||` is documented as it is, a quantity with the dimension of a
  rate compared with an absolute tolerance in the unit of the Hamiltonian (README, manual, article). Planned for the next major
  release: a dimensionless residual `||L rho|| / (||L|| ||rho||)`.
- Exact diagonal solve (`SparseSectorBackend(diagonal_solve=True)`, default). When the generator is exactly
  diagonal in the dyad basis (diagonal Hamiltonian blocks, collapse operators that are diagonal blocks: a closed
  model in its eigenbasis, pure dephasing), every shifted system is solved by the element-wise division
  `b / [(eta - i omega) - L_xy]` instead of GMRES, and the Hilbert resolvent of a diagonal Hamiltonian by
  `b / [(omega + i eta) - E_x]`: the sum over states, exact to rounding, no iteration. Spectra agree with the dense
  backend (explicit inverse) to 1e-12 and with GMRES to 1e-9. `backend.generator_is_diagonal` reports the test;
  `diagnostics["solver"]["method"]` is `"diagonal"` with `"exact": True` (no `iterations_*`, no residual) instead of
  `"gmres"`. `diagonal_solve=False` keeps GMRES with the preconditioner (the tests of the preconditioner use it).
  Models with a collapse operator that moves population or couples coherences, or with matrix-free blocks, are
  unchanged. Example 2 at D = 386, 9 x 9 grid: 2.8 s before, 1.6 s now (default configuration).
- GMRES restart (no change): `resolvent_restart` keeps SciPy's default (20). Measured on unpreconditioned solves of
  the thermal XXZ chain, restart 20/50/100/200 gives 673/321/134/102 iterations (0.12/0.09/0.06/0.06 s) at
  D = 57, eta = 0.03 meV; 5688/3539/1895/128 (0.99/0.98/0.87/0.08 s) at eta = 0.003 meV; but 1210/1081/939/798
  iterations in 1.4/1.7/2.6/4.1 s at D = 163, eta = 0.03 meV, where each iteration costs more and the basis takes
  85 MB at restart 200. No value wins across the cases, and the preconditioner removes the question for the
  models above, so the default is unchanged. The solves of Examples 1 and 3 and of the dissipative chain take 2 or 3
  iterations at any restart.
- `generate_spectrum` (and `generate_nq_spectrum`) evaluate a grid with `strategy="auto"` by
  default: when at least two axes are scanned and the factorized evaluation needs fewer solves, each
  pathway is cut between two scanned intervals, the two parts are evaluated once per point of their
  own axes and contracted by a matrix product (`factorized.py`). The numbers are those of the
  pointwise evaluation up to the accuracy of the linear solves (about 1e-10 for the default GMRES
  tolerance, 1e-15 for the dense backend; the regression snapshots are reproduced to 1e-9); use
  `strategy="pointwise"` for the former evaluation. Measured on the Example 2 model, rephasing
  GSB + SE + ESA, sparse backend, one BLAS thread: 5 x 5 points at D = 57 take 150 -> 10 GMRES solves
  and 27 -> 2.9 s; 3 x 3 points at D = 163 take 54 -> 6 solves and 99 -> 17 s. Example 1 on 21 x 21
  points takes 2646 -> 42 solves. Example 3 on its 151 x 151 grid (seven pathways) takes 302 solves
  instead of about 319 000 (0.04 s dense, 6 s sparse).
- `SparseSectorBackend._detector_vector` no longer loops over the basis vectors.
- The Liouvillian and the sector operators act on the whole density matrix with matrix products
  instead of one Python call per column (`generators.py`, `backends/base.py`, `sparse_sector.py`,
  `dense_liouville.py`). One application of `L` costs `2 + 5 n_c` block-by-matrix products for `n_c`
  dissipative channels (the jump term reuses `C rho` and `rho C^+` for the two anticommutator terms).
  Results are unchanged: the regression snapshots are reproduced to 2e-15 and the Kronecker form of
  the GKSL generator is checked in `tests/api/test_vectorized_action.py`. Measured on the Example 2
  model (sparse backend, no channel, one BLAS thread), before and after: one application of `L`
  1.19 -> 0.064 ms at D = 57 (18x) and 7.4 -> 0.59 ms at D = 163 (12x); 2 x 2 grid points of the
  three rephasing pathways 44 -> 3.4 s at D = 57 (13x) and 254 -> 41 s at D = 163 (6x). The test
  suite runs in 7 s instead of 23 s. `validation/benchmarks/scripts/bench_generator_action.py`
  repeats the measurement.
- Operator blocks and collapse operators may provide `matmat`/`rmatmat` (every SciPy
  `LinearOperator` does); an operator with only `matvec`/`rmatvec` still works and is applied
  column by column. The Hilbert-level `LinearOperator`s built by the backends provide both.
- New backend methods `apply_transition_matrix`, `apply_observable_matrix`,
  `apply_jump_effect_matrix` and `apply_detection_operator_matrix` act on a matrix; the vector
  methods call them. `materialize(operator, chunk=64)` builds the dense matrix by blocks of
  columns; the dense backend builds its transition and observable matrices with one product.
- `generators.operator_matmat` and `operator_rmatmat` apply an operator to a matrix.

### Benchmarks
- Rerun after the exact diagonal solve (Example 2, closed model in its eigenbasis, so default solves are divisions); the
  benchmarks of plain GMRES and of the preconditioner pass `diagonal_solve=False`. At D = 57, 9 x 9 grid: 0.0098 s
  (3.99 s with plain GMRES, 29.3 s dense); at D = 386: 1.6 s and 238 MB (2.8 s before), point by point 6.9 s and 33 MB, so
  the factorized strategy now gains a factor 4 in time and costs seven times the memory. Grid 33 x 33 at D = 57: 0.019 s
  (0.057 s before). Sparse versus dense agreement 3e-16 (Example 2) and 2e-15 (Example 3). `bench_gmres_eta.py exact`
  and `bench_resolved_map.py refresh-sparse` are new.
- All benchmarks of `validation/benchmarks` are recomputed with the factorized evaluation, the vectorized action of
  the generator and the `auto` preconditioner, and the notebooks `example{1,2,3}_benchmark.ipynb` and
  `validation/analysis/example2_analysis.ipynb` read the new results. `bench_scaling.py` measures four
  configurations (dense, plain GMRES, default, default point by point) on a 9 x 9 grid up to D = 386 (peak memory
  above the baseline of the process); `bench_grid_size.py` (new) measures time and solves against the number of grid
  points (log-log slope of the solves 1.00 factorized against 2.00 pointwise; of the time 0.5 to 1.2 against 1.9 to
  2.0); `bench_gmres_eta.py` and the dense-versus-sparse scripts read `result.diagnostics["solver"]`.
- `bench_resolved_map.py` (new): the cost of a *resolved* map at equal resolution (grid spacing a quarter of the
  line width) for the direct evaluation, the sum over states and a time route that is factorized as well (matrix
  transform or zero-padded FFT, bandpass sampling, cheapest (dt, T) that reaches 1e-2, 1e-3 and 1e-4). Result: the
  direct evaluation is comparable to the time route at 1e-2 for strongly damped lines and 4 to 150 times faster at
  1e-3 to 1e-4; the sum over states is faster than both for small models. At D = 386 the factorized strategy divides
  the time by 14 and multiplies the memory by 5 (2.8 s and 238 MB against 38.7 s and 47 MB point by point).
- The earlier claims "x260 from the preconditioner" and "8 to 400 times faster than time propagation" are withdrawn:
  the diagonal preconditioner of a closed model in its eigenbasis is the exact inverse (a sum over states), and
  the comparison with the original QuDPy (136 to 13 838 times, 41 x 41 grid) is against a code that integrates each
  pair of delays separately.

### Added
- Fluorescence on an infinite window: `ObservableSpec.mean_jump(..., time_window=(t0, np.inf))`. The count rate of a
  state tends to `Tr(M rho_ss) Tr(rho)`, so the solver integrates the excess over that stationary background,
  `X = exp(t0 L^+) x` with `(L^+ - |1><<rho_ss|) x = -(m - P m)`, `m = vec(M^+)`, `P m = conj(Tr(M rho_ss)) vec(1)`:
  one dense LU (dense backend) or one GMRES solve (sparse backend, as stiff as the relaxation gap) with the adjoint
  generator, cached per channel and window. When the stationary state has no jumps (zero temperature, decay to the
  ground state) the result is the total count, the quantum yield. It needs a unique stationary state: a degenerate
  kernel raises a `ConvergenceError` (dense backend; the sparse solve may fail to converge); `integration="trapezoid"`
  with an infinite window is a `ValueError`. Tests in `tests/physics/test_infinite_window.py`: the yield of the open
  two-level system is 1 to 1e-9, a non-counted channel gives gamma_r / (gamma_r + gamma_nr), a window starting at t0
  keeps exp(-Gamma t0), the limit of long finite windows, the subtraction of a nonzero background (uphill channel),
  on both backends. Example 1 shows the yield.
- Example 4 (`examples/example4_coupled_modes_detection.ipynb`, model in `validation/models.py`): two coupled
  anharmonic bosonic modes with three levels each (D = 9) and seven GKSL channels (pure dephasing, relaxation
  cascade, downhill population transfer, radiative decay), after the two-mode model of the Perspective on
  detection-dependent dephasing in 2DES (Paiva-Ortega et al., 2026; parameters recovered from the group's scripts).
  It follows the structure of the other examples: model, `decay_rates()` checked against the eigenvalues of the QuTiP
  Liouvillian (5.55 and 8.05 meV for the two diagonal coherences), hand-written and UFSS pathways, coherent emission and
  three action readouts (photoluminescence-type projector with equal and unbalanced weights, excitation number) from one
  propagation, antidiagonal linewidths at two transfer rates, and two verifications: the excitation-number action signal
  equals -i times the polarization to 3e-16 with dissipation present (each readout alternative alone breaks it), and a
  single pathway has the width of the declared rates to about 1 %. Results in `validation/results/data/example4.npz`.
  `tests/physics/test_action_number_identity.py` checks the identity (211 tests, plus 4 in the local
  `tests/consistency/`).
- `validation/benchmarks/scripts/bench_transverse_chain.py`: XXZ ring (Example 1) with a transverse field of 0.3 meV, which
  couples the magnon-number sectors: no sector structure, no truncation in N, the full Hilbert space (D = 2^L) is diagonalized and
  given to `EigenbasisKModel` as one sector (the adapters accept only one Hamiltonian block per sector). Three rephasing
  sequences on a 9 x 9 grid, ground state, 3 BLAS threads, sparse backend: L = 6 (D = 64) 0.02 s, 6 MB; L = 8 (D = 256) 0.7 s,
  110 MB; L = 10 (D = 1 024) 21 s and 1.7 GB with the default `memory_limit_mb=1024`, 22 s and 0.64 GB with 400, 98 s and 0.26 GB
  with 128 (486 solves, the grid is processed in slabs); L = 11 (D = 2 048): diagonalization 10 s, spectrum 676 s, 1.0 GB
  with `memory_limit_mb=520`. Memory grows as D^2 (x16 per two sites): L = 12 needs about 2 GB of vectors at the minimum
  (peak about 4 GB, extrapolated: about 1 h), not run. The generator is diagonal, every solve is a division. Forced plain
  GMRES (`diagonal_solve=False`, `preconditioner=None`) on the same model: 1 570 iterations per solve at L = 5 (D = 32), 2 005 at
  L = 6 (D = 64), 3.5 and 12.6 s instead of milliseconds. Results in `results/data/transverse_chain.json`.
- `validation/pulses.py`: time-domain calculation of the third-order rephasing signal of a closed system driven by Gaussian
  pulses of finite duration, independent of the resolvent formalism (hierarchy of the multilinear terms of the three pulses,
  interaction picture, RK4; the free evolution after the pulses is integrated analytically; rotating-wave or real field).
  It agrees with a non-perturbative propagation with phase cycling and amplitude extrapolation to 1.5e-7 (rotating-wave
  field) and 4.4e-7 (real field) with overlapping pulses, and with QuDPy-FDGF in the impulsive limit (pulses of 0.02 meV^-1)
  to 2e-3. `validation/analysis/example2_pulses.py` computes the 60 x 60 map of the thermal XXZ chain for pulse durations
  from 0.02 to 3 meV^-1 (resumable; one file per map in `validation/results/cache/`), for the error of the impulsive limit and
  of the rotating-wave approximation at THz frequencies. `tests/physics/test_finite_pulses.py` (2 tests, 250 in all).
  Result (`analysis/example2_pulses.ipynb`, carrier on the bright magnon): the impulsive limit costs about 4 % per meV^-1 of
  pulse standard deviation (1e-3 at 0.02, 1 % near 0.25, 12 % at 3) and agrees with QuDPy-FDGF to 1e-3 on the published map
  for the shortest pulses; the RWA costs 9 % for very short pulses and less than 1 % from 0.7 meV^-1 on; the real field
  against QuDPy-FDGF has a minimum of 2.9 % at 0.7 meV^-1 (pulse FWHM 1.1 ps): no duration brings both approximations
  below a few per cent when the carrier sits on the magnon. Not done: a scan of the carrier (real THz sources are
  broader and centred higher).
- `validation/benchmarks/scripts/bench_qudpy_perspective.py` and `..._series.py`: original QuDPy (public `coherence2d`,
  four interactions for the action diagrams) against QuDPy-FDGF on the coupled modes of Example 4, at the grid of the
  Perspective and at a resolved grid; resumable (cases already in `results/data/qudpy_perspective.json` are skipped, the
  propagated signals are kept per diagram in `results/cache/qudpy_perspective`).
- Example 2 (thermal XXZ chain): the notebook computes the complete rephasing response with the eight sequences of
  `pathways_from_signature("-++")` (three are non-zero for the ground state, the five others are activated by the
  thermal state and add 4.4 % of the maximum at 4 K; the scan takes 0.1 s with the exact diagonal solve, 139 s before),
  and locates the bimagnon by total momentum: at K = 0 the lowest level (2.1108 meV, P_nn = 0.67) lies 0.4475 meV below
  the first scattering level (2.5583 meV, P_nn = 0.29) and 0.434 meV below the free pair 2 omega(2 pi / L); the distance
  to 2 omega(0), -0.134 meV, is the displacement of the ESA line from the bright line, not a binding energy.
  `models.EX2_PATHWAYS_ALL` and `models.example2_bimagnon`; `example2.npz` stores the eight maps (`rephasing` is their
  sum, `rephasing_three` the sum of GSB, SE and ESA); `analysis/example2_analysis.ipynb` uses the eight sequences
  (cutoff errors 6.4e-2, 4.3e-3, 2.4e-4 at N_max = 2, 3, 4); `tests/physics/test_chain_bimagnon.py` (3 tests, 248 in all).
- Closed limit of Example 4, fifth order: section 8 of `examples/example4_coupled_modes_detection.ipynb` takes the Hamiltonian
  of Example 4 without dissipation on the ten states with n_a + n_b <= 3, generates the seven fifth-order 2Q pathways with
  UFSS and computes the double-quantum map at eta = 2 meV (six lines, three per mode, at -(2 omega_m + Delta) and omega_m,
  omega_m + Delta, omega_m + 2 Delta); the seven pathways cancel to 2e-13 when the anharmonicity is set to zero.
  `validation/analysis/example4_chi5_analysis.ipynb` checks that the third-order response is the same on three truncations,
  assigns the lines, shows that the mixed state |1,1> appears only through the coupling (1.3e-3 of the maximum at
  J = 1 meV), compares the eight orderings of the last three interactions with and without the three-excitation manifold
  (the all-Ku ordering needs it, the all-Bd ordering never occurs), measures the truncation at two excitations (the maximum
  changes by 2 % and the complex spectrum by 94 % of the maximum), checks the harmonic limit (2e-13 at N_max = 3 and 4, 1.4 at
  N_max = 2, 0.23 with three levels per mode) and that two independent modes add (2e-13). `validation/models.py` has
  `example4_chi5_qobj`, `example4_chi5_model` and `example4_chi5_pathways`;
  `tests/physics/test_closed_limit_two_modes.py` (4 tests, 245 in all) covers the cancellation, its dependence on the
  truncation and the additivity. Results in `validation/results/data/example4_chi5.npz` and `example4_chi5_analysis.json`.
- `validation/analysis/compare_perspective.py` and `example4_analysis.ipynb`: comparison of Example 4 with the data stored in the
  repository of the Perspective (ten samples of each antidiagonal cut, per diagram and for the sum, six scenarios), without
  recomputing anything with QuDPy. QuDPy-FDGF, evaluated at the same ten points of the Perspective's time grid and fitted with
  its estimator, reproduces the stored coherent half widths to 0.35 % on average (1.4 % at most, 48 values) and the amplitudes to
  a few percent, with no parameter to tune. The stored action half widths (Table I: 8.63 and 11.0 meV against 5.16 and 7.10 meV
  coherent) are reproduced to 0.4 % (1.4 % at most) by the four UFSS fluorescence diagrams with the excitation-number detector
  when they are summed with the signs of the raw QuDPy diagrams; with the signs of the Liouville-space formalism the same four
  diagrams give the coherent half widths, as the exact identity S_N = -i S_pol requires (the excited-state-absorption diagram
  Q2b has the opposite sign in the stored sum). The PL projector of the paper gives 8.07 and 10.55 meV (coherent 5.16 and 7.10),
  within 0.6 meV of Table I. The stored unbalanced scenario is compatible with a weight of the two-quantum states of about 0.45
  in the detector, not with weights 2:1 on the one-quantum states. The ten-point estimator gives 5.16 and 7.10 meV for the
  coherent signal where the same fit on a fine cut gives 5.63 and 7.25 meV.
- `solver.steady_state(check_unique=None)`: the stationary state of the generator as a `DensityState`
  (`L rho = 0`, `Tr rho = 1`), from `(L + |r>><<1|) x = r` with `Tr r = 1`: dense LU or matrix-free GMRES
  (restart 100, 100 restart cycles unless `krylov_maxiter`, `ConvergenceError` beyond). Diagnostics in
  `state.metadata["steady_state"]` (method, relative residual, Hermiticity error, smallest eigenvalue,
  evidence about uniqueness). `SteadyStateWarning` when the kernel of `L` is not one-dimensional or the
  result is not a density matrix. The dense backend tests uniqueness for free (reciprocal condition number
  of the LU, estimate eps/rcond < 1e-6; the minimum-norm solution is returned otherwise); the sparse backend
  does so only with `check_unique=True` (a second GMRES solve with a random `r`, limited to 2000 iterations,
  `unique=None` if inconclusive: the test is as stiff as the generator).
- `feed_model(model, context=None, initial_state=None)`: `initial_state` imposes the reference state, a
  `DensityState` or `PureState` (validated like any reference state) or `"steady"` for the stationary state of
  the generator; the model is then not asked for its own state. Backend methods `set_reference_state`,
  `impose_steady_state`, `density_state`. `tests/api/test_steady_state.py` (24 tests: ground state,
  Gibbs state under detailed balance, populations balancing the rates, closed model, decoupled level,
  independence from the reference state, sectors, imposing the state cures the stationarity warning, linear
  response scales with the stationary ground population, the model is not asked for a state that is imposed).
- Diagnostics of the linear solves: `result.diagnostics["solver"]` with `solves`, `iterations_mean`,
  `iterations_max`, `relative_residual_max`, `relative_residual_mean` (true residual `||b - S x|| /
  ||b||`, a backward error), `condition_estimate` (`max (eta + |omega| + ||L||) / eta` over the
  frequency intervals, `||L||` by power iteration in `backend.generator_norm()`) and
  `forward_error_estimate`; `backend.solve_statistics()`; backend option `record_solves=True`
  (one record per solve). The Hilbert-space resolvent reports `iterations` and `relative_residual`.
  The dense backend reports the residual of its direct solves in the factorized evaluation.
  Iterative refinement was evaluated and not kept: with GMRES each refinement step costs as many
  iterations as the solve it corrects, so reaching 5e-13 from 1e-6 takes 36 iterations (one step, summed over
  four frequencies) where a solve with `krylov_tolerance=1e-12` takes 28 and ends at 9e-16 (thermal XXZ chain,
  L = 4, eta = 0.03 meV; same ratio for the dissipative three-site chain with the diagonal preconditioner, 24
  against 18 over six frequencies). Lower `krylov_tolerance` instead.
- `generate_spectrum(..., strategy="auto"|"factorized"|"pointwise", memory_limit_mb=1024.0)`.
  `result.diagnostics` reports `strategy`, `resolvent_solves`, `resolvent_factorizations` (dense
  backend), `propagations`, `maximum_resolvent_residual` (sparse backend) and, for the factorized
  evaluation, `factorization` (cut, axes of each side, vectors stored, slabs).
- Backend primitives of the adjoint chain: `interaction_block`, `interval_block` (propagator or
  resolvent, forward or adjoint, at several coordinates), `detector_vector`,
  `apply_transition_adjoint_matrix`, `liouville_initial_vector`, `factorized_applicable`;
  `reset_counters()` and `counters()`.
- `tests/api/test_factorized_strategy.py`: equality of the two evaluations (both backends, all kinds
  of observable, one to three axes, thermal and dissipative models, fifth order), solve counts,
  slabs, fall-backs.
- Pathway signatures and completeness: `Interaction.sign`, `FrequencyPathway.signature`,
  `pathway_signature()`, `pathways_from_signature(signature, component=, max_manifold=,
  min_manifold=)` which enumerates the `2**n` sequences with a given wave-vector signature.
  `FrequencyPathway.origin` (`"user"`, `"enumerated"`, `"ufss"`). `FrequencyPathway` rejects the
  wrong signature for the components `rephasing` and `nonrephasing` at third order (**change**:
  a hand-written pathway that mixed them now raises `ValueError`).
- `SpectrumResult.null_pathways()` and `result.diagnostics["null_pathways"]`;
  `NullPathwayWarning` for a hand-written pathway that vanishes on the whole grid
  (`generate_spectrum(..., warn_null=False)` silences it).
- `PropagationInterval(signed=True)`: the protocol checks the sign of the coherence order.
- `EigenbasisKModel`: `transition_window=(w_min, w_max)` restricts the automatic split of the
  interaction operator to a band of Bohr frequencies; `transition_summary()` reports the fractions
  of `sum |mu_ij|^2` kept, static, degenerate and outside the window (also in `solver.summary()`);
  `ModelConsistencyWarning` when elements coupling degenerate states are excluded by `rwa_tol`.
  `ExcitationSectorModel.transition_summary()` reports an explicit decomposition. Default
  behavior unchanged.
- Operator checks: the Hamiltonian blocks must be Hermitian (`ModelContractError`, backend option
  `hermiticity_tolerance`, 1e-10) and `J_minus` must be `J_plus^dagger` (`ModelConsistencyWarning` at
  the first use of each operator, option `transition_tolerance`, 1e-8).
- Check of the reference state at `feed_model`: relative residual `||L rho|| / ||rho||`,
  `StationarityWarning` (default) or `StationarityError` above `stationarity_tolerance` (1e-8),
  arguments `check_stationarity="warn"|"error"|"off"` and `stationarity_tolerance` of
  `SpectroscopySolver`, method `stationarity_residual()`, entry in `summary()`.
- `tests/regression/`: stored outputs of eight small calculations (two-level system, dissipative
  and thermal three-site chains, time-domain route, fifth-order bosons) that refactors of the
  numerical core must reproduce to 1e-9.
