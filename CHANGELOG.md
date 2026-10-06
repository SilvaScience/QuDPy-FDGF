# Changelog

Changes to the code of QuDPy-FDGF, newest first. The article and the API reference manual
(`docs/`) are updated with each entry.

## Unreleased

### Breaking changes
- `eta` has no default: `SpectroscopySolver(...)`, `DenseLiouvilleBackend` and `SparseSectorBackend`
  raise a `ValueError` that explains the parameter when it is missing (the former `0.05` has no
  meaning without a unit). Negative, infinite and NaN values are rejected.
- `EigenbasisKModel` and `ExcitationSectorModel`: `boltzmann_constant` no longer defaults to the
  value in eV/K. New argument `energy_unit` (`"eV"`, `"meV"`, `"cm-1"`, `"THz"`; the last is the
  ordinary frequency E/h, k_B/h = 0.0208 THz/K) sets it; `boltzmann_constant` explicit is still
  accepted (in the unit of the Hamiltonian per kelvin); both together, an unknown unit or a
  non-positive constant raise `ValueError`. A temperature above zero without either now raises a
  `ValueError` at `feed_model` instead of silently using eV/K; zero temperature needs neither.
  A model that ran in eV with a thermal state must now say `energy_unit="eV"`.

### Changed
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

### Added
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
