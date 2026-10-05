# Changelog

Changes to the code of QuDPy-FDGF, newest first. The article and the API reference manual
(`docs/`) are updated with each entry.

## Unreleased

### Changed
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
