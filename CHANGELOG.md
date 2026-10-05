# Changelog

Changes to the code of QuDPy-FDGF, newest first. The article and the API reference manual
(`docs/`) are updated with each entry.

## Unreleased

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
