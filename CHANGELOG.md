# Changelog

Changes to the code of QuDPy-FDGF, newest first. The article and the API reference manual
(`docs/`) are updated with each entry.

## Unreleased

### Added
- Check of the reference state at `feed_model`: relative residual `||L rho|| / ||rho||`,
  `StationarityWarning` (default) or `StationarityError` above `stationarity_tolerance` (1e-8),
  arguments `check_stationarity="warn"|"error"|"off"` and `stationarity_tolerance` of
  `SpectroscopySolver`, method `stationarity_residual()`, entry in `summary()`.
- `tests/regression/`: stored outputs of eight small calculations (two-level system, dissipative
  and thermal three-site chains, time-domain route, fifth-order bosons) that refactors of the
  numerical core must reproduce to 1e-9.
