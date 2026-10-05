# QuDPy-FDGF

**QuDPy-FDGF: A Python-Based Tool for Computing Ultrafast Nonlinear Optical
Responses Using Frequency-Domain Green's Functions.**

The distribution is named `qudpy-fdgf` and the import name is `qudpy_fdgf`.
Install it in editable mode from the repository root:

```bash
pip install -e .
```

The example notebooks import the package directly and no longer manipulate
`sys.path`, so this install is required before running them. Add the
`examples` extra (`pip install -e '.[examples]'`) to install QuTiP for the
physical model definitions and UFSS for the fifth-order pathway example.

`qudpy_fdgf` is a generic spectroscopy engine. It contains no physical
model. An external model supplies its sectors, Hamiltonian blocks,
transitions, initial state, and observable.

## Requirements and tested versions

| Package | Required | Used for | Tested with |
|---|---|---|---|
| Python | 3.10 or later | | 3.13.14 |
| NumPy | 1.24 or later | core | 2.5.0 |
| SciPy | 1.10 or later | core (Krylov, GMRES) | 1.18.0 |
| Matplotlib | 3.7 or later | `SpectroscopyPlotter` and figures | 3.11.0 |
| QuTiP | optional | physical operators in the examples | 5.3.0 |
| UFSS | optional | automatic pathway generation (Example 3) | 0.2.5 |
| pytest | optional | test suite | 9.1.1 |
| psutil, nbformat, nbclient | optional | `validation/` (memory, command-line notebooks) | 7.2.2, 5.11.1, 0.11.0 |

Extras of `pip install -e '.[...]'`: `examples` (QuTiP, UFSS, IPython, ipykernel), `validation`
(the previous ones plus psutil, nbformat, nbclient), `dev` (pytest), `manybody` (TeNPy, only for
`qudpy_fdgf.experimental`). All tests and notebooks were run on Windows 11 with the versions
above. BLAS is restricted to one thread (`OMP_NUM_THREADS=1`) for the timings of `validation/`.

## Repository layout

```text
src/qudpy_fdgf/          the package (generic engine, no physical model)
    solver.py            SpectroscopySolver: pathways, protocols, spectra, decay_rates()
    contracts.py         model contract: SectorModel, states, ThermodynamicContext, CollapseChannel
    model_adapters.py    EigenbasisKModel, ExcitationSectorModel (build a model from arrays or Qobj)
    pathways.py          Interaction, FrequencyPathway, translate_ufss_diagrams
    protocols.py         PropagationInterval, SpectroscopyProtocol, standard_nq_protocol
    observables.py       ObservableSpec: polarization, action detection, jump counts
    generators.py        matrix-free Liouville generator actions
    backends/            DenseLiouvilleBackend (reference), SparseSectorBackend (matrix-free)
    diagnostics.py       DecayRates (rates implied by the declared collapse channels)
    results.py           SpectrumResult, PathwayResult, PlotResult
    plotting.py          SpectroscopyPlotter
    capabilities.py, exceptions.py
    experimental/        low-rank and tensor-network prototypes (not part of the released API)
examples/                quickstart.py and the three examples of the article as notebooks
    old_scripts/         earlier versions of the examples, kept until the review is finished
validation/              models.py, analysis/, benchmarks/, results/ (see validation/README.md)
tests/                   api/ (contracts and options), physics/ (closed-form results and limits),
                         regression/ (stored outputs that refactors of the numerical core must reproduce)
docs/                    API reference manual (PDF and LaTeX source)
```

## Reference test run

Install the package with the examples extra, then run from the repository root.

**1. Quick start** (about 2 s). An open two-level system: linear absorption and third-order
rephasing response, each followed by its analytical value.

```bash
python examples/quickstart.py
```

Expected output:

```text
linear peak      : 2.0000 eV
FWHM (numerical) : 0.0202 eV
2(Gamma_2 + eta) : 0.0202 eV
rephasing peak   : (omega1q, omega3) = (-2.000, 2.000) eV
|S| at peak      : 1225.37
2 mu^4/(Gamma_2 + eta)^2 : 1225.37
```

**2. Example 1** (about 10 s): run all cells of `examples/example1_two_level_open.ipynb`. The
cell of Section 7 prints

```text
peak = (-2.000, 2.000) eV
max|S_pop + i S_pol| / max|S_pol| = 0.0e+00
N_fl / S_pop = 0.993262; 1 - exp(-5) = 0.993262
```

and saves its results in `validation/results/data/example1.npz`.

**3. Test suite** (about 15 s).

```bash
python -m pytest
```

Expected: `42 passed` in about 17 s (a local `tests/consistency/` folder, if present, adds 3 tests).

## Examples, tests, and validation

- `examples/quickstart.py`: open two-level system, linear and third-order rephasing responses,
  with analytical checks (about 2 s). It is the Quick start of the API documentation.
- `examples/example{1,2,3}_*.ipynb`: the three examples of the manuscript, as notebooks (see
  `examples/README.md`).
- `python -m pytest`: runs `tests/` in about 15 s (`-m "not slow"` skips the slowest). `tests/api/`
  checks the contracts and options of the library, `tests/physics/` checks closed-form results and
  limiting cases of the manuscript (Example 1 formulas, exact jump integral, decay rates, harmonic
  cancellation at fifth order). `tests/regression/` compares eight small calculations (two-level
  system, dissipative and thermal chains, time-domain route, fifth-order bosons) with stored outputs
  (`data/snapshots.npz`); regenerate them only on purpose with
  `python tests/regression/generate_snapshots.py`.
- `validation/`: analysis notebooks (convergence, analytical limits), benchmark notebooks
  and scripts (time, memory, comparison with the original QuDPy), shared models and results.
  See `validation/README.md`.
- `docs/`: the API reference manual.

## Dependency direction

```text
external model  --->  qudpy_fdgf contracts
script/notebook --->  external model + qudpy_fdgf
qudpy_fdgf --->  no physical model
```

The solver must never import a class such as `SpinOrbitalModel`.

## Minimal model contract

```python
class MyModel:
    def sectors(self):
        ...

    def dimension(self, sector):
        ...

    def hamiltonian_blocks(self, source):
        # {target_sector: operator(target, source)}
        ...

    def transition_blocks(self, operator_name, direction, source):
        # direction is "plus" or "minus"
        ...

    def observable_blocks(self, observable_name, source):
        ...

    def initial_condition(self, context=None):
        # PureState(...) or DensityState(...)
        ...

    def collapse_channels(self, context=None):
        # Tuple of CollapseChannel(name, rate, operator_blocks)
        ...

    def equilibrium_state(self, context):
        # Gibbs state built and partitioned into sectors by the model.
        ...

    def requirements(self):
        # ModelRequirements(...) or dict
        ...

    def capabilities(self):
        # Legacy API retained for compatibility.
        ...
```

Each block may be a NumPy array, a SciPy sparse matrix, or a
`scipy.sparse.linalg.LinearOperator`.

`initial_state()` remains available as a backward-compatible fallback for a
pure state. The solver never builds a thermal basis, chooses thermally
accessible sectors, or invents dissipative rates.

## Initial states and temperature

A mixed state is supplied as ket/bra blocks:

```python
rho0 = DensityState(
    blocks={
        ("bright", "bright"): rho_bb,
        ("dark", "dark"): rho_dd,
        ("bright", "dark"): rho_bd,
        ("dark", "bright"): rho_bd.conj().T,
    }
)
```

Cross-sector blocks support bright--dark, spinor, and momentum coherences. The
solver validates dimensions, Hermiticity, trace, and positivity only for this
physical initial state.

The thermal context is passed to the model:

```python
context = ThermodynamicContext(
    temperature=10.0,
    ensemble="canonical",
    k_ensemble="independent",  # or "global"
)
solver.feed_model(model, context=context)
```

`equilibrium_state(context)` must return the exact thermal state in the
truncated basis built by the model.

## Dynamics convention

The numerical core uses the time-domain generator

\[
\mathcal A\rho=-i[H,\rho]
+\sum_j\gamma_j\left(
L_j\rho L_j^\dagger-\frac12\{L_j^\dagger L_j,\rho\}
\right).
\]

A time interval applies \(e^{t\mathcal A}\). A frequency interval uses only the
direct resolvent

\[
\left[(\eta-i\omega)I-\mathcal A\right]^{-1}.
\]

No global \(-i\) factor is included in this resolvent. The pathway's
perturbative coefficient therefore directly carries the
\(i^n(-1)^{n_B}\) convention, as in the time-domain expansion. No temporal FFT
is used.

Dissipative channels strictly follow the `(L, gamma)` convention:

```python
CollapseChannel(
    name="bright_to_dark",
    rate=gamma_bd,
    operator_blocks={
        "bright": {"dark": L_dark_from_bright},
    },
)
```

The rate must not be included a second time in the operator.

The rate is the Lindblad coefficient, not necessarily a measured rate: a
diagonal channel \(L=\sum_x l_x|x\rangle\langle x|\) damps the coherence
\(|x\rangle\langle y|\) at \(\gamma(l_x-l_y)^2/2\). A pure-dephasing rate
\(\gamma_\phi=1/T_2^*\) therefore requires `rate=gamma_phi/2` with
\(\sigma_z\), but `rate=2*gamma_phi` with a projector or a number operator.
`solver.decay_rates()` reports the rates implied by the declared channels in
the eigenbasis of the Hamiltonian:

```python
rates = solver.decay_rates()                 # modes=True adds exact Liouvillian eigenvalues
rates.population_decay                       # Gamma_x of each eigenstate
rates.coherence_decay[x, y]                  # decay rate of |x><y|
rates.coherences(between=("0", "1"))         # (x, y, omega_xy, gamma_xy) between two sectors
```

For the open two-level system of `examples/quickstart.py` it returns
\(\gamma_1\) for the excited-state population and
\(\Gamma_2=\gamma_1/2+\gamma_\phi\) for the optical coherence.

## Multiple observables and integrated fluorescence

A pathway propagation can be reused for multiple detection schemes:

```python
from qudpy_fdgf import ObservableSpec

fluorescence = ObservableSpec.mean_jump(
    "fluorescence",
    channel="radiative",
    time_window=(0.0, 200.0),
    efficiency=0.35,
)

result = solver.generate_spectrum(
    protocol,
    axes,
    observables={
        "population_a": "P_a",
        "population_b": "P_b",
        "fluorescence": fluorescence,
    },
)
```

The names of GKSL channels available for jump counting are returned by
`solver.jump_channel_names()`. The legacy result remains available through
`result.pathways` and `result.components`. Additional outputs are organized by
observable:

```python
result.observables["population_a"]["P1"]
result.observable_components["fluorescence"]["rephasing"]
```

Action detection can add a projection pulse after a three-interaction pathway.
The pathway core is then propagated only once, and the fourth pulse is applied
only in the detection branch:

```python
action_population = ObservableSpec.action(
    "action_population",
    fourth_interaction={
        "label": "Bu",
        "operator": "light_matter",
        "pulse_index": 3,
    },
    operator="excited_population",
)

action_fluorescence = ObservableSpec.mean_jump(
    "action_fluorescence",
    channel="radiative",
    time_window=(0.0, 200.0),
    fourth_interaction="Bu",
)
```

`fourth_interaction` also accepts a sequence of alternative interactions. Their
contributions are summed with the appropriate ket/bra prefactors. The legacy
form remains valid: if the fourth pulse is already included in
`FrequencyPathway.interactions`, simply omit `fourth_interaction` from the
observable. Both formulations produce the same signal.

An `operator` observable computes `Tr[O rho]` at the end of the pathway. A
`jump_rate` observable computes the instantaneous rate

\[
I_j(t)=\varepsilon\,\gamma_j\operatorname{Tr}[L_j^\dagger L_j\rho(t)],
\]

where \(\varepsilon\) is the detection efficiency, and `integrated_jump`
computes its integrated mean over the requested window. The window starts
after the pathway's final state, and therefore after the protocol's last
interaction and propagation. This is a mean jump count; the API does not yet
generate quantum trajectories or a counting distribution `P(N)`.

By default (`integration="exact"`) the window integral is exact. Because the
trace is linear in the state, the detector is integrated instead of the
state,

\[
N_j=\langle\langle X_j|\rho\rangle\rangle,\qquad
X_j=\varepsilon\,\gamma_j\int_{t_0}^{t_f}\mathcal U^\dagger(t)[L_j^\dagger L_j]\,dt,
\]

and \(X_j\) is computed once per channel and window from the exponential of
an augmented Heisenberg-picture generator (dense `expm` or matrix-free
`expm_multiply`). Each pathway state and grid point then costs one inner
product. `integration="trapezoid"` keeps the former quadrature over `n_steps`
samples of the window.

The `EigenbasisKModel` adapter also accepts multiple named operators through
`observable_op_arrays={"P_a": ..., "P_b": ..., "P_2X": ...}`. Operators must
be supplied in the same basis as the Hamiltonian; the adapter then transforms
them to the eigenbasis.

The dipole decomposition may remain automatic or be supplied explicitly to
follow the UFSS manifolds:

```python
model = EigenbasisKModel(
    H_model=H,
    interaction_op_array=mu,
    j_plus_array=J_plus,
    j_minus_array=J_minus,
)
```

Both explicit arrays must be supplied together and may have shape `(d, d)` or
`(N_k, d, d)`. Without these arguments, the current separation based on the
sign of `Delta_E` and `rwa_tol` remains unchanged.

## Plotter V10

The plotter is separate from the computation and accepts a configuration
dictionary. By convention, `axis_values[0]` is plotted on the vertical axis and
`axis_values[1]` on the horizontal axis, typically giving vertical excitation
and horizontal emission:

```python
import numpy as np
from qudpy_fdgf import SpectroscopyPlotter

# Standard 2D convention: real = absorptive, imag = dispersive.
plotter = SpectroscopyPlotter(detection_phase=np.pi / 2)
plot_params = {
    "source": "pathways",
    "pathways": ["R3", "R1", "R2"],
    "totals": ["1Q"],
    "view": "all",                 # "real", "imag", "abs", or "all"
    "diagonals": "auto",            # y=-x (rephasing) or y=x (non-rephasing)
    "normalization": "row",
    "labels": ("Emission energy (eV)", "Excitation energy (eV)"),
    "title": "2D spectroscopy",
    "show": True,
    "style": {
        "cmap": "RdYlBu_r",
        "abs_cmap": "magma",
        "levels": 30,
        "contour_lines": True,
    },
}
plot_result = plotter.plot_spectrum_result(result, plot_params)
```

To plot multiple pathways with a single component:

```python
plotter.plot_pathways(
    result,
    {
        "pathways": ["R1", "R2", "R3"],
        "detection_phase": np.pi / 2,
        "diagonals": "auto",
        "view": "real",
        "totals": False,
        "normalization": "row",
        "show": True,
    },
)
```

To display all three quadratures side by side for the same pathways:

```python
plotter.plot_real_imag_abs(
    result,
    {
        "pathways": ["R1", "R2", "R3"],
        "detection_phase": np.pi / 2,
        "diagonals": "auto",
        "normalization": "row",
        "style": {"cmap": "RdYlBu_r", "abs_cmap": "magma"},
        "show": True,
    },
)
```

For an additional output, use `source="observables"` and supply
`observable="fluorescence"`. The `plot_pathways_multiorder` method remains
available as a compact V9-inspired interface.

## Setup

```python
from qudpy_fdgf import SpectroscopySolver
from my_spin_orbital_model import SpinOrbitalModel

model = SpinOrbitalModel(...)

solver = SpectroscopySolver(
    backend="sparse_sector",
    eta=0.02,
    krylov_tolerance=1e-10,
)
solver.feed_model(model)
```

## Backends

### `SparseSectorBackend`

- opaque sectors supplied by the model;
- Hamiltonian blocks that may couple multiple sectors;
- Krylov time propagation;
- rank-one ket/bra branches;
- matrix-free Hilbert-space resolvent solved with GMRES;
- exact frequency-domain pathways through matrix-free Liouville action;
- optional diagonal preconditioner for the GMRES solves (`preconditioner="diagonal"`): exact, hence
  one iteration per solve, for a closed model in its eigenbasis; only approximate with collapse
  channels that couple coherences;
- no explicit Liouville matrix.

The frequency-domain path still uses a density vector of size \(D^2\). It
serves as an exact reference and sanity check, but it is not the final scalable
approach for large spin-orbital spaces. Such systems will require a low-rank
algorithm or a more specialized wave-function formulation.

### `DenseLiouvilleBackend`

- reference backend for small systems;
- explicit Liouville space;
- pure or mixed states;
- unitary or Lindblad dynamics;
- time- and frequency-domain intervals;
- useful for validating the sectorized backend.

### Backends not covered by this release

`LowRankLiouvilleBackend`, the tenpy engines (`TenpyDMRGEngine` and
`TenpyTDVPEngine`), and the associated many-body orchestration
(`ManyBodySolver` and `ManyBodyDynamicsSolver`) live in `experimental/`. They
are neither imported nor exposed by `SpectroscopySolver` in this release. They
remain available through explicit imports (for example,
`from qudpy_fdgf.experimental.low_rank_liouville import
LowRankLiouvilleBackend`) for continued development, but they are not
maintained or guaranteed to be stable here.

## Pathway conventions

- `Ku`: positive transition on the ket;
- `Kd`: negative transition on the ket;
- `Bu`: right multiplication by the negative operator, producing a positive
  transition on the bra vector;
- `Bd`: right multiplication by the positive operator, producing a negative
  transition on the bra vector.

An `Interaction` may specify a source sector and a target sector. If they are
omitted, the backend applies every block declared by the model.

## Current limitations

- construction of the exact thermal state remains the model's responsibility;
- only the GKSL/Lindblad form is supported for non-unitary dynamics;
- no general Redfield generator, HEOM, or bath memory;
- the matrix-free frequency-domain resolvent in `SparseSectorBackend` still
  uses a vector of size \(D^2\), without a \(D^2\times D^2\) matrix;
- explicit stationary-mode deflation and the Drazin pseudoinverse are not yet
  available;
- this release covers only `DenseLiouvilleBackend` and
  `SparseSectorBackend`; see "Backends not covered by this release" above;
- construction and physical validation of the basis belong to the model.
