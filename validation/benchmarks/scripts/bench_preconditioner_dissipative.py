"""Diagonal preconditioner against plain GMRES on dissipative models with a diagonal Hamiltonian.

The default ``preconditioner="auto"`` switches the diagonal preconditioner on when the Hamiltonian blocks
are diagonal (eigenbasis), also with collapse operators that couple dyads, where it is only approximate.
This script checks that choice: the iterations and the time per solve of the forced diagonal preconditioner
and of plain GMRES (tolerance 1e-10, restart 20) on shifted solves of three kinds of start vector (the state
after one bra interaction, after two interactions, and a random vector, the worst case), for dissipation rates
from weak to far above the energy scale, and two broadenings.

    python bench_preconditioner_dissipative.py [chain3 | collective | local | all]

Models: the three-site hard-core chain of the tests (rates x 0.1 to x 1000); the six-site XXZ chain of
Example 2 (D = 57, thermal state at 4 K) with collective decay or local dephasing from 0.001 to 10 meV.
Results are merged by group into results/data/preconditioner_dissipative.json (about 3 min per group for the XXZ chain).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # single-thread BLAS, before NumPy is imported
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))               # validation/ (models.py)
from models import DATA, EX2_AXES, example2_chain, example2_context
import json
import time
import warnings

import numpy as np
import qutip as qt

from qudpy_fdgf import ExcitationSectorModel, Interaction, SpectroscopySolver

warnings.simplefilter("ignore")
TOLERANCE = 1e-10


def three_site_chain(scale):
    """Hard-core exciton chain, manifolds 0-2, local dephasing 0.02 and collective decay 0.005, both x scale."""
    n, J = 3, -0.05
    pairs = [(0, 1), (0, 2), (1, 2)]
    eps = 2.0 + 0.01 * np.arange(n)
    H1 = np.diag(eps).astype(complex) + J * (np.eye(n, k=1) + np.eye(n, k=-1))
    H2 = np.diag([eps[i] + eps[j] for i, j in pairs]).astype(complex)
    H2[0, 1] = H2[1, 0] = J
    H2[1, 2] = H2[2, 1] = J
    up10 = np.ones((n, 1), dtype=complex)
    up21 = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=complex)
    number = [{("1", "1"): np.diag(np.eye(n)[s]).astype(complex),
               ("2", "2"): np.diag([1.0 if s in p else 0.0 for p in pairs]).astype(complex)} for s in range(n)]
    lowering = {("0", "1"): up10.conj().T, ("1", "2"): up21.conj().T}
    c_ops = tuple((blocks, 0.02 * scale) for blocks in number) + ((lowering, 0.005 * scale),)
    model = ExcitationSectorModel({"0": np.zeros((1, 1), dtype=complex), "1": H1, "2": H2},
                                  {("1", "0"): up10, ("2", "1"): up21}, c_ops_raw=c_ops, initial_sector="0",
                                  energy_unit="eV")
    return model, None, [1.95, 2.0, 2.01, 2.05, 4.0, 4.02], (0.002, 0.0002)


def xxz_chain(kind, rate, n_sites=6):
    """Example 2 chain with collective lowering or local number dephasing at ``rate`` (meV)."""
    H, N_op, M_minus = example2_chain(n_sites)
    magnons = np.rint(N_op.diag().real).astype(int)
    members = {N: np.flatnonzero(magnons == N) for N in range(5)}
    H_full, M_full = H.full(), M_minus.full()
    hamiltonians = {N: H_full[np.ix_(i, i)] for N, i in members.items()}
    raising = {(N + 1, N): M_full[np.ix_(members[N + 1], members[N])] for N in range(4)}
    if kind == "collective":
        c_ops = (({(N, N + 1): raising[(N + 1, N)].conj().T for N in range(4)}, rate),)
    else:
        c_ops = []
        for j in range(n_sites):
            op = qt.tensor([qt.basis(2, 1).proj() if k == j else qt.qeye(2) for k in range(n_sites)]).full()
            c_ops.append(({(N, N): op[np.ix_(members[N], members[N])] for N in range(5)}, rate))
        c_ops = tuple(c_ops)
    model = ExcitationSectorModel(hamiltonians, raising, initial_sector=0, c_ops_raw=c_ops, energy_unit="meV")
    omegas = [float(EX2_AXES["omega_1q"][i]) for i in (5, 20, 40)]
    return model, example2_context(), omegas, (0.03, 0.003)


def measure(model, context, omegas, eta, preconditioner):
    solver = SpectroscopySolver(backend="sparse_sector", eta=eta, krylov_tolerance=TOLERANCE,
                                preconditioner=preconditioner, check_stationarity="off")
    solver.feed_model(model, context=context) if context is not None else solver.feed_model(model)
    backend = solver.backend
    reference = backend.liouville_initial_vector()
    first = backend.interaction_block(reference[:, None], Interaction("Bu"))[:, 0]
    second = backend.interaction_block(first[:, None], Interaction("Ku"))[:, 0]
    generator = np.random.default_rng(0)
    random_vector = generator.standard_normal(reference.size) + 1j * generator.standard_normal(reference.size)
    backend.reset_counters()
    start = time.perf_counter()
    for vector in (first, second, random_vector):
        for omega in omegas:
            backend._solve_liouville_resolvent(vector, omega, eta)
    elapsed = time.perf_counter() - start
    statistics = backend.solve_statistics()
    return {"iterations_mean": statistics["iterations_mean"], "iterations_max": statistics["iterations_max"],
            "ms_per_solve": 1e3 * elapsed / statistics["solves"], "relative_residual_max": statistics["relative_residual_max"]}


GROUPS = {
    "chain3": [(f"three-site chain, dissipation x{s:g}", {"scale": s}, three_site_chain(s)) for s in (0.1, 1, 10, 100, 1000)],
    "collective": [(f"XXZ L=6, collective decay {r:g} meV", {"rate_meV": r}, xxz_chain("collective", r))
                   for r in (0.001, 0.01, 0.1, 1.0, 10.0)],
    "local": [(f"XXZ L=6, local dephasing {r:g} meV", {"rate_meV": r}, xxz_chain("local", r))
              for r in (0.001, 0.01, 0.1, 1.0, 10.0)],
}

if __name__ == "__main__":
    requested = sys.argv[1] if len(sys.argv) > 1 else "all"
    names = list(GROUPS) if requested == "all" else [requested]
    out = DATA / "preconditioner_dissipative.json"
    DATA.mkdir(parents=True, exist_ok=True)
    report = json.loads(out.read_text()) if out.exists() else {}
    print(f"{'scenario':44s} {'eta':>7s} | {'plain: it mean/max  ms/solve':>31s} | {'diagonal: it mean/max  ms/solve':>31s} | ratio it  time")
    for name in names:
        rows = []
        for label, parameters, (model, context, omegas, etas) in GROUPS[name]:
            for eta in etas:
                plain = measure(model, context, omegas, eta, None)
                diagonal = measure(model, context, omegas, eta, "diagonal")
                row = {"scenario": label, **parameters, "eta": eta, "plain": plain, "diagonal": diagonal,
                       "iteration_ratio": diagonal["iterations_mean"] / plain["iterations_mean"],
                       "time_ratio": diagonal["ms_per_solve"] / plain["ms_per_solve"]}
                rows.append(row)
                print(f"{label:44s} {eta:7.4f} | {plain['iterations_mean']:9.1f}/{plain['iterations_max']:6d} {plain['ms_per_solve']:9.2f} | "
                      f"{diagonal['iterations_mean']:9.1f}/{diagonal['iterations_max']:6d} {diagonal['ms_per_solve']:9.2f} | "
                      f"{row['iteration_ratio']:6.2f} {row['time_ratio']:6.2f}")
        report[name] = {"rows": rows, "worst_iteration_ratio": max(r["iteration_ratio"] for r in rows),
                        "best_iteration_ratio": min(r["iteration_ratio"] for r in rows)}
        out.write_text(json.dumps(report, indent=2))
