"""Model of the former Example 3 (two coupled anharmonic bosonic modes, fifth-order 2Q response), archived.

Taken out of validation/models.py when the examples were renumbered; the archived scripts of this folder
(analysis/fig_example3_*.py, benchmarks/example3_benchmark.ipynb, benchmarks/scripts/bench_example3_dense_sparse.py ...)
used it as ``models.EX3``, ``models.example3_model`` ... and run with the repository as it was at commit ae47e8f.
"""
import numpy as np
import qutip as qt
from qudpy_fdgf import EigenbasisKModel, standard_nq_protocol


EX3 = dict(omega_h=1.53, omega_l=1.58, coupling=0.010, mu_h=1.0, mu_l=0.30,
           U_hh=-0.020, U_ll=-0.020, U_hl=0.030, max_manifold=3)
EX3_FIXED = {"t1": 2.0, "t3": 2.0, "t4": 2.0}
EX3_AXES = {"omega_2q": np.linspace(-3.20, -2.99, 151), "omega_emit": np.linspace(1.42, 1.66, 151)}
EX3_PHASE_DISCRIMINATION = [(0, 1), (0, 1), (1, 0), (1, 0), (1, 0)]
EX3_PROTOCOL = standard_nq_protocol(order=2, nq_interval=2, detection_interval=5, n_interactions=5,
                                    nq_axis="omega_2q", detection_axis="omega_emit")


def example3_qobj(max_manifold=None, anharmonicities=None):
    """Two-mode model restricted to n_h + n_l <= max_manifold, as dense Qobj in manifold order.

    Returns (states, H, J_plus, polarization_h, polarization_l). ``anharmonicities`` is
    (U_hh, U_ll, U_hl); the default is the one of the example.
    """
    n_max = EX3["max_manifold"] if max_manifold is None else max_manifold
    U_hh, U_ll, U_hl = ((EX3["U_hh"], EX3["U_ll"], EX3["U_hl"]) if anharmonicities is None
                        else anharmonicities)
    levels = n_max + 1
    b_h = qt.tensor(qt.destroy(levels), qt.qeye(levels))
    b_l = qt.tensor(qt.qeye(levels), qt.destroy(levels))
    n_h, n_l = b_h.dag() * b_h, b_l.dag() * b_l
    one = qt.qeye([levels, levels])
    H_full = (EX3["omega_h"] * n_h + EX3["omega_l"] * n_l
              + EX3["coupling"] * (b_h.dag() * b_l + b_l.dag() * b_h)
              + 0.5 * U_hh * n_h * (n_h - one) + 0.5 * U_ll * n_l * (n_l - one) + U_hl * n_h * n_l)
    states = sorted(((a, b) for a in range(levels) for b in range(levels) if a + b <= n_max),
                    key=lambda s: (sum(s), -s[0]))
    keep = [a * levels + b for a, b in states]

    def truncate(operator):
        return qt.Qobj(operator.full()[np.ix_(keep, keep)])

    return (tuple(states), truncate(H_full),
            truncate(EX3["mu_h"] * b_h.dag() + EX3["mu_l"] * b_l.dag()),
            truncate(EX3["mu_h"] * (b_h + b_h.dag())), truncate(EX3["mu_l"] * (b_l + b_l.dag())))


def example3_operators(max_manifold=None, anharmonicities=None):
    """NumPy version of example3_qobj: (states, H, J_plus, polarization_h, polarization_l)."""
    states, H, J_plus, P_h, P_l = example3_qobj(max_manifold, anharmonicities)
    return states, H.full(), J_plus.full(), P_h.full(), P_l.full()


def example3_model(max_manifold=None, anharmonicities=None):
    _, H, J_plus, P_h, P_l = example3_qobj(max_manifold, anharmonicities)
    mu = J_plus + J_plus.dag()
    return EigenbasisKModel(H, mu, j_plus_array=J_plus, j_minus_array=J_plus.dag(), detection_op_array=mu,
                            observable_op_arrays={"polarization": mu, "polarization_h": P_h,
                                                  "polarization_l": P_l})


def example3_pathways(max_manifold=None):
    import ufss
    from qudpy_fdgf import translate_ufss_diagrams
    generator = ufss.DiagramGenerator(detection_type="polarization")
    generator.set_phase_discrimination(EX3_PHASE_DISCRIMINATION)
    generator.maximum_manifold = EX3["max_manifold"] if max_manifold is None else max_manifold
    generator.efield_times = [np.array([0.0, 0.0])] * 5
    return translate_ufss_diagrams(generator.get_diagrams(np.arange(5, dtype=float)), component="chi5_2q")
