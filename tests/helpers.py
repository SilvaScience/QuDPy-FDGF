"""Models and constants shared by the tests (imported with ``from helpers import ...``)."""
import numpy as np

from qudpy_fdgf import (
    EigenbasisKModel,
    ExcitationSectorModel,
    FrequencyPathway,
    SpectroscopySolver,
)
from qudpy_fdgf.protocols import standard_nq_protocol

GAMMA_1, GAMMA_PHI = 0.04, 0.03
H = np.diag([0.0, 2.0]).astype(complex)
MU = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
P_E = np.diag([0.0, 1.0]).astype(complex)
L_RAD = np.array([[0, 1], [0, 0]], dtype=complex)
SIGMA_Z = np.diag([-1.0, 1.0]).astype(complex)
PROTOCOL = standard_nq_protocol(order=1, n_interactions=3, nq_interval=1,
                                detection_interval=3, nq_axis="w1", detection_axis="w3")
R1 = FrequencyPathway(name="R1", interactions=("Bu", "Ku", "Bd"), component="rephasing")
REPHASING = (
    FrequencyPathway(name="GSB", interactions=("Bu", "Bd", "Ku"), component="rephasing"),
    FrequencyPathway(name="SE", interactions=("Bu", "Ku", "Bd"), component="rephasing"),
    FrequencyPathway(name="ESA", interactions=("Bu", "Ku", "Ku"), component="rephasing"),
)
BACKENDS = ("dense", "sparse_sector")


def two_level_solver(backend):
    solver = SpectroscopySolver(backend=backend, eta=0.002)
    solver.feed_model(EigenbasisKModel(
        H, MU, c_ops_raw=((L_RAD, GAMMA_1), (SIGMA_Z, GAMMA_PHI / 2)),
        observable_op_arrays={"polarization": MU, "excited_population": P_E}))
    return solver


def three_site_chain(dissipative=True):
    """Hard-core exciton chain, manifolds 0-2, local dephasing and collective decay."""
    n, J = 3, -0.05
    pairs = [(0, 1), (0, 2), (1, 2)]
    eps = 2.0 + 0.01 * np.arange(n)
    H1 = np.diag(eps).astype(complex) + J * (np.eye(n, k=1) + np.eye(n, k=-1))
    H2 = np.diag([eps[i] + eps[j] for i, j in pairs]).astype(complex)
    H2[0, 1] = H2[1, 0] = J          # (0,1) <-> (0,2)
    H2[1, 2] = H2[2, 1] = J          # (0,2) <-> (1,2)
    up10 = np.ones((n, 1), dtype=complex)
    up21 = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=complex)
    number = [{("1", "1"): np.diag(np.eye(n)[s]).astype(complex),
               ("2", "2"): np.diag([1.0 if s in p else 0.0 for p in pairs]).astype(complex)}
              for s in range(n)]
    lowering = {("0", "1"): up10.conj().T, ("1", "2"): up21.conj().T}
    c_ops = tuple((blocks, 0.02) for blocks in number) + ((lowering, 0.005),)
    return ExcitationSectorModel({"0": np.zeros((1, 1), dtype=complex), "1": H1, "2": H2},
                                 {("1", "0"): up10, ("2", "1"): up21},
                                 c_ops_raw=c_ops if dissipative else (),
                                 initial_sector="0")
