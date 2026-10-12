import numpy as np
from qudpy_fdgf import EigenbasisKModel, FrequencyPathway, SpectroscopySolver
from qudpy_fdgf.protocols import (PropagationInterval, SpectroscopyProtocol,
                                  standard_nq_protocol)

# 1. Matrices (hbar = 1, eV), basis ordering (|e>, |g>)
sigma_z = np.diag([1.0, -1.0]).astype(complex)
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_m = np.array([[0, 0], [1, 0]], dtype=complex)   # |g><e|
H, mu = 1.0 * sigma_z, 0.5 * sigma_x
gamma_1, gamma_phi = 0.010, 0.005                     # 1/T1, 1/T2* (eV)

# 2. Model contract: (operator, rate) pairs, rate kept out of the operator
model = EigenbasisKModel(H, mu, c_ops_raw=(
    (sigma_m, gamma_1),          # population decay at gamma_1
    (sigma_z, gamma_phi / 2),    # sigma_z dephases coherences at 2x its rate
))

# 3. Solver
solver = SpectroscopySolver(backend="dense", eta=1e-4)
solver.feed_model(model)

# 4-5. Linear response: one ket interaction, one frequency interval
linear = FrequencyPathway("linear", interactions=[{"label": "Ku"}],
                          component="linear")
protocol = SpectroscopyProtocol(
    intervals=[PropagationInterval("omega1", "frequency")])
w = np.linspace(1.9, 2.1, 4001)
res = solver.generate_spectrum(protocol, {"omega1": w}, pathways=[linear])
A = res.pathways["linear"].imag      # absorptive part
half = w[A >= A.max() / 2]
Gamma_2 = gamma_1 / 2 + gamma_phi
print(f"linear peak      : {w[np.argmax(A)]:.4f} eV")
print(f"FWHM (numerical) : {half[-1] - half[0]:.4f} eV")
print(f"2(Gamma_2 + eta) : {2 * (Gamma_2 + solver.eta):.4f} eV")

# Third-order rephasing (R1 + R2) at t2 = 0: omega_1 and omega_3 resolved
R1 = FrequencyPathway("R1", interactions=[
    {"label": "Bu", "pulse_index": 0}, {"label": "Ku", "pulse_index": 1},
    {"label": "Bd", "pulse_index": 2}], component="rephasing")
R2 = FrequencyPathway("R2", interactions=[
    {"label": "Bu", "pulse_index": 0}, {"label": "Bd", "pulse_index": 1},
    {"label": "Ku", "pulse_index": 2}], component="rephasing")
protocol_2d = standard_nq_protocol(order=1, nq_interval=1, detection_interval=3,
                                   n_interactions=3, nq_axis="omega1q")
axes = {"omega1q": np.linspace(-2.2, -1.8, 81),
        "omega3": np.linspace(1.8, 2.2, 81)}
res2 = solver.generate_spectrum(protocol_2d, axes, pathways=[R1, R2],
                                fixed_coordinates={"t2": 0.0})
S2 = res2.components["rephasing"]
i, j = np.unravel_index(np.argmax(np.abs(S2)), S2.shape)
print(f"rephasing peak   : (omega1q, omega3) = "
      f"({res2.axis_values[0][i]:.3f}, {res2.axis_values[1][j]:.3f}) eV")
print(f"|S| at peak      : {np.abs(S2[i, j]):.2f}")
print(f"2 mu^4/(Gamma_2 + eta)^2 : {2 * 0.5**4 / (Gamma_2 + solver.eta)**2:.2f}")
