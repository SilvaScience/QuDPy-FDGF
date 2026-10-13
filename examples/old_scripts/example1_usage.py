import numpy as np
from qudpy_fdgf import (EigenbasisKModel, FrequencyPathway,
                        ObservableSpec, SpectroscopySolver)
from qudpy_fdgf.protocols import standard_nq_protocol

# 1. Model: matrices in the {|g>, |e>} basis (hbar = 1, eV)
H = np.diag([0.0, 2.0]).astype(complex)
mu = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
P_e = np.diag([0.0, 1.0]).astype(complex)
L_rad = np.array([[0, 1], [0, 0]], dtype=complex)   # |g><e|
sigma_z = np.diag([-1.0, 1.0]).astype(complex)
gamma_1, gamma_phi = 0.04, 0.03
model = EigenbasisKModel(
    H, mu,
    c_ops_raw=((L_rad, gamma_1),        # (operator, rate) pairs
               (sigma_z, gamma_phi / 2)),
    observable_op_arrays={"polarization": mu,
                          "excited_population": P_e})

# 2. Solver: backend and resolvent regularization eta
solver = SpectroscopySolver(backend="dense", eta=0.002)
solver.feed_model(model)             # default: ground state

# 3. Pathways: ordered interaction labels
R1 = FrequencyPathway(
    name="R1", interactions=("Bu", "Ku", "Bd"),
    component="rephasing", detection="polarization")
R2 = FrequencyPathway(
    name="R2", interactions=("Bu", "Bd", "Ku"),
    component="rephasing", detection="polarization")

# 4. Protocol: t1, t3 in frequency; t2 in time
protocol = standard_nq_protocol(
    order=1, n_interactions=3, nq_interval=1,
    detection_interval=3, nq_axis="omega_1q",
    detection_axis="omega_emit")
axes = {"omega_1q": np.linspace(-2.5, -1.5, 41),
        "omega_emit": np.linspace(1.5, 2.5, 41)}

# 5. Observables: one propagation, three detection schemes
radiative = solver.jump_channel_names()[0]      # "c_op_0"
observables = {
    "polarization": "polarization",
    "population": ObservableSpec.action(
        "population", fourth_interaction="Bu",
        operator="excited_population"),
    "fluorescence": ObservableSpec.mean_jump(
        "fluorescence", radiative,
        time_window=(0.0, 5 / gamma_1), fourth_interaction="Bu"),
}
result = solver.generate_spectrum(
    protocol, axes, pathways=(R1, R2),
    fixed_coordinates={"t2": 10.0}, observables=observables)

# 6. Results: complex arrays indexed [omega_1q, omega_emit]
S_pol = result.observables["polarization"]["R1"]
S_pop = result.observables["population"]["R1"]
N_fl = result.observables["fluorescence"]["R1"]
i, j = np.unravel_index(np.argmax(abs(S_pol)), S_pol.shape)
print(f"peak = ({axes['omega_1q'][i]:.3f}, "
      f"{axes['omega_emit'][j]:.3f}) eV")
err = np.max(abs(S_pop + 1j * S_pol)) / np.max(abs(S_pol))
print(f"max|S_pop + i S_pol| / max|S_pol| = {err:.1e}")
ratio = (N_fl[i, j] / S_pop[i, j]).real
print(f"N_fl / S_pop = {ratio:.6f}; "
      f"1 - exp(-5) = {1 - np.exp(-5):.6f}")
