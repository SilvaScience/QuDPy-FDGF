"""Third-order rephasing signal of a closed system driven by Gaussian pulses of finite duration.

A time-domain calculation, independent of the resolvent formalism of QuDPy-FDGF, used to measure the error of its
impulsive limit and of the rotating-wave approximation (validation/analysis/example1_pulses.ipynb).

The system is given in the eigenbasis of H0: the energies, the coupling operator ``mu`` (Hermitian), its raising part
``mu_plus`` (the operator that raises the excitation number; the lowering part is its adjoint) and the stationary state
``rho0``. The field is E(t) = sum_j E_j(t) with three pulses centred at 0, t1 and t1 + t2,

    E_j(t) = g(t - t_j) cos(omega_c (t - t_j) + phi_j),      g(t) = exp(-t^2 / (2 sigma^2)),

written as the sum of a positive-frequency component exp(-i omega_c t) and its complex conjugate. The rephasing signal is
the part of the third-order polarization with the phase dependence exp[i(phi_1 - phi_2 - phi_3)]: pulse 1 enters with the
negative-frequency component, pulses 2 and 3 with the positive-frequency one. Perturbation theory is exact for this part,
and it is computed as the hierarchy of the multilinear terms rho_S, S a subset of the three pulses,

    d rho_S / dt = i sum_{j in S} e_j(t) [mu_j(t), rho_{S \\ j}],     rho_{empty} = rho0,

in the interaction picture of H0 (e_j: the component of pulse j, mu_j the operator it couples to). With ``rwa=True`` the
negative-frequency component couples to the lowering part and the positive-frequency one to the raising part of mu
(rotating-wave approximation); with ``rwa=False`` both couple to the full mu (real field, counter-rotating terms included).
Overlapping pulses are handled exactly: the pulses are integrated together, whatever the delays.

The signal is the polarization Tr(mu rho_123(t)) transformed with the kernel exp[(i omega - eta) t] of the resolvent, from
t3 = 0 (the centre of the third pulse) to infinity: the part during the pulses is integrated numerically, the free
evolution after them analytically. The delays t1 >= 0 are sampled with a fine step near 0 and a step ``dt1`` afterwards,
and integrated with the trapezoid rule. Within about one pulse width of t1 = 0 the pulses overlap and the signal contains the
orderings of the three interactions that are not rephasing sequences; their weight vanishes with the pulse duration.
"""
import numpy as np


def _components(rwa, mu, mu_plus):
    """Operators coupled to the negative-frequency component (pulse 1) and the positive-frequency one (pulses 2, 3)."""
    return (mu_plus.conj().T, mu_plus) if rwa else (mu, mu)


class Hierarchy:
    """Integrator of the multilinear hierarchy for three pulses with a Gaussian envelope."""

    def __init__(self, energies, mu, mu_plus, rho0, sigma, carrier, rwa, dt=0.02):
        self.energies = np.asarray(energies, dtype=float)
        self.mu = np.asarray(mu, dtype=complex)
        self.rho0 = np.asarray(rho0, dtype=complex)
        self.sigma, self.carrier, self.dt = float(sigma), float(carrier), float(dt)
        self.half_window = np.ceil(6.0 * sigma / dt) * dt              # the envelope is below 1.5e-8 beyond 6 sigma
        self.operator_1, self.operator_23 = _components(rwa, self.mu, np.asarray(mu_plus, dtype=complex))
        self.rwa = rwa

    # --------------------------------------------------------------------------------- fields and operators
    def fields(self, t, centres):
        """Components e_j(t) of the three pulses: (negative-frequency for 1, positive-frequency for 2 and 3)."""
        envelope = np.exp(-0.5 * ((t - np.asarray(centres)) / self.sigma) ** 2)
        phase = self.carrier * (t - np.asarray(centres))
        return envelope * np.array([np.exp(1j * phase[0]), np.exp(-1j * phase[1]), np.exp(-1j * phase[2])])

    def interaction_picture(self, operator, t):
        """exp(i H0 t) operator exp(-i H0 t) in the eigenbasis."""
        left = np.exp(1j * self.energies * t)
        return left[:, None] * operator * left.conj()[None, :]

    # --------------------------------------------------------------------------------- derivative
    def derivative(self, t, state, centres, subsets):
        """d rho_S / dt for the subsets S (bit masks); state[0] is rho0 and never changes."""
        e = self.fields(t, centres)
        active = np.abs(e) > 1e-13
        result = np.zeros_like(state)
        if not active.any():
            return result
        operators = {}
        for mask in subsets:
            for j in range(3):
                bit = 1 << j
                if mask & bit and active[j]:
                    operator = self.operator_1 if j == 0 else self.operator_23
                    key = id(operator)
                    if key not in operators:
                        operators[key] = self.interaction_picture(operator, t)
                    other = state[mask ^ bit]
                    a = operators[key]
                    result[mask] += 1j * e[j] * (a @ other - other @ a)
        return result

    def step(self, t, state, centres, subsets):
        dt = self.dt
        k1 = self.derivative(t, state, centres, subsets)
        k2 = self.derivative(t + dt / 2, state + dt / 2 * k1, centres, subsets)
        k3 = self.derivative(t + dt / 2, state + dt / 2 * k2, centres, subsets)
        k4 = self.derivative(t + dt, state + dt * k3, centres, subsets)
        return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    # --------------------------------------------------------------------------------- first order from pulse 1
    def first_order(self):
        """rho_1(t) of the first pulse alone on the grid -L ... +L (interaction picture), and its final value."""
        n = int(round(2 * self.half_window / self.dt))
        times = -self.half_window + self.dt * np.arange(n + 1)
        state = np.zeros((8, *self.rho0.shape), dtype=complex)
        state[0] = self.rho0
        trajectory = np.empty((n + 1, *self.rho0.shape), dtype=complex)
        trajectory[0] = state[1]
        for k in range(n):
            state = self.step(times[k], state, (0.0, 1e9, 1e9), (1,))
            trajectory[k + 1] = state[1]
        return times, trajectory

    # --------------------------------------------------------------------------------- one delay
    def polarization(self, t1, t2, times_1, trajectory_1):
        """Polarization of rho_123 on the integration grid for the delay t1 (pulses 2 and 3 at t1 and t1 + t2).

        Returns (times, P(t), rho_123 at the end in the interaction picture).
        """
        dt, L = self.dt, self.half_window
        centres = (0.0, t1, t1 + t2)
        end = t1 + t2 + L
        state = np.zeros((8, *self.rho0.shape), dtype=complex)
        state[0] = self.rho0
        start = max(t1 - L, -L)
        index = min(int(round((start - times_1[0]) / dt)), trajectory_1.shape[0] - 1)
        state[1] = trajectory_1[index]                                  # pulse 1 alone: free evolution afterwards
        n = int(round((end - start) / dt))
        times = start + dt * np.arange(n + 1)
        values = np.empty(n + 1, dtype=complex)
        subsets = (1, 2, 3, 4, 5, 6, 7)

        def measure(time, rho):
            left = np.exp(1j * self.energies * time)
            return np.sum(self.mu.T * rho * (left.conj()[:, None] * left[None, :]))

        values[0] = measure(times[0], state[7])
        for k in range(n):
            state = self.step(times[k], state, centres, subsets)
            values[k + 1] = measure(times[k + 1], state[7])
        return times, values, state[7], times[-1]


def default_delays(dt1=0.5, t1_max=90.0, fine_step=0.05, fine_until=1.0):
    """Delays of the transform: a fine grid near t1 = 0 and a step ``dt1`` afterwards.

    Near t1 = 0 the pulses overlap and the signal changes over a pulse duration; the fine step keeps the trapezoid rule
    from weighting the first sample, whose value is not that of the delta-pulse response, by a whole ``dt1``.
    """
    fine = np.arange(0.0, fine_until, fine_step)
    coarse = np.arange(round(fine_until / dt1) * dt1 if fine_until % dt1 == 0 else np.ceil(fine_until / dt1) * dt1,
                       t1_max + dt1, dt1)
    return np.concatenate([fine, coarse])


def rephasing_map(energies, mu, mu_plus, rho0, sigma, carrier, rwa, omega1, omega3, eta, t2=0.0, dt=0.02, dt1=0.5,
                  t1_max=None, delays=None):
    """Third-order rephasing signal S(omega1, omega3) on the grid omega1 x omega3 (complex array).

    ``delays`` are the delays t1 >= 0 (default: ``default_delays(dt1, t1_max)`` with t1_max = 9 / eta).
    """
    hierarchy = Hierarchy(energies, mu, mu_plus, rho0, sigma, carrier, rwa, dt)
    omega1, omega3 = np.asarray(omega1, dtype=float), np.asarray(omega3, dtype=float)
    mu = hierarchy.mu
    energies = hierarchy.energies
    omega_ab = energies[:, None] - energies[None, :]
    if delays is None:
        t1_max = 9.0 / eta if t1_max is None else t1_max
        delays = default_delays(dt1, t1_max)
    delays = np.asarray(delays, dtype=float)
    weights_1 = np.gradient(delays)                                  # trapezoid rule on the (non-uniform) delays
    weights_1[0], weights_1[-1] = (delays[1] - delays[0]) / 2, (delays[-1] - delays[-2]) / 2
    weights_1[1:-1] = (delays[2:] - delays[:-2]) / 2
    times_1, trajectory_1 = hierarchy.first_order()
    signal = np.zeros((omega1.size, omega3.size), dtype=complex)
    for t1, weight1 in zip(delays, weights_1):
        times, values, final, t_end = hierarchy.polarization(t1, t2, times_1, trajectory_1)
        # during the pulses: trapezoid from the centre of the third pulse (t3 = 0) to the end of the integration
        centre3 = t1 + t2
        selected = times >= centre3 - 1e-12
        t = times[selected]
        weights = np.full(t.size, hierarchy.dt)
        weights[0] = weights[-1] = hierarchy.dt / 2
        kernel = np.exp((1j * omega3[None, :] - eta) * (t[:, None] - centre3))
        inside = (weights * values[selected]) @ kernel
        # after the pulses: free evolution, integrated analytically
        left = np.exp(-1j * energies * t_end)
        rho_end = final * (left[:, None] * left.conj()[None, :])           # Schrodinger picture at t_end
        amplitude = (mu.T * rho_end).ravel()
        poles = 1.0 / (eta - 1j * (omega3[:, None] - omega_ab.ravel()[None, :]))
        tail = np.exp((1j * omega3 - eta) * (t_end - centre3)) * (poles @ amplitude)
        s3 = inside + tail
        signal += weight1 * np.exp((1j * omega1[:, None] - eta) * t1) * s3[None, :]
    return signal


def scale_and_error(signal, reference):
    """Complex scale c minimizing |signal - c reference| and the relative residual |signal - c reference| / |signal|."""
    c = np.vdot(reference, signal) / np.vdot(reference, reference)
    return c, np.linalg.norm(signal - c * reference) / np.linalg.norm(signal)


def chain_system(max_magnons=4, temperature=4.0):
    """Eigenbasis arrays (energies, mu, mu_plus, rho0) of the XXZ ring of Example 1 truncated at ``max_magnons``.

    ``mu_plus`` is the uniform magnon-raising operator, ``mu`` = mu_plus + mu_plus^dagger and ``rho0`` the canonical state
    on the retained sectors, as for the ExcitationSectorModel of models.example1_model.
    """
    import models

    H, N_op, M = models.example1_chain()
    keep = np.flatnonzero(np.rint(N_op.diag().real).astype(int) <= max_magnons)
    H_kept = H.full()[np.ix_(keep, keep)]
    M_kept = M.full()[np.ix_(keep, keep)]
    energies, vectors = np.linalg.eigh(H_kept)
    mu_plus = vectors.conj().T @ M_kept @ vectors
    weights = np.exp(-energies / (models.BOLTZMANN_MEV_PER_K * temperature))
    return energies, mu_plus + mu_plus.conj().T, mu_plus, np.diag(weights / weights.sum()).astype(complex)

def phase_cycled_polarization(energies, mu, mu_plus, rho0, sigma, carrier, rwa, t1, t2=0.0, amplitude=0.02, cycles=4, dt=0.01):
    """Third-order rephasing polarization by direct propagation with phase cycling, to check ``Hierarchy.polarization``.

    The density matrix is propagated non-perturbatively (RK4, interaction picture) under the real field
    E(t) = 2 a sum_j g(t - t_j) cos(omega_c (t - t_j) + phi_j) (``rwa=False``) or its rotating-wave part (``rwa=True``),
    for ``cycles`` phases of each pulse, and the component exp[i(phi_1 - phi_2 - phi_3)] is projected out. The orders 5 and
    higher are removed by Richardson extrapolation from the amplitudes a and 2a: P3 a^3 = (32 P(a) - P(2a)) / 24.
    Returns (times, P(t)) from the centre of the third pulse to the end of the propagation, on the grid of ``dt``.
    """
    energies = np.asarray(energies, dtype=float)
    mu, mu_plus, rho0 = (np.asarray(x, dtype=complex) for x in (mu, mu_plus, rho0))
    mu_minus = mu_plus.conj().T
    half = np.ceil(6.0 * sigma / dt) * dt
    centres = np.array([0.0, t1, t1 + t2])
    start, end = -half, t1 + t2 + half
    n = int(round((end - start) / dt))
    times = start + dt * np.arange(n + 1)

    def lab(operator, t):
        left = np.exp(1j * energies * t)
        return left[:, None] * operator * left.conj()[None, :]

    def propagate(phases, scale):
        def hamiltonian_term(t):
            """Interaction-picture operator multiplying the field: returns the (Hermitian) coupling matrix V(t) = -H_int."""
            envelope = np.exp(-0.5 * ((t - centres) / sigma) ** 2)
            theta = carrier * (t - centres)
            if rwa:
                positive = np.sum(envelope * np.exp(-1j * (theta + phases)))
                negative = np.conj(positive)
                return scale * (positive * lab(mu_plus, t) + negative * lab(mu_minus, t))
            field = 2 * np.sum(envelope * np.cos(theta + phases))
            return scale * field * lab(mu, t)

        rho = rho0.copy()
        values = np.empty(n + 1, dtype=complex)

        def measure(t, state):
            left = np.exp(1j * energies * t)
            return np.sum(mu.T * state * (left.conj()[:, None] * left[None, :]))

        def derivative(t, state):
            v = hamiltonian_term(t)
            return 1j * (v @ state - state @ v)

        values[0] = measure(times[0], rho)
        for k in range(n):
            t = times[k]
            k1 = derivative(t, rho)
            k2 = derivative(t + dt / 2, rho + dt / 2 * k1)
            k3 = derivative(t + dt / 2, rho + dt / 2 * k2)
            k4 = derivative(t + dt, rho + dt * k3)
            rho = rho + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            values[k + 1] = measure(times[k + 1], rho)
        return values

    grid = 2 * np.pi * np.arange(cycles) / cycles
    results = []
    for scale in (amplitude, 2 * amplitude):
        total = np.zeros(n + 1, dtype=complex)
        for p1 in grid:
            for p2 in grid:
                for p3 in grid:
                    total += propagate(np.array([p1, p2, p3]), scale) * np.exp(-1j * (p1 - p2 - p3))
        results.append(total / cycles**3)
    third = (32 * results[0] - results[1]) / 24 / amplitude**3
    selected = times >= centres[2] - 1e-12
    return times[selected], third[selected]
