"""MPS/SWAP benchmark for noncommuting cross-correlated environments."""

import numpy as np
import oqupy


def lorentz_drude(omega, reorganization=0.2, gamma=5.0):
    """Lorentz-Drude spectral density in angular-frequency units."""
    return (1.0 / np.pi) * 2.0 * reorganization * gamma * omega / \
        (omega**2 + gamma**2)


def underdamped_brownian(omega, reorganization=0.2, omega_0=2.0,
                         gamma=0.2):
    """Underdamped Brownian oscillator spectral density."""
    denominator = (omega**2 - omega_0**2)**2 + gamma**2 * omega**2
    return (1.0 / np.pi) * 2.0 * reorganization * gamma * omega \
        * omega_0**2 / denominator


def run_mps_benchmark(max_steps=20, memory_cutoff=5, progress_type="bar",
                      start_time=0.0, end_time=None, dt=0.1,
                      line_shape="lorentz-drude", rho=0.6,
                      temperature=0.5, initial_state=None,
                      epsrel=1.0e-6):
    if end_time is not None:
        total_steps = (float(end_time) - float(start_time)) / float(dt)
        rounded_steps = round(total_steps)
        if total_steps < 0 or not np.isclose(total_steps, rounded_steps):
            raise ValueError(
                "end_time - start_time must be a non-negative multiple of dt.")
        max_steps = int(rounded_steps)
    if abs(rho) > 1.0:
        raise ValueError("rho must satisfy abs(rho) <= 1")
    if line_shape == "lorentz-drude":
        base_function = lorentz_drude
        cutoff = 1000.0
    elif line_shape == "underdamped-brownian":
        base_function = underdamped_brownian
        cutoff = 8.0
    else:
        raise ValueError("Unknown line shape: " + line_shape)

    scale_z, scale_x = 1.0, 0.8
    spectral_functions = [
        [lambda omega: scale_z**2 * base_function(omega),
         lambda omega: rho * scale_z * scale_x * base_function(omega)],
        [lambda omega: rho * scale_z * scale_x * base_function(omega),
         lambda omega: scale_x**2 * base_function(omega)],
    ]
    correlations = [
        [oqupy.CustomSD(function, cutoff=cutoff, cutoff_type="hard",
                        temperature=temperature, epsrel=epsrel,
                        subdiv_limit=128)
         for function in row]
        for row in spectral_functions
    ]
    operators = [0.5 * oqupy.operators.sigma("z"),
                 0.5 * oqupy.operators.sigma("x")]
    bath = oqupy.Bath(
        operators,
        correlations,
        name="MPS cross-correlated benchmark",
    )
    system = oqupy.System(0.5 * oqupy.operators.sigma("y"))
    parameters = oqupy.TempoParameters(
        dt=dt, dkmax=memory_cutoff, epsrel=epsrel)
    if initial_state is None:
        initial_state = oqupy.operators.spin_dm("z+")
    solver = oqupy.MPSPtTempo(
        system=system,
        bath=bath,
        parameters=parameters,
        initial_state=initial_state,
        start_time=start_time,
        max_steps=max_steps,
        memory_cutoff=memory_cutoff,
    )
    dynamics = solver.compute(progress_type=progress_type)
    traces = np.trace(dynamics.states, axis1=1, axis2=2)
    print("maximum trace error:", np.max(np.abs(traces - 1.0)))
    print("<sigma_z>:", dynamics.expectations(
        oqupy.operators.sigma("z"), real=True)[1])
    return dynamics


if __name__ == "__main__":
    run_mps_benchmark()
