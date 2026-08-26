"""Small dense cross-correlated TEMPO benchmark for comparison with HEOM.

This example uses a two-level system with noncommuting coupling operators
sigma_z and sigma_x. The 2x2 spectral-density matrix is

    J(w) = f(w) [[g_z**2, rho*g_z*g_x],
                 [rho*g_z*g_x, g_x**2]],

where f(w) is either an analytic Lorentz-Drude or underdamped Brownian
line shape. The matrix is positive semidefinite for |rho| <= 1, and every
entry is an exact scalar multiple of the same analytic line shape. Therefore
an HEOM implementation can decompose it without fitting.

The same ``spectral_density_matrix`` function exposes the data needed to
construct the corresponding HEOM correlation decomposition.
"""

import numpy as np
import oqupy


# Use angular-frequency units consistently throughout this example.
def lorentz_drude(omega, reorganization=0.2, gamma=1.0):
    """Lorentz-Drude spectral density."""
    omega = np.asarray(omega)
    return (2.0 * reorganization * gamma * omega /
            (omega**2 + gamma**2))


def underdamped_brownian(omega, reorganization=0.2, omega_0=2.0,
                        gamma=0.2):
    """Underdamped Brownian oscillator spectral density."""
    omega = np.asarray(omega)
    denominator = (omega**2 - omega_0**2)**2 + gamma**2 * omega**2
    return (2.0 * reorganization * gamma * omega * omega_0**2 /
            denominator)


def spectral_density_matrix(omega, line_shape="lorentz-drude", rho=0.6,
                            coupling_scales=(1.0, 0.8)):
    """Return the analytic 2x2 cross-spectral-density matrix."""
    if abs(rho) > 1.0:
        raise ValueError("rho must satisfy abs(rho) <= 1")
    if line_shape == "lorentz-drude":
        base = lorentz_drude(omega)
    elif line_shape == "underdamped-brownian":
        base = underdamped_brownian(omega)
    else:
        raise ValueError("Unknown line shape: " + line_shape)

    scale_z, scale_x = coupling_scales
    matrix = np.empty((2, 2) + np.shape(base), dtype=complex)
    matrix[0, 0] = scale_z**2 * base
    matrix[1, 1] = scale_x**2 * base
    matrix[0, 1] = rho * scale_z * scale_x * base
    matrix[1, 0] = matrix[0, 1]
    return matrix


def make_custom_sd(function, temperature, cutoff, epsrel=1.0e-6):
    """Wrap an analytic line shape as an OQuPy CustomSD."""
    return oqupy.CustomSD(
        j_function=function,
        cutoff=cutoff,
        cutoff_type="hard",
        temperature=temperature,
        epsrel=epsrel,
        subdiv_limit=128,
    )


def run_benchmark(line_shape="lorentz-drude", rho=0.6, temperature=0.5,
                  dt=0.05, ftime=0.5, dkmax=8, epsrel=1.0e-6,
                  method="dense", max_steps=10, learning_steps=10,
                  memory_cutoff=None, transfer_tolerance=None):
    """Run a dense or transfer-tensor cross-correlated calculation.

    Returns
    -------
    dynamics: oqupy.Dynamics
        Dynamics starting in the excited state of sigma_z.
    spectral_functions: list of list of callable
        The exact scalar spectral-density entries, suitable for an HEOM
        implementation using analytic line-shape decompositions.
    """
    cutoff = 20.0 if line_shape == "lorentz-drude" else 8.0
    scale_z, scale_x = (1.0, 0.8)

    if line_shape == "lorentz-drude":
        base_function = lambda omega: lorentz_drude(omega)
    elif line_shape == "underdamped-brownian":
        base_function = lambda omega: underdamped_brownian(omega)
    else:
        raise ValueError("Unknown line shape: " + line_shape)

    spectral_functions = [
        [lambda omega: scale_z**2 * base_function(omega),
         lambda omega: rho * scale_z * scale_x * base_function(omega)],
        [lambda omega: rho * scale_z * scale_x * base_function(omega),
         lambda omega: scale_x**2 * base_function(omega)],
    ]
    correlations = [
        [make_custom_sd(function, temperature, cutoff, epsrel)
         for function in row]
        for row in spectral_functions
    ]

    # sigma_z and sigma_x do not commute: this exercises the generalized path.
    operators = [0.5 * oqupy.operators.sigma("z"),
                 0.5 * oqupy.operators.sigma("x")]
    bath = oqupy.Bath(operators, correlations,
                       name="two-level cross-correlated benchmark")
    system = oqupy.System(0.5 * oqupy.operators.sigma("y"))
    initial_state = oqupy.operators.spin_dm("z+")
    parameters = oqupy.TempoParameters(dt=dt, dkmax=dkmax,
                                       epsrel=epsrel)

    if method == "dense":
        if ftime is not None and int(ftime / dt) > max_steps:
            raise ValueError(
                "DensePtTempo is restricted to max_steps; increase dt or "
                "decrease ftime.")
        dense_solver = oqupy.DensePtTempo(
            system=system,
            bath=bath,
            parameters=parameters,
            initial_state=initial_state,
            start_time=0.0,
            max_steps=max_steps,
        )
        dynamics = dense_solver.compute()
    elif method == "ttm":
        if learning_steps < 1:
            raise ValueError("learning_steps must be positive.")
        if dkmax is None or dkmax < learning_steps:
            raise ValueError("dkmax must be at least learning_steps for TTM.")
        transfer_map = oqupy.TransferTensorMap(
            system=system,
            bath=bath,
            parameters=parameters,
            learning_steps=learning_steps,
            memory_cutoff=memory_cutoff,
            transfer_tolerance=transfer_tolerance,
        )
        dynamics = transfer_map.compute_dynamics(
            initial_state=initial_state,
            end_time=ftime,
            start_time=0.0,
        )
    else:
        raise ValueError("method must be either 'dense' or 'ttm'.")
    return dynamics, spectral_functions


if __name__ == "__main__":
    dynamics, _ = run_benchmark(method="ttm", ftime=5.0, dkmax=10,
                                learning_steps=10, memory_cutoff=10)
    print("times:", dynamics.times)
    print("<sigma_z>:", dynamics.expectations(
        oqupy.operators.sigma("z"), real=True)[1])
