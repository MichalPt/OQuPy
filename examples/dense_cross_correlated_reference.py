"""Dense fitting-free reference for a noncommuting cross-correlated bath.

This is intentionally limited to a few time steps. It is a validation engine,
not a production replacement for PT-TEMPO.
"""

import numpy as np
import oqupy


def lorentz_drude(omega, reorganization=0.1, gamma=1.0):
    """Lorentz-Drude spectral density in angular-frequency units."""
    return 2.0 * reorganization * gamma * omega / (omega**2 + gamma**2)


def run_dense_reference(line_shape="lorentz-drude", steps=4):
    if line_shape != "lorentz-drude":
        raise ValueError("Use line_shape='lorentz-drude' in this example.")

    temperature = 0.5
    cutoff = 20.0
    rho = 0.5
    base = lambda omega: lorentz_drude(omega)
    spectral_functions = [
        [lambda omega: base(omega),
         lambda omega: rho * base(omega)],
        [lambda omega: rho * base(omega),
         lambda omega: base(omega)],
    ]
    correlations = [
        [oqupy.CustomSD(function, cutoff=cutoff,
                        cutoff_type="hard", temperature=temperature,
                        epsrel=1e-5, subdiv_limit=128)
         for function in row]
        for row in spectral_functions
    ]

    operators = [0.5 * oqupy.operators.sigma("z"),
                 0.5 * oqupy.operators.sigma("x")]
    bath = oqupy.Bath(operators, correlations)
    parameters = oqupy.TempoParameters(dt=0.1, dkmax=steps, epsrel=1e-6)
    solver = oqupy.DensePtTempo(
        oqupy.System(0.5 * oqupy.operators.sigma("y")),
        bath,
        parameters,
        oqupy.operators.spin_dm("z+"),
        max_steps=steps,
    )
    dynamics = solver.compute()
    traces = np.trace(dynamics.states, axis1=1, axis2=2)
    print("times:", dynamics.times)
    print("traces:", traces)
    print("<sigma_z>:", dynamics.expectations(
        oqupy.operators.sigma("z"), real=True)[1])
    return dynamics


if __name__ == "__main__":
    run_dense_reference()
