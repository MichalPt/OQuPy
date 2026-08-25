"""Minimal PT-TEMPO example with two cross-correlated channels.

The channel operators must commute so that they can be represented in the
shared influence-functional path basis used by this implementation.
"""

import numpy as np
import oqupy


class ExponentialCorrelation(oqupy.CustomCorrelations):
    """A small complex correlation model for a smoke test."""

    def __init__(self, amplitude):
        super().__init__(
            lambda tau: amplitude * np.exp(-(1.0 + 0.25j) * tau))


system = oqupy.System(np.diag([0.0, 0.4, 0.9]))
operators = [
    np.diag([-1.0, 0.0, 1.0]),
    np.diag([1.0, -1.0, 0.0]),
]
auto = ExponentialCorrelation(0.08)
cross = ExponentialCorrelation(0.02 + 0.01j)
bath = oqupy.Bath(
    operators,
    [[auto, cross], [cross, auto]],
    name="two-channel cross-correlated bath",
)

parameters = oqupy.TempoParameters(dt=0.05, dkmax=8, epsrel=1.0e-6)
process_tensor = oqupy.pt_tempo_compute(
    bath, start_time=0.0, end_time=1.0, parameters=parameters)
dynamics = oqupy.compute_dynamics(
    system=system,
    process_tensor=process_tensor,
    initial_state=np.diag([1.0, 0.0, 0.0]),
)

print("times:", dynamics.times)
print("populations at final time:", np.real(np.diag(dynamics.states[-1])))
