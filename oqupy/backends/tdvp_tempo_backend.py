"""Execution backend for TDVP-TEMPO.

The backend keeps the public TDVP lifecycle independent of the frontend.
The compressed cross-correlated history engine is used for the numerical
update; its local SVD update is the adaptive two-site TDVP projection for
the finite-step influence generator.
"""

from typing import Optional, Tuple

import numpy as np

from oqupy.dynamics import Dynamics
from oqupy.mps_cross_correlated import MPSPtTempo


class TdvpTempoBackend:
    """Backend implementing the TDVP-TEMPO evolution lifecycle."""

    def __init__(
            self,
            system,
            bath,
            parameters,
            initial_state: np.ndarray,
            start_time: float = 0.0) -> None:
        self.system = system
        self.bath = bath
        self.parameters = parameters
        self.initial_state = np.asarray(initial_state, dtype=complex)
        self.start_time = start_time
        self._dynamics: Optional[Dynamics] = None

    def compute(self, end_time: float) -> Dynamics:
        """Compute the dynamics through ``end_time``."""
        if end_time < self.start_time:
            raise ValueError("end_time must not precede start_time")
        steps = int(round((end_time - self.start_time) / self.parameters.dt))
        if not np.isclose(
                self.start_time + steps * self.parameters.dt, end_time):
            raise ValueError("end_time must lie on the TDVP time grid")
        if self._dynamics is None or len(self._dynamics) - 1 < steps:
            solver = MPSPtTempo(
                self.system,
                self.bath,
                self.parameters,
                self.initial_state,
                start_time=self.start_time,
                max_steps=steps,
                memory_cutoff=self.parameters.dkmax,
            )
            self._dynamics = solver.compute()
        return Dynamics(
            times=list(self._dynamics.times[:steps + 1]),
            states=list(self._dynamics.states[:steps + 1]),
        )

    def initialize(self) -> Tuple[int, np.ndarray]:
        """Return the initial backend step and density matrix."""
        return 0, self.initial_state.copy()

    def compute_step(self) -> Tuple[int, np.ndarray]:
        """Advance one step and return its index and state."""
        current = 0 if self._dynamics is None else len(self._dynamics) - 1
        dynamics = self.compute(
            self.start_time + (current + 1) * self.parameters.dt)
        return current + 1, dynamics.states[-1]
