"""MPO-based adaptive two-site TDVP backend for TDVP-TEMPO."""

from typing import Optional, Tuple

import numpy as np
import scipy.linalg

from oqupy.backends.tdvp_environments import TdvpEnvironments
from oqupy.backends.tdvp_krylov import evolve_one_site, evolve_two_site
from oqupy.backends.tdvp_mpo_builder import build_memory_mpo
from oqupy.dense_pt_tempo import ContinuousInfluenceTensor
from oqupy.dynamics import Dynamics
from oqupy.mps_cross_correlated import CrossCorrelatedMPS


def _split(theta: np.ndarray, epsrel: float,
           max_bond_dimension: Optional[int]):
    dl, physical, _, dr = theta.shape
    matrix = theta.transpose(0, 1, 2, 3).reshape(
        dl * physical, physical * dr)
    u, singular, vh = scipy.linalg.svd(matrix, full_matrices=False)
    keep = max(1, int(np.count_nonzero(
        singular > epsrel * singular[0]))) if epsrel > 0 else len(singular)
    if max_bond_dimension is not None:
        keep = min(keep, max_bond_dimension)
    u, singular, vh = u[:, :keep], singular[:keep], vh[:keep]
    return (
        u.reshape(dl, physical, keep),
        np.diag(singular),
        vh.reshape(keep, physical, dr),
    )


class TdvpTempoBackend:
    """Execute adaptive two-site TDVP using a finite-memory MPO."""

    def __init__(self, system, bath, parameters, initial_state: np.ndarray,
                 start_time: float = 0.0) -> None:
        self.system = system
        self.bath = bath
        self.parameters = parameters
        self.initial_state = np.asarray(initial_state, dtype=complex)
        self.start_time = float(start_time)
        self._dynamics: Optional[Dynamics] = None

    def compute(self, end_time: float) -> Dynamics:
        """Compute through a point on the TDVP time grid."""
        if end_time < self.start_time:
            raise ValueError("end_time must not precede start_time")
        steps = int(round(
            (end_time - self.start_time) / self.parameters.dt))
        if not np.isclose(
                self.start_time + steps * self.parameters.dt, end_time):
            raise ValueError("end_time must lie on the TDVP time grid")
        if self._dynamics is None or len(self._dynamics) - 1 < steps:
            self._dynamics = self._compute(steps)
        return Dynamics(
            times=list(self._dynamics.times[:steps + 1]),
            states=list(self._dynamics.states[:steps + 1]))

    def _compute(self, steps: int) -> Dynamics:
        if self.parameters.tdvp_method != "2-site":
            raise NotImplementedError(
                "Only 2-site TDVP is implemented for MPO memory.")
        dimension = self.bath.dimension
        physical = dimension**2
        propagators = self.system.get_propagators(
            self.parameters.dt, self.start_time, None, None)
        influence = ContinuousInfluenceTensor(self.bath, self.parameters)
        unitary = (self.bath.unitary_transform
                   if self.bath.commuting_channels
                   else np.eye(dimension, dtype=complex))
        liouville_transform = np.kron(
            unitary.conjugate().T, unitary.T)
        liouville_inverse = liouville_transform.conjugate().T
        initial_state = liouville_transform @ self.initial_state.reshape(-1)
        mps = CrossCorrelatedMPS(
            physical, self.parameters.epsrel,
            self.parameters.max_bond_dimension)
        states = [self.initial_state.copy()]
        previous_second_half = None

        for step in range(steps):
            first_half, second_half = propagators(step)
            first_half = (
                liouville_transform @ first_half @ liouville_inverse)
            second_half = (
                liouville_transform @ second_half @ liouville_inverse)
            if step == 0:
                first_midpoint = first_half @ initial_state
                mps.tensors = [first_midpoint.reshape(1, physical, 1)]
                mps.apply_one_site(0, influence.gate(0))
            else:
                mps.append_propagated_site(
                    first_half @ previous_second_half)
                length = len(mps.tensors)
                mps.apply_one_site(length - 1, influence.gate(0))
                generators = {
                    distance: -influence.generator(distance) /
                    self.parameters.dt
                    for distance in range(
                        1, min(step, self.parameters.dkmax) + 1)
                }
                mpo = build_memory_mpo(
                    generators, length, physical,
                    self.parameters.max_bond_dimension,
                    self.parameters.sweep_tolerance,
                    active_site=length - 1)
                environments = TdvpEnvironments(mps.tensors, mpo)
                environments.build_left()
                environments.build_right()
                for index in range(length - 1):
                    theta = np.tensordot(
                        mps.tensors[index], mps.tensors[index + 1],
                        axes=(2, 0))
                    theta = evolve_two_site(
                        theta, self.parameters.dt / 2,
                        environments.left[index],
                        environments.right[index + 2],
                        mpo[index], mpo[index + 1])
                    left, center, right = _split(
                        theta, self.parameters.epsrel,
                        self.parameters.max_bond_dimension)
                    mps.tensors[index] = left
                    mps.tensors[index + 1] = np.einsum(
                        "ab,bpr->apr", center, right, optimize=True)
                    environments.update_left(index)
                    if index < length - 2:
                        mps.tensors[index + 1] = evolve_one_site(
                            mps.tensors[index + 1],
                            -self.parameters.dt / 2,
                            environments.left[index + 1],
                            environments.right[index + 2],
                            mpo[index + 1])

                for index in range(length - 2, -1, -1):
                    theta = np.tensordot(
                        mps.tensors[index], mps.tensors[index + 1],
                        axes=(2, 0))
                    theta = evolve_two_site(
                        theta, self.parameters.dt / 2,
                        environments.left[index],
                        environments.right[index + 2],
                        mpo[index], mpo[index + 1])
                    left, center, right = _split(
                        theta, self.parameters.epsrel,
                        self.parameters.max_bond_dimension)
                    mps.tensors[index + 1] = right
                    mps.tensors[index] = np.einsum(
                        "apr,rb->apb", left, center, optimize=True)
                    environments.update_right(index + 1)
                    if index > 0:
                        mps.tensors[index] = evolve_one_site(
                            mps.tensors[index],
                            -self.parameters.dt / 2,
                            environments.left[index],
                            environments.right[index + 1],
                            mpo[index])

            state = liouville_inverse @ (
                second_half @ mps.state_at_last_site())
            states.append(state.reshape(dimension, dimension))
            previous_second_half = second_half

        times = self.start_time + np.arange(len(states)) * self.parameters.dt
        return Dynamics(times=list(times), states=states)

    def initialize(self) -> Tuple[int, np.ndarray]:
        """Return the initial step and state."""
        return 0, self.initial_state.copy()

    def compute_step(self) -> Tuple[int, np.ndarray]:
        """Compute and return one additional time step."""
        current = 0 if self._dynamics is None else len(self._dynamics) - 1
        dynamics = self.compute(
            self.start_time + (current + 1) * self.parameters.dt)
        return current + 1, dynamics.states[-1]
