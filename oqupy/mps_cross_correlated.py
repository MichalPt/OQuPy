"""MPS/SWAP reference solver for noncommuting cross-correlated baths."""

from typing import Optional

import numpy as np
import scipy.linalg

from oqupy.bath import Bath
from oqupy.dense_pt_tempo import GeneralizedInfluenceTensor
from oqupy.dynamics import Dynamics
from oqupy.tempo import TempoParameters
from oqupy.util import get_progress


class CrossCorrelatedMPS:
    """MPS with adjacent-gate SVD compression and exact SWAP routing."""

    def __init__(self, physical_dimension: int, epsrel: float):
        self.physical_dimension = physical_dimension
        self.epsrel = epsrel
        self.tensors = []

    def append_propagated_site(self, propagator):
        """Append a site correlated with the current site's physical index."""
        dimension = self.physical_dimension
        if not self.tensors:
            raise ValueError("Cannot append to an empty MPS.")
        last = self.tensors[-1]
        if last.shape[2] != 1:
            raise ValueError("The MPS must have an open right boundary.")
        left_tensor = np.zeros((last.shape[0], dimension, dimension),
                               dtype=complex)
        for physical in range(dimension):
            left_tensor[:, physical, physical] = last[:, physical, 0]
        new_tensor = np.asarray(propagator, dtype=complex).T.reshape(
            dimension, dimension, 1)
        self.tensors[-1] = left_tensor
        self.tensors.append(new_tensor)

    def apply_one_site(self, site: int, operator):
        op_mat = np.asarray(operator)
        tensor = self.tensors[site]
        res = np.tensordot(op_mat, tensor, axes=([1], [1]))
        self.tensors[site] = res.transpose(1, 0, 2)

    def apply_adjacent_gate(self, site: int, gate, direction="right"):
        """Apply an output/input gate to sites ``site`` and ``site+1``."""
        if direction not in ("left", "right"):
            raise ValueError("direction must be 'left' or 'right'")

        left = self.tensors[site]
        right = self.tensors[site + 1]
        left_dimension, physical = left.shape[0], left.shape[1]
        right_physical, right_dimension = right.shape[1], right.shape[2]

        joint = np.tensordot(left, right, axes=([2], [0]))
        joint_mat = joint.transpose(1, 2, 0, 3).reshape(
            physical * right_physical, left_dimension * right_dimension)

        gate_mat = np.asarray(gate).reshape(physical * right_physical,
                                            physical * right_physical)
        joint_gated = np.dot(gate_mat, joint_mat)
        joint_gated = joint_gated.reshape(physical, right_physical,
                                          left_dimension, right_dimension)
        joint_for_svd = joint_gated.transpose(2, 0, 1, 3).reshape(
            left_dimension * physical, right_physical * right_dimension)

        try:
            u, singular_values, vh = scipy.linalg.svd(
                joint_for_svd, full_matrices=False, lapack_driver='gesdd')
        except scipy.linalg.LinAlgError:
            u, singular_values, vh = scipy.linalg.svd(
                joint_for_svd, full_matrices=False, lapack_driver='gesvd')

        if self.epsrel > 0.0 and singular_values.size > 0:
            threshold = self.epsrel * singular_values[0]
            keep = max(1, int(np.count_nonzero(singular_values > threshold)))
        else:
            keep = singular_values.size

        u = u[:, :keep]
        singular_values = singular_values[:keep]
        vh = vh[:keep, :]

        if direction == "right":
            self.tensors[site] = u.reshape(left_dimension, physical, keep)
            self.tensors[site + 1] = (
                singular_values[:, None] * vh).reshape(
                    keep, right_physical, right_dimension)
        else:
            self.tensors[site] = (u * singular_values).reshape(
                left_dimension, physical, keep)
            self.tensors[site + 1] = vh.reshape(
                keep, right_physical, right_dimension)

    def apply_two_site_gate(self, first_site: int, gate, direction="right"):
        self.apply_adjacent_gate(first_site, gate, direction=direction)

    def swap_adjacent(self, site: int, direction="right"):
        dimension = self.physical_dimension
        swap = np.eye(dimension**2, dtype=complex).reshape(
            dimension, dimension, dimension, dimension)
        swap = swap.transpose(1, 0, 2, 3)
        self.apply_adjacent_gate(site, swap, direction=direction)

    def apply_nonlocal_gate(self, older_site: int, newest_site: int,
                            gate):
        """Route ``newest_site`` next to ``older_site`` and restore ordering."""
        if newest_site <= older_site:
            raise ValueError("newest_site must follow older_site.")
        for site in range(newest_site - 1, older_site, -1):
            self.swap_adjacent(site, direction="left")
        gate_for_pair = np.asarray(gate).transpose(1, 0, 3, 2)
        self.apply_two_site_gate(older_site, gate_for_pair,
                                 direction="right")
        for site in range(older_site + 1, newest_site):
            self.swap_adjacent(site, direction="right")

    def state_at_last_site(self, trace_vector=None):
        """Sum all historical path indices and return the latest state."""
        del trace_vector
        sum_vector = np.ones(self.physical_dimension, dtype=complex)
        if len(self.tensors) == 1:
            return self.tensors[0][0, :, 0]
        tensor = np.tensordot(self.tensors[0], sum_vector, axes=(1, 0))
        for site in range(1, len(self.tensors) - 1):
            tensor = np.tensordot(tensor, self.tensors[site], axes=(-1, 0))
            tensor = np.tensordot(tensor, sum_vector, axes=(1, 0))
        tensor = np.tensordot(tensor, self.tensors[-1], axes=(-1, 0))
        return np.asarray(tensor).reshape(-1)


class MPSPtTempo:
    """Compressed noncommuting cross-correlated TEMPO with SWAP routing."""

    def __init__(self, system, bath: Bath, parameters: TempoParameters,
                 initial_state: np.ndarray, start_time: float = 0.0,
                 max_steps: int = 100, memory_cutoff: Optional[int] = None):
        if not isinstance(bath, Bath):
            raise TypeError("bath must be a Bath instance")
        if bath.commuting_channels:
            raise ValueError("MPSPtTempo requires noncommuting channels.")
        if parameters.dkmax is None:
            raise ValueError("MPSPtTempo requires an explicit dkmax.")
        self.system = system
        self.bath = bath
        self.parameters = parameters
        self.initial_state = np.asarray(initial_state, dtype=complex)
        self.start_time = float(start_time)
        self.max_steps = int(max_steps)
        self.memory_cutoff = (parameters.dkmax if memory_cutoff is None
                              else int(memory_cutoff))
        if not 1 <= self.memory_cutoff <= parameters.dkmax:
            raise ValueError("memory_cutoff must be in [1, dkmax].")

    def compute(self, progress_type: Optional[str] = None) -> Dynamics:
        dimension = self.bath.dimension
        physical_dimension = dimension**2
        propagators = self.system.get_propagators(
            self.parameters.dt, self.start_time, None, None)
        influence = GeneralizedInfluenceTensor(self.bath, self.parameters)
        mps = CrossCorrelatedMPS(physical_dimension, self.parameters.epsrel)
        mps.tensors = [self.initial_state.reshape(1, physical_dimension, 1)]
        trace_vector = np.eye(dimension, dtype=complex).reshape(-1)
        states = [self.initial_state.copy()]
        progress = get_progress(progress_type)(self.max_steps,
                                                "MPS TEMPO propagation:")
        with progress as progress_bar:
            for step in range(self.max_steps):
                first_half, second_half = propagators(step)
                mps.append_propagated_site(first_half)
                newest = len(mps.tensors) - 1
                mps.apply_one_site(newest, influence.gate(0))
                for older in range(max(0, newest - self.memory_cutoff), newest):
                    mps.apply_nonlocal_gate(older, newest,
                                            influence.gate(newest - older))
                mps.apply_one_site(newest, second_half)
                state = mps.state_at_last_site(trace_vector)
                states.append(state.reshape(dimension, dimension))
                progress_bar.update(step + 1)
        times = self.start_time + np.arange(len(states)) * self.parameters.dt
        return Dynamics(times=list(times), states=states)
