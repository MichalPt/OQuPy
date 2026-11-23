"""MPS/SWAP reference solver for noncommuting cross-correlated baths."""

from typing import Optional

import numpy as np
import scipy.linalg

from oqupy.bath import Bath
from oqupy.dense_pt_tempo import GeneralizedInfluenceTensor
from oqupy.dynamics import Dynamics
from oqupy.tempo import TempoParameters
from oqupy.util import get_progress


def _safe_svd(matrix: np.ndarray):
    """Compute economic SVD with exact power-of-2 scaling and gesvd fallback."""
    arr = np.asarray(matrix)
    if arr.ndim != 2:
        raise ValueError("SVD input must be a 2-D array")
    if not np.all(np.isfinite(arr)):
        raise ValueError("array must not contain infs or NaNs")
    if arr.size == 0:
        rows, cols = arr.shape
        k = min(rows, cols)
        return (
            np.empty((rows, k), dtype=arr.dtype),
            np.empty((k,), dtype=np.float64),
            np.empty((k, cols), dtype=arr.dtype),
        )
    if np.iscomplexobj(arr):
        max_abs = float(max(
            np.max(np.abs(arr.real)), np.max(np.abs(arr.imag))))
    else:
        max_abs = float(np.max(np.abs(arr)))
    if max_abs == 0.0:
        return scipy.linalg.svd(
            arr, full_matrices=False, lapack_driver="gesvd",
            check_finite=False)
    _, exp = np.frexp(max_abs)
    exp = max(-1020, min(1020, int(exp)))
    scale = float(np.ldexp(1.0, exp))
    scaled = arr if scale == 1.0 else arr / scale
    try:
        u, singular, vh = scipy.linalg.svd(
            scaled, full_matrices=False, lapack_driver="gesdd",
            check_finite=False)
    except scipy.linalg.LinAlgError:
        u, singular, vh = scipy.linalg.svd(
            scaled, full_matrices=False, lapack_driver="gesvd",
            check_finite=False)
    if scale != 1.0:
        singular = singular * scale
    return u, singular, vh


class CrossCorrelatedMPS:
    """MPS with adjacent-gate SVD compression and exact SWAP routing."""

    def __init__(self, physical_dimension: int, epsrel: float,
                 max_bond_dimension: Optional[int] = None,
                 superoperator_mode: bool = False):
        self.superoperator_mode = bool(superoperator_mode)
        self.state_dimension = physical_dimension
        self.physical_dimension = (
            physical_dimension**2 if self.superoperator_mode
            else physical_dimension)
        self.epsrel = epsrel
        self.max_bond_dimension = max_bond_dimension
        self.tensors = []

    def set_initial_state(self, state_vector: np.ndarray) -> None:
        """Set the first MPS site from a vectorized system state."""
        state = np.asarray(state_vector, dtype=complex)
        if not self.superoperator_mode:
            self.tensors = [state.reshape(1, self.state_dimension, 1)]
            return
        tensor = np.zeros(
            (1, self.state_dimension, self.state_dimension, 1),
            dtype=complex)
        tensor[0, 0, :, 0] = state
        self.tensors = [tensor.reshape(1, self.physical_dimension, 1)]

    def append_propagated_site(self, propagator):
        """Append a site correlated with the current site's physical index."""
        dimension = self.state_dimension
        if not self.tensors:
            raise ValueError("Cannot append to an empty MPS.")
        last = self.tensors[-1]
        if last.shape[2] != 1:
            raise ValueError("The MPS must have an open right boundary.")
        if self.superoperator_mode:
            last_vec = last.reshape(
                last.shape[0], dimension, dimension)[:, 0, :]
            left_tensor = np.zeros(
                (last.shape[0], dimension, dimension, dimension),
                dtype=complex)
            for bond in range(dimension):
                left_tensor[:, bond, :, bond] = last_vec
            new_tensor = np.zeros(
                (dimension, dimension, dimension, 1), dtype=complex)
            new_tensor[:, 0, :, 0] = np.asarray(propagator).T
            self.tensors[-1] = left_tensor.reshape(
                last.shape[0], self.physical_dimension, dimension)
            self.tensors.append(new_tensor.reshape(
                dimension, self.physical_dimension, 1))
            return
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
        if self.superoperator_mode and op_mat.shape == (
                self.state_dimension, self.state_dimension):
            left_dim, _, right_dim = tensor.shape
            reshaped = tensor.reshape(
                left_dim, self.state_dimension, self.state_dimension,
                right_dim)
            result = np.einsum("pq,lsqr->lspr", op_mat, reshaped)
            self.tensors[site] = result.reshape(
                left_dim, self.physical_dimension, right_dim)
            return
        res = np.tensordot(op_mat, tensor, axes=([1], [1]))
        self.tensors[site] = res.transpose(1, 0, 2)

    def apply_adjacent_gate(self, site: int, gate, direction="right"):
        """Apply an output/input gate to sites ``site`` and ``site+1``."""
        if direction not in ("left", "right"):
            raise ValueError("direction must be 'left' or 'right'")

        left = self.tensors[site]
        right = self.tensors[site + 1]
        gate = np.asarray(gate)
        if self.superoperator_mode and gate.shape == (
                self.state_dimension, self.state_dimension,
                self.state_dimension, self.state_dimension):
            eye = np.eye(self.state_dimension, dtype=complex)
            gate = np.einsum(
                "ab,cd,prqs->apcrbqds", eye, eye, gate).reshape(
                    self.physical_dimension, self.physical_dimension,
                    self.physical_dimension, self.physical_dimension)
        left_dimension, physical = left.shape[0], left.shape[1]
        right_physical, right_dimension = right.shape[1], right.shape[2]

        joint = np.tensordot(left, right, axes=([2], [0]))
        joint_mat = joint.transpose(1, 2, 0, 3).reshape(
            physical * right_physical, left_dimension * right_dimension)

        gate_mat = gate.reshape(physical * right_physical,
                               physical * right_physical)
        joint_gated = np.dot(gate_mat, joint_mat)
        joint_gated = joint_gated.reshape(physical, right_physical,
                                          left_dimension, right_dimension)
        joint_for_svd = joint_gated.transpose(2, 0, 1, 3).reshape(
            left_dimension * physical, right_physical * right_dimension)

        u, singular_values, vh = _safe_svd(joint_for_svd)

        if self.epsrel > 0.0 and singular_values.size > 0:
            threshold = self.epsrel * singular_values[0]
            keep = max(1, int(np.count_nonzero(singular_values > threshold)))
        else:
            keep = singular_values.size
        if self.max_bond_dimension is not None:
            keep = min(keep, self.max_bond_dimension)

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
        if self.superoperator_mode:
            sum_vector = np.eye(
                self.state_dimension, dtype=complex).reshape(-1)
        else:
            sum_vector = np.ones(self.physical_dimension, dtype=complex)
        if len(self.tensors) == 1:
            result = self.tensors[0][0, :, 0]
            return (result.reshape(self.state_dimension, self.state_dimension)
                    [0, :] if self.superoperator_mode else result)
        tensor = np.tensordot(self.tensors[0], sum_vector, axes=(1, 0))
        for site in range(1, len(self.tensors) - 1):
            tensor = np.tensordot(tensor, self.tensors[site], axes=(-1, 0))
            tensor = np.tensordot(tensor, sum_vector, axes=(1, 0))
        tensor = np.tensordot(tensor, self.tensors[-1], axes=(-1, 0))
        result = np.asarray(tensor).reshape(-1)
        return (result.reshape(self.state_dimension, self.state_dimension)[0, :]
                if self.superoperator_mode else result)

    def contract_oldest(self) -> None:
        """Trace out the oldest history site and absorb its boundary."""
        if len(self.tensors) < 2:
            raise ValueError("Cannot contract the only MPS site.")
        if self.superoperator_mode:
            vector = np.eye(
                self.state_dimension, dtype=complex).reshape(-1)
        else:
            vector = np.ones(self.physical_dimension, dtype=complex)
        boundary = np.einsum("lpr,p->lr", self.tensors[0], vector)
        self.tensors[1] = np.einsum(
            "lr,rps->lps", boundary, self.tensors[1])
        self.tensors.pop(0)


class MPSPtTempo:
    """Compressed noncommuting cross-correlated TEMPO with SWAP routing."""

    def __init__(self, system, bath: Bath, parameters: TempoParameters,
                 initial_state: np.ndarray, start_time: float = 0.0,
                 max_steps: int = 100, memory_cutoff: Optional[int] = None):
        if not isinstance(bath, Bath):
            raise TypeError("bath must be a Bath instance")
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
        unitary = (self.bath.unitary_transform
                   if self.bath.commuting_channels
                   else np.eye(dimension, dtype=complex))
        liouville_transform = np.kron(
            unitary.conjugate().T, unitary.T)
        liouville_inverse = liouville_transform.conjugate().T
        initial_state = liouville_transform @ self.initial_state.reshape(-1)
        mps = CrossCorrelatedMPS(
            physical_dimension, self.parameters.epsrel,
            superoperator_mode=not self.bath.commuting_channels)
        states = [self.initial_state.copy()]
        previous_second_half = None
        progress = get_progress(progress_type)(self.max_steps,
                                                "MPS TEMPO propagation:")
        with progress as progress_bar:
            for step in range(self.max_steps):
                first_half, second_half = propagators(step)
                first_half = (
                    liouville_transform @ first_half @ liouville_inverse)
                second_half = (
                    liouville_transform @ second_half @ liouville_inverse)
                if step == 0:
                    mps.set_initial_state(
                        first_half @ initial_state)
                    mps.apply_one_site(0, influence.gate(0))
                else:
                    mps.append_propagated_site(
                        first_half @ previous_second_half)
                    newest = len(mps.tensors) - 1
                    mps.apply_one_site(newest, influence.gate(0))
                    for older in range(
                            max(0, newest - self.memory_cutoff), newest):
                        mps.apply_nonlocal_gate(
                            older, newest,
                            influence.gate(newest - older))
                    while len(mps.tensors) > self.memory_cutoff + 1:
                        mps.contract_oldest()
                state = liouville_inverse @ (
                    second_half @ mps.state_at_last_site())
                states.append(state.reshape(dimension, dimension))
                previous_second_half = second_half
                progress_bar.update(step + 1)
        times = self.start_time + np.arange(len(states)) * self.parameters.dt
        return Dynamics(times=list(times), states=states)
