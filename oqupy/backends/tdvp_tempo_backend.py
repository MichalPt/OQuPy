"""MPO-based adaptive two-site TDVP backend for TDVP-TEMPO."""

from typing import Optional, Tuple

import numpy as np
import scipy.linalg

from oqupy.backends.tdvp_environments import TdvpEnvironments
from oqupy.backends.tdvp_krylov import (
    evolve_one_site, evolve_two_site, evolve_zero_site)
from oqupy.backends.tdvp_mpo_builder import build_memory_mpo
from oqupy.dense_pt_tempo import ContinuousInfluenceTensor
from oqupy.dynamics import Dynamics
from oqupy.mps_cross_correlated import CrossCorrelatedMPS
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


def _split(theta: np.ndarray, epsrel: float,
           max_bond_dimension: Optional[int]):
    dl, physical, _, dr = theta.shape
    matrix = theta.transpose(0, 1, 2, 3).reshape(
        dl * physical, physical * dr)
    u, singular, vh = _safe_svd(matrix)
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


def _right_canonicalize(tensors, epsrel: float = 0.0,
                        max_bond_dimension: Optional[int] = None) -> None:
    """Losslessly right-canonicalize all sites except the left boundary."""
    for index in range(len(tensors) - 1, 0, -1):
        tensor = tensors[index]
        left_dimension, physical, right_dimension = tensor.shape
        matrix = tensor.reshape(left_dimension, physical * right_dimension)
        r_mat, q_mat = scipy.linalg.rq(matrix, mode="economic")
        keep = q_mat.shape[0]
        tensors[index] = q_mat.reshape(keep, physical, right_dimension)
        prev = tensors[index - 1]
        tensors[index - 1] = (
            prev.reshape(prev.shape[0] * prev.shape[1], prev.shape[2])
            @ r_mat).reshape(prev.shape[0], prev.shape[1], keep)


def _left_canonicalize(tensors) -> None:
    """Losslessly left-canonicalize all sites except the right boundary."""
    for index in range(len(tensors) - 1):
        tensor = tensors[index]
        left_dimension, physical, right_dimension = tensor.shape
        matrix = tensor.reshape(left_dimension * physical, right_dimension)
        q_mat, r_mat = scipy.linalg.qr(matrix, mode="economic")
        keep = q_mat.shape[1]
        tensors[index] = q_mat.reshape(left_dimension, physical, keep)
        nxt = tensors[index + 1]
        tensors[index + 1] = (
            r_mat @ nxt.reshape(
                nxt.shape[0], nxt.shape[1] * nxt.shape[2])
        ).reshape(keep, nxt.shape[1], nxt.shape[2])


def _apply_mpo(tensors, mpo):
    """Apply an MPO to an MPS using the output/input convention of the backend."""
    sliced = list(mpo)
    sliced[0] = sliced[0][0:1, :, :, :]
    sliced[-1] = sliced[-1][:, 1:2, :, :]
    n_sites = len(sliced)
    fwd = [np.array([True], dtype=bool)]
    for operator in sliced:
        nonzeros = np.any(operator != 0, axis=(2, 3))
        fwd.append(np.any(nonzeros[fwd[-1]], axis=0))
    bwd = [None] * (n_sites + 1)
    bwd[n_sites] = np.array([True], dtype=bool)
    for index in range(n_sites - 1, -1, -1):
        nonzeros = np.any(sliced[index] != 0, axis=(2, 3))
        bwd[index] = np.any(
            nonzeros[:, bwd[index + 1]], axis=1)
    active = [forward & backward
              for forward, backward in zip(fwd, bwd)]
    if any(not np.any(mask) for mask in active):
        return [
            np.zeros((1, tensor.shape[1], 1), dtype=complex)
            for tensor in tensors
        ]
    result = []
    for index, tensor in enumerate(tensors):
        operator = sliced[index][active[index]][:, active[index + 1]]
        left, physical, right = tensor.shape
        mpo_left, mpo_right, output, input_dimension = operator.shape
        if physical == input_dimension:
            contracted = np.tensordot(
                operator, tensor, axes=([3], [1]))
            result.append(contracted.transpose(
                0, 3, 2, 1, 4).reshape(
                    mpo_left * left, output, mpo_right * right))
        elif physical == input_dimension * input_dimension:
            d = input_dimension
            tensor_4d = tensor.reshape(left, d, d, right)
            contracted = np.tensordot(
                operator, tensor_4d, axes=([3], [2]))
            result.append(contracted.transpose(
                0, 3, 4, 2, 1, 5).reshape(
                    mpo_left * left, d * output, mpo_right * right))
        else:
            raise ValueError("MPO and MPS physical dimensions do not match")
    return result


def _gse_expand(mps, mpo, epsrel, max_bond_dimension):
    """Expand a right-canonical MPS with the MPO Krylov direction."""
    if max_bond_dimension is not None:
        min_expandable = next(
            (index for index in range(1, len(mps.tensors))
             if mps.tensors[index].shape[0] < max_bond_dimension),
            None)
        if min_expandable is None:
            return
    else:
        min_expandable = 1
    phi = _apply_mpo(mps.tensors, mpo)
    _left_canonicalize(phi)
    overlap_right = np.eye(1, dtype=complex)
    for index in range(len(mps.tensors) - 1, min_expandable - 1, -1):
        current = mps.tensors[index]
        target = phi[index]
        if target.shape[2] != overlap_right.shape[1]:
            raise ValueError(
                "GSE right-overlap dimension does not match the Krylov "
                "tensor right bond")
        left, physical, right = current.shape
        current_matrix = current.reshape(left, physical * right)
        can_expand = (
            max_bond_dimension is None or left < max_bond_dimension)
        if can_expand:
            mapped_matrix = (
                target.reshape(-1, target.shape[2])
                @ overlap_right.T).reshape(
                    target.shape[0], physical * right)
            mapped_norm = float(np.linalg.norm(mapped_matrix))
            if mapped_norm == 0.0:
                keep = 0
            else:
                projected = mapped_matrix - (
                    mapped_matrix @ current_matrix.conj().T) @ current_matrix
                _, singular, vh = _safe_svd(projected)
                threshold = max(epsrel, 1.0e-12) * mapped_norm
                keep = int(np.count_nonzero(singular > threshold))
                if max_bond_dimension is not None:
                    keep = min(keep, max_bond_dimension - left)
                keep = max(0, keep)
        else:
            expanded = current_matrix
            mps.tensors[index] = current
            keep = 0
        if keep:
            extra = vh[:keep]
            extra = extra - (
                extra @ current_matrix.conj().T) @ current_matrix
            _, s_extra, vh_extra = _safe_svd(extra)
            valid = int(np.count_nonzero(s_extra > 1.0e-10))
            extra = vh_extra[:valid]
            expanded = np.vstack((current_matrix, extra))
            mps.tensors[index] = expanded.reshape(
                left + valid, physical, right)
            previous = mps.tensors[index - 1]
            if valid > 0:
                mps.tensors[index - 1] = np.pad(
                    previous, ((0, 0), (0, 0), (0, valid)))
        else:
            if can_expand:
                expanded = current_matrix
                mps.tensors[index] = current

        if index == min_expandable:
            break

        u_phi, s_phi, vh_phi = _safe_svd(
            target.reshape(target.shape[0], physical * target.shape[2]))
        if s_phi.size == 0 or s_phi[0] == 0.0:
            break
        keep_phi = max(1, int(np.count_nonzero(
            s_phi > max(epsrel, 1.0e-12) * s_phi[0])))
        if max_bond_dimension is not None:
            keep_phi = min(keep_phi, max_bond_dimension)
        phi_right_iso = vh_phi[:keep_phi].reshape(
            keep_phi, physical, target.shape[2])
        phi[index] = phi_right_iso
        us = u_phi[:, :keep_phi] * s_phi[:keep_phi]
        prev_phi = phi[index - 1]
        phi[index - 1] = (
                prev_phi.reshape(-1, prev_phi.shape[2]) @ us
        ).reshape(prev_phi.shape[0], prev_phi.shape[1], keep_phi)
        mapped_iso = (
                phi_right_iso.reshape(-1, target.shape[2])
                @ overlap_right.T
        ).reshape(keep_phi, physical * right)
        overlap_right = expanded.conj() @ mapped_iso.T


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
        self._mps = None
        self._previous_second_half = None
        self._mpo_cache = {}
        self._influence = None
        self._propagators = None
        self._uniform_steps = 0
        self._uniform_dynamics: Optional[Dynamics] = None

    def _reset_uniform_cache(self) -> None:
        self._uniform_dynamics = None
        self._mps = None
        self._previous_second_half = None
        self._mpo_cache = {}
        self._influence = None
        self._propagators = None
        self._uniform_steps = 0

    def compute(self, end_time: Optional[float] = None,
                time_grid: Optional[np.ndarray] = None,
                progress_type: Optional[str] = None) -> Dynamics:
        """Compute through a point on the TDVP time grid."""
        if time_grid is not None:
            grid = np.asarray(time_grid, dtype=float)
            if grid.ndim != 1 or grid.size < 2:
                raise ValueError("time_grid must be a one-dimensional grid")
            if not np.isclose(grid[0], self.start_time):
                raise ValueError("time_grid must start at start_time")
            if np.any(np.diff(grid) <= 0.0):
                raise ValueError("time_grid must be strictly increasing")
            if (np.allclose(np.diff(grid), self.parameters.dt)
                    and np.isclose(grid[0], self.start_time)):
                self._dynamics = self._compute(
                    grid.size - 1, progress_type=progress_type)
                return self._dynamics
            self._reset_uniform_cache()
            self._dynamics = self._compute_time_grid(
                grid, progress_type=progress_type)
            return self._dynamics
        if end_time is None:
            raise ValueError("end_time is required when time_grid is omitted")
        if end_time < self.start_time:
            raise ValueError("end_time must not precede start_time")
        steps = int(round(
            (end_time - self.start_time) / self.parameters.dt))
        if not np.isclose(
                self.start_time + steps * self.parameters.dt, end_time):
            raise ValueError("end_time must lie on the TDVP time grid")
        if self._uniform_dynamics is None or self._uniform_steps < steps:
            self._compute(
                steps, progress_type=progress_type)
        return Dynamics(
            times=list(self._uniform_dynamics.times[:steps + 1]),
            states=list(self._uniform_dynamics.states[:steps + 1]))

    def _compute_time_grid(
            self, time_grid: np.ndarray,
            progress_type: Optional[str] = None) -> Dynamics:
        """Compute TDVP dynamics on an explicitly supplied time grid."""
        if self.parameters.tdvp_method not in ("1-site", "2-site"):
            raise NotImplementedError("Unsupported TDVP method.")
        dimension = self.bath.dimension
        physical = dimension**2
        influence = ContinuousInfluenceTensor(self.bath, self.parameters)
        unitary = (self.bath.unitary_transform
                   if self.bath.commuting_channels
                   else np.eye(dimension, dtype=complex))
        liouville_transform = np.kron(
            unitary.conjugate().T, unitary.T)
        liouville_inverse = liouville_transform.conjugate().T
        initial_state = liouville_transform @ self.initial_state.reshape(-1)
        superop_mode = not self.bath.commuting_channels
        mps = CrossCorrelatedMPS(
            physical, self.parameters.epsrel,
            self.parameters.max_bond_dimension,
            superoperator_mode=superop_mode)
        states = [self.initial_state.copy()]
        previous_second_half = None
        active_intervals = []
        intervals = list(zip(time_grid[:-1], time_grid[1:]))
        active_preview = []
        max_lag = 0.0
        for t_k0, t_k1 in intervals:
            active_preview.append((float(t_k0), float(t_k1)))
            while (self.parameters.dkmax is not None
                   and len(active_preview) > self.parameters.dkmax + 1):
                active_preview.pop(0)
            while (self.parameters.tcut is not None
                   and len(active_preview) > 1
                   and t_k0 - active_preview[0][1] > self.parameters.tcut):
                active_preview.pop(0)
            if active_preview:
                max_lag = max(
                    max_lag, float(t_k1) - active_preview[0][0])
        influence.prepare_splines(max_lag)

        with get_progress(progress_type)(
                time_grid.size - 1,
                "TDVP TEMPO propagation:") as progress_bar:
            for step, (t_k0, t_k1) in enumerate(
                    zip(time_grid[:-1], time_grid[1:])):
                dt = float(t_k1 - t_k0)
                propagator = self.system.get_propagators(
                    dt, float(t_k0), None, None)(0)
                first_half, second_half = propagator
                first_half = liouville_transform @ first_half @ liouville_inverse
                second_half = liouville_transform @ second_half @ liouville_inverse
                if step == 0:
                    midpoint = first_half @ initial_state
                    mps.set_initial_state(midpoint)
                    mps.apply_one_site(
                        0, influence.gate_interval(t_k0, t_k1))
                    active_intervals.append((float(t_k0), float(t_k1)))
                else:
                    mps.append_propagated_site(
                        first_half @ previous_second_half)
                    mps.apply_one_site(
                        len(mps.tensors) - 1,
                        influence.gate_interval(t_k0, t_k1))
                    active_intervals.append((float(t_k0), float(t_k1)))
                    while (self.parameters.dkmax is not None
                           and len(mps.tensors)
                           > self.parameters.dkmax + 1):
                        mps.contract_oldest()
                        active_intervals.pop(0)
                    while (self.parameters.tcut is not None
                           and active_intervals
                           and len(mps.tensors) > 1
                           and t_k0 - active_intervals[0][1]
                           > self.parameters.tcut):
                        mps.contract_oldest()
                        active_intervals.pop(0)
                    length = len(mps.tensors)
                    _right_canonicalize(
                        mps.tensors, self.parameters.epsrel,
                        self.parameters.max_bond_dimension)
                    generators = {
                        length - 1 - index:
                        -influence.generator_interval(
                            t_k0, t_k1, interval[0], interval[1]) / dt
                        for index, interval in enumerate(active_intervals[:-1])
                    }
                    mpo = build_memory_mpo(
                        generators, length, physical,
                        self.parameters.max_bond_dimension,
                        self.parameters.sweep_tolerance,
                        active_site=length - 1)
                    self._tdvp_sweep(mps, mpo, dt)

                state = liouville_inverse @ (
                    second_half @ mps.state_at_last_site())
                states.append(state.reshape(dimension, dimension))
                previous_second_half = second_half
                progress_bar.update(step + 1)

        return Dynamics(times=list(time_grid), states=states)

    def _tdvp_sweep(self, mps, mpo, dt: float) -> None:
        """Run one forward/backward two-site TDVP sweep."""
        if mps.superoperator_mode:
            _gse_expand(
                mps, mpo, self.parameters.epsrel,
                self.parameters.max_bond_dimension)
        if self.parameters.tdvp_method == "1-site":
            self._tdvp_sweep_one_site(mps, mpo, dt)
            return
        length = len(mps.tensors)
        environments = TdvpEnvironments(mps.tensors, mpo)
        environments.build_right()
        for index in range(length - 1):
            theta = np.tensordot(
                mps.tensors[index], mps.tensors[index + 1], axes=(2, 0))
            theta = evolve_two_site(
                theta, dt / 2, environments.left[index],
                environments.right[index + 2], mpo[index], mpo[index + 1])
            left, center, right = _split(
                theta, self.parameters.epsrel,
                self.parameters.max_bond_dimension)
            mps.tensors[index] = left
            mps.tensors[index + 1] = (
                center.diagonal()[:, None, None] * right)
            environments.update_left(index)
            if index < length - 2:
                mps.tensors[index + 1] = evolve_one_site(
                    mps.tensors[index + 1], -dt / 2,
                    environments.left[index + 1],
                    environments.right[index + 2], mpo[index + 1])
        for index in range(length - 2, -1, -1):
            theta = np.tensordot(
                mps.tensors[index], mps.tensors[index + 1], axes=(2, 0))
            theta = evolve_two_site(
                theta, dt / 2, environments.left[index],
                environments.right[index + 2], mpo[index], mpo[index + 1])
            left, center, right = _split(
                theta, self.parameters.epsrel,
                self.parameters.max_bond_dimension)
            mps.tensors[index + 1] = right
            mps.tensors[index] = (
                left * center.diagonal()[None, None, :])
            environments.update_right(index + 1)
            if index > 0:
                mps.tensors[index] = evolve_one_site(
                    mps.tensors[index], -dt / 2,
                    environments.left[index],
                    environments.right[index + 1], mpo[index])

    def _tdvp_sweep_one_site(self, mps, mpo, dt: float) -> None:
        """Run a one-site TDVP sweep with zero-site bond corrections."""
        length = len(mps.tensors)
        environments = TdvpEnvironments(mps.tensors, mpo)
        environments.build_right()
        for index in range(length):
            mps.tensors[index] = evolve_one_site(
                    mps.tensors[index], dt / 2,
                    environments.left[index], environments.right[index + 1],
                    mpo[index])
            if index < length - 1:
                    tensor = mps.tensors[index]
                    left_dim, physical, right_dim = tensor.shape
                    matrix = tensor.reshape(left_dim * physical, right_dim)
                    u, singular, vh = _safe_svd(matrix)
                    keep = min(len(singular), self.parameters.max_bond_dimension
                               or len(singular))
                    u, singular, vh = u[:, :keep], singular[:keep], vh[:keep]
                    mps.tensors[index] = u.reshape(left_dim, physical, keep)
                    center = singular[:, None] * vh
                    environments.update_left(index)
                    center = evolve_zero_site(
                        center, -dt / 2, environments.left[index + 1],
                        environments.right[index + 1],
                        np.eye(mpo[index].shape[1], dtype=complex))
                    nxt = mps.tensors[index + 1]
                    mps.tensors[index + 1] = (
                        center @ nxt.reshape(center.shape[1], -1)
                    ).reshape(center.shape[0], nxt.shape[1], nxt.shape[2])
        for index in range(length - 1, -1, -1):
            mps.tensors[index] = evolve_one_site(
                    mps.tensors[index], dt / 2,
                    environments.left[index], environments.right[index + 1],
                    mpo[index])
            if index > 0:
                    tensor = mps.tensors[index]
                    left_dim, physical, right_dim = tensor.shape
                    matrix = tensor.reshape(left_dim, physical * right_dim)
                    u, singular, vh = _safe_svd(matrix)
                    keep = min(len(singular), self.parameters.max_bond_dimension
                               or len(singular))
                    u, singular, vh = u[:, :keep], singular[:keep], vh[:keep]
                    center = u * singular[None, :]
                    mps.tensors[index] = vh.reshape(
                        keep, physical, right_dim)
                    environments.update_right(index)
                    center = evolve_zero_site(
                        center, -dt / 2, environments.left[index],
                        environments.right[index],
                        np.eye(mpo[index - 1].shape[1], dtype=complex))
                    prev = mps.tensors[index - 1]
                    mps.tensors[index - 1] = (
                        prev.reshape(-1, prev.shape[2]) @ center
                    ).reshape(
                        prev.shape[0], prev.shape[1], center.shape[1])

    def _compute(self, steps: int,
                 progress_type: Optional[str] = None) -> Dynamics:
        if self.parameters.tdvp_method not in ("1-site", "2-site"):
            raise NotImplementedError("Unsupported TDVP method.")
        dimension = self.bath.dimension
        physical = dimension**2
        if self._mps is None or self._uniform_dynamics is None:
            self._propagators = self.system.get_propagators(
                self.parameters.dt, self.start_time, None, None)
            self._influence = ContinuousInfluenceTensor(
                self.bath, self.parameters)
            self._mps = CrossCorrelatedMPS(
                physical, self.parameters.epsrel,
                self.parameters.max_bond_dimension,
                superoperator_mode=not self.bath.commuting_channels)
            self._previous_second_half = None
            self._mpo_cache = {}
            states = [self.initial_state.copy()]
            self._uniform_steps = 0
            self._uniform_dynamics = None
        else:
            states = list(self._uniform_dynamics.states)
            if steps <= self._uniform_steps:
                return self._uniform_dynamics
        propagators = self._propagators
        influence = self._influence
        unitary = (self.bath.unitary_transform
                   if self.bath.commuting_channels
                   else np.eye(dimension, dtype=complex))
        liouville_transform = np.kron(
            unitary.conjugate().T, unitary.T)
        liouville_inverse = liouville_transform.conjugate().T
        initial_state = liouville_transform @ self.initial_state.reshape(-1)
        mps = self._mps
        superop_mode = not self.bath.commuting_channels
        previous_second_half = self._previous_second_half
        mpo_cache = self._mpo_cache
        start_step = self._uniform_steps
        cutoff = (self.parameters.dkmax if self.parameters.dkmax is not None
                  else int(np.ceil(
                      self.parameters.tcut / self.parameters.dt)))
        if steps > start_step:
            max_distance = min(max(steps - 1, 0), cutoff)
            influence.prepare_splines(
                (max_distance + 1) * self.parameters.dt)

        with get_progress(progress_type)(
                steps, "TDVP TEMPO propagation:") as progress_bar:
            for step in range(start_step, steps):
                first_half, second_half = propagators(step)
                first_half = (
                    liouville_transform @ first_half @ liouville_inverse)
                second_half = (
                    liouville_transform @ second_half @ liouville_inverse)
                if step == 0:
                    first_midpoint = first_half @ initial_state
                    mps.set_initial_state(first_midpoint)
                    mps.apply_one_site(0, influence.gate(0))
                else:
                    mps.append_propagated_site(
                        first_half @ previous_second_half)
                    length = len(mps.tensors)
                    mps.apply_one_site(length - 1, influence.gate(0))
                    while len(mps.tensors) > cutoff + 1:
                        mps.contract_oldest()
                    length = len(mps.tensors)
                    _right_canonicalize(
                        mps.tensors,
                        self.parameters.epsrel,
                        self.parameters.max_bond_dimension)
                    generators = {
                        distance: -influence.generator(distance) /
                        self.parameters.dt
                        for distance in range(1, min(step, cutoff) + 1)
                    }
                    if length not in mpo_cache:
                        mpo_cache[length] = build_memory_mpo(
                            generators, length, physical,
                            self.parameters.max_bond_dimension,
                            self.parameters.sweep_tolerance,
                            active_site=length - 1)
                    self._tdvp_sweep(
                        mps, mpo_cache[length], self.parameters.dt)

                state = liouville_inverse @ (
                    second_half @ mps.state_at_last_site())
                states.append(state.reshape(dimension, dimension))
                previous_second_half = second_half
                progress_bar.update(step + 1)

        self._uniform_steps = steps
        self._previous_second_half = previous_second_half
        self._uniform_dynamics = Dynamics(
            times=list(self.start_time
                      + np.arange(len(states)) * self.parameters.dt),
            states=states)
        self._dynamics = self._uniform_dynamics
        return self._uniform_dynamics

    def initialize(self) -> Tuple[int, np.ndarray]:
        """Return the initial step and state."""
        return 0, self.initial_state.copy()

    def compute_step(self) -> Tuple[int, np.ndarray]:
        """Compute and return one additional time step."""
        current = self._uniform_steps
        dynamics = self.compute(
            self.start_time + (current + 1) * self.parameters.dt)
        return current + 1, dynamics.states[-1]
