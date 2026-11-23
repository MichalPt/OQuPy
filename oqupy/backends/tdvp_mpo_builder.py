"""MPO construction helpers for continuous influence generators."""

from typing import Dict, Iterable, List, Optional

import numpy as np
import scipy.linalg


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


def build_memory_mpo(
        generators: Dict[int, np.ndarray],
        length: int,
        physical_dimension: int,
        max_bond_dimension: Optional[int] = None,
        tolerance: float = 0.0,
        active_site: Optional[int] = None) -> List[np.ndarray]:
    """Build an MPO for a finite sum of pairwise memory generators.

    ``generators[d]`` is the two-site generator at distance ``d`` and has
    shape ``(p, p, p, p)``.  The returned tensors use the conventional
    ``(left_bond, right_bond, output, input)`` ordering.  Each pair is
    represented by an operator-Schmidt decomposition and assigned a
    separate finite-state path, so the construction is linear in the number
    of retained pair terms rather than exponential in the memory length.

    Parameters
    ----------
    generators: dict
        Mapping from positive site distance to rank-four generators.
    length: int
        Number of sites in the memory chain.
    physical_dimension: int
        Dimension of one vectorized density operator.
    max_bond_dimension: int, optional
        Maximum number of operator-Schmidt paths retained per pair.
    tolerance: float
        Relative singular-value cutoff.
    """
    if length < 1:
        raise ValueError("length must be positive")
    if physical_dimension < 1:
        raise ValueError("physical_dimension must be positive")
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")

    p = physical_dimension
    identity = np.eye(p, dtype=complex)
    if active_site is not None:
        if active_site < 0 or active_site >= length:
            raise ValueError("active_site must be a valid site index")
        valid_distances = [
            distance for distance in sorted(generators)
            if 1 <= distance < length and active_site >= distance]
        bond = 2
        if valid_distances:
            matrices = []
            for distance in valid_distances:
                array = np.asarray(generators[distance], dtype=complex)
                if array.shape != (p, p, p, p):
                    raise ValueError(
                        "A two-site generator must have shape "
                        f"({p}, {p}, {p}, {p})")
                matrices.append(array.transpose(1, 3, 0, 2).reshape(
                    p * p, p * p))
            stacked = np.vstack(matrices)
            left, singular, right = _safe_svd(stacked)
            cutoff = tolerance * singular[0] if singular.size else 0.0
            keep = int(np.count_nonzero(singular > cutoff))
            keep = max(1, keep)
            if max_bond_dimension is not None:
                if max_bond_dimension < 1:
                    raise ValueError("max_bond_dimension must be positive")
                keep = min(keep, max_bond_dimension)
            bond += keep

        mpo = [np.zeros((bond, bond, p, p), dtype=complex)
               for _ in range(length)]
        for tensor in mpo:
            tensor[0, 0] = identity
            tensor[1, 1] = identity
        if not valid_distances:
            return mpo

        min_first = active_site - max(valid_distances)
        for rank in range(keep):
            path_index = 2 + rank
            mpo[active_site][path_index, 1] = right[rank, :].reshape(p, p)
            for site in range(min_first + 1, active_site):
                mpo[site][path_index, path_index] = identity
            for index, distance in enumerate(valid_distances):
                first = active_site - distance
                block = (left[index * p * p:(index + 1) * p * p, rank]
                         * singular[rank])
                mpo[first][0, path_index] = block.reshape(p, p)
        return mpo

    paths = []
    for distance, generator in generators.items():
        if distance < 1 or distance >= length:
            continue
        array = np.asarray(generator, dtype=complex)
        if array.shape != (p, p, p, p):
            raise ValueError(
                "A two-site generator must have shape "
                f"({p}, {p}, {p}, {p})")
        matrix = array.transpose(1, 3, 0, 2).reshape(p * p, p * p)
        left, singular, right = _safe_svd(matrix)
        cutoff = tolerance * singular[0] if singular.size else 0.0
        keep = int(np.count_nonzero(singular > cutoff))
        keep = max(1, keep)
        if max_bond_dimension is not None:
            if max_bond_dimension < 1:
                raise ValueError("max_bond_dimension must be positive")
            keep = min(keep, max_bond_dimension)
        for rank in range(keep):
            left_op = left[:, rank].reshape(p, p) * singular[rank]
            right_op = right[rank, :].reshape(p, p)
            if active_site is None:
                first_sites = range(length - distance)
            elif active_site >= distance:
                first_sites = (active_site - distance,)
            else:
                first_sites = ()
            for first in first_sites:
                paths.append((first, first + distance, left_op, right_op))

    # The zero state and final state propagate identity operators.  Each
    # finite-state path enters from the initial state and exits into the
    # final state, producing a sum of pair terms.
    bond = 2 + len(paths)
    mpo = [np.zeros((bond, bond, p, p), dtype=complex)
           for _ in range(length)]
    for tensor in mpo:
        tensor[0, 0] = identity
        tensor[1, 1] = identity

    for path_index, (first, second, left_op, right_op) in enumerate(paths, 2):
        mpo[first][0, path_index] = left_op
        for site in range(first + 1, second):
            mpo[site][path_index, path_index] = identity
        mpo[second][path_index, 1] = right_op
    return mpo


def build_tdvp_memory_mpo(
        generators: Iterable[np.ndarray],
        length: int,
        physical_dimension: int,
        max_bond_dimension: Optional[int] = None,
        tolerance: float = 0.0) -> List[np.ndarray]:
    """Convenience wrapper accepting generators ordered by distance."""
    return build_memory_mpo(
        {distance: generator
         for distance, generator in enumerate(generators) if distance > 0},
        length,
        physical_dimension,
        max_bond_dimension,
        tolerance,
    )
