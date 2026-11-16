"""MPO construction helpers for continuous influence generators."""

from typing import Dict, Iterable, List, Optional

import numpy as np


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
        left, singular, right = np.linalg.svd(matrix, full_matrices=False)
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
