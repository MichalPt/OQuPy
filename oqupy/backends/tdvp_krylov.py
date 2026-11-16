"""Matrix-free local Krylov exponentiation for TDVP."""

from typing import Optional, Tuple

import numpy as np
from scipy.sparse.linalg import LinearOperator, expm_multiply


def _two_site_action(vector: np.ndarray, shape: Tuple[int, ...],
                     left: np.ndarray, right: np.ndarray,
                     first: np.ndarray, second: np.ndarray) -> np.ndarray:
    tensor = vector.reshape(shape)
    return np.einsum(
        "awc,byd,wxis,xyjt,cstd->aijb",
        left, right, first, second, tensor, optimize=True).reshape(-1)


def _two_site_adjoint(vector: np.ndarray, shape: Tuple[int, ...],
                      left: np.ndarray, right: np.ndarray,
                      first: np.ndarray,
                      second: np.ndarray) -> np.ndarray:
    tensor = vector.reshape(shape)
    return np.einsum(
        "awc,byd,wxis,xyjt,aijb->cstd",
        left.conj(), right.conj(), first.conj(), second.conj(),
        tensor, optimize=True).reshape(-1)


def two_site_operator(shape: Tuple[int, int, int, int],
                      left: np.ndarray, right: np.ndarray,
                      first: np.ndarray,
                      second: np.ndarray) -> LinearOperator:
    """Create the effective two-site MPO LinearOperator."""
    size = int(np.prod(shape))
    return LinearOperator(
        (size, size),
        matvec=lambda vector: _two_site_action(
            vector, shape, left, right, first, second),
        rmatvec=lambda vector: _two_site_adjoint(
            vector, shape, left, right, first, second),
        dtype=np.complex128,
    )


def evolve_two_site(theta: np.ndarray, time: float, left: np.ndarray,
                    right: np.ndarray, first: np.ndarray,
                    second: np.ndarray,
                    traceA: Optional[float] = None) -> np.ndarray:
    """Apply ``exp(time * H_eff)`` without materialising ``H_eff``."""
    operator = two_site_operator(theta.shape, left, right, first, second)
    return expm_multiply(
        time * operator, theta.reshape(-1), traceA=0.0).reshape(
        theta.shape)


def zero_site_operator(shape: Tuple[int, int], left: np.ndarray,
                       right: np.ndarray, mpo_bond: np.ndarray) -> LinearOperator:
    """Create the effective zero-site operator on an MPS bond."""
    size = int(np.prod(shape))

    def matvec(vector: np.ndarray) -> np.ndarray:
        tensor = vector.reshape(shape)
        result = np.einsum(
            "awc,wx,bxd,cd->ab", left, mpo_bond, right, tensor,
            optimize=True)
        return result.reshape(-1)

    def rmatvec(vector: np.ndarray) -> np.ndarray:
        tensor = vector.reshape(shape)
        result = np.einsum(
            "awc,wx,bxd,ab->cd", left.conj(), mpo_bond.conj(),
            right.conj(), tensor, optimize=True)
        return result.reshape(-1)

    return LinearOperator(
        (size, size), matvec=matvec, rmatvec=rmatvec,
        dtype=np.complex128)


def evolve_zero_site(matrix: np.ndarray, time: float, left: np.ndarray,
                     right: np.ndarray, mpo_bond: np.ndarray) -> np.ndarray:
    """Apply a matrix-free zero-site Krylov exponential."""
    operator = zero_site_operator(matrix.shape, left, right, mpo_bond)
    return expm_multiply(
        time * operator, matrix.reshape(-1), traceA=0.0).reshape(
        matrix.shape)


def one_site_operator(shape: Tuple[int, int, int],
                      left: np.ndarray, right: np.ndarray,
                      mpo_tensor: np.ndarray) -> LinearOperator:
    """Create the effective one-site MPO LinearOperator."""
    size = int(np.prod(shape))

    def matvec(vector: np.ndarray) -> np.ndarray:
        tensor = vector.reshape(shape)
        result = np.einsum(
            "awc,wxis,bxd,csd->aib", left, mpo_tensor, right, tensor,
            optimize=True)
        return result.reshape(-1)

    def rmatvec(vector: np.ndarray) -> np.ndarray:
        tensor = vector.reshape(shape)
        result = np.einsum(
            "awc,wxis,bxd,aib->csd", left.conj(), mpo_tensor.conj(),
            right.conj(), tensor, optimize=True)
        return result.reshape(-1)

    return LinearOperator(
        (size, size), matvec=matvec, rmatvec=rmatvec,
        dtype=np.complex128)


def evolve_one_site(tensor: np.ndarray, time: float,
                    left: np.ndarray, right: np.ndarray,
                    mpo_tensor: np.ndarray) -> np.ndarray:
    """Apply a matrix-free one-site Krylov exponential."""
    operator = one_site_operator(tensor.shape, left, right, mpo_tensor)
    return expm_multiply(
        time * operator, tensor.reshape(-1), traceA=0.0).reshape(
        tensor.shape)
