"""Matrix-free local Krylov exponentiation for TDVP."""

from typing import Optional, Tuple

import numpy as np
import scipy.linalg
from scipy.sparse.linalg import LinearOperator


def _arnoldi_expm(matvec_fn, vector: np.ndarray, time: float,
                  tol: float = 1.0e-9,
                  max_krylov: int = 10) -> np.ndarray:
    """Apply a matrix exponential using a short Arnoldi Krylov space."""
    norm = float(np.linalg.norm(vector))
    if norm == 0.0 or time == 0.0:
        return vector.copy()
    krylov_dim = min(max_krylov, vector.size)
    basis = np.zeros((krylov_dim + 1, vector.size), dtype=np.complex128)
    hessenberg = np.zeros(
        (krylov_dim + 1, krylov_dim), dtype=np.complex128)
    basis[0] = vector / norm

    for column in range(krylov_dim):
        work = matvec_fn(basis[column])
        for row in range(column + 1):
            value = np.vdot(basis[row], work)
            hessenberg[row, column] = value
            work -= value * basis[row]
        for row in range(column + 1):
            correction = np.vdot(basis[row], work)
            hessenberg[row, column] += correction
            work -= correction * basis[row]
        next_norm = float(np.linalg.norm(work))
        hessenberg[column + 1, column] = next_norm
        dimension = column + 1
        coefficients = scipy.linalg.expm(
            time * hessenberg[:dimension, :dimension])[:, 0]
        if next_norm <= tol:
            return norm * (coefficients @ basis[:dimension])
        if abs(time) * next_norm * abs(coefficients[-1]) <= tol:
            return norm * (coefficients @ basis[:dimension])
        basis[column + 1] = work / next_norm

    coefficients = scipy.linalg.expm(
        time * hessenberg[:krylov_dim, :krylov_dim])[:, 0]
    return norm * (coefficients @ basis[:krylov_dim])


def _two_site_action(vector: np.ndarray, shape: Tuple[int, ...],
                     left: np.ndarray, right: np.ndarray,
                     first: np.ndarray, second: np.ndarray) -> np.ndarray:
    tensor = vector.reshape(shape)
    d = first.shape[2]
    if shape[1] == d * d:
        tensor = tensor.reshape(shape[0], d, d, d, d, shape[3])
        mpo_pair = np.einsum("wxis,xyjt->wyijst", first, second)
        tmp = np.einsum("awc,cfsgtd->awfsgtd", left, tensor)
        tmp = np.einsum("wyijst,awfsgtd->ayfijgd", mpo_pair, tmp)
        return np.einsum(
            "byd,ayfijgd->afigjb", right, tmp).reshape(-1)
    tmp = np.einsum("awc,cstd->awstd", left, tensor)
    tmp = np.einsum("wxis,awstd->axitd", first, tmp)
    tmp = np.einsum("xyjt,axitd->ayijd", second, tmp)
    return np.einsum("byd,ayijd->aijb", right, tmp).reshape(-1)


def _two_site_adjoint(vector: np.ndarray, shape: Tuple[int, ...],
                      left: np.ndarray, right: np.ndarray,
                      first: np.ndarray,
                      second: np.ndarray) -> np.ndarray:
    tensor = vector.reshape(shape)
    d = first.shape[2]
    if shape[1] == d * d:
        tensor = tensor.reshape(shape[0], d, d, d, d, shape[3])
        mpo_pair = np.einsum("wxis,xyjt->wyijst", first, second)
        tmp = np.einsum("byd,afigjb->ayfijgd", right.conj(), tensor)
        tmp = np.einsum(
            "wyijst,ayfijgd->awfsgtd", mpo_pair.conj(), tmp)
        return np.einsum(
            "awc,awfsgtd->cfsgtd", left.conj(), tmp).reshape(-1)
    tmp = np.einsum("byd,aijb->ayijd", right.conj(), tensor)
    tmp = np.einsum("xyjt,ayijd->axitd", second.conj(), tmp)
    tmp = np.einsum("wxis,axitd->awstd", first.conj(), tmp)
    return np.einsum("awc,awstd->cstd", left.conj(), tmp).reshape(-1)


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
    shape = theta.shape
    d = first.shape[2]
    if shape[1] == d * d:
        a_dim, _, _, b_dim = shape
        w_l = first.shape[0]
        w_r = second.shape[1]
        left_mat = left.reshape(a_dim * w_l, a_dim)
        right_mat = right.transpose(1, 2, 0).reshape(
            w_r * b_dim, b_dim)

        def matvec(vector: np.ndarray) -> np.ndarray:
            tmp = (left_mat @ vector.reshape(a_dim, -1)).reshape(
                a_dim, w_l, d, d, d, d, b_dim)
            tmp = np.einsum("wxis,awfsgtd->axfigtd", first, tmp)
            tmp = np.einsum("xyjt,axfigtd->afijgyd", second, tmp)
            tmp = tmp.transpose(0, 1, 2, 4, 3, 5, 6)
            return (tmp.reshape(-1, w_r * b_dim) @ right_mat).reshape(-1)

        return _arnoldi_expm(
            matvec, theta.reshape(-1), time,
            tol=1.0e-12, max_krylov=16).reshape(theta.shape)

    mpo_pair = np.einsum("wxis,xyjt->wyijst", first, second)

    def matvec(vector: np.ndarray) -> np.ndarray:
        tensor = vector.reshape(shape)
        tmp = np.einsum("awc,cstd->awstd", left, tensor)
        tmp = np.einsum("wyijst,awstd->ayijd", mpo_pair, tmp)
        return np.einsum("byd,ayijd->aijb", right, tmp).reshape(-1)

    return _arnoldi_expm(
        matvec, theta.reshape(-1), time,
        tol=1.0e-12, max_krylov=16).reshape(theta.shape)


def zero_site_operator(shape: Tuple[int, int], left: np.ndarray,
                       right: np.ndarray, mpo_bond: np.ndarray) -> LinearOperator:
    """Create the effective zero-site operator on an MPS bond."""
    size = int(np.prod(shape))

    def matvec(vector: np.ndarray) -> np.ndarray:
        tensor = vector.reshape(shape)
        tmp = np.einsum("awc,cd->awd", left, tensor)
        tmp = np.einsum("wx,awd->axd", mpo_bond, tmp)
        result = np.einsum("bxd,axd->ab", right, tmp)
        return result.reshape(-1)

    def rmatvec(vector: np.ndarray) -> np.ndarray:
        tensor = vector.reshape(shape)
        tmp = np.einsum("bxd,ab->axd", right.conj(), tensor)
        tmp = np.einsum("wx,axd->awd", mpo_bond.conj(), tmp)
        result = np.einsum("awc,awd->cd", left.conj(), tmp)
        return result.reshape(-1)

    return LinearOperator(
        (size, size), matvec=matvec, rmatvec=rmatvec,
        dtype=np.complex128)


def evolve_zero_site(matrix: np.ndarray, time: float, left: np.ndarray,
                     right: np.ndarray, mpo_bond: np.ndarray) -> np.ndarray:
    """Apply a matrix-free zero-site Krylov exponential."""
    operator = zero_site_operator(matrix.shape, left, right, mpo_bond)
    return _arnoldi_expm(
        operator.matvec, matrix.reshape(-1), time,
        tol=1.0e-12, max_krylov=16).reshape(matrix.shape)


def one_site_operator(shape: Tuple[int, int, int],
                      left: np.ndarray, right: np.ndarray,
                      mpo_tensor: np.ndarray) -> LinearOperator:
    """Create the effective one-site MPO LinearOperator."""
    size = int(np.prod(shape))

    def matvec(vector: np.ndarray) -> np.ndarray:
        tensor = vector.reshape(shape)
        d = mpo_tensor.shape[2]
        if shape[1] == d * d:
            a_dim, _, b_dim = shape
            w_l = mpo_tensor.shape[0]
            w_r = mpo_tensor.shape[1]
            left_mat = left.reshape(a_dim * w_l, a_dim)
            right_mat = right.transpose(1, 2, 0).reshape(
                w_r * b_dim, b_dim)
            tmp = (left_mat @ tensor.reshape(a_dim, d, d, b_dim)
                   .reshape(a_dim, -1)).reshape(
                       a_dim, w_l, d, d, b_dim)
            tmp = np.einsum("wxis,awfsd->axfid", mpo_tensor, tmp)
            tmp = tmp.transpose(0, 2, 3, 1, 4)
            return (tmp.reshape(-1, w_r * b_dim) @ right_mat).reshape(-1)
        tmp = np.einsum("awc,csd->awsd", left, tensor)
        tmp = np.einsum("wxis,awsd->axid", mpo_tensor, tmp)
        result = np.einsum("bxd,axid->aib", right, tmp)
        return result.reshape(-1)

    def rmatvec(vector: np.ndarray) -> np.ndarray:
        tensor = vector.reshape(shape)
        d = mpo_tensor.shape[2]
        if shape[1] == d * d:
            tensor = tensor.reshape(shape[0], d, d, shape[2])
            tmp = np.einsum("bxd,afib->axfid", right.conj(), tensor)
            tmp = np.einsum("wxis,axfid->awfsd", mpo_tensor.conj(), tmp)
            return np.einsum(
                "awc,awfsd->cfsd", left.conj(), tmp).reshape(-1)
        tmp = np.einsum("bxd,aib->axid", right.conj(), tensor)
        tmp = np.einsum("wxis,axid->awsd", mpo_tensor.conj(), tmp)
        result = np.einsum("awc,awsd->csd", left.conj(), tmp)
        return result.reshape(-1)

    return LinearOperator(
        (size, size), matvec=matvec, rmatvec=rmatvec,
        dtype=np.complex128)


def evolve_one_site(tensor: np.ndarray, time: float,
                    left: np.ndarray, right: np.ndarray,
                    mpo_tensor: np.ndarray) -> np.ndarray:
    """Apply a matrix-free one-site Krylov exponential."""
    operator = one_site_operator(tensor.shape, left, right, mpo_tensor)
    return _arnoldi_expm(
        operator.matvec, tensor.reshape(-1), time,
        tol=1.0e-12, max_krylov=16).reshape(tensor.shape)
