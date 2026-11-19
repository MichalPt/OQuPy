"""Left and right contractions for MPO--MPS TDVP sweeps."""

from typing import List

import numpy as np


class TdvpEnvironments:
    """Cache left and right MPO environments for an MPS.

    MPS tensors use ``(left, physical, right)`` and MPO tensors use
    ``(left, right, output, input)``.
    """

    def __init__(self, tensors: List[np.ndarray],
                 mpo: List[np.ndarray]) -> None:
        if len(tensors) != len(mpo):
            raise ValueError("MPS and MPO must have the same length")
        self.tensors = tensors
        self.mpo = mpo
        self.left = [None] * (len(tensors) + 1)
        self.right = [None] * (len(tensors) + 1)
        self.left[0] = np.zeros((1, mpo[0].shape[0], 1), dtype=complex)
        self.left[0][0, 0, 0] = 1.0
        self.right[len(tensors)] = np.zeros(
            (1, mpo[-1].shape[1], 1), dtype=complex)
        self.right[len(tensors)][0, 1, 0] = 1.0

    @staticmethod
    def _left_step(block: np.ndarray, tensor: np.ndarray,
                   operator: np.ndarray) -> np.ndarray:
        d = operator.shape[2]
        if tensor.shape[1] == d * d:
            reshaped = tensor.reshape(tensor.shape[0], d, d, tensor.shape[2])
            tmp = np.einsum("awb,bfsj->wafsj", block, reshaped)
            tmp = np.einsum("wafsj,wxts->axftj", tmp, operator)
            return np.einsum("axftj,afti->ixj", tmp, reshaped.conj())
        tmp = np.einsum("awb,bsj->wasj", block, tensor)
        tmp = np.einsum("wasj,wxts->axtj", tmp, operator)
        return np.einsum("axtj,ati->ixj", tmp, tensor.conj())

    @staticmethod
    def _right_step(block: np.ndarray, tensor: np.ndarray,
                    operator: np.ndarray) -> np.ndarray:
        d = operator.shape[2]
        if tensor.shape[1] == d * d:
            reshaped = tensor.reshape(tensor.shape[0], d, d, tensor.shape[2])
            tmp = np.einsum("bfsj,ixj->bfsxi", reshaped, block)
            tmp = np.einsum("bfsxi,wxts->bwfti", tmp, operator)
            return np.einsum("bwfti,afti->awb", tmp, reshaped.conj())
        tmp = np.einsum("bsj,ixj->bsxi", tensor, block)
        tmp = np.einsum("bsxi,wxts->bwti", tmp, operator)
        return np.einsum("bwti,ati->awb", tmp, tensor.conj())

    def build_right(self) -> None:
        """Build all right environments from right to left."""
        for index in range(len(self.tensors) - 1, -1, -1):
            self.right[index] = self._right_step(
                self.right[index + 1], self.tensors[index], self.mpo[index])

    def build_left(self) -> None:
        """Build all left environments from left to right."""
        for index in range(len(self.tensors)):
            self.left[index + 1] = self._left_step(
                self.left[index], self.tensors[index], self.mpo[index])

    def update_left(self, index: int) -> None:
        """Update the environment immediately to the right of ``index``."""
        self.left[index + 1] = self._left_step(
            self.left[index], self.tensors[index], self.mpo[index])

    def update_right(self, index: int) -> None:
        """Update the environment immediately to the left of ``index``."""
        self.right[index] = self._right_step(
            self.right[index + 1], self.tensors[index], self.mpo[index])
