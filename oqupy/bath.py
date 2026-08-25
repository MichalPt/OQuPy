# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Module on physical information on the bath and its coupling to the system.
"""
from typing import Optional, Text
from copy import copy

import numpy as np
from numpy import ndarray

from oqupy.config import NpDtype
from oqupy.config import DEFAULT_TOLERANCE_DEGENERACY
from oqupy.bath_correlations import BaseCorrelations
from oqupy.base_api import BaseAPIClass
from oqupy.operators import commutator, acommutator


class Bath(BaseAPIClass):
    """
    Represents the bath degrees of freedom with a specific coupling operator
    (to the system) and a specific auto-correlation function.

    Parameters
    ----------
    coupling_operator: np.ndarray
        The system operator to which the bath couples.
    correlations: BaseCorrelations
        The bath's auto correlation function.
    name: str
        An optional name for the bath.
    description: str
        An optional description of the bath.
    """
    def __init__(
            self,
            coupling_operator: ndarray,
            correlations: BaseCorrelations,
            name: Optional[Text] = None,
            description: Optional[Text] = None) -> None:
        """Creates a Bath object. """
        if isinstance(coupling_operator, (list, tuple)) and \
                np.asarray(coupling_operator, dtype=object).ndim == 3:
            try:
                operators = [np.array(operator, dtype=NpDtype)
                             for operator in coupling_operator]
            except Exception as error:
                raise AssertionError("Coupling operator must be numpy array") \
                    from error
        else:
            try:
                operators = [np.array(coupling_operator, dtype=NpDtype)]
            except Exception as error:
                raise AssertionError("Coupling operator must be numpy array") \
                    from error
        assert operators, "At least one coupling operator is required."
        for operator in operators:
            assert len(operator.shape) == 2, \
                "Coupling operator must be a matrix."
            assert operator.shape[0] == operator.shape[1], \
                "Coupling operator must be a square matrix."
            assert np.allclose(operator.conjugate().T, operator), \
                "Coupling operator must be a hermitian matrix."
        self._dimension = operators[0].shape[0]
        assert all(operator.shape == operators[0].shape for operator in operators), \
            "All coupling operators must have the same dimension."
        self._original_coupling_operators = [operator.copy()
                                             for operator in operators]

        # diagonalise the coupling operator
        if len(operators) == 1 and np.allclose(
                np.diag(operators[0].diagonal()), operators[0]):
            v = np.identity(self._dimension)
        else:
            _, v = np.linalg.eigh(operators[0])
        diagonal_operators = [v.conjugate().T @ operator @ v
                              for operator in operators]
        self._commuting_channels = all(
            np.allclose(operator, np.diag(operator.diagonal()))
            for operator in diagonal_operators)
        if self._commuting_channels:
            self._coupling_operators = [np.diag(operator.diagonal())
                                        for operator in diagonal_operators]
        else:
            self._coupling_operators = self._original_coupling_operators
            self._unitary = np.identity(self._dimension)
        self._coupling_operator = self._coupling_operators[0]
        self._unitary = v

        # identify degeneracies in eigensystem of coupling operator
        coupling_comm = np.array([
            commutator(operator).diagonal()
            for operator in self._coupling_operators])
        coupling_acomm = np.array([
            acommutator(operator).diagonal()
            for operator in self._coupling_operators])
        self._coupling_comm = coupling_comm[0] if len(operators) == 1 \
            else coupling_comm
        self._coupling_acomm = coupling_acomm[0] if len(operators) == 1 \
            else coupling_acomm

        self._north_degeneracy_map = _row_degeneracy([self._coupling_comm,
                                                      self._coupling_acomm])
        self._west_degeneracy_map = _row_degeneracy([self._coupling_comm])

        # input check for correlations.
        if isinstance(correlations, BaseCorrelations):
            assert len(operators) == 1, \
                "Multiple coupling operators require a correlation matrix."
            correlation_matrix = [[correlations]]
        else:
            correlation_matrix = [list(row) for row in correlations]
            assert len(correlation_matrix) == len(operators) and all(
                len(row) == len(operators) for row in correlation_matrix), \
                "Correlations must be a square matrix matching operators."
        assert all(isinstance(correlation, BaseCorrelations)
                   for row in correlation_matrix for correlation in row), \
            "Correlations must contain BaseCorrelations instances."
        self._correlations_matrix = [
            [copy(correlation) for correlation in row]
            for row in correlation_matrix]
        self._correlations = self._correlations_matrix[0][0]

        super().__init__(name, description)

    def __str__(self) -> Text:
        ret = []
        ret.append(super().__str__())
        ret.append("  dimension     = {} \n".format(self.dimension))
        ret.append("  correlations  = {} \n".format(self.correlations.name))
        return "".join(ret)

    @property
    def coupling_operator(self) -> np.ndarray:
        """The diagonalised system operator to which the bath couples. """
        return self._coupling_operator.copy()

    @property
    def coupling_operators(self) -> list:
        """The system coupling operators for all channels."""
        if self._commuting_channels:
            return [operator.copy() for operator in self._coupling_operators]
        return [operator.copy() for operator in self._original_coupling_operators]

    @property
    def commuting_channels(self) -> bool:
        """Whether all multi-channel coupling operators commute."""
        return self._commuting_channels

    @property
    def unitary_transform(self) -> np.ndarray:
        """The unitary that makes the coupling operator diagonal. """
        return self._unitary.copy()

    @property
    def dimension(self) -> np.ndarray:
        """Hilbert space dimension of the coupling operator. """
        return copy(self._dimension)

    @property
    def correlations(self) -> BaseCorrelations:
        """The correlations of the bath. """
        return copy(self._correlations)

    @property
    def correlations_matrix(self) -> list:
        """The correlation-function matrix for all coupling channels."""
        return [[copy(correlation) for correlation in row]
                for row in self._correlations_matrix]

    @property
    def coupling_acomm(self) -> np.ndarray:
        """Diagonal elements of the anti-commutator of the coupling
        operator. """
        return self._coupling_acomm.copy()

    @property
    def coupling_comm(self) -> np.ndarray:
        """Diagonal elements of the commutator of the coupling
        operator. """
        return self._coupling_comm.copy()

    @property
    def north_degeneracy_map(self) -> np.ndarray:
        """Map to minimal set of indices for influence tensors in
        north-south direction according to simultaneous degeneracies in
        sums & differences of eigenvalues of coupling operator (minimal
        dimension is number of unique values in this map).
        Used by a Tempo computation if unique==True only. """
        return copy(self._north_degeneracy_map)

    @property
    def west_degeneracy_map(self) -> np.ndarray:
        """Map to minimal set of indices for influence tensors in
        west-east direction according to degeneracies in sums of
        eigenvalues of coupling operator (minimal dimension is number
        of unique values in this map).
        Used by a Tempo computation if unique==True only. """
        return copy(self._west_degeneracy_map)


def _row_degeneracy(matrix):
    """Finds the row degeneracy of matrix. Returns array of
    indices mapping full space to non-degenerate rows (repeated
    indices indicate row degeneracy in the original matrix)."""
    mat = np.array(matrix).round(decimals=DEFAULT_TOLERANCE_DEGENERACY)
    return_map = np.unique(mat.T,return_inverse=True,axis=0)[1]
    return return_map
