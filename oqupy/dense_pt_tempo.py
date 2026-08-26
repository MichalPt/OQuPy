"""Dense reference engine for cross-correlated noncommuting environments.

This module deliberately keeps the complete Liouville-space history and is
restricted to small numbers of time steps. It is intended as a correctness
reference for the compressed PT-TEMPO backend, not as a production solver.
"""

from typing import List, Optional, Tuple

import numpy as np
from scipy.linalg import expm

from oqupy.bath import Bath
from oqupy.dynamics import Dynamics
from oqupy.tempo import TempoParameters


class GeneralizedInfluenceTensor:
    """Build joint one-site and two-site influence gates."""

    def __init__(self, bath: Bath, parameters: TempoParameters):
        self._bath = bath
        self._parameters = parameters
        self._dimension = bath.dimension
        self._physical_dimension = self._dimension**2
        self._operators = bath.coupling_operators
        self._commutators = []
        self._anticommutators = []
        identity = np.identity(self._dimension)
        for operator in self._operators:
            self._commutators.append(
                np.kron(operator, identity) -
                np.kron(identity, operator.T))
            self._anticommutators.append(
                np.kron(operator, identity) +
                np.kron(identity, operator.T))

    @property
    def physical_dimension(self) -> int:
        """Dimension of one vectorized density-matrix index."""
        return self._physical_dimension

    def _eta(self, distance: int) -> np.ndarray:
        if distance == 0:
            shape = "upper-triangle"
            time_1 = 0.0
            time_2 = None
        else:
            shape = "square"
            time_1 = distance * self._parameters.dt
            time_2 = None
        return np.array([
            [correlation.correlation_2d_integral(
                delta=self._parameters.dt,
                time_1=time_1,
                time_2=time_2,
                shape=shape)
             for correlation in row]
            for row in self._bath.correlations_matrix],
            dtype=complex,
        )

    def gate(self, distance: int) -> np.ndarray:
        """Return a one-site or two-site influence gate.

        For ``distance == 0`` the result has shape ``(D**2, D**2)``.
        Otherwise it has shape ``(D**2, D**2, D**2, D**2)`` with the first
        two axes as output indices and the last two as input indices.
        """
        eta = self._eta(distance)
        if distance == 0:
            generator = np.zeros((self._physical_dimension,
                                  self._physical_dimension), dtype=complex)
            for index_i, commutator_i in enumerate(self._commutators):
                for index_j, commutator_j in enumerate(self._commutators):
                    generator += eta[index_i, index_j].real \
                        * commutator_i @ commutator_j
                    generator += 1j * eta[index_i, index_j].imag \
                        * commutator_i @ self._anticommutators[index_j]
            return expm(-generator)

        generator = np.zeros((self._physical_dimension**2,
                              self._physical_dimension**2), dtype=complex)
        for index_i, commutator_i in enumerate(self._commutators):
            for index_j, commutator_j in enumerate(self._commutators):
                generator += eta[index_i, index_j].real \
                    * np.kron(commutator_i, commutator_j)
                generator += 1j * eta[index_i, index_j].imag \
                    * np.kron(commutator_i,
                              self._anticommutators[index_j])
        return expm(-generator).reshape(
            self._physical_dimension,
            self._physical_dimension,
            self._physical_dimension,
            self._physical_dimension)


class DensePtTempo:
    """Uncompressed finite-history cross-correlated PT-TEMPO reference."""

    def __init__(self, system, bath: Bath, parameters: TempoParameters,
                 initial_state: np.ndarray, start_time: float = 0.0,
                 max_steps: int = 10):
        if not isinstance(bath, Bath):
            raise TypeError("bath must be a Bath instance")
        if not isinstance(parameters, TempoParameters):
            raise TypeError("parameters must be TempoParameters")
        if bath.commuting_channels:
            raise ValueError(
                "DensePtTempo is intended for noncommuting channel tests.")
        if parameters.dkmax is None:
            raise ValueError("DensePtTempo requires an explicit dkmax.")
        self._system = system
        self._bath = bath
        self._parameters = parameters
        self._initial_state = np.asarray(initial_state, dtype=complex)
        self._start_time = float(start_time)
        self._max_steps = int(max_steps)

    @staticmethod
    def _apply_one_site(history, site, operator):
        history = np.moveaxis(history, site, 0)
        shape = history.shape
        transformed = operator @ history.reshape(shape[0], -1)
        return np.moveaxis(transformed.reshape(shape), 0, site)

    @staticmethod
    def _apply_two_site(history, first_site, second_site, gate):
        history = np.moveaxis(history, (first_site, second_site), (0, 1))
        shape = history.shape
        transformed = gate.reshape(shape[0] * shape[1],
                                   shape[0] * shape[1]) \
            @ history.reshape(shape[0] * shape[1], -1)
        history = transformed.reshape(shape)
        return np.moveaxis(history, (0, 1), (first_site, second_site))

    def compute(self) -> Dynamics:
        """Compute at most ``max_steps`` steps without tensor compression."""
        num_steps = self._max_steps
        propagators = self._system.get_propagators(
            self._parameters.dt, self._start_time, None, None)
        influence = GeneralizedInfluenceTensor(self._bath, self._parameters)
        history = self._initial_state.reshape(self._bath.dimension**2)
        states = [self._initial_state.copy()]

        for step in range(num_steps):
            first_half, second_half = propagators(step)
            history = np.einsum("...a,ba->...ab", history, first_half)
            history = self._apply_one_site(
                history, step + 1, influence.gate(0))
            memory = min(step + 1, self._parameters.dkmax)
            for distance in range(1, memory + 1):
                history = self._apply_two_site(
                    history,
                    step + 1,
                    step + 1 - distance,
                    influence.gate(distance),
                )
            history = self._apply_one_site(
                history, step + 1, second_half)
            state = history.sum(axis=tuple(range(history.ndim - 1)))
            state = state.reshape(self._bath.dimension, self._bath.dimension)
            states.append(state)

        times = self._start_time + np.arange(len(states)) * self._parameters.dt
        return Dynamics(times=list(times), states=states)
