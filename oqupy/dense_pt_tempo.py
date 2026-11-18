"""Dense reference engine for cross-correlated noncommuting environments.

This module deliberately keeps the complete Liouville-space history and is
restricted to small numbers of time steps. It is intended as a correctness
reference for the compressed PT-TEMPO backend, not as a production solver.
"""

from typing import List, Optional, Tuple

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import expm

from oqupy.bath import Bath
from oqupy.dynamics import Dynamics
from oqupy.tempo import TempoParameters

_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(16)


def _integrate_tent_1d(func, a, b, c, d, t_k0, t_k1, t_m0, t_m1):
    result = 0.0j
    for x0, x1 in ((a, b), (b, c), (c, d)):
        if x1 - x0 <= 1.0e-15:
            continue
        samples = (0.5 * (x1 - x0) * _GL_NODES
                   + 0.5 * (x0 + x1))
        weights = 0.5 * (x1 - x0) * _GL_WEIGHTS
        try:
            values = np.asarray(func(samples), dtype=complex)
            if values.shape != samples.shape:
                raise TypeError
        except (TypeError, ValueError):
            values = np.array([func(float(sample)) for sample in samples],
                              dtype=complex)
        tent = (np.minimum(t_k1, t_m1 + samples)
                - np.maximum(t_k0, t_m0 + samples))
        result += np.dot(weights, values * tent)
    return result


class GeneralizedInfluenceTensor:
    """Build joint one-site and two-site influence gates."""

    def __init__(self, bath: Bath, parameters: TempoParameters):
        self._bath = bath
        self._parameters = parameters
        self._dimension = bath.dimension
        self._physical_dimension = self._dimension**2
        self._operators = bath.coupling_operators
        self._eta_cache = {}
        self._generator_cache = {}
        self._gate_cache = {}
        self._upper_triangle_cache = {}
        self._interval_eta_cache = {}
        self._correlation_splines = {}
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

    @staticmethod
    def _channel_key(correlation):
        """Return a stable key for equivalent correlation channels."""
        for attribute in ("correlation_function", "j_function"):
            function = getattr(correlation, attribute, None)
            if function is not None:
                function = getattr(function, "pyfunc", function)
                return (attribute, id(function))
        function = getattr(correlation, "correlation", None)
        owner = getattr(function, "__self__", None)
        if owner is not None:
            return ("correlation", id(owner))
        j_function = getattr(correlation, "j_function", None)
        if j_function is not None:
            return ("j_function", id(j_function))
        return ("correlation", id(function), id(correlation))

    def _interpolated_correlation(self, correlation, max_tau: float):
        """Return a cached vectorized spline for a scalar correlation."""
        key = self._channel_key(correlation)
        cached = self._correlation_splines.get(key)
        if cached is not None and max_tau <= cached[0]:
            return cached[1]
        function = getattr(correlation, "correlation", None)
        if not callable(function):
            return None
        max_tau = max(float(max_tau), np.finfo(float).eps)
        grid = np.linspace(0.0, max_tau, 513)
        try:
            values = np.asarray(function(grid), dtype=complex)
            if values.shape != grid.shape:
                raise TypeError
        except (TypeError, ValueError):
            values = np.array([function(float(value)) for value in grid],
                              dtype=complex)
        spline = CubicSpline(grid, values, extrapolate=False)
        self._correlation_splines[key] = (max_tau, spline)
        return spline

    def _upper_triangle(self, tau: float) -> np.ndarray:
        """Return the matrix of upper-triangle integrals at ``tau``."""
        if tau <= 0.0 or np.isclose(tau, 0.0):
            return np.zeros(
                (len(self._operators), len(self._operators)), dtype=complex)
        key = round(float(tau), 14)
        if key not in self._upper_triangle_cache:
            values = {}
            for row in self._bath.correlations_matrix:
                for correlation in row:
                    channel = self._channel_key(correlation)
                    if channel not in values:
                        values[channel] = correlation.correlation_2d_integral(
                            delta=key, time_1=0.0, time_2=None,
                            shape="upper-triangle")
            self._upper_triangle_cache[key] = np.asarray([
                [values[self._channel_key(correlation)] for correlation in row]
                for row in self._bath.correlations_matrix], dtype=complex)
        return self._upper_triangle_cache[key]

    def eta_interval(self, t_k0: float, t_k1: float,
                     t_m0: Optional[float] = None,
                     t_m1: Optional[float] = None) -> np.ndarray:
        """Return the influence integral for one interval pair."""
        if t_m0 is None or t_m1 is None:
            return self._upper_triangle(t_k1 - t_k0)
        if np.isclose(t_k0, t_m0) and np.isclose(t_k1, t_m1):
            return self._upper_triangle(t_k1 - t_k0)
        if t_k0 >= t_m1:
            widths = (round(t_k1 - t_k0, 12),
                      round(t_m1 - t_m0, 12),
                      round(t_k0 - t_m0, 12))
            if widths not in self._interval_eta_cache:
                values = []
                a = t_k0 - t_m1
                d = t_k1 - t_m0
                b = min(t_k0 - t_m0, t_k1 - t_m1)
                c = max(t_k0 - t_m0, t_k1 - t_m1)
                for row in self._bath.correlations_matrix:
                    value_row = []
                    for correlation in row:
                        function = self._interpolated_correlation(
                            correlation, d)
                        if not callable(function):
                            value_row = []
                            break
                        value_row.append(_integrate_tent_1d(
                            function, a, b, c, d,
                            t_k0, t_k1, t_m0, t_m1))
                    if not value_row:
                        values = []
                        break
                    values.append(value_row)
                if values:
                    self._interval_eta_cache[widths] = np.asarray(
                        values, dtype=complex)
            if widths in self._interval_eta_cache:
                return self._interval_eta_cache[widths]
        return (self._upper_triangle(t_k1 - t_m0)
                - self._upper_triangle(t_k0 - t_m0)
                - self._upper_triangle(t_k1 - t_m1)
                + self._upper_triangle(t_k0 - t_m1))

    def _generator_from_eta(self, eta: np.ndarray,
                            distance: int) -> np.ndarray:
        if distance == 0:
            generator = np.zeros(
                (self._physical_dimension, self._physical_dimension),
                dtype=complex)
            for index_i, commutator_i in enumerate(self._commutators):
                for index_j, commutator_j in enumerate(self._commutators):
                    generator += eta[index_i, index_j].real \
                        * commutator_i @ commutator_j
                    generator += 1j * eta[index_i, index_j].imag \
                        * commutator_i @ self._anticommutators[index_j]
            return generator
        generator = np.zeros(
            (self._physical_dimension**2, self._physical_dimension**2),
            dtype=complex)
        for index_i, commutator_i in enumerate(self._commutators):
            for index_j, commutator_j in enumerate(self._commutators):
                generator += eta[index_i, index_j].real \
                    * np.kron(commutator_i, commutator_j)
                generator += 1j * eta[index_i, index_j].imag \
                    * np.kron(commutator_i,
                              self._anticommutators[index_j])
        return generator.reshape(
            self._physical_dimension, self._physical_dimension,
            self._physical_dimension, self._physical_dimension)

    def generator_interval(self, t_k0: float, t_k1: float,
                           t_m0: Optional[float] = None,
                           t_m1: Optional[float] = None) -> np.ndarray:
        """Return the continuous generator for an interval pair."""
        distance = 0 if t_m0 is None or t_m1 is None else 1
        return self._generator_from_eta(
            self.eta_interval(t_k0, t_k1, t_m0, t_m1), distance)

    def gate_interval(self, t_k0: float, t_k1: float) -> np.ndarray:
        """Return the influence gate for one non-uniform interval."""
        generator = self.generator_interval(t_k0, t_k1)
        return expm(-generator)

    @property
    def physical_dimension(self) -> int:
        """Dimension of one vectorized density-matrix index."""
        return self._physical_dimension

    def _eta(self, distance: int) -> np.ndarray:
        if distance in self._eta_cache:
            return self._eta_cache[distance]
        if distance == 0:
            shape = "upper-triangle"
            time_1 = 0.0
            time_2 = None
        else:
            shape = "square"
            time_1 = distance * self._parameters.dt
            time_2 = None
        values = {}
        for row in self._bath.correlations_matrix:
            for correlation in row:
                channel = self._channel_key(correlation)
                if channel not in values:
                    values[channel] = correlation.correlation_2d_integral(
                        delta=self._parameters.dt,
                        time_1=time_1,
                        time_2=time_2,
                        shape=shape)
        eta = np.asarray([
            [values[self._channel_key(correlation)] for correlation in row]
            for row in self._bath.correlations_matrix], dtype=complex)
        self._eta_cache[distance] = eta
        return eta

    def gate(self, distance: int) -> np.ndarray:
        """Return a one-site or two-site influence gate.

        For ``distance == 0`` the result has shape ``(D**2, D**2)``.
        Otherwise it has shape ``(D**2, D**2, D**2, D**2)`` with the first
        two axes as output indices and the last two as input indices.
        """
        if distance in self._gate_cache:
            return self._gate_cache[distance]
        generator = self.generator(distance)
        if distance == 0:
            gate = expm(-generator)
        else:
            gate = expm(-generator.reshape(
            self._physical_dimension**2,
            self._physical_dimension**2)).reshape(
                self._physical_dimension,
                self._physical_dimension,
                self._physical_dimension,
                self._physical_dimension)
        self._gate_cache[distance] = gate
        return gate

    def generator(self, distance: int) -> np.ndarray:
        """Return the un-exponentiated influence generator."""
        if distance in self._generator_cache:
            return self._generator_cache[distance]
        generator = self._generator_from_eta(self._eta(distance), distance)
        self._generator_cache[distance] = generator
        return generator


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
        history = None
        states = [self._initial_state.copy()]
        previous_second_half = None

        for step in range(num_steps):
            first_half, second_half = propagators(step)
            if step == 0:
                history = first_half @ self._initial_state.reshape(-1)
                history = self._apply_one_site(
                    history, 0, influence.gate(0))
            else:
                history = np.einsum(
                    "...a,ba->...ab",
                    history, first_half @ previous_second_half)
                newest = step
                history = self._apply_one_site(
                    history, newest, influence.gate(0))
                memory = min(step, self._parameters.dkmax)
                for distance in range(1, memory + 1):
                    history = self._apply_two_site(
                        history,
                        newest,
                        newest - distance,
                        influence.gate(distance))
            readout = self._apply_one_site(history, step, second_half)
            state = readout.sum(axis=tuple(range(readout.ndim - 1)))
            state = state.reshape(self._bath.dimension, self._bath.dimension)
            states.append(state)
            previous_second_half = second_half

        times = self._start_time + np.arange(len(states)) * self._parameters.dt
        return Dynamics(times=list(times), states=states)
class ContinuousInfluenceTensor(GeneralizedInfluenceTensor):
    """Expose continuous influence generators for TDVP evolution.

    The generalized influence tensor stores ``expm(-G)``.  TDVP requires
    ``G`` itself, while retaining the same channel ordering and tensor-leg
    conventions.

    Parameters
    ----------
    bath: Bath
        Bath defining the coupling channels and correlations.
    parameters: TempoParameters
        TEMPO discretisation and memory parameters.
    """

    def _eta(self, distance: int) -> np.ndarray:
        if distance in self._eta_cache:
            return self._eta_cache[distance]
        if getattr(self._bath, "correlations_matrix", None) is not None:
            return super()._eta(distance)
        if distance == 0:
            shape = "upper-triangle"
            time_1 = 0.0
            time_2 = None
        else:
            shape = "square"
            time_1 = distance * self._parameters.dt
            time_2 = None
        eta = self._bath.correlations.correlation_2d_integral(
            delta=self._parameters.dt,
            time_1=time_1,
            time_2=time_2,
            shape=shape)
        result = np.array([[eta]], dtype=complex)
        self._eta_cache[distance] = result
        return result

    def generator(self, distance: int) -> np.ndarray:
        """Return the un-exponentiated influence generator."""
        return super().generator(distance)
