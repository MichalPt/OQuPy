"""Public TDVP-TEMPO interface."""

from typing import Optional, Text

import numpy as np

from oqupy.base_api import BaseAPIClass
from oqupy.bath import Bath
from oqupy.config import LIOUVILLIAN_EPSREL, LIOUVILLIAN_SUBDIV_LIMIT
from oqupy.dynamics import Dynamics
from oqupy.tempo import TempoParameters
from oqupy.backends.tdvp_tempo_backend import TdvpTempoBackend


def make_time_grid(start_time: float, end_time: float,
                   dt_initial: float, dt_max: float,
                   t_initial: float = 0.0,
                   growth_factor: float = 1.2) -> np.ndarray:
    """Construct a smoothly expanding, strictly increasing time grid."""
    if end_time <= start_time:
        raise ValueError("end_time must be greater than start_time")
    if dt_initial <= 0.0 or dt_max < dt_initial:
        raise ValueError("require 0 < dt_initial <= dt_max")
    if t_initial < 0.0:
        raise ValueError("t_initial must be non-negative")
    if growth_factor <= 1.0:
        raise ValueError("growth_factor must be greater than one")

    end = float(end_time)
    transition = min(end, start_time + t_initial)
    grid = [float(start_time)]
    current = float(start_time)
    while current < transition and not np.isclose(current, transition):
        current += min(dt_initial, transition - current)
        grid.append(current)

    step = float(dt_initial)
    while current < end and not np.isclose(current, end):
        step = min(step * growth_factor, dt_max)
        current += min(step, end - current)
        grid.append(current)
    grid[-1] = end
    return np.asarray(grid, dtype=float)


class TdvpTempoParameters(TempoParameters):
    """Parameters controlling continuous TDVP-TEMPO propagation.

    Parameters
    ----------
    tdvp_method: str
        Either ``"1-site"`` or ``"2-site"``.
    max_bond_dimension: int, optional
        Maximum adaptive history bond dimension.
    sweep_tolerance: float
        Relative tolerance used during local TDVP compression.
    """

    def __init__(
            self,
            dt: float,
            epsrel: float,
            tcut: Optional[float] = None,
            dkmax: Optional[int] = None,
            add_correlation_time: Optional[float] = None,
            liouvillian_epsrel: float = LIOUVILLIAN_EPSREL,
            liouvillian_subdiv_limit: int = LIOUVILLIAN_SUBDIV_LIMIT,
            tdvp_method: str = "2-site",
            max_bond_dimension: Optional[int] = None,
            sweep_tolerance: float = 1.0e-8,
            name: Optional[Text] = None,
            description: Optional[Text] = None) -> None:
        super().__init__(
            dt, epsrel, tcut, dkmax, add_correlation_time,
            liouvillian_epsrel, liouvillian_subdiv_limit, name, description)
        if tdvp_method not in ("1-site", "2-site"):
            raise ValueError("tdvp_method must be '1-site' or '2-site'")
        if max_bond_dimension is not None and max_bond_dimension < 1:
            raise ValueError("max_bond_dimension must be positive")
        if sweep_tolerance < 0.0:
            raise ValueError("sweep_tolerance must be non-negative")
        self._tdvp_method = tdvp_method
        self._max_bond_dimension = max_bond_dimension
        self._sweep_tolerance = sweep_tolerance

    @property
    def tdvp_method(self) -> str:
        return self._tdvp_method

    @property
    def max_bond_dimension(self) -> Optional[int]:
        return self._max_bond_dimension

    @property
    def sweep_tolerance(self) -> float:
        return self._sweep_tolerance


class TdvpTempo(BaseAPIClass):
    """TDVP-TEMPO evolution for commuting and cross-correlated baths."""

    def __init__(
            self,
            system,
            bath,
            parameters: TdvpTempoParameters,
            initial_state: np.ndarray,
            start_time: float = 0.0,
            name: Optional[Text] = None,
            description: Optional[Text] = None) -> None:
        if not isinstance(parameters, TdvpTempoParameters):
            raise TypeError("parameters must be TdvpTempoParameters")
        if not isinstance(bath, Bath):
            raise TypeError("bath must be a Bath instance")
        if parameters.dkmax is None and parameters.tcut is None:
            raise ValueError("TdvpTempo requires dkmax or tcut")
        state = np.asarray(initial_state, dtype=complex)
        if state.shape != (system.dimension, system.dimension):
            raise ValueError("initial_state has incompatible system dimension")
        super().__init__(name, description)
        self._backend = TdvpTempoBackend(
            system, bath, parameters, state, start_time)

    def compute(self, end_time: Optional[float] = None,
                time_grid: Optional[np.ndarray] = None,
                progress_type: Optional[str] = None) -> Dynamics:
        """Compute dynamics on a uniform or explicitly supplied time grid.

        Parameters
        ----------
        end_time: float, optional
            End point for the existing uniform grid.
        time_grid: numpy.ndarray, optional
            Strictly increasing absolute time points beginning at
            ``start_time``.  If supplied, this takes precedence over
            ``end_time``.
        progress_type: str, optional
            Progress-bar type passed to :func:`oqupy.util.get_progress`.
        """
        return self._backend.compute(end_time, time_grid, progress_type)

    def get_dynamics(self) -> Optional[Dynamics]:
        """Return the currently computed dynamics, if available."""
        return self._backend._dynamics
