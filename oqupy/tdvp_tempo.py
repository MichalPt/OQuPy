"""Public TDVP-TEMPO interface."""

from typing import Optional, Text

import numpy as np

from oqupy.base_api import BaseAPIClass
from oqupy.bath import Bath
from oqupy.config import LIOUVILLIAN_EPSREL, LIOUVILLIAN_SUBDIV_LIMIT
from oqupy.dynamics import Dynamics
from oqupy.tempo import TempoParameters
from oqupy.backends.tdvp_tempo_backend import TdvpTempoBackend


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
        if parameters.dkmax is None:
            raise ValueError("TdvpTempo requires an explicit dkmax")
        state = np.asarray(initial_state, dtype=complex)
        if state.shape != (system.dimension, system.dimension):
            raise ValueError("initial_state has incompatible system dimension")
        super().__init__(name, description)
        self._backend = TdvpTempoBackend(
            system, bath, parameters, state, start_time)

    def compute(self, end_time: float) -> Dynamics:
        """Compute and return dynamics through ``end_time``."""
        return self._backend.compute(end_time)

    def get_dynamics(self) -> Optional[Dynamics]:
        """Return the currently computed dynamics, if available."""
        return self._backend._dynamics
