"""Transfer Tensor Method built on the dense noncommuting TEMPO reference."""

from collections import deque
import warnings
from typing import List, Optional

import numpy as np

from oqupy.bath import Bath
from oqupy.dense_pt_tempo import DensePtTempo
from oqupy.dynamics import Dynamics
from oqupy.tempo import TempoParameters
from oqupy.util import get_progress


class TransferTensorMap:
    """Learn and apply transfer tensors for a small noncommuting bath model.

    The dynamical maps are learned from the complete canonical operator basis
    using :class:`DensePtTempo`. After learning, long-time propagation uses a
    rolling buffer and no longer stores the exponentially growing dense path
    history. This method is intended for time-independent systems and baths.

    Parameters
    ----------
    system: System
        Time-independent system used for learning and propagation.
    bath: Bath
        Noncommuting cross-correlated bath.
    parameters: TempoParameters
        Must contain an explicit ``dkmax`` at least as large as
        ``learning_steps``.
    learning_steps: int
        Number of dynamical maps to learn.
    memory_cutoff: Optional[int]
        Number of transfer tensors retained during long-time propagation.
        Defaults to ``learning_steps``. If supplied, it must not exceed
        ``learning_steps``.
    transfer_tolerance: Optional[float]
        Optional norm threshold for automatically selecting a shorter memory.
        The cutoff is the last tensor before the final consecutive tail of
        tensors below this threshold.
    """

    def __init__(self, system, bath: Bath, parameters: TempoParameters,
                 learning_steps: int = 10,
                 memory_cutoff: Optional[int] = None,
                 transfer_tolerance: Optional[float] = None):
        if not isinstance(bath, Bath):
            raise TypeError("bath must be a Bath instance")
        if not isinstance(parameters, TempoParameters):
            raise TypeError("parameters must be TempoParameters")
        if bath.commuting_channels:
            raise ValueError(
                "TransferTensorMap is intended for noncommuting channels.")
        if parameters.dkmax is None:
            raise ValueError("TransferTensorMap requires an explicit dkmax.")
        self._system = system
        self._bath = bath
        self._parameters = parameters
        self._learning_steps = int(learning_steps)
        if self._learning_steps < 1:
            raise ValueError("learning_steps must be positive.")
        if parameters.dkmax < self._learning_steps:
            raise ValueError("dkmax must be at least learning_steps.")
        self._memory_cutoff = memory_cutoff
        if memory_cutoff is not None and not 1 <= memory_cutoff <= self._learning_steps:
            raise ValueError("memory_cutoff must be between 1 and learning_steps.")
        if transfer_tolerance is not None and \
                (transfer_tolerance <= 0.0 or
                 not np.isfinite(transfer_tolerance)):
            raise ValueError("transfer_tolerance must be positive and finite.")
        self._transfer_tolerance = transfer_tolerance
        self._transfer_tensors = None
        self._dynamical_maps = None

    @property
    def learning_steps(self) -> int:
        """Number of learned time steps."""
        return self._learning_steps

    @property
    def transfer_tensors(self) -> Optional[List[np.ndarray]]:
        """Copies of the learned transfer tensors, or None before learning."""
        if self._transfer_tensors is None:
            return None
        return [tensor.copy() for tensor in self._transfer_tensors]

    @property
    def dynamical_maps(self) -> Optional[List[np.ndarray]]:
        """Copies of the learned dynamical maps, including the identity map."""
        if self._dynamical_maps is None:
            return None
        return [mapping.copy() for mapping in self._dynamical_maps]

    @property
    def transfer_norms(self) -> Optional[np.ndarray]:
        """Frobenius norms of the learned transfer tensors."""
        if self._transfer_tensors is None:
            return None
        return np.array([np.linalg.norm(tensor, ord="fro")
                         for tensor in self._transfer_tensors])

    def learn(self) -> List[np.ndarray]:
        """Learn dynamical maps and transfer tensors from canonical states."""
        dimension = self._bath.dimension
        liouville_dimension = dimension**2
        maps = [np.eye(liouville_dimension, dtype=complex)]
        for step in range(1, self._learning_steps + 1):
            maps.append(np.zeros((liouville_dimension, liouville_dimension),
                                 dtype=complex))

        for column in range(liouville_dimension):
            basis_state = np.zeros((dimension, dimension), dtype=complex)
            basis_state.flat[column] = 1.0
            dynamics = DensePtTempo(
                self._system,
                self._bath,
                self._parameters,
                basis_state,
                max_steps=self._learning_steps,
            ).compute()
            for step in range(1, self._learning_steps + 1):
                maps[step][:, column] = dynamics.states[step].flat

        tensors = []
        for step in range(1, self._learning_steps + 1):
            tensor = maps[step].copy()
            for previous in range(1, step):
                tensor -= tensors[step - previous - 1] @ maps[previous]
            tensors.append(tensor)

        self._dynamical_maps = maps
        self._transfer_tensors = tensors
        if self._transfer_tolerance is not None and \
            self._selected_memory() == self._learning_steps:
            warnings.warn(
            "Transfer tensors did not show a decayed tail below "
            "transfer_tolerance; TTM extrapolation may be unreliable.",
            UserWarning)
        return self.transfer_tensors

    def _selected_memory(self) -> int:
        if self._memory_cutoff is not None:
            return self._memory_cutoff
        if self._transfer_tolerance is None:
            return self._learning_steps
        norms = self.transfer_norms
        for index in range(len(norms)):
            if np.all(norms[index:] < self._transfer_tolerance):
                return max(1, index)
        return self._learning_steps

    def compute_dynamics(self, initial_state: np.ndarray, end_time: float,
                         start_time: float = 0.0,
                         progress_type: Optional[str] = None) -> Dynamics:
        """Propagate an initial state using learned transfer tensors."""
        if self._transfer_tensors is None:
            self.learn()
        initial_state = np.asarray(initial_state, dtype=complex)
        dimension = self._bath.dimension
        if initial_state.shape != (dimension, dimension):
            raise ValueError("initial_state must have shape (D, D).")
        dt = self._parameters.dt
        num_steps = int((float(end_time) - float(start_time)) / dt)
        if num_steps < 0:
            raise ValueError("end_time must not precede start_time.")

        memory = self._selected_memory()
        tensors = self._transfer_tensors[:memory]
        states = [initial_state.copy()]
        initial_vector = initial_state.reshape(-1)
        vector_states = deque([initial_vector],
                              maxlen=memory + 1)
        progress = get_progress(progress_type)(num_steps,
                                                "TTM propagation:")
        with progress as progress_bar:
            for step in range(1, num_steps + 1):
                if step <= self._learning_steps:
                    state_vector = self._dynamical_maps[step] \
                        @ initial_vector
                else:
                    state_vector = np.zeros(dimension**2, dtype=complex)
                    for lag, tensor in enumerate(tensors, start=1):
                        if lag <= len(vector_states):
                            state_vector += tensor @ vector_states[-lag]
                vector_states.append(state_vector)
                states.append(state_vector.reshape(dimension, dimension))
                progress_bar.update(step)

        times = float(start_time) + np.arange(num_steps + 1) * dt
        return Dynamics(times=list(times), states=states)

    def trace_preservation_error(self) -> float:
        """Return the maximum learned-map trace error."""
        if self._transfer_tensors is None:
            self.learn()
        identity_vector = np.eye(self._bath.dimension).reshape(-1)
        return max(np.linalg.norm(
            identity_vector.conj() @ mapping - identity_vector.conj())
            for mapping in self._dynamical_maps)
