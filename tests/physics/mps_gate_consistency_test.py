"""Operator-level consistency checks for the cross-correlated MPS gates."""

import numpy as np

import oqupy
from oqupy.mps_cross_correlated import CrossCorrelatedMPS


def test_adjacent_influence_gate_matches_dense_contraction():
    correlation = oqupy.CustomCorrelations(lambda tau: 0.02 * np.exp(-tau))
    cross = oqupy.CustomCorrelations(lambda tau: 0.01 * np.exp(-tau))
    bath = oqupy.Bath(
        [0.5 * oqupy.operators.sigma("z"),
         0.5 * oqupy.operators.sigma("x")],
        [[correlation, cross], [cross, correlation]],
    )
    parameters = oqupy.TempoParameters(dt=0.1, dkmax=1, epsrel=1e-12)
    gate = oqupy.GeneralizedInfluenceTensor(bath, parameters).gate(1)
    rng = np.random.default_rng(7)
    older = rng.normal(size=4) + 1j * rng.normal(size=4)
    newest = rng.normal(size=4) + 1j * rng.normal(size=4)
    old_new_state = np.outer(older, newest)
    dense = gate.reshape(16, 16) @ old_new_state.reshape(-1)
    dense = dense.reshape(4, 4)

    mps = CrossCorrelatedMPS(4, 1e-15)
    mps.tensors = [older[None, :, None], newest[None, :, None]]
    mps.apply_adjacent_gate(0, gate.transpose(1, 0, 3, 2))
    result = np.einsum("lpr,rqs->lpqs", mps.tensors[0], mps.tensors[1])
    result = result[0, :, :, 0]
    np.testing.assert_allclose(result, dense, atol=1e-10)


def test_mps_history_readout_matches_dense_reference():
    correlation = oqupy.CustomCorrelations(lambda tau: 0.02 * np.exp(-tau))
    cross = oqupy.CustomCorrelations(lambda tau: 0.01 * np.exp(-tau))
    bath = oqupy.Bath(
        [0.5 * oqupy.operators.sigma("z"),
         0.5 * oqupy.operators.sigma("x")],
        [[correlation, cross], [cross, correlation]],
    )
    system = oqupy.System(0.5 * oqupy.operators.sigma("y"))
    parameters = oqupy.TempoParameters(dt=0.1, dkmax=2, epsrel=1e-12)
    initial_state = oqupy.operators.spin_dm("z+")
    mps_dynamics = oqupy.MPSPtTempo(
        system, bath, parameters, initial_state,
        max_steps=3, memory_cutoff=2).compute(progress_type="silent")
    dense_dynamics = oqupy.DensePtTempo(
        system, bath, parameters, initial_state, max_steps=3).compute()
    np.testing.assert_allclose(
        mps_dynamics.states, dense_dynamics.states, atol=1e-5)
