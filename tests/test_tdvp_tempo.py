"""Regression tests for TDVP-TEMPO cross-correlated environments."""

import numpy as np

import oqupy


def _cross_correlated_bath():
    correlation = oqupy.CustomCorrelations(
        lambda tau: 0.02 * np.exp(-tau))
    operators = [
        0.5 * oqupy.operators.sigma("x"),
        0.5 * oqupy.operators.sigma("y"),
    ]
    return oqupy.Bath(
        operators,
        [[correlation, correlation], [correlation, correlation]],
    )


def test_tdvp_matches_dense_and_mps_references():
    bath = _cross_correlated_bath()
    system = oqupy.System(0.5 * oqupy.operators.sigma("z"))
    initial_state = oqupy.operators.spin_dm("z+")
    parameters = oqupy.TdvpTempoParameters(
        dt=0.1, epsrel=1e-10, dkmax=2, max_bond_dimension=32)

    tdvp = oqupy.TdvpTempo(
        system, bath, parameters, initial_state).compute(0.2)
    dense = oqupy.DensePtTempo(
        system, bath, parameters, initial_state, max_steps=2).compute()
    mps = oqupy.MPSPtTempo(
        system, bath, parameters, initial_state, max_steps=2).compute()

    np.testing.assert_allclose(tdvp.states, dense.states, atol=1e-9)
    np.testing.assert_allclose(tdvp.states, mps.states, atol=1e-9)
