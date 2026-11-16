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


def _compare_with_tempo(coupling_operator):
    correlations = oqupy.CustomCorrelations(
        lambda tau: 0.02 * np.exp(-tau))
    bath = oqupy.Bath(0.5 * coupling_operator, correlations)
    system = oqupy.System(0.5 * oqupy.operators.sigma("z"))
    initial_state = oqupy.operators.spin_dm("z+")
    parameters = oqupy.TdvpTempoParameters(
        dt=0.1, epsrel=1e-10, dkmax=2, max_bond_dimension=32)
    tdvp = oqupy.TdvpTempo(
        system, bath, parameters, initial_state).compute(0.2)
    tempo_parameters = oqupy.TempoParameters(
        dt=0.1, epsrel=1e-10, dkmax=2)
    tempo = oqupy.Tempo(
        system, bath, tempo_parameters, initial_state,
        start_time=0.0).compute(0.2)
    return np.max(np.abs(tdvp.states - tempo.states))


def test_tdvp_matches_original_tempo_for_diagonal_coupling():
    assert _compare_with_tempo(oqupy.operators.sigma("z")) < 1e-10


def test_tdvp_matches_original_tempo_for_nondiagonal_coupling():
    assert _compare_with_tempo(oqupy.operators.sigma("x")) < 1e-10


def test_tdvp_cross_correlated_multistep_evolution_is_finite():
    bath = _cross_correlated_bath()
    system = oqupy.System(0.5 * oqupy.operators.sigma("z"))
    initial_state = oqupy.operators.spin_dm("z+")
    parameters = oqupy.TdvpTempoParameters(
        dt=0.1, epsrel=1e-10, dkmax=2, max_bond_dimension=32)

    dynamics = oqupy.TdvpTempo(
        system, bath, parameters, initial_state).compute(0.2)

    assert np.all(np.isfinite(dynamics.states))


def test_midpoint_references_agree_for_cross_correlated_bath():
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

    assert np.max(np.abs(tdvp.states - dense.states)) < 1e-12
    assert np.max(np.abs(tdvp.states - mps.states)) < 1e-12


def test_tdvp_numerical_correlations_match_original_tempo():
    correlations = oqupy.CustomCorrelations(
        lambda tau: 0.02 * np.exp(-tau))
    bath = oqupy.Bath(
        0.5 * oqupy.operators.sigma("x"), correlations)
    system = oqupy.System(0.5 * oqupy.operators.sigma("z"))
    initial_state = oqupy.operators.spin_dm("z+")
    parameters = oqupy.TdvpTempoParameters(
        dt=0.1, epsrel=1e-10, dkmax=2, max_bond_dimension=32)
    tdvp = oqupy.TdvpTempo(
        system, bath, parameters, initial_state).compute(0.2)
    tempo = oqupy.Tempo(
        system, bath,
        oqupy.TempoParameters(dt=0.1, epsrel=1e-10, dkmax=2),
        initial_state,
        start_time=0.0).compute(0.2)

    assert np.max(np.abs(tdvp.states - tempo.states)) < 1e-10
