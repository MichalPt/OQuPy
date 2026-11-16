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


def test_tdvp_multichannel_commuting_bath_in_nondiagonal_basis():
    correlation = oqupy.CustomCorrelations(
        lambda tau: 0.02 * np.exp(-tau))
    bath = oqupy.Bath(
        [oqupy.operators.sigma("x"),
         0.5 * oqupy.operators.sigma("x")],
        [[correlation, correlation], [correlation, correlation]])

    assert bath.commuting_channels
    assert np.allclose(
        bath.coupling_operators[0],
        np.diag(np.diag(bath.coupling_operators[0])))
    assert np.allclose(
        bath.coupling_operators[1],
        np.diag(np.diag(bath.coupling_operators[1])))
    assert not np.allclose(bath.unitary_transform, np.eye(2))

    system = oqupy.System(0.5 * oqupy.operators.sigma("z"))
    initial_state = oqupy.operators.spin_dm("z+")
    parameters = oqupy.TdvpTempoParameters(
        dt=0.1, epsrel=1e-10, dkmax=2, max_bond_dimension=32)
    dynamics = oqupy.TdvpTempo(
        system, bath, parameters, initial_state).compute(0.2)

    assert np.all(np.isfinite(dynamics.states))


def test_tdvp_uniform_time_grid_matches_uniform_api():
    correlations = oqupy.CustomCorrelations(
        lambda tau: 0.02 * np.exp(-tau))
    bath = oqupy.Bath(0.5 * oqupy.operators.sigma("x"), correlations)
    system = oqupy.System(0.5 * oqupy.operators.sigma("z"))
    initial_state = oqupy.operators.spin_dm("z+")
    parameters = oqupy.TdvpTempoParameters(
        dt=0.1, epsrel=1e-10, dkmax=2, max_bond_dimension=32)
    tempo = oqupy.TdvpTempo(system, bath, parameters, initial_state)
    uniform = tempo.compute(0.5)
    grid = tempo.compute(time_grid=np.linspace(0.0, 0.5, 6))

    assert np.max(np.abs(uniform.states - grid.states)) < 1e-14


def test_tdvp_variable_time_grid_is_finite():
    correlations = oqupy.CustomCorrelations(
        lambda tau: 0.02 * np.exp(-tau))
    bath = oqupy.Bath(0.5 * oqupy.operators.sigma("x"), correlations)
    system = oqupy.System(0.5 * oqupy.operators.sigma("z"))
    initial_state = oqupy.operators.spin_dm("z+")
    parameters = oqupy.TdvpTempoParameters(
        dt=0.1, epsrel=1e-10, dkmax=3, max_bond_dimension=32)
    grid = np.array([0.0, 0.02, 0.04, 0.2, 0.3, 0.4, 0.5])
    dynamics = oqupy.TdvpTempo(
        system, bath, parameters, initial_state).compute(time_grid=grid)

    assert np.array_equal(dynamics.times, grid)
    assert np.all(np.isfinite(dynamics.states))


def test_tdvp_variable_grid_improves_coarse_grid_accuracy():
    correlations = oqupy.CustomCorrelations(
        lambda tau: 0.5 * np.exp(-4.0 * tau))
    bath = oqupy.Bath(0.5 * oqupy.operators.sigma("x"), correlations)
    system = oqupy.System(2.0 * oqupy.operators.sigma("z"))
    initial_state = oqupy.operators.spin_dm("z+")

    fine_parameters = oqupy.TdvpTempoParameters(
        dt=0.02, epsrel=1e-8, dkmax=3, max_bond_dimension=16)
    coarse_parameters = oqupy.TdvpTempoParameters(
        dt=0.1, epsrel=1e-8, dkmax=3, max_bond_dimension=16)
    fine = oqupy.TdvpTempo(
        system, bath, fine_parameters, initial_state).compute(1.0)
    coarse = oqupy.TdvpTempo(
        system, bath, coarse_parameters, initial_state).compute(1.0)
    grid = oqupy.make_time_grid(
        0.0, 1.0, 0.02, 0.1, t_initial=0.2, growth_factor=1.2)
    variable = oqupy.TdvpTempo(
        system, bath, coarse_parameters, initial_state).compute(
            time_grid=grid)

    fine_error = np.max(np.abs(variable.states[-1] - fine.states[-1]))
    coarse_error = np.max(np.abs(coarse.states[-1] - fine.states[-1]))
    assert fine_error < coarse_error
