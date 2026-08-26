"""Tests for the dense noncommuting cross-correlation reference engine."""

import numpy as np

import oqupy


def _zero_cross_bath():
    zero = oqupy.CustomCorrelations(lambda tau: 0.0)
    operators = [0.5 * oqupy.operators.sigma("z"),
                 0.5 * oqupy.operators.sigma("x")]
    return oqupy.Bath(operators, [[zero, zero], [zero, zero]])


def test_dense_reference_zero_bath():
    bath = _zero_cross_bath()
    parameters = oqupy.TempoParameters(dt=0.1, dkmax=3, epsrel=1e-10)
    dynamics = oqupy.DensePtTempo(
        oqupy.System(0.5 * oqupy.operators.sigma("y")),
        bath,
        parameters,
        oqupy.operators.spin_dm("z+"),
        max_steps=3,
    ).compute()
    np.testing.assert_allclose(
        np.trace(dynamics.states, axis1=1, axis2=2), 1.0)
    expected = oqupy.compute_dynamics(
        system=oqupy.System(0.5 * oqupy.operators.sigma("y")),
        initial_state=oqupy.operators.spin_dm("z+"),
        dt=0.1,
        num_steps=3,
    )
    np.testing.assert_allclose(dynamics.states, expected.states)


def test_generalized_influence_gate_shape():
    bath = _zero_cross_bath()
    parameters = oqupy.TempoParameters(dt=0.1, dkmax=2, epsrel=1e-6)
    tensor = oqupy.GeneralizedInfluenceTensor(bath, parameters)
    assert tensor.gate(0).shape == (4, 4)
    assert tensor.gate(1).shape == (4, 4, 4, 4)


def test_dense_reference_cross_correlations_preserve_trace():
    correlation = oqupy.CustomCorrelations(
        lambda tau: 0.05 * np.exp(-tau))
    bath = oqupy.Bath(
        [0.5 * oqupy.operators.sigma("z"),
         0.5 * oqupy.operators.sigma("x")],
        [[correlation, correlation], [correlation, correlation]],
    )
    parameters = oqupy.TempoParameters(dt=0.1, dkmax=2, epsrel=1e-6)
    dynamics = oqupy.DensePtTempo(
        oqupy.System(0.5 * oqupy.operators.sigma("y")),
        bath,
        parameters,
        oqupy.operators.spin_dm("z+"),
        max_steps=2,
    ).compute()
    np.testing.assert_allclose(
        np.trace(dynamics.states, axis1=1, axis2=2), 1.0, atol=1e-12)
