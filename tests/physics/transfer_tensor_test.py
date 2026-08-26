"""Tests for the transfer-tensor layer."""

import numpy as np

import oqupy


def _zero_cross_bath():
    zero = oqupy.CustomCorrelations(lambda tau: 0.0)
    operators = [0.5 * oqupy.operators.sigma("z"),
                 0.5 * oqupy.operators.sigma("x")]
    return oqupy.Bath(operators, [[zero, zero], [zero, zero]])


def test_transfer_tensor_zero_bath_long_time():
    zero = oqupy.CustomCorrelations(lambda tau: 0.0)
    bath = oqupy.Bath(
        [0.5 * oqupy.operators.sigma("z"),
         0.5 * oqupy.operators.sigma("x")],
        [[zero, zero], [zero, zero]],
    )
    system = oqupy.System(0.5 * oqupy.operators.sigma("y"))
    parameters = oqupy.TempoParameters(dt=0.1, dkmax=3, epsrel=1e-10)
    transfer_map = oqupy.TransferTensorMap(
        system, bath, parameters, learning_steps=3)
    dynamics = transfer_map.compute_dynamics(
        oqupy.operators.spin_dm("z+"), end_time=1.0)
    expected = oqupy.compute_dynamics(
        system=system,
        initial_state=oqupy.operators.spin_dm("z+"),
        dt=0.1,
        num_steps=10,
    )
    np.testing.assert_allclose(dynamics.states, expected.states, atol=1e-9)
    assert transfer_map.trace_preservation_error() < 1e-9


def test_transfer_tensor_properties():
    zero = oqupy.CustomCorrelations(lambda tau: 0.0)
    bath = oqupy.Bath(
        [0.5 * oqupy.operators.sigma("z"),
         0.5 * oqupy.operators.sigma("x")],
        [[zero, zero], [zero, zero]],
    )
    parameters = oqupy.TempoParameters(dt=0.1, dkmax=2, epsrel=1e-8)
    transfer_map = oqupy.TransferTensorMap(
        oqupy.System(0.5 * oqupy.operators.sigma("y")),
        bath,
        parameters,
        learning_steps=2,
    )
    tensors = transfer_map.learn()
    assert len(tensors) == 2
    assert all(tensor.shape == (4, 4) for tensor in tensors)
    assert len(transfer_map.dynamical_maps) == 3
    assert transfer_map.transfer_norms.shape == (2,)


def test_transfer_tensor_memory_arguments():
    bath = _zero_cross_bath()
    parameters = oqupy.TempoParameters(dt=0.1, dkmax=3, epsrel=1e-8)
    with np.testing.assert_raises(ValueError):
        oqupy.TransferTensorMap(
            oqupy.System(0.5 * oqupy.operators.sigma("y")),
            bath, parameters, learning_steps=2, memory_cutoff=3)
    with np.testing.assert_raises(ValueError):
        oqupy.TransferTensorMap(
            oqupy.System(0.5 * oqupy.operators.sigma("y")),
            bath, parameters, learning_steps=2, transfer_tolerance=0.0)
