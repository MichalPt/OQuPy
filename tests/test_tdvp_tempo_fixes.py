"""Focused regression tests for TDVP-TEMPO fixes."""

import numpy as np
import pytest

import oqupy
from oqupy.backends.tdvp_tempo_backend import (
    _apply_mpo, _gse_expand, _right_canonicalize)
from oqupy.backends.tdvp_environments import TdvpEnvironments
from oqupy.backends.tdvp_krylov import _arnoldi_expm
from oqupy.bath_correlations import BaseCorrelations
from oqupy.dense_pt_tempo import GeneralizedInfluenceTensor
from oqupy.mps_cross_correlated import CrossCorrelatedMPS


def _noncommuting_problem(method="2-site"):
    correlation = oqupy.CustomCorrelations(
        lambda tau: 0.02 * np.exp(-tau))
    bath = oqupy.Bath(
        [0.5 * oqupy.operators.sigma("z"),
         0.5 * oqupy.operators.sigma("x")],
        [[correlation, correlation], [correlation, correlation]])
    system = oqupy.System(0.5 * oqupy.operators.sigma("y"))
    parameters = oqupy.TdvpTempoParameters(
        dt=0.1, epsrel=1e-12, dkmax=5, max_bond_dimension=64,
        tdvp_method=method)
    return bath, system, parameters, oqupy.operators.spin_dm("z+")


def test_noncommuting_gse_has_no_lag_three_error_jump():
    bath, system, parameters, initial_state = _noncommuting_problem()
    tdvp = oqupy.TdvpTempo(
        system, bath, parameters, initial_state).compute(
            0.6, progress_type="silent")
    dense = oqupy.DensePtTempo(
        system, bath, parameters, initial_state, max_steps=6).compute()
    errors = [
        np.max(np.abs(tdvp.states[index] - dense.states[index]))
        for index in range(1, 7)]
    assert max(errors) < 1e-7
    assert errors[2] < 1e-7
    for state in tdvp.states:
        assert abs(np.trace(state) - 1.0) < 1e-10
        assert np.linalg.norm(state - state.conjugate().T) < 1e-10


def test_noncommuting_one_site_gse_expands_and_converges():
    bath, system, parameters, initial_state = _noncommuting_problem("1-site")
    tdvp = oqupy.TdvpTempo(
        system, bath, parameters, initial_state).compute(
            0.6, progress_type="silent")
    dense = oqupy.DensePtTempo(
        system, bath, parameters, initial_state, max_steps=6).compute()
    assert np.max(np.abs(tdvp.states - dense.states)) < 3e-7


def test_variable_grid_tcut_keeps_preceding_interval():
    correlation = oqupy.CustomCorrelations(
        lambda tau: 0.02 * np.exp(-tau))
    bath = oqupy.Bath(
        [0.5 * oqupy.operators.sigma("z"),
         0.5 * oqupy.operators.sigma("x")],
        [[correlation, correlation], [correlation, correlation]])
    parameters = oqupy.TdvpTempoParameters(
        dt=0.1, tcut=0.3, epsrel=1e-10, max_bond_dimension=32)
    grid = np.array([0.0, 0.1, 0.2, 0.5, 0.6])
    dynamics = oqupy.TdvpTempo(
        oqupy.System(0.5 * oqupy.operators.sigma("y")),
        bath, parameters, oqupy.operators.spin_dm("z+")).compute(
            time_grid=grid)
    assert np.array_equal(dynamics.times, grid)
    assert np.all(np.isfinite(dynamics.states))


def test_spectral_interpolation_respects_cutoff_type():
    for cutoff_type in ("hard", "exponential", "gaussian"):
        correlation = oqupy.CustomSD(
            lambda omega: 0.1 * omega,
            cutoff=3.0, cutoff_type=cutoff_type, temperature=0.5)
        bath = oqupy.Bath(0.5 * oqupy.operators.sigma("z"), correlation)
        tensor = GeneralizedInfluenceTensor(
            bath, oqupy.TempoParameters(dt=0.05, epsrel=1e-10, dkmax=2))
        spline = tensor._interpolated_correlation(correlation, 0.5)
        points = np.array([0.0, 0.1, 0.3, 0.5])
        assert np.allclose(
            spline(points), np.array([
                correlation.correlation(float(point)) for point in points]),
            rtol=2e-5,
            atol=2e-7)


def test_spectral_channel_cache_includes_parameters():
    j_function = lambda omega: 0.1 * omega
    first = oqupy.CustomSD(
        j_function, cutoff=1.0, cutoff_type="exponential", temperature=0.2)
    second = oqupy.CustomSD(
        j_function, cutoff=2.0, cutoff_type="exponential", temperature=0.2)
    assert GeneralizedInfluenceTensor._channel_key(first) != \
        GeneralizedInfluenceTensor._channel_key(second)


def test_noncommuting_bath_uses_identity_unitary():
    bath = oqupy.Bath(
        [oqupy.operators.sigma("x"), oqupy.operators.sigma("z")],
        [[oqupy.CustomCorrelations(lambda tau: np.exp(-tau))] * 2] * 2)
    assert np.allclose(bath.unitary_transform, np.eye(2))


def _dense_mps(tensors):
    state = tensors[0][0]
    for tensor in tensors[1:]:
        state = np.tensordot(state, tensor, axes=(-1, 0))
    return state[..., 0]


def _lift_operator(operator):
    dimension = operator.shape[0]
    return np.kron(np.eye(dimension), operator)


@pytest.mark.parametrize("length", [2, 3, 4, 5])
@pytest.mark.parametrize("superoperator_mode", [False, True])
def test_apply_mpo_matches_dense_boundary_projected_mpo(
        length, superoperator_mode):
    rng = np.random.default_rng(42 + length)
    base_dimension = 2
    physical = base_dimension**2 if superoperator_mode else base_dimension
    tensors = []
    for index in range(length):
        left = 1 if index == 0 else 2
        right = 1 if index == length - 1 else 2
        tensor = 0.1 * rng.normal(size=(left, physical, right))
        tensors.append(tensor + 1j * rng.normal(size=tensor.shape))

    mpo_bond = 3
    mpo = []
    for _ in range(length):
        operator = 0.1 * rng.normal(size=(mpo_bond, mpo_bond,
                                           base_dimension, base_dimension))
        mpo.append(operator + 1j * rng.normal(size=operator.shape))

    applied = _apply_mpo(tensors, mpo)
    actual = _dense_mps(applied).reshape(-1)
    state = _dense_mps(tensors).reshape(-1)
    expected = np.zeros_like(actual)
    for path in np.ndindex(*(mpo_bond,) * (length - 1)):
        mpo_path = (0,) + path + (1,)
        matrix = np.array([[1.0 + 0.0j]])
        for index in range(length):
            local = mpo[index][
                mpo_path[index], mpo_path[index + 1]]
            if superoperator_mode:
                local = _lift_operator(local)
            matrix = np.kron(matrix, local)
        expected += matrix @ state
    assert np.max(np.abs(actual - expected)) < 1e-12


def _random_right_canonical_mps(length=4):
    rng = np.random.default_rng(123)
    mps = CrossCorrelatedMPS(2, 1e-13, max_bond_dimension=None)
    mps.tensors = []
    for index in range(length):
        left = 1 if index == 0 else 2
        right = 1 if index == length - 1 else 2
        tensor = rng.normal(size=(left, 2, right))
        mps.tensors.append(tensor + 1j * rng.normal(size=tensor.shape))
    _right_canonicalize(mps.tensors)
    return mps


def _complex_mpo(length=4):
    rng = np.random.default_rng(321)
    mpo = []
    for index in range(length):
        tensor = rng.normal(size=(3, 3, 2, 2)).astype(complex)
        tensor += 1j * rng.normal(size=tensor.shape)
        mpo.append(tensor)
    return mpo


def test_gse_preserves_state_and_right_orthonormality():
    mps = _random_right_canonical_mps()
    before = _dense_mps(mps.tensors)
    _gse_expand(mps, _complex_mpo(), 1e-13, None)
    after = _dense_mps(mps.tensors)
    assert np.max(np.abs(before - after)) < 1e-13
    for tensor in mps.tensors[1:]:
        matrix = tensor.reshape(tensor.shape[0], -1)
        assert np.linalg.norm(
            matrix @ matrix.conj().T - np.eye(matrix.shape[0])) < 1e-12


def test_gse_identity_mpo_does_not_append_noise_vectors():
    mps = _random_right_canonical_mps()
    old_shapes = [tensor.shape for tensor in mps.tensors]
    mpo = [np.zeros((2, 2, 2, 2), dtype=complex)
           for _ in mps.tensors]
    mpo[0][0, 1] = np.eye(2)
    for tensor in mpo[1:]:
        tensor[1, 1] = np.eye(2)
    _gse_expand(mps, mpo, 0.0, None)
    assert [tensor.shape for tensor in mps.tensors] == old_shapes


def test_arnoldi_max_depth_warns_instead_of_recursing_forever():
    matrix = np.array([[0.0, 100.0], [-100.0, 0.0]], dtype=complex)
    with pytest.warns(RuntimeWarning, match="max_depth"):
        result = _arnoldi_expm(
            lambda vector: matrix @ vector, np.array([1.0, 0.0j]),
            1.0, tol=1e-30, max_krylov=1, max_depth=0)
    assert np.all(np.isfinite(result))


class _TabulatedCorrelations(BaseCorrelations):
    def __init__(self, table):
        self.table = np.asarray(table)
        super().__init__()

    def correlation(self, tau, epsrel=None, subdiv_limit=None):
        del epsrel, subdiv_limit
        return np.exp(-tau) * self.table[0]


def test_channel_key_fingerprints_custom_array_state():
    first = _TabulatedCorrelations([1.0, 2.0])
    second = _TabulatedCorrelations([1.0, 3.0])
    assert GeneralizedInfluenceTensor._channel_key(first) != \
        GeneralizedInfluenceTensor._channel_key(second)


def test_mps_tempo_transforms_commuting_non_diagonal_initial_state():
    correlation = oqupy.CustomCorrelations(
        lambda tau: 0.02 * np.exp(-tau))
    bath = oqupy.Bath(0.5 * oqupy.operators.sigma("x"), correlation)
    system = oqupy.System(0.5 * oqupy.operators.sigma("z"))
    initial_state = oqupy.operators.spin_dm("z+")
    parameters = oqupy.TdvpTempoParameters(
        dt=0.1, epsrel=1e-10, dkmax=3, max_bond_dimension=32)
    mps = oqupy.MPSPtTempo(
        system, bath, parameters, initial_state, max_steps=3).compute(
            progress_type="silent")
    tdvp = oqupy.TdvpTempo(
        system, bath, parameters, initial_state).compute(
            0.3, progress_type="silent")
    assert np.max(np.abs(mps.states - tdvp.states)) < 1e-6


def test_mixed_uniform_and_variable_tdvp_calls_reset_the_right_cache():
    correlation = oqupy.CustomCorrelations(
        lambda tau: 0.02 * np.exp(-tau))
    bath = oqupy.Bath(0.5 * oqupy.operators.sigma("x"), correlation)
    system = oqupy.System(0.5 * oqupy.operators.sigma("z"))
    initial_state = oqupy.operators.spin_dm("z+")
    parameters = oqupy.TdvpTempoParameters(
        dt=0.1, epsrel=1e-10, dkmax=3, max_bond_dimension=16)
    grid = np.array([0.0, 0.1, 0.25, 0.3])

    mixed = oqupy.TdvpTempo(
        system, bath, parameters, initial_state)
    mixed.compute(time_grid=grid, progress_type="silent")
    short = mixed.compute(0.2, progress_type="silent")
    long = mixed.compute(0.4, progress_type="silent")
    fresh = oqupy.TdvpTempo(
        system, bath, parameters, initial_state).compute(
            0.4, progress_type="silent")
    assert np.max(np.abs(short.states - fresh.states[:3])) < 1e-12
    assert np.max(np.abs(long.states - fresh.states)) < 1e-12

    mixed = oqupy.TdvpTempo(
        system, bath, parameters, initial_state)
    mixed.compute(0.2, progress_type="silent")
    mixed.compute(time_grid=grid, progress_type="silent")
    _, resumed = mixed._backend.compute_step()
    expected = oqupy.TdvpTempo(
        system, bath, parameters, initial_state).compute(
            0.1, progress_type="silent").states[-1]
    assert np.max(np.abs(resumed - expected)) < 1e-12


def test_splines_are_prepared_once_for_uniform_run():
    correlation = oqupy.CustomCorrelations(
        lambda tau: 0.02 * np.exp(-tau))
    bath = oqupy.Bath(0.5 * oqupy.operators.sigma("x"), correlation)
    parameters = oqupy.TdvpTempoParameters(
        dt=0.1, epsrel=1e-10, dkmax=4, max_bond_dimension=16)
    backend = oqupy.TdvpTempo(
        oqupy.System(0.5 * oqupy.operators.sigma("z")),
        bath, parameters, oqupy.operators.spin_dm("z+"))._backend
    backend.compute(0.4, progress_type="silent")
    assert len(backend._influence._correlation_splines) == 1
    assert backend._influence._correlation_splines[
        GeneralizedInfluenceTensor._channel_key(correlation)][0] >= 0.4


@pytest.mark.parametrize("superoperator_mode", [False, True])
def test_environment_blas_contractions_are_exact(superoperator_mode):
    rng = np.random.default_rng(987)
    d = 2
    physical = d * d if superoperator_mode else d
    left_dimension = right_dimension = 2
    mpo_left, mpo_right = 3, 4
    tensor = rng.normal(
        size=(left_dimension, physical, right_dimension)
    ) + 1j * rng.normal(
        size=(left_dimension, physical, right_dimension))
    operator = rng.normal(
        size=(mpo_left, mpo_right, d, d)
    ) + 1j * rng.normal(size=(mpo_left, mpo_right, d, d))
    left_block = rng.normal(
        size=(left_dimension, mpo_left, left_dimension)
    ) + 1j * rng.normal(
        size=(left_dimension, mpo_left, left_dimension))
    right_block = rng.normal(
        size=(right_dimension, mpo_right, right_dimension)
    ) + 1j * rng.normal(
        size=(right_dimension, mpo_right, right_dimension))

    if superoperator_mode:
        reshaped = tensor.reshape(
            left_dimension, d, d, right_dimension)
        tmp = np.einsum("awb,bfsj->wafsj", left_block, reshaped)
        tmp = np.einsum("wafsj,wxts->axftj", tmp, operator)
        left_reference = np.einsum(
            "axftj,afti->ixj", tmp, reshaped.conj())
        tmp = np.einsum(
            "bfsj,ixj->bfsxi", reshaped, right_block)
        tmp = np.einsum("bfsxi,wxts->bwfti", tmp, operator)
        right_reference = np.einsum(
            "bwfti,afti->awb", tmp, reshaped.conj())
    else:
        tmp = np.einsum("awb,bsj->wasj", left_block, tensor)
        tmp = np.einsum("wasj,wxts->axtj", tmp, operator)
        left_reference = np.einsum(
            "axtj,ati->ixj", tmp, tensor.conj())
        tmp = np.einsum("bsj,ixj->bsxi", tensor, right_block)
        tmp = np.einsum("bsxi,wxts->bwti", tmp, operator)
        right_reference = np.einsum(
            "bwti,ati->awb", tmp, tensor.conj())

    assert np.max(np.abs(
        TdvpEnvironments._left_step(
            left_block, tensor, operator) - left_reference)) < 1e-13
    assert np.max(np.abs(
        TdvpEnvironments._right_step(
            right_block, tensor, operator) - right_reference)) < 1e-13
