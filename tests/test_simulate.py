"""Unit tests for the public simulation helpers and neural-mass models."""

import numpy as np
import pytest

from pyeeg.simulate import (
    CTRNN,
    HopfOscillator,
    JansenRit,
    JansenRitExtended,
    JRNetwork,
    Kuramoto,
    NeuralMassNetwork,
    NeuralMassNode,
    Phasor,
    WilsonCowan,
    dummy_trf_kernel,
    simulate_ar,
    simulate_pulse_inputs,
    simulate_smooth_input,
    simulate_trf_output,
    simulate_var,
    simulate_var_from_cov,
)


def test_ar_is_reproducible_and_has_requested_length():
    x = simulate_ar(2, [0.4, -0.1], 100, seed=3)
    np.testing.assert_array_equal(x, simulate_ar(2, [0.4, -0.1], 100, seed=3))
    assert x.shape == (100,)


def test_var_accepts_order_one_matrix_and_is_reproducible():
    coef = np.array([[0.2, 0.1], [-0.1, 0.3]])
    x = simulate_var(1, coef, nobs=40, ndim=2, seed=4)
    np.testing.assert_array_equal(x, simulate_var(1, coef, nobs=40, ndim=2, seed=4))
    assert x.shape == (40, 2)
    with pytest.raises(AssertionError):
        simulate_var(2, coef, nobs=10, ndim=2)


def test_var_from_cov_runs_for_multiple_lags():
    x = simulate_var_from_cov(
        np.stack([np.eye(2), 0.5 * np.eye(2)]), nobs=20, ndim=2, seed=1
    )
    assert x.shape == (20, 2)


def test_trf_and_input_helpers():
    t, kernel = dummy_trf_kernel(srate=100)
    assert t.shape == kernel.shape
    _, pulses = simulate_pulse_inputs(n_events=10, dur=2, srate=100, seed=2)
    assert pulses.shape == (200,)
    assert np.count_nonzero(pulses) == 10
    _, smooth = simulate_smooth_input(dur=2, srate=100, seed=2)
    assert smooth.shape == (200,)
    assert simulate_trf_output(t, kernel, pulses, srate=100).shape == pulses.shape


def test_new_neural_mass_nodes_have_consistent_shapes():
    for model in (HopfOscillator(dt=0.001), Phasor(dt=0.001), WilsonCowan(dt=0.001)):
        states, output = model.simulate(tmax=0.01)
        assert states.shape == (10, model.nstates)
        assert output.shape == (10, 1)
        assert np.isfinite(states).all() and np.isfinite(output).all()


def test_node_kwargs_and_predefined_couplings():
    W = np.array([[0.0, 1.0], [1.0, 0.0]])
    network = NeuralMassNetwork(
        N=2,
        W=W,
        node_dynamics=Phasor,
        node_kwargs={"frequency": 7.0},
        coupling="diffusive",
    )
    assert all(node.frequency == 7.0 for node in network.nodes)
    network.step()
    assert np.isfinite([node.read_out() for node in network.nodes]).all()
    with pytest.raises(ValueError):
        NeuralMassNetwork(N=2, W=W, node_dynamics=Phasor, coupling="unknown")


def test_kuramoto_network_simulate():
    network = Kuramoto(N=2, W=np.array([[0.0, 1.0], [1.0, 0.0]]), dt=0.001)
    output = network.simulate(tmax=0.01, x0=np.zeros(2))
    assert output.shape == (10, 2) and np.isfinite(output).all()


def test_generic_network_forwards_dt_and_resets_nodes():
    network = NeuralMassNetwork(
        N=2,
        W=np.array([[0.0, 1.0], [1.0, 0.0]]),
        node_dynamics=HopfOscillator,
        dt=0.005,
    )
    assert all(node.dt == 0.005 for node in network.nodes)
    network.step()
    network.reset()
    assert all(np.all(node.x == 0) for node in network.nodes)


def test_ctrnn_has_working_defaults_and_shapes():
    model = CTRNN(N=2, W=np.zeros((2, 2)), dt=0.01)
    output, states, rates = model.simulate(tmax=0.05)
    assert output.shape == (5, 1) and states.shape == rates.shape == (5, 2)
    assert np.isfinite(output).all()


def test_jansen_rit_models_have_consistent_shapes_and_scalar_input():
    for model in (JansenRit(dt=0.001), JansenRitExtended(dt=0.001)):
        states, output = model.simulate(tmax=0.01, P=100)
        assert states.shape == (10, model.nstates) and output.shape == (10, 1)
        assert np.isfinite(states).all()


def test_jr_network_and_reset():
    network = JRNetwork(N=2, dt=0.001, delay=0.002)
    assert network.simulate(tmax=0.01).shape == (10, 2)
    network.reset()
    np.testing.assert_array_equal(network.K, network.W)
    np.testing.assert_array_equal(network.delayed_states, np.zeros((2, 1)))


def test_abstract_simulation_interfaces_raise():
    with pytest.raises(NotImplementedError):
        NeuralMassNode().simulate()
    with pytest.raises(NotImplementedError):
        NeuralMassNetwork(1, np.zeros((1, 1))).simulate()
