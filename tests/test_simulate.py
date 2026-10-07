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
    _resolve_coupling,
    _simulate_node,
    diffusive_coupling,
    dummy_trf_kernel,
    kuramoto_coupling,
    linear_coupling,
    simulate_ar,
    simulate_pulse_inputs,
    simulate_smooth_input,
    simulate_trf_output,
    simulate_var,
    simulate_var_from_cov,
)
from pyeeg.utils import sigmoid


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


# ---------------------------------------------------------------------------
# Coupling functions
# ---------------------------------------------------------------------------


def test_linear_coupling_matches_matrix_product():
    readouts = np.array([1.0, 2.0, 3.0])
    W = np.array([[0.0, 1.0, 0.5], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    np.testing.assert_allclose(linear_coupling(readouts, W), W @ readouts)
    # the phases argument is accepted for interface uniformity and ignored
    np.testing.assert_allclose(
        linear_coupling(readouts, W, phases=np.arange(3.0)), W @ readouts
    )


def test_linear_coupling_zero_and_identity_connectivity():
    readouts = np.array([1.0, -2.0, 0.5])
    np.testing.assert_allclose(
        linear_coupling(readouts, np.zeros((3, 3))), np.zeros(3)
    )
    np.testing.assert_allclose(linear_coupling(readouts, np.eye(3)), readouts)


def test_diffusive_coupling_matches_definition():
    readouts = np.array([1.0, 2.0, 3.0])
    W = np.array([[0.0, 1.0, 0.5], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    expected = W @ readouts - W.sum(axis=1) * readouts
    np.testing.assert_allclose(diffusive_coupling(readouts, W), expected)
    # hand-computed two-node example: each node receives the difference
    W2 = np.array([[0.0, 1.0], [1.0, 0.0]])
    np.testing.assert_allclose(diffusive_coupling([1.0, 2.0], W2), [1.0, -1.0])


def test_diffusive_coupling_conserves_total_input_for_symmetric_W():
    rng = np.random.default_rng(0)
    W = rng.standard_normal((4, 4))
    W = W + W.T  # symmetric connectivity
    readouts = rng.standard_normal(4)
    np.testing.assert_allclose(
        diffusive_coupling(readouts, W).sum(), 0.0, atol=1e-12
    )


def test_diffusive_coupling_zero_and_identity_connectivity():
    readouts = np.array([1.0, -2.0, 0.5])
    np.testing.assert_allclose(
        diffusive_coupling(readouts, np.zeros((3, 3))), np.zeros(3)
    )
    # identity connectivity couples each node only to itself -> no net input
    np.testing.assert_allclose(diffusive_coupling(readouts, np.eye(3)), np.zeros(3))


def test_kuramoto_coupling_matches_sine_phase_differences():
    W = np.array([[0.0, 1.0], [1.0, 0.0]])
    phases = np.array([0.0, np.pi / 2])
    # node 0 receives sin(pi/2 - 0) = 1, node 1 receives sin(0 - pi/2) = -1
    np.testing.assert_allclose(kuramoto_coupling([0.0, 0.0], W, phases), [1.0, -1.0])
    # readouts are accepted for interface uniformity but unused
    np.testing.assert_allclose(kuramoto_coupling([5.0, -3.0], W, phases), [1.0, -1.0])


def test_kuramoto_coupling_sums_to_zero_for_symmetric_W():
    rng = np.random.default_rng(1)
    W = rng.standard_normal((4, 4))
    W = W + W.T  # symmetric connectivity
    phases = rng.uniform(0, 2 * np.pi, 4)
    np.testing.assert_allclose(
        kuramoto_coupling(np.zeros(4), W, phases).sum(), 0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        kuramoto_coupling(np.zeros(4), np.zeros((4, 4)), phases), np.zeros(4)
    )


def test_resolve_coupling_names_and_errors():
    assert _resolve_coupling("linear") is linear_coupling
    assert _resolve_coupling("diffusive") is diffusive_coupling
    assert _resolve_coupling("kuramoto") is kuramoto_coupling
    custom = lambda readouts, connectivity, phases=None: connectivity @ readouts
    assert _resolve_coupling(custom) is custom
    with pytest.raises(ValueError):
        _resolve_coupling("unknown")
    with pytest.raises(TypeError):
        _resolve_coupling(42)


# ---------------------------------------------------------------------------
# HopfOscillator
# ---------------------------------------------------------------------------


def test_hopf_oscillator_oscillates_at_configured_frequency():
    model = HopfOscillator(a=0.01, frequency=10.0, dt=0.001, seed=1)
    states, outputs = model.simulate(x0=[0.1, 0.0], tmax=2.0)
    steady = outputs[1000:, 0]
    # non-zero variance: the oscillator is actually oscillating
    assert steady.std() > 0.1
    # two zero crossings per period -> 10 Hz over the 1 s steady window
    crossings = np.sum(np.signbit(steady[1:]) != np.signbit(steady[:-1]))
    assert crossings / 2 == pytest.approx(10.0, abs=0.5)


def test_hopf_oscillator_read_out_is_scalar_x_state():
    model = HopfOscillator(a=0.01, frequency=10.0, dt=0.001, seed=1)
    model.x = np.array([0.3, -0.4])
    assert isinstance(model.read_out(), float)
    assert model.read_out() == pytest.approx(0.3)


def test_hopf_oscillator_input_changes_trajectory():
    kwargs = dict(a=0.01, frequency=10.0, dt=0.001, seed=1)
    _, out_free = HopfOscillator(**kwargs).simulate(x0=[0.1, 0.0], tmax=1.0)
    _, out_driven = HopfOscillator(**kwargs).simulate(x0=[0.1, 0.0], tmax=1.0, I=0.5)
    assert not np.allclose(out_free, out_driven)


def test_hopf_oscillator_damped_for_negative_a():
    model = HopfOscillator(a=-100.0, frequency=10.0, dt=0.001, seed=1)
    states, _ = model.simulate(x0=[0.5, 0.0], tmax=0.1)
    radius = np.hypot(states[:, 0], states[:, 1])
    assert radius[-1] < 0.01 * radius[0]


def test_hopf_oscillator_large_dt_stability_limit():
    # moderate dt (10x the default): the explicit Euler integration stays finite
    model = HopfOscillator(a=0.01, frequency=10.0, dt=0.01, seed=1)
    states, _ = model.simulate(x0=[0.1, 0.0], tmax=1.0)
    assert np.isfinite(states).all()
    # excessive dt (50x the default): the integration diverges (NaN), which
    # documents the stability limit of the explicit Euler scheme
    model = HopfOscillator(a=0.01, frequency=10.0, dt=0.05, seed=1)
    with np.errstate(over="ignore", invalid="ignore"):
        states, _ = model.simulate(x0=[0.1, 0.0], tmax=2.0)
    assert not np.isfinite(states).all()


# ---------------------------------------------------------------------------
# Phasor
# ---------------------------------------------------------------------------


def test_phasor_phase_advances_at_configured_frequency():
    model = Phasor(frequency=10.0, dt=0.001, seed=1)
    states, _ = model.simulate(x0=[0.0], tmax=1.0)
    phase = np.unwrap(states[:, 0])
    duration = (len(states) - 1) * model.dt
    freq = (phase[-1] - phase[0]) / (2 * np.pi * duration)
    assert freq == pytest.approx(10.0, rel=1e-6)
    # the phase state is wrapped into [0, 2*pi)
    assert states[:, 0].min() >= 0.0
    assert states[:, 0].max() < 2 * np.pi


def test_phasor_input_shifts_frequency():
    model = Phasor(frequency=10.0, dt=0.001, seed=1)
    states, _ = model.simulate(x0=[0.0], tmax=1.0, I=2 * np.pi * 5)
    phase = np.unwrap(states[:, 0])
    duration = (len(states) - 1) * model.dt
    freq = (phase[-1] - phase[0]) / (2 * np.pi * duration)
    assert freq == pytest.approx(15.0, rel=1e-6)


def test_phasor_read_out_is_sine_of_phase_in_unit_interval():
    model = Phasor(frequency=10.0, dt=0.001, seed=1)
    states, outputs = model.simulate(x0=[0.3], tmax=0.5)
    assert isinstance(model.read_out(), float)
    assert np.abs(outputs).max() <= 1.0
    np.testing.assert_allclose(outputs[:, 0], np.sin(states[:, 0]))


def test_phasor_and_hopf_reject_negative_frequency():
    with pytest.raises(ValueError):
        Phasor(frequency=-1.0)
    with pytest.raises(ValueError):
        HopfOscillator(frequency=-1.0)


# ---------------------------------------------------------------------------
# _simulate_node engine
# ---------------------------------------------------------------------------


def test_simulate_node_output_length_and_initial_state():
    node = Phasor(frequency=10.0, dt=0.001, seed=1)
    states, outputs = _simulate_node(node, [0.3], 0.5, 0.0, 0.0)
    assert states.shape == (500, 1)
    assert outputs.shape == (500, 1)
    np.testing.assert_allclose(states[0], [0.3])
    np.testing.assert_allclose(outputs[0], [np.sin(0.3)])


def test_simulate_node_scalar_and_array_input_equivalent():
    node_scalar = Phasor(frequency=10.0, dt=0.001, seed=1)
    node_array = Phasor(frequency=10.0, dt=0.001, seed=1)
    _, out_scalar = _simulate_node(node_scalar, None, 0.5, 0.0, 1.0)
    _, out_array = _simulate_node(node_array, None, 0.5, 0.0, np.full(500, 1.0))
    np.testing.assert_array_equal(out_scalar, out_array)


def test_simulate_node_noise_is_stochastic_but_reproducible():
    node_a = Phasor(frequency=10.0, dt=0.001, seed=7)
    node_b = Phasor(frequency=10.0, dt=0.001, seed=7)
    node_c = Phasor(frequency=10.0, dt=0.001, seed=7)
    _, out_a = _simulate_node(node_a, None, 0.1, 0.5, 0.0)
    _, out_b = _simulate_node(node_b, None, 0.1, 0.5, 0.0)
    _, out_c = _simulate_node(node_c, None, 0.1, 0.0, 0.0)
    np.testing.assert_array_equal(out_a, out_b)  # same seed -> identical
    assert not np.allclose(out_a, out_c)  # noise perturbs the trajectory


def test_simulate_node_validates_inputs():
    node = Phasor(frequency=10.0, dt=0.001, seed=1)
    with pytest.raises(ValueError):
        _simulate_node(node, None, 0.0005, 0.0, 0.0)  # tmax < dt
    with pytest.raises(ValueError):
        _simulate_node(node, [1.0, 2.0], 0.1, 0.0, 0.0)  # wrong x0 shape


# ---------------------------------------------------------------------------
# WilsonCowan
# ---------------------------------------------------------------------------


def test_wilson_cowan_excitatory_input_drives_e_above_i():
    model = WilsonCowan(dt=0.001, seed=1)  # default P=1 (excitatory input)
    states, _ = model.simulate(tmax=1.0)
    steady = states[900:]
    assert steady[:, 0].mean() > steady[:, 1].mean()  # e > i
    assert steady[:, 0].mean() == pytest.approx(1.0, abs=0.05)  # near saturation


def test_wilson_cowan_read_out_is_e_minus_i():
    model = WilsonCowan(dt=0.001, seed=1)
    states, outputs = model.simulate(tmax=0.5)
    np.testing.assert_allclose(outputs[:, 0], states[:, 0] - states[:, 1])


def test_wilson_cowan_input_amplitude_scales_excitation():
    # moderate weights avoid sigmoid saturation so the input has a clear effect
    kwargs = dict(
        tau_e=0.01, tau_i=0.02, w_ee=2.0, w_ei=1.0, w_ie=1.5, w_ii=0.5,
        dt=0.001, seed=1,
    )
    steady_e = []
    for P in (0.0, 2.0):
        states, _ = WilsonCowan(P=P, **kwargs).simulate(tmax=1.0)
        steady_e.append(states[900:, 0].mean())
    assert steady_e[1] > steady_e[0]  # more input -> more excitation
    # with no input the two populations balance (e == i, readout -> 0)
    states, outputs = WilsonCowan(P=0.0, **kwargs).simulate(tmax=1.0)
    assert states[900:, 0].mean() == pytest.approx(
        states[900:, 1].mean(), abs=1e-9
    )
    assert outputs[900:, 0].mean() == pytest.approx(0.0, abs=1e-9)


def test_wilson_cowan_stable_and_bounded_for_dt_below_tau():
    model = WilsonCowan(dt=0.004, seed=1)  # dt < tau (0.008)
    states, outputs = model.simulate(tmax=1.0)
    assert np.isfinite(states).all()
    # activities are sigmoid-bounded rates and stay within [0, 1]
    assert states.min() >= 0.0 and states.max() <= 1.0
    # the dynamics converge to a fixed point
    n = len(outputs)
    assert outputs[int(0.9 * n):, 0].std() < 1e-9


def test_wilson_cowan_large_dt_escapes_sigmoid_bounds():
    # dt >> tau makes the explicit Euler integration unstable: the states
    # escape the [0, 1] sigmoid bounds (documented stability limit)
    model = WilsonCowan(dt=0.05, seed=1)
    states, _ = model.simulate(tmax=0.2)
    assert np.abs(states).max() > 10.0


def test_wilson_cowan_validates_parameters():
    with pytest.raises(ValueError):
        WilsonCowan(tau_e=0.0)
    with pytest.raises(ValueError):
        WilsonCowan(dt=0.001).simulate(tmax=0.0005)  # tmax < dt


# ---------------------------------------------------------------------------
# Kuramoto network
# ---------------------------------------------------------------------------


def _order_parameter(network):
    phases = np.array([node.x[0] for node in network.nodes])
    return np.abs(np.mean(np.exp(1j * phases)))


def test_kuramoto_strong_coupling_synchronizes_phases():
    N = 5
    W = np.ones((N, N)) - np.eye(N)
    x0 = np.array([0.0, 1.0, 2.5, 4.0, 5.5])  # spread, asymmetric phases
    r_start = np.abs(np.mean(np.exp(1j * x0)))
    network = Kuramoto(
        N=N, W=W, coupling_strength=10.0, frequency=10.0, dt=0.001, seed=42
    )
    output = network.simulate(tmax=1.0, x0=x0)
    assert output.shape == (1000, N)
    assert r_start < 0.5
    assert _order_parameter(network) == pytest.approx(1.0, abs=1e-3)


def test_kuramoto_zero_coupling_keeps_phases_independent():
    N = 5
    x0 = np.array([0.0, 1.0, 2.5, 4.0, 5.5])
    network = Kuramoto(
        N=N, W=np.zeros((N, N)), coupling_strength=10.0, frequency=10.0,
        dt=0.001, seed=42,
    )
    output = network.simulate(tmax=1.0, x0=x0)
    # identical natural frequencies + no coupling -> constant phase differences
    phases = np.array([node.x[0] for node in network.nodes])
    diffs = np.angle(np.exp(1j * (phases[:, None] - phases[None, :])))
    diffs0 = np.angle(np.exp(1j * (x0[:, None] - x0[None, :])))
    np.testing.assert_allclose(diffs, diffs0, atol=1e-9)
    assert _order_parameter(network) == pytest.approx(
        np.abs(np.mean(np.exp(1j * x0))), rel=1e-9
    )
    # the node readouts stay different (phases remain spread)
    assert not np.allclose(output[-50:, 0], output[-50:, 1])


def test_kuramoto_simulate_validates_inputs():
    network = Kuramoto(N=2, dt=0.001, seed=1)
    with pytest.raises(ValueError):
        network.simulate(tmax=0.0005)  # tmax < dt
    with pytest.raises(ValueError):
        network.simulate(tmax=0.1, x0=np.zeros(3))  # wrong x0 shape
    # phases outside [0, 2*pi) are wrapped into range
    network.simulate(tmax=0.01, x0=np.array([2 * np.pi + 0.5, -1.0]))
    assert all(0.0 <= node.x[0] < 2 * np.pi for node in network.nodes)


# ---------------------------------------------------------------------------
# CTRNN
# ---------------------------------------------------------------------------


def test_ctrnn_state_decays_with_unit_time_constant():
    # tau * x' = -x + W o + I with W=0 and no input -> x(t) = x0 * exp(-t/tau)
    # with the implicit tau = 1 s of the model
    model = CTRNN(N=2, W=np.zeros((2, 2)), dt=0.001, seed=1)
    _, states, _ = model.simulate(x0=[1.0, -1.0], tmax=1.0)
    np.testing.assert_allclose(
        states[-1], np.exp(-1.0) * np.array([1.0, -1.0]), rtol=1e-2
    )
    np.testing.assert_allclose(
        states[500], np.exp(-0.5) * np.array([1.0, -1.0]), rtol=1e-2
    )


def test_ctrnn_rates_are_sigmoid_outputs_in_unit_interval():
    W = np.array([[0.0, 1.0, -0.5], [0.5, 0.0, 1.0], [1.0, 0.0, 0.0]])
    model = CTRNN(N=3, W=W, dt=0.001, seed=1)
    _, states, rates = model.simulate(tmax=0.5, I=lambda t: np.sin(2 * np.pi * t))
    assert rates.min() >= 0.0 and rates.max() <= 1.0
    np.testing.assert_allclose(rates, sigmoid(states), rtol=1e-12)
    # regression: the first row of the returned rates used to stay at zero
    np.testing.assert_allclose(rates[0], sigmoid(states[0]))


def test_ctrnn_read_out_is_bounded_projection():
    model = CTRNN(N=2, W=np.array([[0.0, 1.0], [1.0, 0.0]]), dt=0.001, seed=1)
    model.simulate(tmax=0.5)
    # zero-initialised readout matrix -> 2 * sigmoid(0) - 1 = 0
    assert model.read_out().shape == (1,)
    np.testing.assert_allclose(model.read_out(), np.zeros(1))
    # a non-zero readout matrix projects the rates into [-1, 1]
    model.readout_W = np.ones((1, 2))
    expected = 2 * sigmoid(model.readout_W @ model.o) - 1
    np.testing.assert_allclose(model.read_out(), expected)
    assert np.all(np.abs(model.read_out()) <= 1.0)


def test_ctrnn_simulate_is_reproducible_with_noise():
    kwargs = dict(N=2, W=np.zeros((2, 2)), dt=0.001, seed=3)
    first = CTRNN(**kwargs).simulate(tmax=0.2, noise=0.5)
    second = CTRNN(**kwargs).simulate(tmax=0.2, noise=0.5)
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)


def test_ctrnn_simulate_rejects_tmax_below_dt():
    # regression: used to crash with IndexError instead of ValueError
    model = CTRNN(N=2, W=np.zeros((2, 2)), dt=0.001, seed=1)
    with pytest.raises(ValueError):
        model.simulate(tmax=0.0005)


# ---------------------------------------------------------------------------
# JansenRit
# ---------------------------------------------------------------------------


def test_jansen_rit_produces_alpha_oscillation_for_physiological_input():
    model = JansenRit(dt=0.0001, seed=1)
    states, outputs = model.simulate(tmax=1.0, P=220)
    assert np.isfinite(states).all() and np.isfinite(outputs).all()
    steady = outputs[5000:, 0]
    assert steady.std() > 0.1  # oscillating, not a fixed point
    # dominant frequency in the alpha band (~10 Hz)
    freqs = np.fft.rfftfreq(len(steady), d=model.dt)
    peak = freqs[np.argmax(np.abs(np.fft.rfft(steady - steady.mean())))]
    assert peak == pytest.approx(10.0, abs=1.0)


def test_jansen_rit_read_out_is_excitatory_minus_inhibitory():
    model = JansenRit(dt=0.0001, seed=1)
    states, outputs = model.simulate(tmax=0.2, P=150)
    np.testing.assert_allclose(outputs[:, 0], states[:, 1] - states[:, 2])


def test_jansen_rit_input_amplitude_changes_output_magnitude():
    steady_std = []
    for P in (100, 220):
        model = JansenRit(dt=0.0001, seed=1)
        _, outputs = model.simulate(tmax=1.0, P=P)
        steady_std.append(outputs[5000:, 0].std())
    # P=100 settles to a fixed point, P=220 sustains alpha oscillations
    assert steady_std[0] < 0.01
    assert steady_std[1] > 0.1


def test_jansen_rit_noise_is_applied_and_reproducible():
    # regression: the noise argument of simulate() used to be silently ignored
    kwargs = dict(dt=0.0001, seed=1)
    noisy_a = JansenRit(**kwargs).simulate(tmax=0.05, P=150, noise=5.0)[0]
    noisy_b = JansenRit(**kwargs).simulate(tmax=0.05, P=150, noise=5.0)[0]
    quiet = JansenRit(**kwargs).simulate(tmax=0.05, P=150, noise=0.0)[0]
    np.testing.assert_array_equal(noisy_a, noisy_b)  # same seed -> identical
    assert not np.allclose(noisy_a, quiet)  # noise perturbs the state


def test_jansen_rit_simulate_validates_inputs():
    model = JansenRit(dt=0.0001, seed=1)
    with pytest.raises(ValueError):
        model.simulate(tmax=0.01, x0=np.zeros(3))  # wrong x0 shape
    with pytest.raises(ValueError):
        # regression: used to crash with IndexError instead of ValueError
        model.simulate(tmax=0.00005)  # tmax < dt


# ---------------------------------------------------------------------------
# JansenRitExtended
# ---------------------------------------------------------------------------


def test_jansen_rit_extended_w_selects_subpopulation():
    # w=1 -> readout is the slow subpopulation difference (as in JansenRit)
    model = JansenRitExtended(w=1.0, dt=0.0001, seed=1)
    states, outputs = model.simulate(tmax=0.5, P=150)
    np.testing.assert_allclose(outputs[:, 0], states[:, 1] - states[:, 2])
    # w=0 -> readout is the fast subpopulation difference
    model = JansenRitExtended(w=0.0, dt=0.0001, seed=1)
    states, outputs = model.simulate(tmax=0.5, P=150)
    np.testing.assert_allclose(outputs[:, 0], states[:, 7] - states[:, 8])
    # 0 < w < 1 -> readout mixes both subpopulations
    model = JansenRitExtended(w=0.5, dt=0.0001, seed=1)
    states, outputs = model.simulate(tmax=0.5, P=150)
    np.testing.assert_allclose(
        outputs[:, 0],
        0.5 * (states[:, 1] - states[:, 2]) + 0.5 * (states[:, 7] - states[:, 8]),
    )


def test_jansen_rit_extended_slow_subpopulation_matches_jansen_rit():
    # with w=1 the slow subpopulation evolves exactly like the plain model
    extended = JansenRitExtended(w=1.0, dt=0.0001, seed=1)
    states_ext, outputs_ext = extended.simulate(tmax=0.5, P=150)
    plain = JansenRit(dt=0.0001, seed=1)
    states_plain, outputs_plain = plain.simulate(tmax=0.5, P=150)
    np.testing.assert_array_equal(states_ext[:, :6], states_plain)
    np.testing.assert_allclose(outputs_ext[:, 0], outputs_plain[:, 0])


def test_jansen_rit_extended_differs_from_plain_dynamics():
    kwargs = dict(dt=0.0001, seed=1)
    _, out_ext = JansenRitExtended(w=0.5, **kwargs).simulate(tmax=0.5, P=150)
    _, out_plain = JansenRit(**kwargs).simulate(tmax=0.5, P=150)
    assert not np.allclose(out_ext, out_plain)


def test_jansen_rit_extended_noise_and_validation():
    # regression: the noise argument of simulate() used to be silently ignored
    kwargs = dict(dt=0.0001, seed=1)
    noisy_a = JansenRitExtended(**kwargs).simulate(tmax=0.05, P=150, noise=5.0)[0]
    noisy_b = JansenRitExtended(**kwargs).simulate(tmax=0.05, P=150, noise=5.0)[0]
    np.testing.assert_array_equal(noisy_a, noisy_b)
    model = JansenRitExtended(dt=0.0001, seed=1)
    with pytest.raises(ValueError):
        model.simulate(tmax=0.01, x0=np.zeros(3))  # wrong x0 shape
    with pytest.raises(ValueError):
        # regression: used to crash with IndexError instead of ValueError
        model.simulate(tmax=0.00005)  # tmax < dt


# ---------------------------------------------------------------------------
# JRNetwork
# ---------------------------------------------------------------------------


def test_jr_network_coupling_changes_dynamics():
    kwargs = dict(N=2, dt=0.001, delay=0.01, seed=42)
    coupled = JRNetwork(W=np.array([[0, 1], [0, 0]]), **kwargs).simulate(
        tmax=0.05, P=220, sigma_p=22
    )
    uncoupled = JRNetwork(W=np.zeros((2, 2)), **kwargs).simulate(
        tmax=0.05, P=220, sigma_p=22
    )
    assert coupled.shape == uncoupled.shape == (50, 2)
    assert not np.allclose(coupled, uncoupled)


def test_jr_network_delay_changes_dynamics():
    W = np.array([[0, 1], [0, 0]])
    short_delay = JRNetwork(N=2, W=W, dt=0.001, delay=0.002, seed=42).simulate(
        tmax=0.05, P=220, sigma_p=22
    )
    long_delay = JRNetwork(N=2, W=W, dt=0.001, delay=0.02, seed=42).simulate(
        tmax=0.05, P=220, sigma_p=22
    )
    assert not np.allclose(short_delay, long_delay)


def test_jr_network_reset_restores_initial_state():
    network = JRNetwork(
        N=2, W=np.array([[0, 1], [0, 0]]), dt=0.001, delay=0.01, seed=42
    )
    network.simulate(tmax=0.02, P=220, sigma_p=22)
    network.reset()
    assert all(np.all(node.x == 0) for node in network.nodes)
    np.testing.assert_array_equal(network.delayed_states, np.zeros((2, 1)))
    np.testing.assert_array_equal(network.K, network.W)


def test_jr_network_update_connectivity_normalises_by_rate_std():
    network = JRNetwork(
        N=2, W=np.array([[0.0, 0.5], [0.5, 0.0]]), dt=0.001, seed=42
    )
    history = np.array([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]])  # shape (N, ntimes)
    network.update_connectivity(history, sigma_p=1.0)
    sigma_rate = np.std(history, axis=1)
    expected = np.array(
        [
            [0.0, np.sqrt(2 * 0.5 - 0.25) / sigma_rate[0]],
            [np.sqrt(2 * 0.5 - 0.25) / sigma_rate[1], 0.0],
        ]
    )
    np.testing.assert_allclose(network.K, expected)


def test_jr_network_update_connectivity_constant_history_stays_finite():
    # regression: a constant rate history (std = 0) used to divide by zero
    # and fill K with inf/NaN
    network = JRNetwork(
        N=2, W=np.array([[0.0, 0.5], [0.5, 0.0]]), dt=0.001, seed=42
    )
    network.update_connectivity(np.ones((2, 5)), sigma_p=1.0)
    assert np.isfinite(network.K).all()
    # the un-normalised coupling strength is used when there is nothing to
    # normalise by
    np.testing.assert_allclose(
        network.K, np.array([[0.0, np.sqrt(0.75)], [np.sqrt(0.75), 0.0]])
    )


def test_jr_network_accepts_list_connectivity():
    # regression: W was stored as given, so a list W crashed in step()
    network = JRNetwork(N=2, W=[[0, 1], [0, 0]], dt=0.001, delay=0.01, seed=42)
    output = network.simulate(tmax=0.02, P=220, sigma_p=22)
    assert output.shape == (20, 2)
    assert np.isfinite(output).all()


def test_jr_network_uncoupled_deterministic_reproducible():
    kwargs = dict(N=2, W=np.zeros((2, 2)), dt=0.001, seed=42)
    first = JRNetwork(**kwargs).simulate(tmax=0.02, P=220, sigma_p=0)
    second = JRNetwork(**kwargs).simulate(tmax=0.02, P=220, sigma_p=0)
    np.testing.assert_array_equal(first, second)
    assert np.isfinite(first).all()


# ---------------------------------------------------------------------------
# NeuralMassNetwork with coupling
# ---------------------------------------------------------------------------


def _phase(node):
    return np.arctan2(node.x[1], node.x[0])


def _run_hopf_pair(W, steps=1000):
    network = NeuralMassNetwork(
        N=2,
        W=W,
        node_dynamics=HopfOscillator,
        node_kwargs={"a": 0.01, "frequency": 10.0},
        dt=0.001,
        seed=42,
        coupling="linear",
    )
    network.nodes[0].x = np.array([0.1, 0.0])
    network.nodes[1].x = np.array([0.0, 0.1])  # 90 degrees apart
    before = [node.x.copy() for node in network.nodes]
    readouts, phase_diffs = [], []
    for _ in range(steps):
        network.step()
        readouts.append([node.read_out() for node in network.nodes])
        diff = _phase(network.nodes[0]) - _phase(network.nodes[1])
        phase_diffs.append(np.angle(np.exp(1j * diff)))
    return network, before, np.asarray(readouts), np.asarray(phase_diffs)


def test_network_step_advances_all_nodes():
    network, before, _, _ = _run_hopf_pair(np.zeros((2, 2)), steps=1)
    assert all(
        not np.array_equal(initial, node.x)
        for initial, node in zip(before, network.nodes, strict=True)
    )


def test_coupled_hopf_pair_phase_relationship_changes():
    _, _, readouts_unc, diffs_unc = _run_hopf_pair(np.zeros((2, 2)))
    _, _, _, diffs_coupled = _run_hopf_pair(np.array([[0.0, 1.0], [1.0, 0.0]]))
    # uncoupled: the 90-degree phase difference is preserved exactly
    np.testing.assert_allclose(diffs_unc, diffs_unc[0], atol=1e-9)
    assert not np.allclose(readouts_unc[-50:, 0], readouts_unc[-50:, 1])
    # coupled: mutual coupling pulls the oscillators toward synchrony
    assert abs(diffs_coupled[-1]) < abs(diffs_coupled[0])
    assert abs(diffs_coupled[-1] - diffs_coupled[0]) > 0.5


def test_network_diffusive_coupling_of_synchronized_nodes_is_zero():
    # identical readouts -> diffusive coupling input is zero -> same as uncoupled
    kwargs = dict(
        N=2, dt=0.001, seed=42, node_dynamics=Phasor,
        node_kwargs={"frequency": 7.0},
    )
    diffusive = NeuralMassNetwork(
        W=np.array([[0.0, 1.0], [1.0, 0.0]]), coupling="diffusive", **kwargs
    )
    uncoupled = NeuralMassNetwork(
        W=np.zeros((2, 2)), coupling="linear", **kwargs
    )
    for network in (diffusive, uncoupled):
        network.nodes[0].x = np.array([1.0])
        network.nodes[1].x = np.array([1.0])  # identical phases/readouts
    diffusive.step()
    uncoupled.step()
    np.testing.assert_allclose(
        [node.read_out() for node in diffusive.nodes],
        [node.read_out() for node in uncoupled.nodes],
    )


def test_network_validates_inputs():
    with pytest.raises(ValueError):
        NeuralMassNetwork(N=2, W=np.zeros((3, 3)), node_dynamics=Phasor)
    with pytest.raises(ValueError):
        NeuralMassNetwork(
            N=2, W=np.zeros((2, 2)), node_dynamics=Phasor, node_kwargs={"dt": 0.5}
        )
    with pytest.raises(RuntimeError):
        NeuralMassNetwork(N=2, W=np.zeros((2, 2))).step()  # no node_dynamics


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_zero_input_edge_cases():
    # Hopf: exactly zero state + zero input stays at the (unstable) origin
    model = HopfOscillator(a=0.01, frequency=10.0, dt=0.001, seed=1)
    states, outputs = model.simulate(x0=[0.0, 0.0], tmax=0.1, I=0.0)
    np.testing.assert_array_equal(states, np.zeros((100, 2)))
    np.testing.assert_array_equal(outputs, np.zeros((100, 1)))
    # Phasor: zero input still rotates at its natural frequency
    model = Phasor(frequency=10.0, dt=0.001, seed=1)
    _, outputs = model.simulate(x0=[0.0], tmax=0.1, I=0.0)
    assert outputs.std() > 0.1
    # Wilson-Cowan: zero input settles to the balanced fixed point (e == i)
    model = WilsonCowan(
        tau_e=0.01, tau_i=0.02, w_ee=2.0, w_ei=1.0, w_ie=1.5, w_ii=0.5,
        P=0.0, dt=0.001, seed=1,
    )
    _, outputs = model.simulate(tmax=1.0)
    assert outputs[900:, 0].mean() == pytest.approx(0.0, abs=1e-9)
