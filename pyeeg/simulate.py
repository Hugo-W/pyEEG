"""
=====================
Simulations utilities
=====================

Simulate MEEG-like signals with different connectivity patterns or methods.

.. TODO::

    - Follow up on these:
        - RNN of rate models: `RNN of rate models <https://elifesciences.org/articles/69499>`_
        - ctRNN revision, see `ctRNN revision <https://www.nature.com/articles/s42256-023-00748-9#Sec9>`_
    - Simulation based on connectivity matrix: implement a network of N nodes simply using the connectivity matrix and node-specific implementations
    - Neural mass models to be added:
        - Wilson-Cowan

.. References::

    - Jansen-Rit model:
        - `Comparing individual and group-level simulated neurophysiological brain connectivity using the Jansen and Rit neural mass model, 2023 <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10473283/>`_
        - `Evaluation of different measures of functional connectivity using a neural mass model, 2004 <https://pdf.sciencedirectassets.com/272508/1-s2.0-S1053811900X00973/1-s2.0-S1053811903006566/main.pdf>`_
        - `A neural mass model for MEG/EEG:: coupling and neuronal dynamics, 2003 <https://pdf.sciencedirectassets.com/272508/1-s2.0-S1053811900X00924/1-s2.0-S1053811903004579/main.pdf>`_

.. Updates::

    - **22/11/2023**: Network class added and JR network implemented
    - **17/11/2023**: Added Jansen-Rit model
    - **10/11/2023**: Initial commit (AR and VAR simulations)
"""

import numpy as np

from ._logging import LOGGER
from .utils import poisson_onsets_fixed_N, scisig, sigmoid


def simulate_ar(order, coefs, n, sigma=1, seed=42):
    """
    Simulate an autoregressive process of order `order`.

    Parameters
    ----------
    order : int
        The order of the autoregressive process.
    coefs : array_like
        The coefficients of the autoregressive process. The first element is the coefficient of the lag (t-1).
    n : int
        The number of samples to simulate.
    sigma : float
        The standard deviation of the additive noise process.

    Returns
    -------
    x : array_like
        The simulated time series. Shape (n,).
    """
    rng = np.random.default_rng(seed)
    x = np.zeros(n + order)
    for i in range(n + order):
        if i < order:
            x[i] = rng.standard_normal() * sigma
        else:
            x[i] = np.dot(coefs[::-1], x[i - order : i]) + rng.standard_normal() * sigma
    return x[order:]


def simulate_var(order, coef, nobs=500, ndim=2, seed=42, verbose=False):
    """
    Simulate a VAR model of order `order`.

    The VAR model is defined as:

    .. math::
        x_t = A_1 x_{t-1} + A_2 x_{t-2} + ... + A_p x_{t-p} + \\epsilon_t
        x_t = \\sum_{i=1}^p A_i x_{t-i} + \\epsilon_t

    where :math:`x_t` is a vector of shape (ndim, 1), :math:`A_i` is a matrix of shape (ndim, ndim).

    Parameters
    ----------
    order : int
        The order of the VAR model.
    coef : array_like
        The coefficients of the VAR model. Shape (order, ndim, ndim). If
        ``order == 1`` and ``coef`` is 2D, it is reshaped to (1, ndim, ndim).
    nobs : int
        The number of observations to simulate.
    ndim : int
        The number of dimensions of the VAR process.
    seed : int
        The random seed used to initialise the process.
    verbose : bool
        Whether to log information about the simulation.

    Returns
    -------
    data : ndarray
        The simulated VAR time series. Shape (nobs, ndim).

    Notes
    -----
    The coefficients at a given lag are such as :math:`C_ij` is i->j, so it will be the coefficients for dimension j!
    For example, each row of the first column are determining the contributions of each component onto the first component.
    """
    rng = np.random.default_rng(seed)
    if order == 1 and coef.ndim == 2:
        coef = coef[None, :, :]
    assert coef.shape == (order, ndim, ndim), (
        "coef must be of shape (order, ndim, ndim)"
    )
    data = np.zeros((nobs + order, ndim))

    if verbose:
        LOGGER.info(
            f"Simulating VAR({order}) model with {ndim} dimensions and {nobs} observations"
        )
        LOGGER.info(f"Data shape: {data.shape}")

    data[:, :] = rng.standard_normal(size=data.shape)  # initialize with noise
    for t in range(order, nobs + order):
        for lag in range(order):
            data[t] += (
                data[t - (lag + 1)] @ coef[lag]
            )  # here if I multiply from the left, I get the contributions row wise instead of column wise

    return data[order:, :]


def simulate_var_from_cov(cov, nobs=500, ndim=2, seed=42, verbose=False):
    """
    Simulate a VAR model of order `order` from a covariance matrix.

    The VAR model is defined as:

    .. math::
        x_t = A_1 x_{t-1} + A_2 x_{t-2} + ... + A_p x_{t-p} + \\epsilon_t
        x_t = \\sum_{i=1}^p A_i x_{t-i} + \\epsilon_t

    where :math:`x_t` is a vector of shape (ndim, 1), :math:`A_i` is a matrix of shape (ndim, ndim).

    Parameters
    ----------
    cov : array_like
        The covariance matrices of the VAR model. Shape (order, ndim, ndim).
        The order of the model is inferred from ``cov.shape[0]``.
    nobs : int
        The number of observations to simulate.
    ndim : int
        The number of dimensions of the VAR process.
    seed : int
        The random seed used to initialise the process.
    verbose : bool
        Whether to log information about the simulation.

    Returns
    -------
    data : ndarray
        The simulated VAR time series. Shape (nobs, ndim).

    Notes
    -----
    The coefficients at a given lag are such as :math:`C_ij` is i->j, so it will be the coefficients for dimension j!
    For example, each row of the first column are determining the contributions of each component onto the first component.
    """
    rng = np.random.default_rng(seed)
    order = cov.shape[0]
    assert cov.shape == (order, ndim, ndim), "cov must be of shape (order, ndim, ndim)"
    data = np.zeros((nobs + order, ndim))

    if verbose:
        LOGGER.info(
            f"Simulating VAR({order}) model with {ndim} dimensions and {nobs} observations"
        )
        LOGGER.info(f"Data shape: {data.shape}")

    data[:, :] = rng.standard_normal(size=data.shape)  # initialize with noise
    for t in range(order, nobs + order):
        for lag in range(order):
            data[t] += (
                data[t - (lag + 1)] @ np.linalg.cholesky(cov[lag])
            )  # here if I multiply from the left, I get the contributions row wise instead of column wise

    return data[order:, :]


def linear_coupling(readouts, connectivity, phases=None):
    """
    Linear projection of node readouts through the connectivity matrix.

    Parameters
    ----------
    readouts : array_like
        The node readouts. Shape (N,).
    connectivity : array_like
        The connectivity matrix. Shape (N, N).
    phases : array_like, optional
        Ignored; accepted for a uniform coupling-function interface.

    Returns
    -------
    input : ndarray
        The coupling input to each node. Shape (N,).
    """
    return np.asarray(connectivity) @ np.asarray(readouts)


def diffusive_coupling(readouts, connectivity, phases=None):
    """
    Weighted diffusive coupling of scalar node readouts.

    The input to node *i* is ``sum_j W_ij (readout_j - readout_i)``, where
    ``W_ij`` is the coupling strength from node *j* to node *i*.

    Parameters
    ----------
    readouts : array_like
        The node readouts. Shape (N,).
    connectivity : array_like
        The connectivity matrix. Shape (N, N).
    phases : array_like, optional
        Ignored; accepted for a uniform coupling-function interface.

    Returns
    -------
    input : ndarray
        The coupling input to each node. Shape (N,).
    """
    readouts = np.asarray(readouts)
    connectivity = np.asarray(connectivity)
    return connectivity @ readouts - connectivity.sum(axis=1) * readouts


def kuramoto_coupling(readouts, connectivity, phases):
    """
    Sinusoidal phase-difference (Kuramoto) coupling.

    The input to node *i* is ``sum_j W_ij sin(phase_j - phase_i)``.

    Parameters
    ----------
    readouts : array_like
        The node readouts. Shape (N,). Unused in the coupling computation but
        accepted for a uniform coupling-function interface.
    connectivity : array_like
        The connectivity matrix. Shape (N, N).
    phases : array_like
        The phase of each node. Shape (N,).

    Returns
    -------
    input : ndarray
        The coupling input to each node. Shape (N,).
    """
    phases = np.asarray(phases)
    connectivity = np.asarray(connectivity)
    return np.sum(connectivity * np.sin(phases[None, :] - phases[:, None]), axis=1)


_COUPLING_FUNCTIONS = {
    "linear": linear_coupling,
    "diffusive": diffusive_coupling,
    "kuramoto": kuramoto_coupling,
}


def _resolve_coupling(coupling):
    """
    Resolve a coupling specification into a coupling function.

    Parameters
    ----------
    coupling : str or callable
        Either a supported coupling name (``"linear"``, ``"diffusive"``,
        ``"kuramoto"``) or a callable with signature
        ``f(readouts, connectivity, phases=None)``.

    Returns
    -------
    coupling_function : callable
        The resolved coupling function.

    Raises
    ------
    ValueError
        If ``coupling`` is a string that is not a supported coupling name.
    TypeError
        If ``coupling`` is neither a supported name nor callable.
    """
    if isinstance(coupling, str):
        try:
            return _COUPLING_FUNCTIONS[coupling]
        except KeyError as error:
            raise ValueError(f"Unknown coupling function: {coupling!r}") from error
    if not callable(coupling):
        raise TypeError("coupling must be a supported name or callable")
    return coupling


class NeuralMassNode:
    """
    Abstract base class for a single neural-mass node.

    Defines the interface to be implemented for the simulation of a single
    node: a child class must provide a :meth:`step` method that advances the
    internal state by one integration step, an optional :meth:`read_out`
    method returning a scalar readout, and may override :meth:`simulate` to
    produce a full time series.

    Parameters
    ----------
    dt : float
        The integration time step in seconds.
    seed : int
        The random seed used to initialise the node's random number generator.
    """

    def __init__(self, dt=0.001, seed=42):
        self.dt = dt  # sampling rate
        self.seed = seed  # random seed

    def simulate(self):
        """
        Simulate the node and return its time series.

        Raises
        ------
        NotImplementedError
            This method must be implemented in the child class.
        """
        raise NotImplementedError("This method must be implemented in the child class")

    def step(self):
        """
        Advance the node by one integration step.

        Raises
        ------
        NotImplementedError
            This method must be implemented in the child class.
        """
        raise NotImplementedError("This method must be implemented in the child class")


class NeuralMassNetwork:
    """
    Abstract base class for a network of coupled neural-mass nodes.

    A network is made of ``N`` nodes, each instantiated from a node-dynamics
    class (e.g. :class:`Phasor`, :class:`HopfOscillator`,
    :class:`WilsonCowan`), coupled through a connectivity matrix ``W`` and a
    coupling function. Each step, the scalar readout of every node is combined
    through the coupling function into an input that is fed back to the nodes.

    Parameters
    ----------
    N : int
        The number of nodes in the network.
    W : array_like
        The connectivity matrix. Shape (N, N). Entry ``W[i, j]`` is the
        coupling strength from node *j* to node *i*.
    delay : float
        The delay between nodes in seconds. Stored for compatibility; not
        used by the default coupling scheme.
    node_dynamics : class, optional
        The class of the node dynamics used to instantiate the ``N`` nodes.
        Must accept ``dt`` and ``seed`` keyword arguments. If ``None``, no
        nodes are instantiated and :meth:`step` will raise a RuntimeError.
    dt : float
        The integration time step in seconds.
    seed : int
        The random seed used to initialise the network's random number
        generator and the per-node seeds.
    node_kwargs : dict, optional
        Extra keyword arguments passed to the ``node_dynamics`` constructor.
        Cannot override ``dt`` or ``seed``.
    coupling : str or callable
        Either a supported coupling name (``"linear"``, ``"diffusive"``,
        ``"kuramoto"``) or a callable with signature
        ``f(readouts, connectivity, phases=None)`` returning the coupling
        input to each node. Default is ``"linear"``.

    Raises
    ------
    ValueError
        If ``W`` does not have shape (N, N), or if ``node_kwargs`` attempts
        to override ``dt`` or ``seed``.
    """

    def __init__(
        self,
        N,
        W,
        delay=0,
        node_dynamics=None,
        dt=0.001,
        seed=42,
        node_kwargs=None,
        coupling="linear",
    ):
        self.rng = np.random.default_rng(seed)
        self.N = N  # number of neurons/nodes
        self.W = np.asarray(W, dtype=float)
        if self.W.shape != (N, N):
            raise ValueError(f"W must have shape ({N}, {N})")
        self.K = (
            self.W.copy()
        )  # updated connectivity in case of normalisation by activity std
        self.delay = delay  # delay
        self.dt = dt  # sampling rate
        self.seed = seed  # random seed
        self.node_dynamics = node_dynamics  # node dynamics instance
        self.node_kwargs = {} if node_kwargs is None else dict(node_kwargs)
        if {"dt", "seed"} & self.node_kwargs.keys():
            raise ValueError("node_kwargs cannot override dt or seed")
        self.coupling_function = _resolve_coupling(coupling)
        if node_dynamics is not None:
            seeds = self.rng.integers(0, np.iinfo(np.int32).max, size=N)
            self.nodes = [
                node_dynamics(dt=dt, seed=int(s), **self.node_kwargs) for s in seeds
            ]

    def simulate(self):
        """
        Simulate the network and return its outputs.

        Raises
        ------
        NotImplementedError
            This method must be implemented in the child class.
        """
        raise NotImplementedError("This method must be implemented in the child class")

    def step(self):
        """
        Advance the network by one integration step.

        The readout of every node is collected, combined through the coupling
        function into one input per node, and each node is advanced with its
        coupling input.

        Returns
        -------
        outs : ndarray
            The node readouts before the step. Shape (N,).

        Raises
        ------
        RuntimeError
            If ``node_dynamics`` was not provided at construction.
        ValueError
            If the coupling function does not return a vector of shape (N,).
        """
        if not hasattr(self, "nodes"):
            raise RuntimeError("node_dynamics must be provided to use step()")
        outs = np.asarray([node.read_out() for node in self.nodes], dtype=float)
        phases = np.asarray([node.x[0] for node in self.nodes], dtype=float)
        inputs = np.asarray(self.coupling_function(outs, self.K, phases), dtype=float)
        if inputs.shape != (self.N,):
            raise ValueError(f"coupling function must return shape ({self.N},)")
        for node, node_input in zip(self.nodes, inputs):
            node.step(I=node_input)
        return outs

    def reset(self):
        """
        Reset the network to its initial state.

        The state of every node is set to zero and the connectivity matrix
        ``K`` is restored to the original connectivity ``W``.
        """
        if hasattr(self, "nodes"):
            for node in self.nodes:
                node.x = np.zeros(node.nstates, dtype=float)
        self.K = self.W.copy()


class HopfOscillator(NeuralMassNode):
    """
    Two-dimensional Stuart-Landau (Hopf normal-form) oscillator.

    The state ``(x, y)`` evolves according to

    .. math::
        \\dot{x} = (a - r^2) x - \\omega y + I, \\quad
        \\dot{y} = (a - r^2) y + \\omega x + I

    with :math:`r^2 = x^2 + y^2` and :math:`\\omega = 2 \\pi f`. For
    :math:`a > 0` the origin is unstable and the oscillator converges to a
    limit cycle of radius :math:`\\sqrt{a}` at frequency :math:`f`.

    Parameters
    ----------
    a : float
        The bifurcation parameter. Positive values yield a stable limit
        cycle, negative values a damped oscillator.
    frequency : float
        The oscillation frequency in Hz. Must be non-negative.
    dt : float
        The integration time step in seconds.
    seed : int
        The random seed used to initialise the node's random number generator.

    Raises
    ------
    ValueError
        If ``frequency`` is negative.
    """

    def __init__(self, a=0.01, frequency=10.0, dt=0.001, seed=42):
        super().__init__(dt=dt, seed=seed)
        if frequency < 0:
            raise ValueError("frequency must be non-negative")
        self.a, self.frequency = float(a), float(frequency)
        self.omega = 2 * np.pi * self.frequency
        self.nstates, self.x = 2, np.zeros(2)
        self.rng = np.random.default_rng(seed)

    def step(self, I=0.0, noise=0.0):
        """
        Advance the oscillator by one integration step (Euler method).

        Parameters
        ----------
        I : float
            The external (coupling) input to the oscillator.
        noise : float
            The standard deviation of the additive noise, scaled by
            ``sqrt(dt)`` per sample.
        """
        x, y = self.x
        r2 = x * x + y * y
        self.x += self.dt * np.array(
            [(self.a - r2) * x - self.omega * y + I, (self.a - r2) * y + self.omega * x]
        )
        if noise:
            self.x += np.sqrt(self.dt) * noise * self.rng.standard_normal(2)

    def read_out(self):
        """
        Return the scalar readout of the oscillator.

        Returns
        -------
        readout : float
            The first state variable ``x``.
        """
        return float(self.x[0])

    def simulate(self, x0=None, tmax=1.0, noise=0.0, I=0.0):
        """
        Simulate the oscillator and return its states and readout.

        Parameters
        ----------
        x0 : array_like, optional
            The initial state. Shape (2,). Defaults to zeros.
        tmax : float
            The duration of the simulation in seconds.
        noise : float
            The standard deviation of the additive noise.
        I : float or array_like
            The external input. A scalar is applied at every step, an array
            of shape (n,) is indexed per step.

        Returns
        -------
        states : ndarray
            The simulated states. Shape (n, 2).
        outputs : ndarray
            The simulated scalar readouts. Shape (n, 1).
        """
        return _simulate_node(self, x0, tmax, noise, I)


class Phasor(NeuralMassNode):
    """
    One-dimensional phase oscillator with sinusoidal readout.

    The phase :math:`\\phi` evolves according to
    :math:`\\dot{\\phi} = \\omega + I` where :math:`\\omega = 2 \\pi f`, and
    the readout is :math:`\\sin(\\phi)`.

    Parameters
    ----------
    frequency : float
        The oscillation frequency in Hz. Must be non-negative.
    dt : float
        The integration time step in seconds.
    seed : int
        The random seed used to initialise the node's random number generator.

    Raises
    ------
    ValueError
        If ``frequency`` is negative.
    """

    def __init__(self, frequency=10.0, dt=0.001, seed=42):
        super().__init__(dt=dt, seed=seed)
        if frequency < 0:
            raise ValueError("frequency must be non-negative")
        self.frequency, self.omega = float(frequency), 2 * np.pi * float(frequency)
        self.nstates, self.x = 1, np.zeros(1)
        self.rng = np.random.default_rng(seed)

    def step(self, I=0.0, noise=0.0):
        """
        Advance the phase by one integration step (Euler method).

        Parameters
        ----------
        I : float
            The external (coupling) input to the oscillator.
        noise : float
            The standard deviation of the additive noise, scaled by
            ``sqrt(dt)`` per sample.
        """
        self.x[0] += self.dt * (self.omega + I)
        if noise:
            self.x[0] += np.sqrt(self.dt) * noise * self.rng.standard_normal()
        self.x[0] %= 2 * np.pi

    def read_out(self):
        """
        Return the scalar readout of the oscillator.

        Returns
        -------
        readout : float
            The sine of the phase, in [-1, 1].
        """
        return float(np.sin(self.x[0]))

    def simulate(self, x0=None, tmax=1.0, noise=0.0, I=0.0):
        """
        Simulate the oscillator and return its phase and readout.

        Parameters
        ----------
        x0 : array_like, optional
            The initial phase. Shape (1,). Defaults to zero.
        tmax : float
            The duration of the simulation in seconds.
        noise : float
            The standard deviation of the additive noise.
        I : float or array_like
            The external input. A scalar is applied at every step, an array
            of shape (n,) is indexed per step.

        Returns
        -------
        states : ndarray
            The simulated phases. Shape (n, 1).
        outputs : ndarray
            The simulated scalar readouts. Shape (n, 1).
        """
        return _simulate_node(self, x0, tmax, noise, I)


def _simulate_node(node, x0, tmax, noise, input_signal):
    """
    Shared simulation engine for the simple neural-mass nodes.

    Sets the initial state of ``node`` (validating its shape), then iterates
    the node's :meth:`~NeuralMassNode.step` forward in time, recording the
    states and scalar readouts at every sample.

    Parameters
    ----------
    node : NeuralMassNode
        The node to simulate. Must expose ``nstates``, ``dt``, ``x``, a
        ``step(I=..., noise=...)`` method and a ``read_out()`` method.
    x0 : array_like or None
        The initial state. Shape (node.nstates,). If ``None``, the state is
        initialised to zeros.
    tmax : float
        The duration of the simulation in seconds. Must be at least ``dt``.
    noise : float
        The standard deviation of the additive noise applied at each step.
    input_signal : float or array_like
        The external input. A scalar is applied at every step, an array of
        shape (n,) is indexed per step.

    Returns
    -------
    states : ndarray
        The simulated states. Shape (n, node.nstates).
    outputs : ndarray
        The simulated scalar readouts. Shape (n, 1).

    Raises
    ------
    ValueError
        If ``x0`` does not have shape (node.nstates,), or if ``tmax`` is
        smaller than ``dt``.
    """
    if x0 is None:
        x0 = np.zeros(node.nstates)
    node.x = np.asarray(x0, dtype=float).copy()
    if node.x.shape != (node.nstates,):
        raise ValueError(f"x0 must have shape ({node.nstates},)")
    n = int(tmax / node.dt)
    if n < 1:
        raise ValueError("tmax must be at least dt")
    states, outputs = np.zeros((n, node.nstates)), np.zeros((n, 1))
    states[0], outputs[0, 0] = node.x, node.read_out()
    for i in range(1, n):
        value = input_signal if np.ndim(input_signal) == 0 else input_signal[i]
        node.step(I=value, noise=noise)
        states[i], outputs[i, 0] = node.x, node.read_out()
    return states, outputs


class WilsonCowan(NeuralMassNode):
    """
    Two-population excitatory/inhibitory Wilson-Cowan rate model.

    The mean activities of the excitatory (:math:`e`) and inhibitory
    (:math:`i`) populations evolve according to

    .. math::
        \\tau_e \\dot{e} = -e + f(w_{ee} e - w_{ie} i + P + I)
        \\tau_i \\dot{i} = -i + f(w_{ei} e - w_{ii} i)

    where :math:`f` is the (sigmoidal) nonlinearity. The readout is the
    difference :math:`e - i` between the two populations.

    Parameters
    ----------
    tau_e : float
        The time constant of the excitatory population in seconds. Must be
        positive.
    tau_i : float
        The time constant of the inhibitory population in seconds. Must be
        positive.
    w_ee : float
        The excitatory-to-excitatory coupling weight.
    w_ei : float
        The excitatory-to-inhibitory coupling weight.
    w_ie : float
        The inhibitory-to-excitatory coupling weight.
    w_ii : float
        The inhibitory-to-inhibitory coupling weight.
    P : float
        The constant external input to the excitatory population.
    dt : float
        The integration time step in seconds.
    seed : int
        The random seed used to initialise the node's random number generator.
    nonlinearity : callable
        The activation function applied to the population drives. Default is
        :func:`~pyeeg.utils.sigmoid`.

    Raises
    ------
    ValueError
        If ``tau_e`` or ``tau_i`` is not positive.
    """

    def __init__(
        self,
        tau_e=0.008,
        tau_i=0.008,
        w_ee=12.0,
        w_ei=4.0,
        w_ie=13.0,
        w_ii=11.0,
        P=1.0,
        dt=0.001,
        seed=42,
        nonlinearity=sigmoid,
    ):
        super().__init__(dt=dt, seed=seed)
        if tau_e <= 0 or tau_i <= 0:
            raise ValueError("time constants must be positive")
        self.tau_e, self.tau_i = float(tau_e), float(tau_i)
        self.w_ee, self.w_ei, self.w_ie, self.w_ii = map(
            float, (w_ee, w_ei, w_ie, w_ii)
        )
        self.P, self.nonlinearity = P, nonlinearity
        self.nstates, self.x = 2, np.zeros(2)
        self.rng = np.random.default_rng(seed)

    def step(self, I=0.0, noise=0.0, P=None):
        """
        Advance the model by one integration step (Euler method).

        Parameters
        ----------
        I : float
            The external (coupling) input to the excitatory population.
        noise : float
            The standard deviation of the additive noise, scaled by
            ``sqrt(dt)`` per sample.
        P : float, optional
            The external input to the excitatory population for this step.
            If ``None``, the constant input set at construction is used.
        """
        e, inh = self.x
        drive_e = self.w_ee * e - self.w_ie * inh + (self.P if P is None else P) + I
        drive_i = self.w_ei * e - self.w_ii * inh
        rates = np.asarray([self.nonlinearity(drive_e), self.nonlinearity(drive_i)])
        self.x += self.dt * np.asarray(
            [(-e + rates[0]) / self.tau_e, (-inh + rates[1]) / self.tau_i]
        )
        if noise:
            self.x += np.sqrt(self.dt) * noise * self.rng.standard_normal(2)

    def read_out(self):
        """
        Return the scalar readout of the model.

        Returns
        -------
        readout : float
            The difference ``e - i`` between the excitatory and inhibitory
            population activities.
        """
        return float(self.x[0] - self.x[1])

    def simulate(self, x0=None, tmax=1.0, noise=0.0, P=None):
        """
        Simulate the model and return its states and readout.

        Parameters
        ----------
        x0 : array_like, optional
            The initial state. Shape (2,). Defaults to zeros.
        tmax : float
            The duration of the simulation in seconds.
        noise : float
            The standard deviation of the additive noise.
        P : float or array_like, optional
            The external input to the excitatory population. A scalar is
            applied at every step, an array of shape (n,) is indexed per
            step. If ``None``, the constant input set at construction is
            used.

        Returns
        -------
        states : ndarray
            The simulated states. Shape (n, 2).
        outputs : ndarray
            The simulated scalar readouts. Shape (n, 1).
        """
        if x0 is None:
            x0 = np.zeros(self.nstates)
        self.x = np.asarray(x0, dtype=float).copy()
        if self.x.shape != (self.nstates,):
            raise ValueError(f"x0 must have shape ({self.nstates},)")
        n = int(tmax / self.dt)
        if n < 1:
            raise ValueError("tmax must be at least dt")
        states = np.zeros((n, self.nstates))
        outputs = np.zeros((n, 1))
        states[0], outputs[0, 0] = self.x, self.read_out()
        for i in range(1, n):
            current_P = self.P if P is None else (P if np.ndim(P) == 0 else P[i])
            self.step(P=current_P, noise=noise)
            states[i], outputs[i, 0] = self.x, self.read_out()
        return states, outputs


class Kuramoto(NeuralMassNetwork):
    """
    Convenience network of :class:`Phasor` nodes with Kuramoto coupling.

    Each node is a :class:`Phasor` oscillator with the same ``frequency``.
    The phase of node *i* evolves as

    .. math::
        \\dot{\\phi}_i = \\omega + \\sum_j K_{ij} \\sin(\\phi_j - \\phi_i)

    with ``K = coupling_strength * W``, and the readout of each node is
    :math:`\\sin(\\phi_i)`.

    Parameters
    ----------
    N : int
        The number of oscillators in the network.
    W : array_like, optional
        The connectivity matrix. Shape (N, N). If ``None``, an all-zero
        matrix is used (uncoupled oscillators).
    coupling_strength : float
        The global scaling applied to ``W`` to obtain the effective coupling
        matrix ``K``.
    frequency : float
        The natural frequency of every oscillator in Hz.
    dt : float
        The integration time step in seconds.
    seed : int
        The random seed used to initialise the network's random number
        generator and the per-node seeds.
    **node_kwargs
        Extra keyword arguments passed to the :class:`Phasor` constructor.
    """

    def __init__(
        self,
        N=2,
        W=None,
        coupling_strength=1.0,
        frequency=10.0,
        dt=0.001,
        seed=42,
        **node_kwargs,
    ):
        if W is None:
            W = np.zeros((N, N))
        node_kwargs = {"frequency": frequency, **node_kwargs}
        super().__init__(
            N=N,
            W=W,
            node_dynamics=Phasor,
            dt=dt,
            seed=seed,
            node_kwargs=node_kwargs,
            coupling="kuramoto",
        )
        self.coupling_strength = float(coupling_strength)
        self.K = self.coupling_strength * self.W

    def simulate(self, tmax=1.0, x0=None):
        """
        Simulate the network and return the readout of every node.

        Parameters
        ----------
        tmax : float
            The duration of the simulation in seconds. Must be at least
            ``dt``.
        x0 : array_like, optional
            The initial phases. Shape (N,). If given, each phase is wrapped
            into [0, 2*pi). If ``None``, the phases are initialised to zero.

        Returns
        -------
        output : ndarray
            The simulated readouts. Shape (n, N).

        Raises
        ------
        ValueError
            If ``tmax`` is smaller than ``dt``, or if ``x0`` does not have
            shape (N,).
        """
        n = int(tmax / self.dt)
        if n < 1:
            raise ValueError("tmax must be at least dt")
        if x0 is not None:
            x0 = np.asarray(x0, dtype=float)
            if x0.shape != (self.N,):
                raise ValueError(f"x0 must have shape ({self.N},)")
            for node, phase in zip(self.nodes, x0):
                node.x[0] = phase % (2 * np.pi)
        output = np.zeros((n, self.N))
        output[0] = [node.read_out() for node in self.nodes]
        for i in range(1, n):
            self.step()
            output[i] = [node.read_out() for node in self.nodes]
        return output


class CTRNN(NeuralMassNetwork):
    """
    Continuous Time Recurrent Neural Network (CTRNN) model.

    The state :math:`x` of the network evolves according to

    .. math::
        \\tau \\dot{x} = -x + W o + I + \\theta

    where :math:`o = f(x + \\theta)` is the output of the network through
    the nonlinearity :math:`f`, :math:`I` is the external input projected
    through a (trainable) input matrix, and :math:`\\theta` is a bias term.
    A (trainable) readout matrix projects the node outputs to the desired
    output dimension, rescaled into the range [-1, 1] when the nonlinearity
    is a sigmoid.

    Parameters
    ----------
    N : int
        The number of neurons/nodes.
    W : array_like
        The connectivity matrix. Shape (N, N).
    input_dim : int
        The dimension of the external input. The input is projected through
        a zero-initialised matrix of shape (N, input_dim).
    output_dim : int
        The dimension of the readout. The node outputs are projected through
        a zero-initialised matrix of shape (output_dim, N).
    dt : float
        The integration time step in seconds.
    seed : int
        The random seed used to initialise the network's random number
        generator.
    nonlinearity : callable
        The nonlinearity function applied to the network state. Default is
        sigmoid (e.g. can use :func:`np.tanh`).
    theta : array_like, optional
        The bias term. Shape (N,). If ``None``, a zero bias is used.
    """

    def __init__(
        self,
        N,
        W,
        input_dim=1,
        output_dim=1,
        dt=0.001,
        seed=42,
        nonlinearity=sigmoid,
        theta=None,
    ):
        super().__init__(N=N, W=W, dt=dt, seed=seed)
        self.nonlinearity = nonlinearity  # nonlinearity function
        self.output_dim = output_dim
        self.readout_W = np.zeros((output_dim, N))  # readout matrix
        self.input_W = np.zeros((N, input_dim))  # input matrix
        self.theta = theta if theta is not None else np.zeros((N,))
        self.x = np.zeros((N,))  # state of the network
        self.o = np.zeros((N,))  # output of the network

    def step(self, I=None, noise=0.0):
        """
        Compute one step of the CTRNN model.

        Parameters
        ----------
        I : float or array_like, optional
            The external input. A scalar is broadcast to all input
            dimensions, an array must have shape (input_dim,). If ``None``,
            a zero input is used.
        noise : array_like or float
            The additive noise applied to the state. Typically an array of
            shape (N,) scaled by ``sqrt(dt)``, but any broadcastable value
            is accepted.
        """
        if I is None:
            I = np.zeros((self.input_W.shape[1],))
        elif np.isscalar(I):
            I = np.ones((self.input_W.shape[1],)) * I
        self.x = (
            self.x + self.dt * (-self.x + self.W @ self.o + self.input_W @ I) + noise
        )
        self.o = self.nonlinearity(self.x + self.theta)

    def read_out(self):
        """
        Compute the readout of the network.

        Returns
        -------
        readout : ndarray
            The projected output. Shape (output_dim,). This is in the range
            -1 to 1 if the nonlinearity is a sigmoid.
        """
        return 2 * self.nonlinearity(self.readout_W @ self.o) - 1

    def simulate(self, x0=None, tmax=1.0, noise=0.0, I=lambda t: 0.0):
        """
        Simulate the CTRNN model and monitor the output.

        Parameters
        ----------
        x0 : array_like, optional
            The initial state of the system. Shape (N,). Defaults to zeros.
        tmax : float
            The maximum time to simulate.
        noise : float
            The standard deviation of the noise to add to the system.
        I : callable
            The external input as a function of time, ``I(t)``, evaluated at
            every sample. Defaults to a constant zero input.

        Returns
        -------
        O : ndarray
            The simulated readout. Shape (n, output_dim).
        x : ndarray
            The simulated state. Shape (n, N).
        o : ndarray
            The simulated node outputs (after the nonlinearity). Shape (n, N).
        """
        rng = np.random.default_rng(self.seed)
        n = int(tmax / self.dt)
        x = np.zeros((n, self.N))
        o = np.zeros((n, self.N))
        O = np.zeros((n, self.output_dim))
        if x0 is None:
            x0 = np.zeros(self.N)
        x0 = np.asarray(x0, dtype=float)
        if x0.shape != (self.N,):
            raise ValueError(f"x0 must have shape ({self.N},)")
        x[0] = x0
        self.x = x0.copy()
        self.o = self.nonlinearity(self.x + self.theta)
        O[0] = self.read_out()
        dt_noise = np.sqrt(self.dt) * noise
        for i in range(1, n):
            self.step(
                I=I(i * self.dt),
                noise=rng.standard_normal(size=self.x.shape) * dt_noise,
            )
            x[i] = self.x
            o[i] = self.o
            O[i] = self.read_out()

        return O, x, o


class JansenRit(NeuralMassNode):
    r"""
    Jansen-Rit model.

    3 populations: excitatory, inhibitory and pyramidal:

    .. code-block:: text

        ___________    ___________
        |         |    |         |
        !  Inhib  !    !  Excit  !
        |         |    |         |
        -----------    -----------
        C2,C4 \             / C1, C3
               \___________/
                |         |
                ! Pyramid !
                |         |
                -----------

    Parameters and typical values as in Grimbert & Faugeras, 2006:

    - **C1, C2, C3, C4**: Average number of synapses between populations: ``135 * [1, 0.8, 0.25, 0.25]``
    - **tau_e**: Time scale for excitatory population: ``100 ms``
    - **tau_i**: Time scale for inhibitory population: ``50 ms``
    - **G_exc**: Average excitatory synaptic gain: ``3.25``
    - **G_inh**: Average inhibitory synaptic gain: ``22``
    - **rmax**: Amplitude of sigmoid: ``5 s^-1``
    - **beta**: Slope of sigmoid: ``0.56 mV^-1``
    - **theta**: Threshold of sigmoid: ``6 mV``
    - **Conduction velocity**: ``10 m/s``
    - **h**: Integration time step: ``0.0001 s`` (by default)
    - **P**: External input to each of the neural masses: ``150 Hz`` (constant input)
    - **Coupling**: Coupling between the neural masses: ``[0.1:0.012:0.292]``

    This table is from the paper:
    `Kulik et al, Network Neurosci. (2023) <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10473283/>`_

    """

    def __init__(self, dt=0.0001, seed=42, nonlinearity=sigmoid):
        """
        Parameters
        ----------
        dt : float
            The integration time step in seconds.
        seed : int
            The random seed used to initialise the node's random number
            generator.
        nonlinearity : callable
            The nonlinearity function applied to the population drives.
            Default is :func:`~pyeeg.utils.sigmoid`.
        """
        super().__init__(
            dt, seed
        )  # this is a single node (cortical column with 3 sub-populations)
        n_synapses = 135  # number of synapses between populations
        self.C_1 = (
            1.0 * n_synapses
        )  # probability of connection between excitatory and pyramidal populations
        self.C_2 = 0.8 * n_synapses
        self.C_3 = 0.25 * n_synapses
        self.C_4 = 0.25 * n_synapses
        self.tau_exc = 1 / 100  # time scale for excitatory population ~10ms
        self.tau_inh = 1 / 50  # time scale for inhibitory population ~20ms
        self.G_exc = 3.25  # average excitatory synaptic gain (mV)
        self.G_inh = 22  # average inhibitory synaptic gain
        self.rmax = 5  # amplitude of sigmoid in Hz (max firing rate)
        self.beta = 0.56  # slope of sigmoid (mV^-1)
        self.theta = 6  # threshold of sigmoid (mV)
        self.v = 10  # conduction velocity
        self.P = 150  # external input to each of the neural masses
        # self.Coupling = 0.1 # coupling between the neural masses (global coupling strength)
        self.nstates = 6  # number of state variables

        self.x = np.zeros((6,))  # state of the network
        self.S = lambda x: nonlinearity(
            x, rmax=self.rmax, beta=self.beta, x0=self.theta
        )  # nonlinearity function

    def step(self, I=0.0):
        """
        Compute one step of the Jansen-Rit model.

        Parameters
        ----------
        I : float
            The external input to the excitatory population.
        """
        # 0: pyramidal, 1: excitatory, 2: inhibitory
        x0, x1, x2, xdot0, xdot1, xdot2 = self.x  # unpack the state
        # Input received by each population
        # x1 - x2: difference between excitatory and inhibitory activity, which is the input received by the pyramidal population interpreted as the average potential of pyramidal populations
        # self.C_1 * x0: input received by the excitatory population
        # self.C_3 * x0: input received by the inhibitory population
        firing_rates = self.S(np.asarray([x1 - x2, self.C_1 * x0, self.C_3 * x0]))
        input_excitatory = (
            self.C_2 * firing_rates[1] + I
        )  # contribution from other nodes will go here
        xdot0_next = (
            xdot0
            + self.dt
            * (self.G_exc * 1.0 * firing_rates[0] - 2 * xdot0 - x0 / self.tau_exc)
            / self.tau_exc
        )  # pyramidal cell
        xdot1_next = (
            xdot1
            + self.dt
            * (self.G_exc * input_excitatory - 2 * xdot1 - x1 / self.tau_exc)
            / self.tau_exc
        )  # excitatory stellate cell
        xdot2_next = (
            xdot2
            + self.dt
            * (self.G_inh * self.C_4 * firing_rates[2] - 2 * xdot2 - x2 / self.tau_inh)
            / self.tau_inh
        )  # inhibitory interneuron
        x0_next = x0 + xdot0 * self.dt
        x1_next = x1 + xdot1 * self.dt
        x2_next = x2 + xdot2 * self.dt
        self.x = np.array(
            [x0_next, x1_next, x2_next, xdot0_next, xdot1_next, xdot2_next]
        )

    def read_out(self):
        return self.x[1] - self.x[2]

    def simulate(self, x0=None, tmax=1.0, noise=0.0, P=None):
        """
        Simulate the Jansen-Rit model and monitor the output.

        Parameters
        ----------
        x0 : array_like, optional
            The initial state of the system. Shape (6,). Defaults to zeros.
        tmax : float
            The maximum time to simulate.
        noise : float
            The standard deviation of the noise to add to the system.
        P : float or array_like, optional
            The external input to the excitatory population. A scalar is
            applied at every step, an array of shape (n,) is indexed per
            step. If ``None``, the constant input set at construction is
            used.

        Returns
        -------
        x : ndarray
            The simulated state. Shape (n, 6).
        o : ndarray
            The simulated readout (excitatory minus inhibitory). Shape (n, 1).
        """
        rng = np.random.default_rng(self.seed)
        n = int(tmax / self.dt)
        x = np.zeros((n, 6))
        o = np.zeros((n, 1))
        if x0 is None:
            x0 = np.zeros(self.nstates)
        self.x = np.asarray(x0, dtype=float).copy()
        if self.x.shape != (self.nstates,):
            raise ValueError(f"x0 must have shape ({self.nstates},)")
        x[0] = self.x
        o[0] = self.read_out()
        dt_noise = np.sqrt(self.dt) * noise
        for i in range(1, n):
            current_P = self.P if P is None else (P if np.ndim(P) == 0 else P[i])
            self.step(I=current_P)
            # noise = rng.standard_normal(size=self.x.shape) * dt_noise
            x[i] = self.x
            o[i] = self.read_out()
        return x, o


class JansenRitExtended(NeuralMassNode):
    """
    Jansen-Rit model - Exctended version: dual kinetic model.

    See `David & Friston, NeuroImage (2003) <https://pdf.sciencedirectassets.com/272508/1-s2.0-S1053811900X00924/1-s2.0-S1053811903004579/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEKH%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIQDklbniEg%2BIxtbMhqre0GLXBUY61F7QwRGlTcS0Tw50mAIgPijMAG0JRY86ILcJl5khJAbWRSGrqqz8rRNKkeVXhtMqvAUI6v%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAFGgwwNTkwMDM1NDY4NjUiDDWGjReDtp%2FZt%2BvT4SqQBUgH%2FBzAz1ffqLsZXij%2Fhvyo540aF9iKjI5qqwn%2FClzZq1dP%2BofFqhpbLq%2FbRaP98Et%2Bs5VZoAoNHhC8dMDRlrxY47ZuTWehsc2C8ZUcz6D9lYArh5ggiCQZMm9040OWVNDarLG8631K4g0HFpEHwsubfZoIUgs5XABH%2FyF1NE2zXo3JXhU%2FKwsZHVqqidrV0nv%2B9IJ5%2BigTmMVePAINzRUuQjnFgbyqNvqMwUsNDi92QSN7u%2BdJi0ksGwqTWyBCM7MUrwK%2FisZjWmQcQeCx%2FuLyhE77tU6x7gpR%2BGdE%2BOmiISZDROqxwq%2FmsC%2B%2FR%2BMsFubuDhdTI8n8kllkh079IhICKgEzhzFLWgmNxXdbxBS6vQi13IA5SNlQuuzArMu0Z1GkM9EwbXRUI3u4oeOMtGOslgCYdgsSJpebcg8zOg2ueLBgHSfubsGGJ5l5SxOR%2FIVF6tHjzLoCz2njmpmtSr%2Bmi56T9qKmkUC4sAoYnbpllz3yd%2F4%2B2Xh3AruYCVWyJ%2FVLgTbrPXYpihKXryoL1DNw94iDb%2FWutOEqh46F64lPV96HjCvNME4smEYyyPKz8DWYdMcU6U35qVbk%2BXlv%2FoDv7AIjCJi3l9U98IL%2BiZMAnCN8u88BtwF8o7PrMAMfRzjQl53oJEt%2BCfnIMZhwy0raTnU3Sb61tmgBqoIrY7NFmWjrAmqNJilwO16T7RIJkjRhufHocRzquTEj6F5NyPrir%2FeVwlg3%2BM8kgmsdsiIfLR9tkvkXCcSx1M4%2FwdK9id37RAh5D%2Fcw0jAcqEinfgj%2Bw4Z9MX5FU91B9pOJIBk%2BGRraBuxSsxznjeQ%2BlOZVwtQ7nVwE%2Bk0XUHkMv075s9RxWPIrg48wrQMqJUhLaW7zMK%2B%2B7KoGOrEBKbm0kOWBELm7BGve3m4Vj6De%2BonWolZfqid1yachKSVCGGly07GpJptQJqCsQftOtoRw966JD257GoZ%2BssIo%2FrdAhYGJXSWfvpbzGVT3etLIjbrApoBoQMzw%2BVTy1I%2FyomAtv2yXcop%2FNGlQyOkREmdcc3cbpVi78%2FPiChzyI3ESzp4hwWnNKs2cFoQE15W8GGT0FxkBk0QoXJN2MiPdAS%2Fd7Nrw4MOMPaseHhIfQx81&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20231120T101219Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYXMTQIC7Q%2F20231120%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=29cd9a0fade336c06381b528450f2b797b8b67eaa9ecc3e029ceb182ac9dcf40&hash=d99e790b5d5df95dc08507ca2341780d14b640ae4c5e57d1b6e1cac92280cb90&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1053811903004579&tid=spdf-2465ea3f-65c5-4f0d-9fd9-eaf43d018e8d&sid=3f6dcc697a1fe84db69a8074585332d50fa6gxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=080f57525b01065d0459&rr=828fe9f6bfba5c4b&cc=nl>`_

    We model two parallel subpopulations with different kinematics in order to capture multiband or broadband dynamics.
    """

    def __init__(self, w=0.5, dt=0.0001, seed=42, nonlinearity=sigmoid):
        """
        Parameters
        ----------
        w : float
            The relative contribution of the first (slow) subpopulation.
            The second (fast) subpopulation contributes ``1 - w``.
        dt : float
            The integration time step in seconds.
        seed : int
            The random seed used to initialise the node's random number
            generator.
        nonlinearity : callable
            The nonlinearity function applied to the population drives.
            Default is :func:`~pyeeg.utils.sigmoid`.
        """
        super().__init__(
            dt, seed
        )  # this is a single node (cortical column with 3 sub-populations)

        # ~ 10 Hz dynamics
        self.tau_exc_1 = 1 / 100  # time scale for excitatory population ~10ms
        self.tau_inh_1 = 1 / 50  # time scale for inhibitory population ~20ms
        # ~ 43 Hz dynamics
        self.tau_exc_2 = 0.0046  # time scale for excitatory population ~4.6ms
        self.tau_inh_2 = 0.0029  # time scale for inhibitory population ~2.9ms
        self.G_exc_1 = 3.25  # average excitatory synaptic gain (mV)
        self.G_inh_1 = 22  # average inhibitory synaptic gain
        self.G_exc_2 = 2 * 3.25  # average excitatory synaptic gain (mV)
        self.G_inh_2 = 150  # average inhibitory synaptic gain
        self.w = w  # relative contribution of the first subpopulation

        # The rest is the same as the Jansen-Rit model
        n_synapses = 135  # number of synapses between populations
        self.C_1 = (
            1.0 * n_synapses
        )  # probability of connection between excitatory and pyramidal populations
        self.C_2 = 0.8 * n_synapses
        self.C_3 = 0.25 * n_synapses
        self.C_4 = 0.25 * n_synapses
        self.rmax = 5  # amplitude of sigmoid in Hz (max firing rate)
        self.beta = 0.56  # slope of sigmoid (mV^-1)
        self.theta = 6  # threshold of sigmoid (mV)
        self.v = 10  # conduction velocity
        self.P = 150  # external input to each of the neural masses
        # self.Coupling = 0.1 # coupling between the neural masses (global coupling strength)

        self.x = np.zeros((2 * 6,))  # state of the network
        self.nstates = 2 * 6
        self.S = lambda x: nonlinearity(
            x, rmax=self.rmax, beta=self.beta, x0=self.theta
        )  # nonlinearity function

    def step(self, I=0.0):
        """
        Compute one step of the extended Jansen-Rit model.

        Parameters
        ----------
        I : float
            The external input to the excitatory population.
        """
        # 0: pyramidal, 1: excitatory, 2: inhibitory
        (
            x0_1,
            x1_1,
            x2_1,
            xdot0_1,
            xdot1_1,
            xdot2_1,
            x0_2,
            x1_2,
            x2_2,
            xdot0_2,
            xdot1_2,
            xdot2_2,
        ) = self.x  # unpack the state
        # Input received by each population
        # x1 - x2: difference between excitatory and inhibitory activity, which is the input received by the pyramidal population interpreted as the average potential of pyramidal populations
        # self.C_1 * x0: input received by the excitatory population
        # self.C_3 * x0: input received by the inhibitory population
        firing_rates = self.S(
            np.asarray(
                [
                    self.w * (x1_1 - x2_1) + (1 - self.w) * (x1_2 - x2_2),
                    self.C_1 * (self.w * x0_1 + (1 - self.w) * x0_2),
                    self.C_3 * (self.w * x0_1 + (1 - self.w) * x0_2),
                ]
            )
        )
        input_excitatory = (
            self.C_2 * firing_rates[1] + I
        )  # contribution from other nodes will go here
        # pop 1
        xdot0_next_1 = (
            xdot0_1
            + self.dt
            * (
                self.G_exc_1 * 1.0 * firing_rates[0]
                - 2 * xdot0_1
                - x0_1 / self.tau_exc_1
            )
            / self.tau_exc_1
        )
        xdot1_next_1 = (
            xdot1_1
            + self.dt
            * (self.G_exc_1 * input_excitatory - 2 * xdot1_1 - x1_1 / self.tau_exc_1)
            / self.tau_exc_1
        )
        xdot2_next_1 = (
            xdot2_1
            + self.dt
            * (
                self.G_inh_1 * self.C_4 * firing_rates[2]
                - 2 * xdot2_1
                - x2_1 / self.tau_inh_1
            )
            / self.tau_inh_1
        )
        x0_next_1 = x0_1 + xdot0_1 * self.dt
        x1_next_1 = x1_1 + xdot1_1 * self.dt
        x2_next_1 = x2_1 + xdot2_1 * self.dt
        # Pop 2
        xdot0_next_2 = (
            xdot0_2
            + self.dt
            * (
                self.G_exc_2 * 1.0 * firing_rates[0]
                - 2 * xdot0_2
                - x0_2 / self.tau_exc_2
            )
            / self.tau_exc_2
        )
        xdot1_next_2 = (
            xdot1_2
            + self.dt
            * (self.G_exc_2 * input_excitatory - 2 * xdot1_2 - x1_2 / self.tau_exc_2)
            / self.tau_exc_2
        )
        xdot2_next_2 = (
            xdot2_2
            + self.dt
            * (
                self.G_inh_2 * self.C_4 * firing_rates[2]
                - 2 * xdot2_2
                - x2_2 / self.tau_inh_2
            )
            / self.tau_inh_2
        )
        x0_next_2 = x0_2 + xdot0_2 * self.dt
        x1_next_2 = x1_2 + xdot1_2 * self.dt
        x2_next_2 = x2_2 + xdot2_2 * self.dt
        self.x = np.array(
            [
                x0_next_1,
                x1_next_1,
                x2_next_1,
                xdot0_next_1,
                xdot1_next_1,
                xdot2_next_1,
                x0_next_2,
                x1_next_2,
                x2_next_2,
                xdot0_next_2,
                xdot1_next_2,
                xdot2_next_2,
            ]
        )

    def read_out(self):
        return self.w * (self.x[1] - self.x[2]) + (1 - self.w) * (self.x[7] - self.x[8])

    def simulate(self, x0=None, tmax=1.0, noise=0.0, P=None):
        """
        Simulate the extended Jansen-Rit model and monitor the output.

        Parameters
        ----------
        x0 : array_like, optional
            The initial state of the system. Shape (12,). Defaults to zeros.
        tmax : float
            The maximum time to simulate.
        noise : float
            The standard deviation of the noise to add to the system.
        P : float or array_like, optional
            The external input to the excitatory population. A scalar is
            applied at every step, an array of shape (n,) is indexed per
            step. If ``None``, the constant input set at construction is
            used.

        Returns
        -------
        x : ndarray
            The simulated state. Shape (n, 12).
        o : ndarray
            The simulated readout (weighted excitatory minus inhibitory).
            Shape (n, 1).
        """
        rng = np.random.default_rng(self.seed)
        n = int(tmax / self.dt)
        x = np.zeros((n, 12))
        o = np.zeros((n, 1))
        if x0 is None:
            x0 = np.zeros(self.nstates)
        self.x = np.asarray(x0, dtype=float).copy()
        if self.x.shape != (self.nstates,):
            raise ValueError(f"x0 must have shape ({self.nstates},)")
        x[0] = self.x
        o[0] = self.read_out()
        dt_noise = np.sqrt(self.dt) * noise
        for i in range(1, n):
            current_P = self.P if P is None else (P if np.ndim(P) == 0 else P[i])
            self.step(I=current_P)
            # noise = rng.standard_normal(size=self.x.shape) * dt_noise
            x[i] = self.x
            o[i] = self.read_out()
        return x, o


class JRNetwork(NeuralMassNetwork):
    """
    Network of coupled Jansen-Rit extended neural-mass nodes.

    Each node is an :class:`JansenRitExtended` cortical column; nodes are
    coupled through the connectivity matrix ``W`` with a delay ``delay`` and
    activity-dependent normalisation of the coupling strengths.

    Notes
    -----
    Two types of networks are modelled in the literature: either one when the mean input and variance of the input
    are controlled for each node, such that the input received from connected nodes is normalised and will relatively
    shut down contribution from external input, or one where the input is not normalised and the external input is
    simply summed over the input from connected nodes.
    The latter is seen in [2]_ and [3]_, while the former is seen in [4]_ & [5]_.

        References
        ----------

        - [1]_ Jansen, B. H., & Rit, V. G. (1995). Electroencephalogram and visual evoked potential generation in a mathematical model of coupled cortical columns. *Biological cybernetics*, 73(4), 357-366.
        - [2]_ Kazemi & Jamali, (2022). Phase synchronization and measure of criticality in a network of neural mass models. `Nature <https://www.nature.com/articles/s41598-022-05285-w>`_.
        - [3]_ Forrester et al. (2020). Network Neuroscience. The role of node dynamics in shaping emergent functional connectivity patterns in the brain. `PMC <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7286301/#bib70>`_.
        - [4]_ David & Friston (2006). *NeuroImage*. A Neural mass model for MEG/EEG: coupling and neuronal dynamics. `ScienceDirect <https://www.sciencedirect.com/science/article/pii/S1053811903004579?ref=cra_js_challenge&fr=RR-1>`_.
        - [5]_ David et al., (2004). Evaluation of different measures of functional connectivity using a neural mass model. `ScienceDirect <https://www.sciencedirect.com/science/article/pii/S1053811903006566?ref=pdf_download&fr=RR-2&rr=8279b478ba0328ac#APP1>`_.

    .. [1] Jansen, B. H., & Rit, V. G. (1995). Electroencephalogram and visual evoked potential generation in a mathematical model of coupled cortical columns. *Biological cybernetics*, 73(4), 357-366.
    .. [2] Kazemi & Jamali, (2022). Phase synchronization and measure of criticality in a network of neural mass models. `Nature <https://www.nature.com/articles/s41598-022-05285-w>`_.
    .. [3] Forrester et al. (2020). Network Neuroscience. The role of node dynamics in shaping emergent functional connectivity patterns in the brain. `PMC <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7286301/#bib70>`_.
    .. [4] David & Friston (2006). *NeuroImage*. A Neural mass model for MEG/EEG: coupling and neuronal dynamics. `ScienceDirect <https://www.sciencedirect.com/science/article/pii/S1053811903004579?ref=cra_js_challenge&fr=RR-1>`_.
    .. [5] David et al., (2004). Evaluation of different measures of functional connectivity using a neural mass model. `ScienceDirect <https://www.sciencedirect.com/science/article/pii/S1053811903006566?ref=pdf_download&fr=RR-2&rr=8279b478ba0328ac#APP1>`_.

    """

    def __init__(
        self,
        N=2,
        W=np.asarray([[0, 1], [0, 0]]),
        delay=0.01,
        w=0.8,
        node_dynamics=None,
        dt=0.001,
        seed=42,
    ):
        """
        Parameters
        ----------
        N : int
            The number of neurons/nodes.
        W : array_like
            The connectivity matrix. Shape (N, N). E.g. W = np.asarray([[0, 1], [0, 0]]) means that node 1 is connected to node 2, while node 2 is not connected to node 1.
        delay : float
            The delay between nodes in seconds. Default is 10ms.
        w : float or array_like
            The relative contribution of the first (slow) subpopulation of
            each :class:`JansenRitExtended` node. A scalar is applied to all
            nodes, an array must have shape (N,).
        node_dynamics : class, optional
            Ignored; kept for interface compatibility with
            :class:`NeuralMassNetwork`. Nodes are always
            :class:`JansenRitExtended` instances.
        dt : float
            The integration time step in seconds.
        seed : int
            The random seed used to initialise the network's random number
            generator and the per-node seeds.
        """
        self.rng = np.random.default_rng(seed)
        self.N = N  # number of neurons/nodes
        self.W = W  # connectivity matrix (W_ij is the connection from i to j, between 0 and 1, relative contribution)
        self.K = (
            W.copy()
        )  # updated connectivity in case of normalisation by activity std
        self.delay = delay  # delay (10ms)
        self.dt = dt  # sampling rate
        self.seed = seed  # random seed
        if not np.isscalar(w):  # if w is a scalar, then it is the same for all nodes
            w = w * np.ones((self.N,))
        self.nodes = [
            JansenRitExtended(w=w, dt=dt, seed=self.rng.integers(k + seed))
            for k in range(N)
        ]  # get different systems/rng for each node
        self.S = self.nodes[0].S  # nonlinearity function
        # self.delayed_states = np.zeros((N, self.nodes[0].nstates)) # delayed states of the nodes (state of the nodes at t-dt)
        self.delayed_states = np.zeros(
            (N, 1)
        )  # delayed states of the nodes (state of the nodes at t-dt) / readout only

    def update_connectivity(self, x, sigma_p=1):
        """
        Update the coupling matrix ``K`` from the history of node outputs.

        The coupling strengths are normalised by the standard deviation of
        each node's firing rate so that the mean input to each node is
        conserved across coupling strengths (see David & Friston, 2004).

        Parameters
        ----------
        x : array_like
            The firing-rate outputs of all nodes, needed to compute the
            standard deviation. Shape (N, ntimes).
        sigma_p : float or array_like
            The standard deviation of the external input fluctuations. A
            scalar is applied to all nodes, an array must have shape (N,).
        """
        # See ref (David & Friston 2004: A Neural Mass model for M/EEG: coupling and neuronal dynamics)
        if np.isscalar(
            sigma_p
        ):  # if sigma_p is a scalar, then it is the same for all nodes
            sigma_p = sigma_p * np.ones((self.N,))

        if np.ndim(x) <= 1 or x.shape[1] <= 1:
            sigma_rate = np.ones((self.N,))
        else:
            sigma_rate = np.std(x, axis=1)
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    # k12_star[i] is the normalisation factor for connection from i to j
                    self.K[i, j] = (
                        sigma_p[j]
                        * np.sqrt(2 * self.W[i, j] - self.W[i, j] ** 2)
                        / sigma_rate[i]
                    )
        # Then update self.K

    def simulate(self, tmax=1.0, P=220, sigma_p=22):
        """
        Simulate the network and return the readout of every node.

        Parameters
        ----------
        tmax : float
            The duration of the simulation in seconds.
        P : float
            The constant external input to each node.
        sigma_p : float
            The standard deviation of the external input fluctuations.

        Returns
        -------
        outs : ndarray
            The simulated readouts. Shape (nsamples, N). The number of
            samples is ``int(tmax / dt)``.
        """
        outs = []
        t = np.arange(0, tmax, self.dt)
        nsamples = len(t)
        tdelay, kdelay = 0, 0
        for k, tt in enumerate(t):
            outs.append(
                self.step(P=P, sigma_p=sigma_p, history_outs=self.S(np.asarray(outs).T))
            )
            tdelay += self.dt
            kdelay += 1
            if tdelay >= self.delay:
                self.delayed_states = outs[-kdelay]
                tdelay, kdelay = 0, 0
        return np.asarray(outs)

    def step(self, P=220, sigma_p=22, history_outs=None):
        """
        Advance the network by one integration step.

        The coupling matrix is updated from the history of the outputs, then
        every node receives the external input ``P`` plus a fluctuating
        external input plus the inter-area contributions from the other
        nodes.

        Parameters
        ----------
        P : float
            The constant external input to each node.
        sigma_p : float
            The standard deviation of the external input fluctuations.
        history_outs : array_like, optional
            The firing-rate outputs of all nodes up to the current time,
            used to update the coupling matrix. Shape (N, ntimes).

        Returns
        -------
        outs : ndarray
            The readout of every node after the step. Shape (N,).
        """
        if np.isscalar(
            sigma_p
        ):  # if sigma_p is a scalar, then it is the same for all nodes
            sigma_p = sigma_p * np.ones((self.N,))

        self.update_connectivity(
            history_outs, sigma_p=sigma_p
        )  # update connectivity based on the history of the outputs
        external_input_fluctuation = (
            sigma_p * self.rng.standard_normal(size=(self.N,))
        ) * (1 - self.W.sum(axis=0))
        interarea_contributions = (
            self.S(self.delayed_states).ravel() - 3.84
        ) @ self.K  # using normalised connectivity
        outs = []
        for k, n in enumerate(self.nodes):
            # Below y_1 and y_2 corresponds to the fast and slow dynamics respectively of the extended Jansen-Rit model
            #  I = p + (1-k21) * p̃ + k21_star *(S(w*y_1(t - δ) + (1-w)*y_2(t - δ)) - a) where "a" is the mean firing rate
            # we remove the mean firing rate to ensure mean input is conserved with different coupling strengths
            # a = 3.5 in the reference paper, my measured mean is more around 3.84
            n.step(
                I=P
                + external_input_fluctuation[k]  # noise input scaled : (1-k21) * p̃
                + interarea_contributions[
                    k
                ]  # k21_star * (S(w*y_1(t - δ) + (1-w)*y_2(t - δ)) - a)
            )  # the input to each node is the output of all the other nodes
            outs.append(n.read_out())
            # for i in range(self.N):
            #     I += self.K[i, n] * (outs[i] - 3.84)
        return np.asarray(outs)

    def reset(self):
        """
        Reset the network to its initial state.

        The delayed states are zeroed, the state of every node is set to
        zero, and the coupling matrix ``K`` is restored to the original
        connectivity ``W``.
        """
        self.delayed_states = np.zeros((self.N, 1))
        for n in self.nodes:
            n.x = np.zeros((n.nstates,))
        self.K = self.W.copy()


def dummy_trf_kernel(
    tmin=-0.2, tmax=0.5, srate=100, tloc=0.1, sigma=0.1, kernel_type="gaussian"
):
    """
    Dummy kernel for testing purposes.

    Parameters
    ----------
    tmin : float
        The minimum time of the kernel.
    tmax : float
        The maximum time of the kernel.
    srate : int
        The sampling rate of the kernel.
    tloc : float
        The location of the peak of the kernel.
    sigma : float
        The standard deviation, or width, of the kernel.

    Returns
    -------
    kernel : array_like
        The kernel. Shape (n,).
    """
    t = np.arange(tmin, tmax, 1 / srate)
    if kernel_type == "gaussian":
        return t, np.exp(-0.5 * ((t - tloc) / sigma) ** 2) / (
            sigma * np.sqrt(2 * np.pi)
        )
    elif kernel_type == "exponential":
        return t, np.exp(-np.abs(t - tloc) / sigma) / (2 * sigma)
    elif kernel_type == "bipolar":
        sigma /= 2  # half the width of the kernel
        tloc += sigma  # shift the location of the peak to the right
        ker = np.exp(-0.5 * ((t - tloc) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        return t, np.diff(np.r_[0, ker])  # derivative of gaussian


def simulate_smooth_input(dur=1.0, srate=100, fmax=10, seed=12):
    """
    Simulate a smooth input signal.

    Parameters
    ----------
    dur : float
        The duration of the signal in seconds.
    srate : int
        The sampling rate of the signal.
    fmax : float
        The maximum frequency of the signal.

    Returns
    -------
    t : array_like
        The time vector. Shape (n,).
    x : array_like
        The simulated signal. Shape (n,).
    """
    rng = np.random.default_rng(seed)
    n = int(dur * srate)
    t = np.arange(0, dur, 1 / srate)
    x = rng.standard_normal(size=n)
    b, a = scisig.butter(4, fmax / (srate / 2), btype="low")
    x = scisig.filtfilt(b, a, x)  # filter the signal
    return t, x


def simulate_pulse_inputs(n_events=100, dur=30.0, srate=100, seed=12):
    """
    Simulate a pulse input signal.

    Parameters
    ----------
    n_events : int
        Number of events to simulate.
    dur : float
        The duration of the signal in seconds.
    srate : int
        The sampling rate of the signal.

    Returns
    -------
    t : array_like
        The time vector. Shape (n,).
    x : array_like
        The simulated signal. Shape (n,).
    """
    n = int(dur * srate)
    t = np.arange(0, dur, 1 / srate)
    x = np.zeros(n)
    # Generate Poisson distributed events
    event_times = poisson_onsets_fixed_N(n_events, dur, seed=seed)
    x[(event_times * srate).astype(int)] = 1  # set the events to 1
    return t, x


def simulate_trf_output(tkernel, kernel, input, srate=100):
    """
    Simulate the output of a kernel given an input signal.

    Parameters
    ----------
    tkernel : array_like
        The time vector of the kernel. Shape (nker,).
    kernel : array_like
        The kernel. Shape (nker,).
    input : array_like
        The input signal. Shape (n,).
    srate : int
        The sampling rate of the signal.

    Returns
    -------
    output : array_like
        The output signal. Shape (n,).
    """
    n = len(input)
    tmin, tmax = tkernel[0], tkernel[-1]
    # if tker is not symmetric around 0, we need to pad the kernel
    if np.abs(tmin) > np.abs(tmax):
        kernel = np.pad(
            kernel,
            (0, int(np.abs(tmin) * srate) - int(np.abs(tmax) * srate)),
            "constant",
        )
    elif np.abs(tmax) > np.abs(tmin):
        kernel = np.pad(
            kernel,
            (int(np.abs(tmax) * srate) - int(np.abs(tmin) * srate), 0),
            "constant",
        )
    return np.convolve(input, kernel, mode="same")
