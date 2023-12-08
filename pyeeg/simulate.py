# -*- coding: utf-8 -*-
"""
=====================
Simulations utilities
=====================

Simulate MEEG-like signals with different connectivity patterns or methods.

TODO:
    - Follow up on those:
        - RNN of rate models: https://elifesciences.org/articles/69499
        - ctRNN revision, see https://www.nature.com/articles/s42256-023-00748-9#Sec9
    - Simulation based on connectivity matrix: I should be anbleto implement a network
    of N nodes simply using the the connectivity matrix and node-specific implementations
    - Neural mass models to be added:
        - Wilson-Cowan
    - Simulations of TRF-based signals

References:
    - Janseen-Rit model: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10473283/, See https://pdf.sciencedirectassets.com/272508/1-s2.0-S1053811900X00973/1-s2.0-S1053811903006566/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEAAaCXVzLWVhc3QtMSJHMEUCIEBRfmnMkb2j8ut2p46cc2y6emk9Jl4srBYEgaVRmPS%2FAiEAtKpadf9qZgbW0vxfxuG0FxR2CXU8iGh9M9VZ6T36r%2BMqswUISBAFGgwwNTkwMDM1NDY4NjUiDAOu3ZZjVACJ4G%2B%2F4yqQBbt8wNEj1LK7ozKjn8OBKbG3gcdZaU0Sg%2FLJMMQfgZnBK0iFT%2BHicPKc%2BEzpvhyikXNHGv%2FXtokrD%2Favx5YbL%2B7rKox9FRVvl9pVFrdaSkf%2BhkE2ACQM6nxOFXVbfpsd0QSEGYH40O8EL%2F5FQOxBcuLU2SE1wA9xSjpHnQB0CM1Q2WxF67v0WAkiSIuwAA5hwfscGL%2BUzRidKmTWC8B8lzPKki0D0jZVngPHRHuawuZbmR07LHye1pgPBacqY%2B3DBrOrIjgZXU%2FMzPw1kgcou%2Fd0nJnJgfEFotmAuhj%2FgbLPuOu0ROhYeJeCUTFQ9cXwaIkN%2FpGPwV3EtLvkH7QvEpsfKDMOR1iB4YYMh1oe75O7lx6hJ5qWpHCFgEh0Xg5Z%2BatTlrifNzMyr5isVLVrUZElzdY%2B3ZCZKaX%2FA6zJBp0Y%2FsDCvV4IU5K2GHOzTWyimbTQd2WUk%2BM4FEpn9U38zsTnEA5I5%2F5il8dMHS4fRDCF0dEarsdRbi%2BRXuLnFU8%2FTkkRyMGeqh4nIt8%2F7MyvrWE5D2YqSwImBeknhAVQ%2BF%2BHUWqU9tNVrnbuchks%2BZWB4E2%2BkEJeUBtFlswq5W5DuPMVAzT0IbCkITPSLvHtyAuAvPD2S3zfduMgPu98lwYfijcDEdS58uc05WQiUIHg8H3u%2F46DHT8MbHFp7B8FgkWqYWzof1yCIA6dOzjD9AvkSjGCxtbUqiDgGV88pzvnS9CKEX%2BKt1k5bLebz6MERTnPBd%2BTbrFGpHJlzKB%2FCbfVIkEzqvZQIKCkMUm7I8JfainxdL3y4JjKMxi75aoenMZSPOhBRPNQFglT43ZYjuZXNitc4P1KpEDkIGkS7iSdeU5576MRicX3fID9feeeMGlYMPf8yKoGOrEBRrcOQuw9PMvlCvY0kUumTwNahcqJuAbPVCj5UFZKOf9bBMEOcC1Xbi5r%2FCoOYPPf9yyQIuQ4pk5UTOs3E%2BBHrMqCboKDNcLFdkc6oWOtD8yF%2BaZddbsrfT2%2FdIWe80mM65qPFbm374EtTfoo8mGY%2F8QwlzJfEUYFSUgpZr5%2B32O2pJNnNIGQxL3suTL9hEGSvZAaxEgE3TcP073vpVxDHZzWPai90pW%2FtrJCH0v9vkdi&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20231113T164609Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYRX55LHJ7%2F20231113%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=efb4397d32d0283d891f7f7b6b54cf6086f166c7f49d7c73cd85383c2ace1215&hash=c12ba03b2ecaf8322bf34c85889ef48d9263bd5c22c8fe4fdc4fd77c9ddd950c&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1053811903006566&tid=spdf-fd17f1b2-9263-4484-94b2-caa5ff6aeb65&sid=1f34fbf92ab4c94b979becc2279c4fddfea3gxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=080f56555356560150&rr=82587d3bad7f66a5&cc=nl
        or https://pdf.sciencedirectassets.com/272508/1-s2.0-S1053811900X00924/1-s2.0-S1053811903004579/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEAAaCXVzLWVhc3QtMSJIMEYCIQCLOeRLcqnzm1kuLuu9gAR4xW20REgfwC%2BUG7p0FPu14gIhANRW6LXlFS2LON%2F%2BgS1jbn%2F7a8w6IHV7gKbTbNV5wqjnKrIFCEkQBRoMMDU5MDAzNTQ2ODY1Igy1V1bpLECYjeUFuUEqjwVSOS4pH6j2ew1gfsyDFISqu%2FmCuyDnXJ%2BkanuTqSOt35WOcurB7Pczd0RdZV4dTzoTbW0ZoPMv6dCNi8tfQ4oaeaxWA3mN0Vs%2FIaYMy8eyCOkgxTFBPEOxGmXFV1hBWQhmske05vPkhjTf37BfdI7Ipik7o%2BHMGfIKoQKO8zybRghfnsSyV3crhDAbQx9GF%2BOe%2FKNp7X5Yx06OV953HqhGiu%2Fs5wITv2fSRaAvoLHoMkiYLo8B0jNIynMY71cB9p68YgUPZ%2FAxfUiSHT%2FwjTywZO1sPhVSB0Bm2HqCDh5soRiZ07MylsEDbYbnrW%2B7aCVBRKykjaQ9IFK4OwiaE%2BK7WQm6pdztg3oLmY5Su%2FWU90iUY41Ju1dUGmhsFfO8Q8WTlnnZ5GBRzGcgqKxuYcKHWxUBieD5U6w2FSynBti2ryxOFiiGtXI%2BeODXN1Ea0qoR9cT7HLExDbFyL101aLiyP%2FTSFjq%2Buygyfwl0yk6%2FkJdgWZYB3Xllmi9s7uc3sZ3KsZo5P4m8we2fz5XrMDpCLsiuTMfUwfZatWon8GXyysOSOHwoCVvS%2Btn%2BnZxRGbNTcNvvRmSQz5TGJ02lKzNOC5Bd0y9fKCfuNSXVcL1FutS2oJtonpy0iiLc3mgczpXspnK3kJaSAJo%2FFXaW8OVupYm3C8hQx5E9wCb1c%2FdSCLcaitk%2FZA3eGjbIXjcUiTHuiit6N20F6VhzUNTkSuM7FZzNH%2Fkl7Cmt72Z13TDiG%2BUkrD3RA2dePMS5XvqkwiWvLWK%2Bbljweot6RbbwfRoLc4ouc8Fy4pVsUhlABP6iAFGRpz7lGEUDo6cxpYp1J8aoUbpnibvPR1SdHUSz%2BGbq7PXxmjEO%2Bu0Lu2D7JZqAMO2LyaoGOrABbIrfm40nIMybDuTZ4mK%2BoaIvNb9Vc0xLy1Nypl6sNZoLJl4oVqI0WYNwtHXNhpbGBR8jq2%2F%2FL24tih289cLG8S5GV07PE5oeIJj7DfywGo8PPko2O6vViv9%2BZo4i5RIEn259RUd2IXplHZd57rggnFTY4ac8m9CEbjtk7gTW7HF7DNFaFMRv6D8K%2F5hpLKGHjyF%2BZhxZ79fcAN%2B5TgvVIBYRWS4Qi%2F%2B6NMhnU28QTVo%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20231113T164603Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY742FACEQ%2F20231113%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=da2d7fe1b02d5c068a1f3c7c8f2fdc64b8350e1f909000d3d6906267796778a2&hash=82bcc505cc92fe4f322349938bc58f4add1575143448707ba5b16def04ceade7&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1053811903004579&tid=spdf-2c8c5161-288d-41b8-83ac-547a9870612e&sid=1f34fbf92ab4c94b979becc2279c4fddfea3gxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=080f56555356560152&rr=82587d1569b866a5&cc=nl

Update:
    - 22/11/2023: Network class added and JR network implemented
    - 17/11/2023: added Jansen-Rit model
    - 10/11/2023: initial commit (AR and VAR simulations)
"""
import numpy as np
from .utils import sigmoid

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
    for i in range(n+order):
        if i < order:
            x[i] = rng.standard_normal()* sigma
        else:
            x[i] = np.dot(coefs[::-1], x[i-order:i]) + rng.standard_normal()*sigma
    return x[order:]

def simulate_var(order, coef, nobs=500, ndim=2, seed=42, verbose=False):
    """
    Simulate a VAR model of order `order`.

    The VAR model is defined as:

    .. math::
        x_t = A_1 x_{t-1} + A_2 x_{t-2} + ... + A_p x_{t-p} + \\epsilon_t
        x_t = \\sum_{i=1}^p A_i x_{t-i} + \\epsilon_t

    where :math:`x_t` is a vector of shape (ndim, 1), :math:`A_i` is a matrix of shape (ndim, ndim)

    .. note::
        The coefficients at a given lag are such as :math:`C_ij` is i->j, so it will be the coefficients for dimension j!
        For example, each row of the first column are determining the contributions of each component onto the first component.
    """
    rng = np.random.default_rng(seed)
    if order == 1 and coef.ndim == 2:
        coef = coef[None, :, :]
    assert coef.shape == (order, ndim, ndim), "coef must be of shape (order, ndim, ndim)"
    data = np.zeros((nobs+order, ndim))

    if verbose:
        print(f"Simulating VAR({order}) model with {ndim} dimensions and {nobs} observations")
        print(f"Data shape: {data.shape}")

    data[:, :] = rng.standard_normal(size=data.shape) # initialize with noise
    for t in range(order, nobs+order):
        for lag in range(order):
            data[t] += data[t-(lag+1)] @ coef[lag] # here if I multiply from the left, I get the contributions row wise instead of column wise

    return data[order:, :]

def simulate_var_from_cov(cov, nobs=500, ndim=2, seed=42, verbose=False):
    """
    Simulate a VAR model of order `order` from a covariance matrix.

    The VAR model is defined as:

    .. math::
        x_t = A_1 x_{t-1} + A_2 x_{t-2} + ... + A_p x_{t-p} + \\epsilon_t
        x_t = \\sum_{i=1}^p A_i x_{t-i} + \\epsilon_t

    where :math:`x_t` is a vector of shape (ndim, 1), :math:`A_i` is a matrix of shape (ndim, ndim)

    .. note::
        The coefficients at a given lag are such as :math:`C_ij` is i->j, so it will be the coefficients for dimension j!
        For example, each row of the first column are determining the contributions of each component onto the first component.
    """
    rng = np.random.default_rng(seed)
    order = cov.shape[0]
    assert cov.shape == (order, ndim, ndim), "cov must be of shape (order, ndim, ndim)"
    data = np.zeros((nobs+order, ndim))

    if verbose:
        print(f"Simulating VAR({order}) model with {ndim} dimensions and {nobs} observations")
        print(f"Data shape: {data.shape}")

    data[:, :] = rng.standard_normal(size=data.shape) # initialize with noise
    for t in range(order, nobs+order):
        for lag in range(order):
            data[t] += data[t-(lag+1)] @ np.linalg.cholesky(cov[lag]) # here if I multiply from the left, I get the contributions row wise instead of column wise

    return data[order:, :]

class NeuralMassNode(object):
    """
    Abstract class for neural mass models.
    Defines the function to be implemented for the simulation.
    """
    def __init__(self, dt=0.001, seed=42):
        self.dt = dt # sampling rate
        self.seed = seed # random seed

    def simulate(self):
        raise NotImplementedError("This method must be implemented in the child class")
    
    def step(self):
        raise NotImplementedError("This method must be implemented in the child class")
    
class NeuralMassNetwork(object):
    """
    Abstract class for neural mass models.
    Defines the function to be implemented for the simulation.
    """
    def __init__(self, N, W, delay=0, node_dynamics=None, dt=0.001, seed=42):
        self.rng = np.random.default_rng(seed)
        self.N = N # number of neurons/nodes
        self.W = W # connectivity matrix
        self.K = W # updated connectivity in case of normalisation by activity std
        self.delay = delay # delay 
        self.dt = dt # sampling rate
        self.seed = seed # random seed
        self.node_dynamics = node_dynamics # node dynamics instance
        if node_dynamics is not None:
            self.nodes = [node_dynamics(seed=self.rng.integers(0, k+1)) for k in range(N)] # get different systems/rng for each node

    def simulate(self):
        raise NotImplementedError("This method must be implemented in the child class")
    
    def step(self):
        outs = []
        for n in self.nodes:
            outs.append(n.read_out())
            # incorporate delays here
            n.step(I=self.K @ np.asarray(outs)) # the input to each node is the output of all the other nodes
    
class CTRNN(NeuralMassNetwork):
    """
    Continuous Time Recurrent Neural Network (CTRNN) model.

    .. math::
        \\tau \\dot{x} = -x + W o + I + \\theta

    """
    def __init__(self, N, W, input_dim=1, output_dim=1, dt=0.001, seed=42, nonlinearity=sigmoid, theta=None):
        """
        Parameters
        ----------
        N : int
            The number of neurons/nodes.
        W : array_like
            The connectivity matrix. Shape (N, N).
        nonlinearity : callable
            The nonlinearity function to apply to the network. Default is sigmoid (e.g. can use func:`np.tanh`)    
        """
        super().__init__(N=N, W=W, dt=dt, seed=seed)
        self.nonlinearity = nonlinearity # nonlinearity function
        self.readout_W = np.zeros((output_dim, N)) # readout matrix
        self.input_W = np.zeros((N, input_dim)) # input matrix
        self.theta = theta if theta is not None else np.zeros((N,))
        self.x = np.zeros((N,)) # state of the network
        self.o = np.zeros((N,)) # output of the network

    def step(self, I=None, noise=0.):
        """
        Compute one step of the CTRNN model.
        """
        if I is None:
            I = np.zeros((self.input_W.shape[1],))
        elif np.isscalar(I):
            I = np.ones((self.input_W.shape[1],)) * I
        self.x = self.x + self.dt * (-self.x + self.W @ self.o + self.input_W @ I) + noise
        self.o = self.nonlinearity(self.x + self.theta)

    def read_out(self):
        return 2 * self.nonlinearity( self.readout_W @ self.o) - 1 # this is in the range -1 to 1 if the nonlinearity is sigmoid

    def simulate(self, x0, tmax=1, noise=0., I=lambda t: 0.):
        """
        Simulate the CTRNN model and monitor the output.

        Parameters
        ----------
        x0 : array_like
            The initial state of the system. Shape (N,).
        tmax : float
            The maximum time to simulate.
        noise : float
            The standard deviation of the noise to add to the system.
        
        Returns
        -------
        x : array_like
            The simulated time series. Shape (n, N).
        """
        rng = np.random.default_rng(self.seed)
        n = int(tmax / self.dt)
        x = np.zeros((n, self.N))
        o = np.zeros((n, self.N))
        O = np.zeros((n,))
        x[0] = x0
        self.x = x0
        dt_noise = np.sqrt(self.dt) * noise
        for i in range(1, n):
            self.step(I=I(i*self.dt), noise = rng.standard_normal(size=self.x.shape) * dt_noise)
            x[i] = self.x
            o[i] = self.o
            O[i] = self.read_out()
            
        return O, x, o
    
    def read_out(self):
        return 2 * self.nonlinearity( self.readout_W @ self.o) - 1
    
class JansenRit(NeuralMassNode):
    """
    Jansen-Rit model.

    3 populations: excitatory, inhibitory and pyramidal:
    
    ```ascii
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
    ```

    Parameters and typical values as in Grimbert & Faugeras, 2006:
    - C1, C2, C3, C4	Average number of synapses between populations	135 * [1 0.8 0.25 0.25]
    - tau_e             Time scale for excitatory population	        100 ms
    - tau_i	            Time scale for inhibitory population	        50 ms
    - G_exc	            Average excitatory synaptic gain	            3.25
    - G_inh	            Average inhibitory synaptic gain	            22
    - rmax	            Amplitude of sigmoid	                        5 s^-1
    - beta	            Slope of sigmoid	                            0.56 mV^-1
    - theta	            Threshold of sigmoid	                        6 mV
    - Conduction velocity	 	                                        10 m/s
    - h	                Integration time step	                        0.0001 (s) by default
    - P	                External input to each of the neural masses	    150 (Hz, whihc is a contant input)
    - Coupling	        Coupling between the neural masses	            [0.1:0.012:0.292]

    This table is from the paper:
    [This paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10473283/)
    
    """
    def __init__(self, dt=0.0001, seed=42, nonlinearity=sigmoid):
        super().__init__(dt, seed) # this is a single node (cortical column with 3 sub-populations)
        n_synapses = 135 # number of synapses between populations
        self.C_1 = 1. * n_synapses # probability of connection between excitatory and pyramidal populations
        self.C_2 = 0.8 * n_synapses
        self.C_3 = 0.25 * n_synapses
        self.C_4 = 0.25 * n_synapses
        self.tau_exc = 1/100 # time scale for excitatory population ~10ms
        self.tau_inh = 1/50 # time scale for inhibitory population ~20ms
        self.G_exc = 3.25 # average excitatory synaptic gain (mV)
        self.G_inh = 22 # average inhibitory synaptic gain
        self.rmax = 5 # amplitude of sigmoid in Hz (max firing rate)
        self.beta = 0.56 # slope of sigmoid (mV^-1)
        self.theta = 6 # threshold of sigmoid (mV)
        self.v = 10 # conduction velocity
        self.P = 150 # external input to each of the neural masses
        #self.Coupling = 0.1 # coupling between the neural masses (global coupling strength)
        self.nstates = 6 # number of state variables

        self.x = np.zeros((6,)) # state of the network
        self.S = lambda x: nonlinearity(x, rmax=self.rmax, beta=self.beta, x0=self.theta) # nonlinearity function

    def step(self, I=0.):
        """
        Compute one step of the Jansen-Rit model.
        """
        # 0: pyramidal, 1: excitatory, 2: inhibitory
        x0, x1, x2, xdot0, xdot1, xdot2 = self.x # unpack the state
        # Input received by each population
        # x1 - x2: difference between excitatory and inhibitory activity, which is the input received by the pyramidal population interpreted as the average potential of pyramidal populations
        # self.C_1 * x0: input received by the excitatory population
        # self.C_3 * x0: input received by the inhibitory population
        firing_rates = self.S(np.asarray([x1 -x2, self.C_1 * x0, self.C_3 * x0]))
        input_excitatory = self.C_2 * firing_rates[1] + I # contribution from other nodes will go here
        xdot0_next = xdot0 + self.dt * (self.G_exc * 1.0      * firing_rates[0] - 2 * xdot0 - x0/self.tau_exc ) / self.tau_exc # pyramidal cell
        xdot1_next = xdot1 + self.dt * (self.G_exc * input_excitatory           - 2 * xdot1 - x1/self.tau_exc) / self.tau_exc  # excitatory stellate cell
        xdot2_next = xdot2 + self.dt * (self.G_inh * self.C_4 * firing_rates[2] - 2 * xdot2 - x2/self.tau_inh) / self.tau_inh # inhibitory interneuron
        x0_next = x0 + xdot0 * self.dt
        x1_next = x1 + xdot1 * self.dt
        x2_next = x2 + xdot2 * self.dt
        self.x = np.array([x0_next, x1_next, x2_next, xdot0_next, xdot1_next, xdot2_next])

    def read_out(self):
        return  self.x[1] - self.x[2]

    def simulate(self, x0, tmax=1, noise=0., P=None):
        """
        Simulate the Jansen-Rit model and monitor the output.

        Parameters
        ----------
        x0 : array_like
            The initial state of the system. Shape (N,).
        tmax : float
            The maximum time to simulate.
        noise : float
            The standard deviation of the noise to add to the system.
        
        Returns
        -------
        x : array_like
            The simulated time series. Shape (n, N).
        """
        rng = np.random.default_rng(self.seed)
        n = int(tmax / self.dt)
        x = np.zeros((n, 6))
        o = np.zeros((n, 1))
        self.x = x0
        x[0] = x0
        o[0] = self.read_out()
        dt_noise = np.sqrt(self.dt) * noise
        for i in range(1, n):
            self.step(I = self.P if P is None else P[i])
            # noise = rng.standard_normal(size=self.x.shape) * dt_noise
            x[i] = self.x           
            o[i] = self.read_out()
        return x, o
        
class JansenRitExtended(NeuralMassNode):
    """
    Jansen-Rit model - Exctended version: dual kinetic model.

    See https://pdf.sciencedirectassets.com/272508/1-s2.0-S1053811900X00924/1-s2.0-S1053811903004579/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEKH%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIQDklbniEg%2BIxtbMhqre0GLXBUY61F7QwRGlTcS0Tw50mAIgPijMAG0JRY86ILcJl5khJAbWRSGrqqz8rRNKkeVXhtMqvAUI6v%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAFGgwwNTkwMDM1NDY4NjUiDDWGjReDtp%2FZt%2BvT4SqQBUgH%2FBzAz1ffqLsZXij%2Fhvyo540aF9iKjI5qqwn%2FClzZq1dP%2BofFqhpbLq%2FbRaP98Et%2Bs5VZoAoNHhC8dMDRlrxY47ZuTWehsc2C8ZUcz6D9lYArh5ggiCQZMm9040OWVNDarLG8631K4g0HFpEHwsubfZoIUgs5XABH%2FyF1NE2zXo3JXhU%2FKwsZHVqqidrV0nv%2B9IJ5%2BigTmMVePAINzRUuQjnFgbyqNvqMwUsNDi92QSN7u%2BdJi0ksGwqTWyBCM7MUrwK%2FisZjWmQcQeCx%2FuLyhE77tU6x7gpR%2BGdE%2BOmiISZDROqxwq%2FmsC%2B%2FR%2BMsFubuDhdTI8n8kllkh079IhICKgEzhzFLWgmNxXdbxBS6vQi13IA5SNlQuuzArMu0Z1GkM9EwbXRUI3u4oeOMtGOslgCYdgsSJpebcg8zOg2ueLBgHSfubsGGJ5l5SxOR%2FIVF6tHjzLoCz2njmpmtSr%2Bmi56T9qKmkUC4sAoYnbpllz3yd%2F4%2B2Xh3AruYCVWyJ%2FVLgTbrPXYpihKXryoL1DNw94iDb%2FWutOEqh46F64lPV96HjCvNME4smEYyyPKz8DWYdMcU6U35qVbk%2BXlv%2FoDv7AIjCJi3l9U98IL%2BiZMAnCN8u88BtwF8o7PrMAMfRzjQl53oJEt%2BCfnIMZhwy0raTnU3Sb61tmgBqoIrY7NFmWjrAmqNJilwO16T7RIJkjRhufHocRzquTEj6F5NyPrir%2FeVwlg3%2BM8kgmsdsiIfLR9tkvkXCcSx1M4%2FwdK9id37RAh5D%2Fcw0jAcqEinfgj%2Bw4Z9MX5FU91B9pOJIBk%2BGRraBuxSsxznjeQ%2BlOZVwtQ7nVwE%2Bk0XUHkMv075s9RxWPIrg48wrQMqJUhLaW7zMK%2B%2B7KoGOrEBKbm0kOWBELm7BGve3m4Vj6De%2BonWolZfqid1yachKSVCGGly07GpJptQJqCsQftOtoRw966JD257GoZ%2BssIo%2FrdAhYGJXSWfvpbzGVT3etLIjbrApoBoQMzw%2BVTy1I%2FyomAtv2yXcop%2FNGlQyOkREmdcc3cbpVi78%2FPiChzyI3ESzp4hwWnNKs2cFoQE15W8GGT0FxkBk0QoXJN2MiPdAS%2Fd7Nrw4MOMPaseHhIfQx81&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20231120T101219Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYXMTQIC7Q%2F20231120%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=29cd9a0fade336c06381b528450f2b797b8b67eaa9ecc3e029ceb182ac9dcf40&hash=d99e790b5d5df95dc08507ca2341780d14b640ae4c5e57d1b6e1cac92280cb90&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1053811903004579&tid=spdf-2465ea3f-65c5-4f0d-9fd9-eaf43d018e8d&sid=3f6dcc697a1fe84db69a8074585332d50fa6gxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=080f57525b01065d0459&rr=828fe9f6bfba5c4b&cc=nl

    We model two parallel subpopulations with different kinematics in order to capture multiband or broadband dynamics.
    """
    def __init__(self, w=0.5, dt=0.0001, seed=42, nonlinearity=sigmoid):
        super().__init__(dt, seed) # this is a single node (cortical column with 3 sub-populations)
        
        # ~ 10 Hz dynamics
        self.tau_exc_1 = 1/100 # time scale for excitatory population ~10ms
        self.tau_inh_1 = 1/50 # time scale for inhibitory population ~20ms
        # ~ 43 Hz dynamics
        self.tau_exc_2 = 0.0046 # time scale for excitatory population ~4.6ms
        self.tau_inh_2 = 0.0029 # time scale for inhibitory population ~2.9ms
        self.G_exc_1 = 3.25 # average excitatory synaptic gain (mV)
        self.G_inh_1 = 22 # average inhibitory synaptic gain
        self.G_exc_2 = 2*3.25 # average excitatory synaptic gain (mV)
        self.G_inh_2 = 150 # average inhibitory synaptic gain
        self.w = w # relative contribution of the first subpopulation
        
        # The rest is the same as the Jansen-Rit model
        n_synapses = 135 # number of synapses between populations
        self.C_1 = 1. * n_synapses # probability of connection between excitatory and pyramidal populations
        self.C_2 = 0.8 * n_synapses
        self.C_3 = 0.25 * n_synapses
        self.C_4 = 0.25 * n_synapses
        self.rmax = 5 # amplitude of sigmoid in Hz (max firing rate)
        self.beta = 0.56 # slope of sigmoid (mV^-1)
        self.theta = 6 # threshold of sigmoid (mV)
        self.v = 10 # conduction velocity
        self.P = 150 # external input to each of the neural masses
        #self.Coupling = 0.1 # coupling between the neural masses (global coupling strength)

        self.x = np.zeros((2 * 6,)) # state of the network
        self.nstates = 2 * 6
        self.S = lambda x: nonlinearity(x, rmax=self.rmax, beta=self.beta, x0=self.theta) # nonlinearity function

    def step(self, I=0.):
        """
        Compute one step of the Jansen-Rit model.
        """
        # 0: pyramidal, 1: excitatory, 2: inhibitory
        x0_1, x1_1, x2_1, xdot0_1, xdot1_1, xdot2_1, \
        x0_2, x1_2, x2_2, xdot0_2, xdot1_2, xdot2_2 = self.x # unpack the state
        # Input received by each population
        # x1 - x2: difference between excitatory and inhibitory activity, which is the input received by the pyramidal population interpreted as the average potential of pyramidal populations
        # self.C_1 * x0: input received by the excitatory population
        # self.C_3 * x0: input received by the inhibitory population
        firing_rates = self.S(np.asarray([self.w*(x1_1 -x2_1) + (1 - self.w)*(x1_2 -x2_2),
                                          self.C_1 * (self.w * x0_1 + (1 - self.w) * x0_2),
                                          self.C_3 * (self.w * x0_1 + (1 - self.w) * x0_2)]))
        input_excitatory = self.C_2 * firing_rates[1] + I # contribution from other nodes will go here
        # pop 1
        xdot0_next_1 = xdot0_1 + self.dt * (self.G_exc_1 * 1.0      * firing_rates[0] - 2 * xdot0_1 - x0_1/self.tau_exc_1 ) / self.tau_exc_1
        xdot1_next_1 = xdot1_1 + self.dt * (self.G_exc_1 * input_excitatory           - 2 * xdot1_1 - x1_1/self.tau_exc_1) / self.tau_exc_1
        xdot2_next_1 = xdot2_1 + self.dt * (self.G_inh_1 * self.C_4 * firing_rates[2] - 2 * xdot2_1 - x2_1/self.tau_inh_1) / self.tau_inh_1
        x0_next_1 = x0_1 + xdot0_1 * self.dt
        x1_next_1 = x1_1 + xdot1_1 * self.dt
        x2_next_1 = x2_1 + xdot2_1 * self.dt
        # Pop 2
        xdot0_next_2 = xdot0_2 + self.dt * (self.G_exc_2 * 1.0      * firing_rates[0] - 2 * xdot0_2 - x0_2/self.tau_exc_2 ) / self.tau_exc_2
        xdot1_next_2 = xdot1_2 + self.dt * (self.G_exc_2 * input_excitatory           - 2 * xdot1_2 - x1_2/self.tau_exc_2) / self.tau_exc_2
        xdot2_next_2 = xdot2_2 + self.dt * (self.G_inh_2 * self.C_4 * firing_rates[2] - 2 * xdot2_2 - x2_2/self.tau_inh_2) / self.tau_inh_2
        x0_next_2 = x0_2 + xdot0_2 * self.dt
        x1_next_2 = x1_2 + xdot1_2 * self.dt
        x2_next_2 = x2_2 + xdot2_2 * self.dt
        self.x = np.array([x0_next_1, x1_next_1, x2_next_1, xdot0_next_1, xdot1_next_1, xdot2_next_1,
                           x0_next_2, x1_next_2, x2_next_2, xdot0_next_2, xdot1_next_2, xdot2_next_2])

    def read_out(self):
        return  self.w * (self.x[1] - self.x[2]) + (1 - self.w) * (self.x[7] - self.x[8])

    def simulate(self, x0, tmax=1, noise=0., P=None):
        """
        Simulate the Jansen-Rit model and monitor the output.

        Parameters
        ----------
        x0 : array_like
            The initial state of the system. Shape (N,).
        tmax : float
            The maximum time to simulate.
        noise : float
            The standard deviation of the noise to add to the system.
        
        Returns
        -------
        x : array_like
            The simulated time series. Shape (n, N).
        """
        rng = np.random.default_rng(self.seed)
        n = int(tmax / self.dt)
        x = np.zeros((n, 12))
        o = np.zeros((n, 1))
        self.x = x0
        x[0] = x0
        o[0] = self.read_out()
        dt_noise = np.sqrt(self.dt) * noise
        for i in range(1, n):
            self.step(I = self.P if P is None else P[i])
            # noise = rng.standard_normal(size=self.x.shape) * dt_noise
            x[i] = self.x           
            o[i] = self.read_out()
        return x, o

class JRNetwork(NeuralMassNetwork):
    """
    Abstract class for neural mass models.
    Defines the function to be implemented for the simulation.

    Notes
    -----
    Two types of networks are modelled in the literature: either one when the mean input and variance of the input
    are controlled for each node, such that the input received from connected nodes is normalised and will relatively
    shut down contribution from external input, or one where the input is not normalised and the external input is
    simply summed over the input from connected nodes.
    The latter is seen in [2] and [3], while the former is seen in [4] & [5].

    References
    ----------

    [1] Jansen, B. H., & Rit, V. G. (1995). Electroencephalogram and visual evoked potential generation in a mathematical model of coupled cortical columns. Biological cybernetics, 73(4), 357-366.
    [2] Kazemi & Jamali, (2022), Phase synchronization and measure of criticality in a network od neural mass models. https://www.nature.com/articles/s41598-022-05285-w
    [3] Forrester et al. (2020), Network Neuroscience. The role of node dynamics in shaping emergent functinoal connectivity patterns in the brain. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7286301/#bib70
    [4] David & Friston (2006), NeuroImage. A Neural mass model for MEG/EEG: coupling and neuonal dynamics. https://www.sciencedirect.com/science/article/pii/S1053811903004579?ref=cra_js_challenge&fr=RR-1
    [5] David et al., (2004). Evaluation of different measures of funcitonal connectivity using a neural mass model. https://www.sciencedirect.com/science/article/pii/S1053811903006566?ref=pdf_download&fr=RR-2&rr=8279b478ba0328ac#APP1
    
    """
    def __init__(self, N=2, W=np.asarray([[0, 1], [0, 0]]), delay=0.01, w=0.8, node_dynamics=None, dt=0.001, seed=42):
        """
        Parameters
        ----------
        N : int
            The number of neurons/nodes.
        W : array_like
            The connectivity matrix. Shape (N, N). E.g. W = np.asarray([[0, 1], [0, 0]]) means that node 1 is connected to node 2, while node 2 is not connected to node 1. 
        delay : float
            The delay between nodes in seconds. Default is 10ms.
        """
        self.rng = np.random.default_rng(seed)
        self.N = N # number of neurons/nodes
        self.W = W # connectivity matrix (W_ij is the connection from i to j, between 0 and 1, relative contribution)
        self.K = W.copy() # updated connectivity in case of normalisation by activity std
        self.delay = delay # delay (10ms)
        self.dt = dt # sampling rate
        self.seed = seed # random seed
        if not np.isscalar(w): # if w is a scalar, then it is the same for all nodes
            w = w * np.ones((self.N,))
        self.nodes = [JansenRitExtended(w=w, dt=dt, seed=self.rng.integers(k+seed)) for k in range(N)] # get different systems/rng for each node
        self.S = self.nodes[0].S # nonlinearity function
        # self.delayed_states = np.zeros((N, self.nodes[0].nstates)) # delayed states of the nodes (state of the nodes at t-dt)
        self.delayed_states = np.zeros((N, 1)) # delayed states of the nodes (state of the nodes at t-dt) / readout only

    def update_connectivity(self, x, sigma_p=1):
        """
        x represents the firing rates output of all nodes, needed to compute the standard deviation
        Shape of x: (N, ntimes)
        """
        # See ref (David & Friston 2004: A Neural Mass model for M/EEG: coupling and neuronal dynamics)
        if np.isscalar(sigma_p): # if sigma_p is a scalar, then it is the same for all nodes
            sigma_p = sigma_p * np.ones((self.N,))
        
        if np.ndim(x) <= 1 or x.shape[1] <= 1:
            sigma_rate = np.ones((self.N,))
        else:
            sigma_rate = np.std(x, axis=1)
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    # k12_star[i] is the normalisation factor for connection from i to j
                    self.K[i, j] = sigma_p[j] * np.sqrt(2 * self.W[i, j] - self.W[i, j]**2) / sigma_rate[i]
        # Then update self.K

    def simulate(self, tmax=1, P=220, sigma_p=22):
        outs = []
        t = np.arange(0, tmax, self.dt)
        nsamples = len(t)
        tdelay, kdelay = 0, 0
        for k, tt in enumerate(t):
            outs.append(self.step(P=P, sigma_p=sigma_p, history_outs=self.S(np.asarray(outs).T)))
            tdelay += self.dt
            kdelay += 1
            if tdelay >= self.delay:
                self.delayed_states = outs[-kdelay]
                tdelay, kdelay = 0, 0
        return np.asarray(outs)

    
    def step(self, P=220, sigma_p=22, history_outs=None):
        if np.isscalar(sigma_p): # if sigma_p is a scalar, then it is the same for all nodes
            sigma_p = sigma_p * np.ones((self.N,))

        self.update_connectivity(history_outs, sigma_p=sigma_p) # update connectivity based on the history of the outputs
        external_input_fluctuation = (sigma_p * self.rng.standard_normal(size=(self.N,))) * (1 - self.W.sum(axis=0))  
        interarea_contributions = (self.S(self.delayed_states).ravel() - 3.84) @ self.K # using normalised connectivity
        outs = []
        for k, n in enumerate(self.nodes):
            # Below y_1 and y_2 corresponds to the fast and slow dynamics respectively of the extended Jansen-Rit model
            #  I = p + (1-k21) * p̃ + k21_star *(S(w*y_1(t - δ) + (1-w)*y_2(t - δ)) - a) where "a" is the mean firing rate
            # we remove the mean firing rate to ensure mean input is conserved with different coupling strengths
            # a = 3.5 in the reference paper, my measured mean is more around 3.84
            n.step(I=P + 
                   external_input_fluctuation[k]  + # noise input scaled : (1-k21) * p̃
                   interarea_contributions[k] # k21_star * (S(w*y_1(t - δ) + (1-w)*y_2(t - δ)) - a)
                   ) # the input to each node is the output of all the other nodes
            outs.append(n.read_out())
            # for i in range(self.N):
            #     I += self.K[i, n] * (outs[i] - 3.84)
        return np.asarray(outs)
    
    def reset(self):
        self.delayed_states = np.zeros((self.N, 1))
        for n in self.nodes:
            n.x = np.zeros((n.nstates,))
        self.K = self.W.copy()
