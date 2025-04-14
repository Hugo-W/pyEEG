import numpy as np
import platform
import os
import logging

logging.basicConfig(level=logging.ERROR)
LOGGER = logging.getLogger(__name__.split('.')[0])
LOGGER.setLevel('INFO')

try:
    # Attempt to import the compiled Python extension module
    from pyeeg.bin import gammatone_c
    LOGGER.info("Successfully loaded gammatone_c Python extension module.")
except ImportError:
    # Fallback to loading prebuilt shared library using ctypes
    import ctypes

    LOGGER.warning("Failed to load gammatone_c Python extension module. Falling back to prebuilt binary.")

    # Determine the shared library extension based on the operating system
    system = platform.system()
    if system == "Windows":
        ext = ".dll"
    elif system == "Darwin":
        ext = ".dylib"
    else:
        ext = ".so"

    # Load the shared library
    lib_path = os.path.join(os.path.dirname(__file__), f"bin/gammatone_c{ext}")
    gammatone_lib = ctypes.CDLL(lib_path)

    # Define the argument and return types
    gammatone_lib.gammatone_c.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int,
        ctypes.c_double, ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)
    ]

def gammatone_filter(x, fs, cf, hrect=0):
    """
    Apply the gammatone filter to an input signal.

    Parameters:
    x : array_like
        Input signal.
    fs : int
        Sampling frequency in Hz.
    cf : float
        Centre frequency of the filter in Hz.
    hrect : int, optional
        Half-wave rectifying if hrect = 1 (default is 0).

    Returns:
    bm : ndarray
        Basilar membrane displacement.
    env : ndarray
        Instantaneous envelope.
    instp : ndarray
        Instantaneous phase (unwrapped radian).
    instf : ndarray
        Instantaneous frequency (Hz).
    """
    x = np.asarray(x, dtype=np.float64)
    nsamples = len(x)

    # Prepare output placeholders
    bm = np.zeros(nsamples, dtype=np.float64)
    env = np.zeros(nsamples, dtype=np.float64)
    instp = np.zeros(nsamples, dtype=np.float64)
    instf = np.zeros(nsamples, dtype=np.float64)

    if 'gammatone_c' in globals():
        # Use the Python extension module
        gammatone_c.gammatone_c(x, fs, cf, hrect, bm, env, instp, instf)
    else:
        # Use the fallback ctypes-based shared library
        x_ctypes = x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        bm_ctypes = bm.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        env_ctypes = env.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        instp_ctypes = instp.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        instf_ctypes = instf.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        gammatone_lib.gammatone_c(
            x_ctypes, nsamples, fs, cf, hrect,
            bm_ctypes, env_ctypes, instp_ctypes, instf_ctypes
        )

    return bm, env, instp, instf

# Example usage
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    fs = 16000
    t = np.arange(0, 1, 1/fs)
    x = np.sin(2 * np.pi * 440 * t)
    cf = 440
    bm, env, instp, instf = gammatone_filter(x, fs, cf)
    
    plt.figure()
    plt.subplot(4, 1, 1)
    plt.plot(t, x)  # Input signal
    plt.title("Input signal")
    plt.subplot(4, 1, 2)
    plt.plot(t, bm)  # Basilar membrane displacement
    plt.title("Basilar membrane displacement")
    plt.subplot(4, 1, 3)
    plt.plot(t, env)  # Instantaneous envelope
    plt.title("Instantaneous envelope")
    plt.subplot(4, 1, 4)
    plt.plot(t, instf)  # Instantaneous frequency
    plt.title("Instantaneous frequency")
    plt.tight_layout()
    plt.show()