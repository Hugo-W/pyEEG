import numpy as np
import logging
import platform
import os

logging.basicConfig(level=logging.ERROR)
LOGGER = logging.getLogger(__name__.split('.')[0])
LOGGER.setLevel('INFO')

try:
    # Attempt to import the compiled Python extension module
    from pyeeg.bin import makeRateMap_c
    LOGGER.info("Successfully loaded makeRateMap_c Python extension module.")
except ImportError:
    # Fallback to loading prebuilt shared library using ctypes
    import ctypes

    LOGGER.warning("Failed to load makeRateMap_c Python extension module. Falling back to prebuilt binary.")

    # Determine the shared library extension based on the operating system
    system = platform.system()
    if system == "Windows":
        ext = ".dll"
    elif system == "Darwin":
        ext = ".dylib"
    else:
        ext = ".so"

    # Load the shared library
    lib_path = os.path.join(os.path.dirname(__file__), f"bin/makeRateMap_c{ext}")
    makeRateMap_lib = ctypes.CDLL(lib_path)

    # Define the argument and return types
    makeRateMap_lib.makeRateMap.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int,
        ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_double,
        ctypes.c_double, ctypes.c_char_p, ctypes.POINTER(ctypes.c_double)
    ]

def erb_rate_to_hz(erb_rate):
    return (10**(erb_rate / 21.4) - 1.0) / 4.37e-3

def hz_to_erb_rate(hz):
    return 21.4 * np.log10(4.37e-3 * hz + 1.0)

def generate_cfs(lowcf, highcf, numchans):
    low_erb = 21.4 * np.log10(4.37e-3 * lowcf + 1.0)
    high_erb = 21.4 * np.log10(4.37e-3 * highcf + 1.0)
    erb_space = (high_erb - low_erb) / (numchans - 1)
    cfs = [erb_rate_to_hz(low_erb + i * erb_space) for i in range(numchans)]
    return np.array(cfs)

def make_rate_map(x, fs, lowcf, highcf, numchans, frameshift, ti, compression):
    """
    Compute the rate map (aka cochleogram) of an input signal using a gammatone filterbank.
    The rate map is a matrix where each row corresponds to a frequency channel
    and each column corresponds to a time frame.

    Parameters:
    -----------
    x : array_like
        Input signal.
    fs : int
        Sampling frequency in Hz (e.g., 8000).
    lowcf : float
        Centre frequency of the lowest filter in Hz (e.g., 50).
    highcf : float
        Centre frequency of the highest filter in Hz (e.g., 3500).
    numchans : int
        Number of channels in the filterbank (e.g., 32).
    frameshift : float
        Interval between successive frames in ms (e.g., 10).
    ti : float
        Temporal integration in ms (e.g., 8).
    compression : str
        Type of compression ['cuberoot', 'log', 'none'] (e.g., 'cuberoot').

    Returns:
    --------
    ratemap : ndarray
        The computed rate map.

    Example:
    --------
    ratemap = make_rate_map(x, 8000, 50, 3500, 32, 10, 8, 'cuberoot')
    """
    x = np.asarray(x, dtype=np.float64)
    numframes = int(np.ceil(len(x) / (frameshift * fs / 1000)))
    ratemap = np.zeros((numchans, numframes), dtype=np.float64, order='F')

    if 'makeRateMap_c' in globals():
        # Use the Python extension module
        makeRateMap_c.makeRateMap(
            x, len(x), fs, lowcf, highcf, numchans, frameshift, ti, compression, ratemap
        )
    else:
        # Use the fallback ctypes-based shared library
        x_ctypes = x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        ratemap_ctypes = ratemap.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        compression_bytes = compression.encode('utf-8')

        makeRateMap_lib.makeRateMap(
            x_ctypes, len(x), fs, lowcf, highcf, numchans, frameshift, ti,
            compression_bytes, ratemap_ctypes
        )

    return ratemap

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Prepare input data
    fs = 8000
    t = np.arange(0, 0.5, 1/fs)
    x = np.sin(2 * np.pi * 440 * t)
    lowcf = 50
    highcf = 3500
    numchans = 16
    frameshift = 10
    ti = 8
    compression = "cuberoot"

    # Call the wrapper function
    ratemap = make_rate_map(x, fs, lowcf, highcf, numchans, frameshift, ti, compression)
    t_map = np.arange(0, len(x) / fs * 1000, frameshift)
    freqs = generate_cfs(lowcf, highcf, numchans)

    # Show results
    plt.imshow(ratemap, aspect='auto', origin='lower', cmap='plasma')
    plt.xlabel('Time (ms)')
    plt.xticks(np.arange(0, len(t_map), 10), [f'{time:.0f}' for time in t_map[::10]])
    plt.ylabel('Frequency (Hz)')
    plt.yticks(np.arange(0, numchans, 2), [f'{freq:.0f}' for freq in freqs[::2]])
    plt.title('Rate Map')
    plt.colorbar(label='Rate (spikes/s)')
    plt.show()