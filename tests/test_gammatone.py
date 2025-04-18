from pyeeg.gammatone import gammatone_filter
import numpy as np

# Sample input signal
fs = 4000  # Sampling frequency
cf = 400  # Centre frequency
t = np.arange(0, 1, 1/fs)  # Time vector
x = np.sin(2 * np.pi * 350 * t)  # Example signal (sine wave)
nsamples = len(x)
# Prepare output placeholders
bm = np.zeros(nsamples, dtype=np.float64)
env = np.zeros(nsamples, dtype=np.float64)
instp = np.zeros(nsamples, dtype=np.float64)
instf = np.zeros(nsamples, dtype=np.float64)

# Low-level C extensions:
try:
    # Attempt to import the compiled Python extension module
    from pyeeg.bin import gammatone_c
    print("Successfully loaded gammatone_c Python extension module.")

    # Test directly gammatone_c from gammatone_c
    def gammatone_c_from_extension():
        """
        Test the gammatone filter function with a sample input signal.
        This loads one of the libraries...

        >>> gammatone_c_from_extension()
        array([0.00000000e+00, 2.42658114e-05, 1.82263060e-04, ...,
               3.36750713e-01, 4.66512862e-01, 4.58784493e-01], shape=(4000,))

        """
        # Call the gammatone filter function
       # Use the compiled Python extension module
        gammatone_c.gammatone_c(
            x, nsamples, fs, cf, 0,
            bm, env, instp, instf
        )
        return bm
    
except ImportError:
    # Fallback to loading prebuilt shared library using ctypes
    import ctypes
    from pyeeg.gammatone import gammatone_lib
    # Test directly gammatone_c from gammatone_lib

    # Use the fallback ctypes-based shared library
    x_ctypes = x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    bm_ctypes = bm.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    env_ctypes = env.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    instp_ctypes = instp.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    instf_ctypes = instf.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    def gammatone_c_from_shared_lib():
        """
        Test the gammatone filter function with a sample input signal.
        This loads one of the libraries...

        >>> gammatone_c_from_shared_lib()
        array([0.00000000e+00, 2.42658114e-05, 1.82263060e-04, ...,
               3.36750713e-01, 4.66512862e-01, 4.58784493e-01], shape=(4000,))

        """
        # Call the gammatone filter function
        gammatone_lib.gammatone_c(
                    x_ctypes, nsamples, fs, cf, 0,
                    bm_ctypes, env_ctypes, instp_ctypes, instf_ctypes
                )
        return bm

def c_gammatone():
    """
    Test the gammatone filter function with a sample input signal.
    This loads one of the libraries...

    >>> c_gammatone()
    array([0.00000000e+00, 2.42658114e-05, 1.82263060e-04, ...,
           3.36750713e-01, 4.66512862e-01, 4.58784493e-01], shape=(4000,))

    """
    # Call the gammatone filter function
    bm, env, instp, instf = gammatone_filter(x, fs, cf)
    return bm

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    c_gammatone()