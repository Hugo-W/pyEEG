"""Centralized logging configuration for the pyeeg library.

All modules should import LOGGER from here instead of calling
getLogger directly. Use set_log_level() to control verbosity.
"""
import logging
import sys

LOGGER = logging.getLogger("pyeeg")

# Standard library pattern: attach a NullHandler so that no "No handlers
# could be found" warning is emitted when the user hasn't configured logging.
# Messages still propagate to the root logger (configured by basicConfig in
# __init__.py), so set_log_level('INFO') will show output.
if not LOGGER.handlers:
    LOGGER.addHandler(logging.NullHandler())


def set_log_level(level="WARNING"):
    """Set the logging level for the entire pyeeg library.

    Parameters
    ----------
    level : str or int
        Logging level: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL',
        or the corresponding integer (e.g. logging.DEBUG).

    Examples
    --------
    >>> from pyeeg import set_log_level
    >>> set_log_level('INFO')   # show informational messages
    >>> set_log_level('WARNING')  # silence everything except warnings
    """
    if isinstance(level, str):
        level = level.upper()
    LOGGER.setLevel(level)


def get_logger():
    """Return the package-level logger used by all pyeeg modules."""
    return LOGGER