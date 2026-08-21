"""
Dashboard subpackage for pyEEG.

This subpackage provides a web-based dashboard for exploring TRF (Temporal Response Function)
analysis with EEG/MEG data and features.
"""

from .app import create_app
from . import server

__all__ = ['create_app', 'server']
