"""
Encoder module - compatibility shim for test suite.

This module provides a compatibility layer for the test suite which expects
src.models.encoder.MultiViewEncoder to exist.
"""

# Import from the actual implementation
from .ucl_tsc_model import MultiViewEncoder, UCLTSCModel

__all__ = ['MultiViewEncoder', 'UCLTSCModel']
