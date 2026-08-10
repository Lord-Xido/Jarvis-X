"""Jarvis-X deterministic geometric computing research runtime.

The package is explicit rather than an implicit namespace package so static
analysis, editable installs, wheels, and runtime imports resolve sibling modules
through one stable ``jarvisx.*`` package identity.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
