"""Compatibility shim for legacy editable installs.

All project metadata lives in ``pyproject.toml`` so packaging configuration has
one authoritative source.
"""

from setuptools import setup


if __name__ == "__main__":
    setup()
