"""
io — input file parsing.

For M1, we reuse the existing Python parsers from python-current/ as-is.
They're pure Python, no GPU dependency, and they handle the existing input
formats (mdrun.params, ff.dat, connectivity.dat, coords.inp).

The boundary between numpy-state (parsed) and jax-state (simulation) is in
main.py's load_simulation function: parse with numpy, convert to jax at the end.
"""

from .loader import load_simulation_state

__all__ = ['load_simulation_state']
