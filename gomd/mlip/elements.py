"""Atomic numbers for the MLIP, from the force-field masses.

APPLE&P type labels ('O= 1', 'Fs 1', 'S- 1', 'Lp 1') are not element symbols,
so the element of each site is taken from its mass. Lone-pair / dummy sites
(mass 0) get 0 and are never sent to the MLIP.
"""
from __future__ import annotations
import numpy as np

# Standard atomic weights (IUPAC, abridged), Z -> (symbol, weight)
_WEIGHTS = {
    1: ('H', 1.008), 2: ('He', 4.0026), 3: ('Li', 6.94), 4: ('Be', 9.0122),
    5: ('B', 10.81), 6: ('C', 12.011), 7: ('N', 14.007), 8: ('O', 15.999),
    9: ('F', 18.998), 10: ('Ne', 20.180), 11: ('Na', 22.990), 12: ('Mg', 24.305),
    13: ('Al', 26.982), 14: ('Si', 28.085), 15: ('P', 30.974), 16: ('S', 32.06),
    17: ('Cl', 35.45), 18: ('Ar', 39.948), 19: ('K', 39.098), 20: ('Ca', 40.078),
    25: ('Mn', 54.938), 26: ('Fe', 55.845), 27: ('Co', 58.933), 28: ('Ni', 58.693),
    29: ('Cu', 63.546), 30: ('Zn', 65.38), 35: ('Br', 79.904), 37: ('Rb', 85.468),
    38: ('Sr', 87.62), 53: ('I', 126.904), 55: ('Cs', 132.905),
}
SYMBOLS = {z: s for z, (s, _) in _WEIGHTS.items()}


def numbers_from_masses(mass, ldummy, tol=0.05):
    """Atomic number per site (0 for lone-pair sites).

    Raises ValueError for a mass that matches no element within `tol` amu,
    or more than one.
    """
    mass = np.asarray(mass, dtype=float)
    ldummy = np.asarray(ldummy, dtype=bool)
    z_of = {}
    for m in np.unique(np.round(mass[~ldummy], 6)):
        hits = [z for z, (_, w) in _WEIGHTS.items() if abs(w - m) <= tol]
        if len(hits) != 1:
            raise ValueError(
                f"cannot assign an element to mass {m} amu "
                f"(matches: {[SYMBOLS[z] for z in hits] or 'none'}); "
                f"add it to gomd/mlip/elements.py")
        z_of[m] = hits[0]
    numbers = np.zeros(mass.shape[0], dtype=np.int64)
    real = np.flatnonzero(~ldummy)
    numbers[real] = [z_of[m] for m in np.round(mass[real], 6)]
    return numbers


def formula(numbers):
    zs, n = np.unique(numbers[numbers > 0], return_counts=True)
    return ' '.join(f"{SYMBOLS[int(z)]}{int(c)}" for z, c in zip(zs, n))
