"""
Isotope data and helper utilities for handling isotopologue labels.
"""

from __future__ import annotations

import math
import re
from typing import Iterable, List, Sequence, Tuple

# Accurate isotopic masses (in atomic mass units) for elements referenced in the project.
ISOTOPE_MASSES = {
    # Boron
    ("B", 10): 10.01293695,
    ("B", 11): 11.00930536,
    # Carbon
    ("C", 12): 12.0000000,
    ("C", 13): 13.00335483521,
    ("C", 14): 14.003241988,
    # Hydrogen
    ("H", 1): 1.00782503223,
    ("H", 2): 2.01410177812,
    ("H", 3): 3.01604928199,
    # Nitrogen
    ("N", 14): 14.00307400446,
    ("N", 15): 15.00010889888,
    # Oxygen
    ("O", 16): 15.99491461956,
    ("O", 17): 16.99913175650,
    ("O", 18): 17.99915961286,
    # Fluorine
    ("F", 19): 18.998403163,
    # Phosphorus
    ("P", 31): 30.973761998,
    # Sulfur
    ("S", 32): 31.9720711744,
    ("S", 33): 32.9714589098,
    ("S", 34): 33.9678669025,
    ("S", 36): 35.96708076,
    # Chlorine
    ("Cl", 35): 34.96885271,
    ("Cl", 37): 36.96590260,
    # Bromine
    ("Br", 79): 78.9183376,
    ("Br", 81): 80.9162897,
    # Iodine
    ("I", 127): 126.904473,
    ("I", 129): 128.9049837,
    # Potassium
    ("K", 39): 38.963706487,
    ("K", 40): 39.96399817,
    ("K", 41): 40.96182526,
    # Lithium
    ("Li", 6): 6.0151228874,
    ("Li", 7): 7.0160034366,
    # Sodium
    ("Na", 23): 22.98976928,
    # Magnesium
    ("Mg", 24): 23.985041697,
    ("Mg", 25): 24.985836976,
    ("Mg", 26): 25.982592968,
    # Calcium
    ("Ca", 40): 39.962590863,
    ("Ca", 42): 41.958617828,
    ("Ca", 43): 42.958766430,
    ("Ca", 44): 43.955481543,
    ("Ca", 46): 45.953688,
    ("Ca", 48): 47.952522904,
}

MOST_ABUNDANT = {
    "B": 11,
    "C": 12,
    "H": 1,
    "N": 14,
    "O": 16,
    "F": 19,
    "Cl": 35,
    "S": 32,
    "Br": 79,
    "I": 127,
    "P": 31,
    "K": 39,
    "Li": 7,
    "Na": 23,
    "Mg": 24,
    "Ca": 40,
}


def parse_isotopologue_label(label: str) -> List[Tuple[int, str]]:
    """
    Parse an isotopologue label (e.g. ``1H2H``) into a list of ``(mass_number, element)`` tuples.
    """
    return [(int(mass), elem) for mass, elem in re.findall(r"(\d+)([A-Z][a-z]?)", label)]


def isotopologue_to_smiles(label: str) -> str:
    """
    Convert an isotopologue label into a simplistic SMILES string that preserves isotope tagging.
    Suitable for diatomics where the connectivity is linear.
    """
    fragments = [f"[{mass}{elem}]" for mass, elem in parse_isotopologue_label(label)]
    return "".join(fragments)


def isotopologue_to_formula(label: str) -> str:
    """
    Convert an isotopologue label into a condensed chemical formula without isotope annotations.
    Homonuclear diatomics are suffixed with ``2`` (e.g. ``1H1H`` -> ``H2``).
    """
    elements = [elem for _, elem in parse_isotopologue_label(label)]
    if len(elements) == 2 and elements[0] == elements[1]:
        return f"{elements[0]}2"
    return "".join(elements)


def compute_reduced_mass(pairs: Sequence[Tuple[int, str]]) -> float:
    """
    Compute the reduced mass (amu) for a diatomic isotopologue defined by ``pairs``.
    """
    if len(pairs) != 2:
        raise ValueError("Reduced mass is only defined here for diatomics.")
    (mass1, elem1), (mass2, elem2) = pairs
    m1 = ISOTOPE_MASSES[(elem1, mass1)]
    m2 = ISOTOPE_MASSES[(elem2, mass2)]
    return (m1 * m2) / (m1 + m2)


def degeneracy_clusters(
    frequencies: Iterable[float], tolerance: float = 1.0
) -> List[List[float]]:
    """
    Group vibrational frequencies into degeneracy clusters where adjacent values differ
    by at most ``tolerance`` (cm^-1).
    """
    cleaned: List[float] = []
    for value in frequencies:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(numeric):
            continue
        cleaned.append(numeric)
    if not cleaned:
        return []
    clusters: List[List[float]] = [[cleaned[0]]]
    for value in cleaned[1:]:
        if abs(value - clusters[-1][-1]) <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return clusters


__all__ = [
    "ISOTOPE_MASSES",
    "MOST_ABUNDANT",
    "parse_isotopologue_label",
    "isotopologue_to_smiles",
    "isotopologue_to_formula",
    "compute_reduced_mass",
    "degeneracy_clusters",
]
