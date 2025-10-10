"""
Data structures and helpers for working with molecular constants sourced from Richet et al. (1977).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from scipy import constants as sc

from . import isotopes


@dataclass
class MolecularConstants:
    """
    Container for an isotopologue's spectroscopic constants.
    """

    isotopologue: str
    harmonic: List[float] = field(default_factory=list)
    anharmonic_poly: List[float] = field(default_factory=list)
    anharmonic_dia: List[float] = field(default_factory=list)
    rotational_constants: Dict[str, float] = field(default_factory=dict)
    delta: List[float] = field(default_factory=list)
    alpha: List[float] = field(default_factory=list)
    other_constants: Dict[str, float] = field(default_factory=dict)
    symmetry: Optional[float] = None
    mass: Optional[float] = None
    moments_of_inertia: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mass is None:
            self.mass = self._compute_isotopologue_mass()
        if not self.moments_of_inertia:
            self._compute_moments_of_inertia()
        if self.alpha and "B0 (cm-1)" in self.rotational_constants:
            self.delta.append(self.alpha[0] / self.rotational_constants["B0 (cm-1)"])

    def _compute_isotopologue_mass(self) -> float:
        pairs = isotopes.parse_isotopologue_label(self.isotopologue)
        total_mass = 0.0
        for mass_number, element in pairs:
            try:
                total_mass += isotopes.ISOTOPE_MASSES[(element, mass_number)]
            except KeyError as exc:
                raise ValueError(f"No isotopic mass found for {mass_number}{element}") from exc
        return total_mass

    def _compute_moments_of_inertia(self) -> None:
        self.moments_of_inertia = {}
        for key, value in self.rotational_constants.items():
            if key not in {"A0 (cm-1)", "B0 (cm-1)", "C0 (cm-1)"}:
                continue
            if value == 0.0:
                self.moments_of_inertia[f"I_{key[0]}"] = np.inf
                continue
            inertia = sc.h / (8 * sc.pi**2 * sc.c * 100 * value)
            self.moments_of_inertia[f"I_{key[0]}"] = inertia


def _is_nan(value) -> bool:
    try:
        return np.isnan(value)
    except TypeError:
        return False


def build_molecular_constants(row: pd.Series) -> MolecularConstants:
    isotopologue_name = row.iloc[0]
    constants_dict = row.drop(row.index[0]).to_dict()

    harmonic: List[float] = []
    anharmonic_poly: List[float] = []
    anharmonic_dia: List[float] = []
    rotational_constants: Dict[str, float] = {}
    delta: List[float] = []
    alpha: List[float] = []
    other: Dict[str, float] = {}
    symmetry = None

    for key, value in constants_dict.items():
        if _is_nan(value):
            continue
        if key.startswith("w1x1"):
            anharmonic_dia.append(value)
        elif key.startswith("w") and not key.startswith("w1x1"):
            harmonic.append(value)
        elif key.startswith("x"):
            anharmonic_poly.append(value)
        elif key in {"A0 (cm-1)", "B0 (cm-1)", "C0 (cm-1)"}:
            rotational_constants[key] = value
        elif key.startswith("d"):
            delta.append(value)
        elif key.startswith("a"):
            alpha.append(value)
        elif key.startswith("Sym"):
            symmetry = value
        else:
            other[key] = value

    return MolecularConstants(
        isotopologue=isotopologue_name,
        harmonic=harmonic,
        anharmonic_poly=anharmonic_poly,
        anharmonic_dia=anharmonic_dia,
        rotational_constants=rotational_constants,
        delta=delta,
        alpha=alpha,
        other_constants=other,
        symmetry=symmetry,
    )


def load_molecular_constants_from_excel(
    file_path: str | Path, sheet_name: str | None = None
) -> Dict[str, MolecularConstants]:
    """
    Load an Excel sheet and construct a dictionary of ``MolecularConstants`` keyed by isotopologue label.
    """
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    records = {}
    for _, row in df.iterrows():
        mc = build_molecular_constants(row)
        records[mc.isotopologue] = mc
    return records


_FLOAT_PATTERN = re.compile(r"([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)\s*$")


def load_molecular_constants_from_fit_output(file_path: str | Path) -> MolecularConstants:
    """
    Parse a diatomic ``Fit_1D`` text output (e.g., ``14C18O.7pointfit.out``) and build a
    ``MolecularConstants`` instance. The parser targets the CCSD(T) 7-point fit format that
    lists bond length, rotational constant, vibration-rotation constant, harmonic frequency,
    and anharmonicity.
    """
    path = Path(file_path)
    lines = path.read_text().splitlines()

    def extract(label: str) -> Optional[float]:
        for line in lines:
            if label in line:
                match = _FLOAT_PATTERN.search(line)
                if match:
                    return float(match.group(1))
        return None

    bond_length = extract("Bond length (Angstroms)")
    rotational_mhz = extract("Rotational constant (MHz)")
    vibration_rot_cm = extract("Vibration-rotation constant (cm-1)")
    centrifugal_khz = extract("Centrifugal distortion constant (kHz)")
    harmonic_cm = extract("Harmonic frequency (cm-1)")
    anharmonic_cm = extract("Anharmonicity constant (cm-1)")

    rotational_constants = {}
    if rotational_mhz is not None:
        rotational_constants["B0 (cm-1)"] = rotational_mhz * 1.0e6 / (sc.c * 100.0)

    harmonic = [harmonic_cm] if harmonic_cm is not None else []
    anharmonic_dia = [anharmonic_cm] if anharmonic_cm is not None else []
    alpha = [vibration_rot_cm] if vibration_rot_cm is not None else []

    other_constants: Dict[str, float] = {}
    if bond_length is not None:
        other_constants["Bond length (Angstroms)"] = bond_length
    if rotational_mhz is not None:
        other_constants["Rotational constant (MHz)"] = rotational_mhz
    if centrifugal_khz is not None:
        other_constants["Centrifugal distortion constant (kHz)"] = centrifugal_khz

    isotopologue = path.name.split(".")[0]

    return MolecularConstants(
        isotopologue=isotopologue,
        harmonic=harmonic,
        anharmonic_poly=[],
        anharmonic_dia=anharmonic_dia,
        rotational_constants=rotational_constants,
        delta=[],
        alpha=alpha,
        other_constants=other_constants,
    )


def canonicalize_isotopologue(label: str) -> str:
    parts = isotopes.parse_isotopologue_label(label)
    sorted_parts = sorted(parts, key=lambda item: (item[1], item[0]))
    return "".join(f"{mass}{elem}" for mass, elem in sorted_parts)


def is_most_common_isotopologue(label: str) -> bool:
    for mass, elem in isotopes.parse_isotopologue_label(label):
        if isotopes.MOST_ABUNDANT.get(elem) != mass:
            return False
    return True


def possible_isotopes(element: str) -> List[int]:
    return sorted(mass for el, mass in isotopes.ISOTOPE_MASSES if el == element)


def generate_isotopic_variants(
    mc: MolecularConstants, include_original: bool = False
) -> Iterable[Dict[str, float]]:
    """
    Generate isotopic variants using Richet eqs. 55-58 scaling rules.
    """
    pairs = isotopes.parse_isotopologue_label(mc.isotopologue)
    if len(pairs) != 2:
        return []

    (mass1, elem1), (mass2, elem2) = pairs
    reduced_ref = isotopes.compute_reduced_mass(pairs)
    iso_masses1 = possible_isotopes(elem1)
    iso_masses2 = possible_isotopes(elem2)

    variants = []

    for m1 in iso_masses1:
        for m2 in iso_masses2:
            if not include_original and m1 == mass1 and m2 == mass2:
                continue
            label = f"{m1}{elem1}{m2}{elem2}"
            reduced_new = isotopes.compute_reduced_mass([(m1, elem1), (m2, elem2)])
            rho = np.sqrt(reduced_ref / reduced_new)

            entry = {"Molecule": label}

            if mc.harmonic:
                entry["w1 (cm-1)"] = mc.harmonic[0] * rho
            if mc.anharmonic_dia:
                entry["w1x1 (cm-1)"] = mc.anharmonic_dia[0] * rho**2
            if mc.rotational_constants.get("B0 (cm-1)") is not None:
                entry["B0 (cm-1)"] = mc.rotational_constants["B0 (cm-1)"] * rho**2
            if mc.alpha:
                entry["a1 (cm-1)"] = mc.alpha[0] * rho**3
            entry.update({k: v for k, v in mc.other_constants.items() if k not in entry})
            variants.append(entry)

    return variants


__all__ = [
    "MolecularConstants",
    "load_molecular_constants_from_excel",
    "load_molecular_constants_from_fit_output",
    "canonicalize_isotopologue",
    "is_most_common_isotopologue",
    "generate_isotopic_variants",
]
