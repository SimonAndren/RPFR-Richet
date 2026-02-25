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


# Mapping from common molecule names (Exp values sheet) to isotopologue labels.
_EXP_NAME_TO_LABEL = {
    "H2": "1H1H", "D2": "2H2H", "HT": "1H3H", "T2": "3H3H",
    "HD": "1H2H", "DT": "2H3H",
    "HF": "1H19F", "DF": "2H19F", "TF": "3H19F",
    "H35Cl": "1H35Cl", "D35Cl": "2H35Cl",
    # "T35Cl": "3H35Cl",  # SKIP: Exp values sheet has wrong data (copy of H35Cl). Needs correction.
    # T35Cl will instead be generated via reduced-mass scaling from H35Cl.
    "F2": "19F19F",
    "NaF": "23Na19F", "Na35Cl": "23Na35Cl",
    "7LiF": "7Li19F",
    "39KF": "39K19F",
    "CO": "12C16O",
}

# Homonuclear diatomics get symmetry = 2.
_HOMONUCLEAR = {
    "1H1H", "2H2H", "3H3H", "19F19F", "35Cl35Cl", "37Cl37Cl",
    "14N14N", "15N15N", "16O16O", "17O17O", "18O18O",
    "32S32S", "33S33S", "34S34S", "36S36S",
}


def _parse_exp_name(name: str) -> str:
    """Convert an experimental molecule name to an isotopologue label."""
    name = str(name).strip()
    if name in _EXP_NAME_TO_LABEL:
        return _EXP_NAME_TO_LABEL[name]
    # Already contains mass numbers (e.g. "35Cl35Cl", "14N14N", "24Mg16O")
    pairs = isotopes.parse_isotopologue_label(name)
    if pairs:
        return "".join(f"{m}{e}" for m, e in pairs)
    return name


def _parse_numeric(val) -> Optional[float]:
    """Parse numeric values that may contain '*10-' notation or dashes."""
    if _is_nan(val):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if not s or s == "-":
        return None
    s = s.replace("*10-", "e-").replace("×10-", "e-").replace("x10-", "e-")
    try:
        return float(s)
    except ValueError:
        return None


def load_experimental_constants(
    file_path: str | Path, sheet_name: str = "Exp values"
) -> Dict[str, MolecularConstants]:
    """
    Load experimental spectroscopic constants and return a dict of
    ``MolecularConstants`` keyed by isotopologue label.

    The sheet is expected to have columns: Molecule, Atomic mass (x2),
    Re (A), Be (cm-1), αe (cm-1), De (cm-1), ωe (cm-1), ωexe (cm-1).

    Equilibrium constants are converted to ground-state:
    ``B0 = Be - αe / 2``.
    """
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

    # Find the header row (contains "Molecule")
    header_idx = None
    for i, row in df.iterrows():
        if any(str(v).strip() == "Molecule" for v in row.values if not _is_nan(v)):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Could not find header row with 'Molecule' in Exp values sheet")

    # Molecules to skip (known bad data in the sheet).
    _SKIP_EXP_NAMES = {"T35Cl"}

    records: Dict[str, MolecularConstants] = {}

    for i in range(header_idx + 1, len(df)):
        row = df.iloc[i]
        mol_name = row.iloc[0]
        if _is_nan(mol_name) or str(mol_name).strip() == "":
            continue
        if str(mol_name).strip() in _SKIP_EXP_NAMES:
            continue

        label = _parse_exp_name(mol_name)

        # Parse numeric values (columns 4-8 correspond to Be, αe, De, ωe, ωexe)
        Be = _parse_numeric(row.iloc[4]) or 0.0
        alpha_e = _parse_numeric(row.iloc[5]) or 0.0
        omega_e = _parse_numeric(row.iloc[7]) or 0.0
        omega_e_xe = _parse_numeric(row.iloc[8]) or 0.0
        if omega_e == 0.0:
            continue

        B0 = Be - alpha_e / 2.0

        symmetry = 2.0 if label in _HOMONUCLEAR else None

        mc = MolecularConstants(
            isotopologue=label,
            harmonic=[omega_e],
            anharmonic_poly=[],
            anharmonic_dia=[omega_e_xe],
            rotational_constants={"B0 (cm-1)": B0},
            delta=[],
            alpha=[alpha_e],
            symmetry=symmetry,
        )
        records[label] = mc

    return records


def generate_all_variants(
    parent_constants: Dict[str, MolecularConstants],
) -> Dict[str, MolecularConstants]:
    """
    From a dict of parent molecule constants, generate all isotopic variants
    via reduced-mass scaling and return a combined lookup dict.

    The parent molecules are included in the output (with ``include_original=True``).
    """
    lookup: Dict[str, MolecularConstants] = {}

    for label, mc in parent_constants.items():
        # Include the parent itself
        lookup[label] = mc

        # Generate scaled variants
        variants = generate_isotopic_variants(mc, include_original=False)
        for entry in variants:
            vlabel = entry["Molecule"]
            if vlabel in lookup:
                continue
            symmetry = 2.0 if vlabel in _HOMONUCLEAR else None
            vmc = MolecularConstants(
                isotopologue=vlabel,
                harmonic=[entry["w1 (cm-1)"]] if "w1 (cm-1)" in entry else [],
                anharmonic_poly=[],
                anharmonic_dia=[entry["w1x1 (cm-1)"]] if "w1x1 (cm-1)" in entry else [],
                rotational_constants=(
                    {"B0 (cm-1)": entry["B0 (cm-1)"]} if "B0 (cm-1)" in entry else {}
                ),
                delta=[],
                alpha=[entry["a1 (cm-1)"]] if "a1 (cm-1)" in entry else [],
                symmetry=symmetry,
            )
            lookup[vlabel] = vmc

    return lookup


def load_amila_rpfr(
    file_path: str | Path, sheet_name: str = "Amila"
) -> pd.DataFrame:
    """
    Parse Amila's RPFR sheet into a tidy DataFrame.

    Returns a DataFrame with columns: ``Molecule``, ``SSI``, ``NSI``,
    ``Method``, ``Translational``, ``ZPE (total)``, ``Excited (harmonic)``,
    ``Excited (anharmonic)``, ``Rotational (diatomic)``,
    ``Rotational-vibrational``, ``RPFR 273.15K``.
    """
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

    contrib_cols = [
        "Translational",
        "ZPE (total)",
        "Excited (harmonic)",
        "Excited (anharmonic)",
        "Rotational (diatomic)",
        "Rotational-vibrational",
        "RPFR 273.15K",
    ]

    rows = []
    current_method = None

    for i in range(len(df)):
        cell = df.iloc[i, 0]
        if _is_nan(cell) or str(cell).strip() == "":
            continue
        cell_str = str(cell).strip()

        # Skip header row
        if cell_str == "Molecule":
            continue

        # Detect method labels
        if cell_str.startswith("CCSD(T)"):
            current_method = cell_str
            continue

        # Parse data row
        mol = cell_str
        if "/" in mol:
            ssi_label, nsi_label = mol.split("/", 1)
        else:
            # Malformed entries without "/" - skip or try to split at midpoint
            continue

        values = {}
        for j, col in enumerate(contrib_cols):
            raw = df.iloc[i, j + 1]
            try:
                values[col] = float(raw)
            except (ValueError, TypeError):
                values[col] = np.nan

        row = {"Molecule": mol, "SSI": ssi_label, "NSI": nsi_label, "Method": current_method}
        row.update(values)
        rows.append(row)

    return pd.DataFrame(rows)


# Homonuclear diatomic notation in benchmark: "1H2" means 1H1H, etc.
_BENCH_HOMONUCLEAR = {
    "1H2": "1H1H",
    "14N2": "14N14N",
    "16O2": "16O16O",
    "32S2": "32S32S",
    "35Cl2": "35Cl35Cl",
}


def _parse_bench_name(name: str) -> str:
    """Convert a benchmark molecule name to an isotopologue label."""
    name = str(name).strip().replace("\xa0", "")
    if name in _BENCH_HOMONUCLEAR:
        return _BENCH_HOMONUCLEAR[name]
    # Already a valid isotopologue label (e.g., "12C16O", "1H35Cl")
    pairs = isotopes.parse_isotopologue_label(name)
    if len(pairs) == 2:
        return "".join(f"{m}{e}" for m, e in pairs)
    return name


def load_benchmark_we(
    file_path: str | Path,
    sheet_name: str = "Benchmark",
) -> Dict[str, Dict[str, float]]:
    """
    Load benchmark harmonic frequencies from an Excel file.

    Returns a dict mapping isotopologue label to {method_name: ωe_value}.
    """
    raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

    # Method columns: col_idx → method_name
    method_cols = {
        2: "Amila TQ5 (Molpro, 7pt fit)",
        4: "Piotr TQ5 (Molpro, 7pt fit)",
        6: "Piotr TQ5 (Molpro, full path)",
        8: "Piotr 7Z (Gaussian, full path)",
        10: "Piotr F12 (Molpro/Orca, full path)",
        12: "Simon (CCSD(T)/aug-cc-pv5Z)",
    }

    result: Dict[str, Dict[str, float]] = {}

    # Data rows start at row 4 (rows 0-3 are headers)
    for i in range(4, len(raw)):
        mol_raw = raw.iloc[i, 0]
        if _is_nan(mol_raw) or str(mol_raw).strip() == "":
            continue
        mol_name = str(mol_raw).strip().replace("\xa0", "")
        if not any(c.isdigit() for c in mol_name):
            continue

        label = _parse_bench_name(mol_name)
        methods: Dict[str, float] = {}

        # Also store the experimental value
        exp_val = raw.iloc[i, 1]
        try:
            methods["Experiment"] = float(exp_val)
        except (ValueError, TypeError):
            pass

        for col_idx, method_name in method_cols.items():
            if col_idx >= raw.shape[1]:
                continue
            val = raw.iloc[i, col_idx]
            try:
                methods[method_name] = float(val)
            except (ValueError, TypeError):
                continue

        if methods:
            result[label] = methods

    return result


def build_benchmark_parents(
    benchmark_we: Dict[str, Dict[str, float]],
    exp_parents: Dict[str, MolecularConstants],
) -> Dict[str, Dict[str, MolecularConstants]]:
    """
    For each benchmark method, create parent ``MolecularConstants`` using
    the benchmark ωe and experimental values for all other constants.

    Returns method_name → {isotopologue_label: MolecularConstants}.
    """
    # Collect all method names
    all_methods: set = set()
    for methods_dict in benchmark_we.values():
        all_methods.update(k for k in methods_dict if k != "Experiment")

    result: Dict[str, Dict[str, MolecularConstants]] = {}

    for method in sorted(all_methods):
        parents: Dict[str, MolecularConstants] = {}
        for label, methods_dict in benchmark_we.items():
            we_val = methods_dict.get(method)
            if we_val is None:
                continue
            exp_mc = exp_parents.get(label)
            if exp_mc is None:
                continue
            # Clone the experimental constants but replace ωe
            mc = MolecularConstants(
                isotopologue=exp_mc.isotopologue,
                harmonic=[we_val],
                anharmonic_poly=list(exp_mc.anharmonic_poly),
                anharmonic_dia=list(exp_mc.anharmonic_dia),
                rotational_constants=dict(exp_mc.rotational_constants),
                delta=[],  # will be recomputed in __post_init__
                alpha=list(exp_mc.alpha),
                symmetry=exp_mc.symmetry,
            )
            parents[label] = mc
        result[method] = parents

    return result


__all__ = [
    "MolecularConstants",
    "load_molecular_constants_from_excel",
    "load_molecular_constants_from_fit_output",
    "load_experimental_constants",
    "load_amila_rpfr",
    "generate_all_variants",
    "canonicalize_isotopologue",
    "is_most_common_isotopologue",
    "generate_isotopic_variants",
    "load_benchmark_we",
    "build_benchmark_parents",
]
