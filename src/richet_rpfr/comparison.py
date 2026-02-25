"""
Comparison utilities for experimental vs. computed molecular constants.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from .molecular_constants import MolecularConstants, canonicalize_isotopologue
from .partition_functions import PartitionFunctionCalculator


DEFAULT_COLUMNS = ["w1 (cm-1)", "w1x1 (cm-1)", "B0 (cm-1)", "a1 (cm-1)"]


def compare_experimental_and_computed(
    df_experimental: pd.DataFrame,
    df_computed: pd.DataFrame,
    columns: Sequence[str] = DEFAULT_COLUMNS,
) -> pd.DataFrame:
    """
    Return a row-per-constant comparison between experimental and generated isotopologues.
    """
    merged = pd.merge(
        df_experimental[["Molecule", *columns]],
        df_computed[["Molecule", *columns]],
        on="Molecule",
        suffixes=("_exp", "_comp"),
        how="inner",
    )
    records = []
    for _, row in merged.iterrows():
        for column in columns:
            exp = row[f"{column}_exp"]
            comp = row[f"{column}_comp"]
            diff = comp - exp if pd.notna(exp) and pd.notna(comp) else None
            records.append(
                {
                    "Molecule": row["Molecule"],
                    "Constant": column,
                    "Experimental_value": exp,
                    "Computed_value": comp,
                    "Difference": diff,
                }
            )
    return pd.DataFrame(records)


def drop_duplicate_isotopologues(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows representing the same isotopologue regardless of ordering of atoms (e.g. ``10B11B`` vs ``11B10B``).
    """
    df = df.copy()
    if "Molecule" not in df.columns:
        return df
    df["Canonical"] = df["Molecule"].apply(canonicalize_isotopologue)
    df = df.drop_duplicates(subset="Canonical", keep="first")
    return df.drop(columns="Canonical")


AMILA_CONTRIB_COLS = [
    "Translational",
    "ZPE (total)",
    "Excited (harmonic)",
    "Excited (anharmonic)",
    "Rotational (diatomic)",
    "Rotational-vibrational",
    "RPFR 273.15K",
]


def compute_experimental_rpfr(
    amila_df: pd.DataFrame,
    exp_lookup: Dict[str, MolecularConstants],
    temperature: float = 273.15,
) -> pd.DataFrame:
    """
    Compute RPFR contributions from experimental constants for each
    isotopologue pair in *amila_df*.

    Parameters
    ----------
    amila_df : DataFrame
        Output of ``load_amila_rpfr`` with ``SSI`` and ``NSI`` columns.
    exp_lookup : dict
        Isotopologue label → ``MolecularConstants``, typically from
        ``generate_all_variants(load_experimental_constants(...))``.
    temperature : float
        Temperature in Kelvin (default 273.15 K).

    Returns a DataFrame with the same rows/columns as Amila's data but
    values computed from experimental constants.
    """
    rows = []
    for _, arow in amila_df.iterrows():
        nsi_label = arow["NSI"]
        ssi_label = arow["SSI"]

        nsi_mc = exp_lookup.get(nsi_label)
        ssi_mc = exp_lookup.get(ssi_label)

        row = {
            "Molecule": arow["Molecule"],
            "SSI": ssi_label,
            "NSI": nsi_label,
            "Method": arow["Method"],
        }

        if nsi_mc is None or ssi_mc is None:
            for col in AMILA_CONTRIB_COLS:
                row[col] = np.nan
            rows.append(row)
            continue

        calc = PartitionFunctionCalculator(
            temperature=temperature, NSI=nsi_mc, SSI=ssi_mc
        )
        calc.calculate_all()

        row["Translational"] = calc.Q_ratio_trans
        row["ZPE (total)"] = calc.Q_ratio_ZPE_total
        row["Excited (harmonic)"] = calc.Q_ratio_excited_harmonic
        row["Excited (anharmonic)"] = calc.Q_ratio_excited_anharmonic
        row["Rotational (diatomic)"] = getattr(calc, "Q_ratio_rotational_diatomic", 1.0)
        row["Rotational-vibrational"] = calc.Q_ratio_rotational_vibrational
        row["RPFR 273.15K"] = calc.Q_tot

        rows.append(row)

    return pd.DataFrame(rows)


def compare_amila_to_experimental(
    amila_df: pd.DataFrame,
    exp_rpfr_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge Amila and experimental RPFR DataFrames and compute per-mil
    differences for each contribution.

    Returns a DataFrame with columns for each contribution showing
    Amila value, Experimental value, and Difference (permil).
    """
    merged = amila_df[["Molecule", "Method"] + AMILA_CONTRIB_COLS].merge(
        exp_rpfr_df[["Molecule", "Method"] + AMILA_CONTRIB_COLS],
        on=["Molecule", "Method"],
        suffixes=("_amila", "_exp"),
    )

    result = merged[["Molecule", "Method"]].copy()
    for col in AMILA_CONTRIB_COLS:
        amila_val = merged[f"{col}_amila"]
        exp_val = merged[f"{col}_exp"]
        result[f"{col} (Amila)"] = amila_val
        result[f"{col} (Exp)"] = exp_val
        result[f"{col} (Diff ‰)"] = ((amila_val - exp_val) / exp_val) * 1000.0

    return result


def compute_benchmark_rpfr(
    amila_df: pd.DataFrame,
    benchmark_parents: Dict[str, Dict[str, "MolecularConstants"]],
    temperature: float = 273.15,
) -> pd.DataFrame:
    """
    Compute RPFRs for each benchmark method.

    Parameters
    ----------
    amila_df : DataFrame
        Output of ``load_amila_rpfr`` — defines which isotopologue pairs to compute.
    benchmark_parents : dict
        method_name → {isotopologue_label: MolecularConstants}, from
        ``build_benchmark_parents``.
    temperature : float
        Temperature in Kelvin.

    Returns a long-format DataFrame with columns: Molecule, SSI, NSI,
    Amila_Method, Benchmark_Method, + AMILA_CONTRIB_COLS.
    """
    from .molecular_constants import generate_all_variants

    all_rows = []

    for bench_method, parents in benchmark_parents.items():
        lookup = generate_all_variants(parents)

        for _, arow in amila_df.iterrows():
            nsi_label = arow["NSI"]
            ssi_label = arow["SSI"]

            nsi_mc = lookup.get(nsi_label)
            ssi_mc = lookup.get(ssi_label)

            row = {
                "Molecule": arow["Molecule"],
                "SSI": ssi_label,
                "NSI": nsi_label,
                "Amila_Method": arow["Method"],
                "Benchmark_Method": bench_method,
            }

            if nsi_mc is None or ssi_mc is None:
                for col in AMILA_CONTRIB_COLS:
                    row[col] = np.nan
                all_rows.append(row)
                continue

            calc = PartitionFunctionCalculator(
                temperature=temperature, NSI=nsi_mc, SSI=ssi_mc
            )
            calc.calculate_all()

            row["Translational"] = calc.Q_ratio_trans
            row["ZPE (total)"] = calc.Q_ratio_ZPE_total
            row["Excited (harmonic)"] = calc.Q_ratio_excited_harmonic
            row["Excited (anharmonic)"] = calc.Q_ratio_excited_anharmonic
            row["Rotational (diatomic)"] = getattr(
                calc, "Q_ratio_rotational_diatomic", 1.0
            )
            row["Rotational-vibrational"] = calc.Q_ratio_rotational_vibrational
            row["RPFR 273.15K"] = calc.Q_tot

            all_rows.append(row)

    return pd.DataFrame(all_rows)


__all__ = [
    "compare_experimental_and_computed",
    "drop_duplicate_isotopologues",
    "compute_experimental_rpfr",
    "compare_amila_to_experimental",
    "compute_benchmark_rpfr",
    "AMILA_CONTRIB_COLS",
]
