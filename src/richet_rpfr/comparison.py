"""
Comparison utilities for experimental vs. computed molecular constants.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

import pandas as pd

from .molecular_constants import canonicalize_isotopologue


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


__all__ = ["compare_experimental_and_computed", "drop_duplicate_isotopologues"]
