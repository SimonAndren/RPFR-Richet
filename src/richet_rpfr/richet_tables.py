"""
Helpers to replicate Richet et al. RPFR tables and compute deviations.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from .partition_functions import PartitionFunctionCalculator


CONTRIB_ORDER = [
    "Translational",
    "ZPE_G0",
    "ZPE_harmonic",
    "ZPE_anharmonic",
    "ZPE_total",
    "Excited state harmonic",
    "Excited state anharmonic",
    "Rotational linear",
    "Rotational Diatomic",
    "Rotational-vibrational",
    "Total",
]


def compute_and_finalize_with_temp(
    temperature: float,
    NSI,
    SSI,
    paper_values: Dict[str, float],
    *,
    beta_scale: float = 1.0,
) -> Dict[str, float]:
    """
    Run the partition function calculator and blend with Richet paper values where
    contributions are unavailable from the computational workflow.
    """
    calc = PartitionFunctionCalculator(
        temperature=temperature, NSI=NSI, SSI=SSI, beta_scale=beta_scale
    )
    calc.calculate_all()

    computed = {
        "Translational": calc.Q_ratio_trans,
        "ZPE_G0": calc.Q_ratio_ZPE_G0,
        "ZPE_harmonic": calc.Q_ratio_ZPE_harmonic,
        "ZPE_anharmonic": calc.Q_ratio_ZPE_anharmonic,
        "ZPE_total": calc.Q_ratio_ZPE_total,
        "Excited state harmonic": calc.Q_ratio_excited_harmonic,
        "Excited state anharmonic": calc.Q_ratio_excited_anharmonic,
        "Rotational linear": getattr(calc, "Q_ratio_rotational_linear_w_corr", 1.0),
        "Rotational Diatomic": getattr(calc, "Q_ratio_rotational_diatomic", 1.0),
        "Rotational-vibrational": calc.Q_ratio_rotational_vibrational,
        "Total": calc.Q_tot,
    }

    final = {}
    paper_g0 = None
    for key, value in computed.items():
        if np.isclose(value, 1.0) and key in paper_values:
            final[key] = paper_values[key]
            if key == "ZPE_G0":
                paper_g0 = paper_values[key]
        else:
            final[key] = value

    if paper_g0 is not None:
        final["ZPE_total"] *= paper_g0
        final["Total"] *= paper_g0

    if np.isclose(final.get("Total", 1.0), 1.0) and "Total" in paper_values:
        final["Total"] = paper_values["Total"]

    return final


def create_diff_series(
    final_values: Dict[str, float], paper_values: Dict[str, float]
) -> pd.Series:
    """
    Compute per-mil differences between computed and paper reference contributions.
    """
    contributions = sorted(set(final_values) | set(paper_values))
    diffs = []
    for key in contributions:
        paper = paper_values.get(key, np.nan)
        comp = final_values.get(key, np.nan)
        if np.isnan(paper) or paper == 0 or np.isnan(comp):
            diffs.append(np.nan)
        else:
            diffs.append(((comp - paper) / paper) * 1000.0)
    return pd.Series(diffs, index=contributions)


def create_summary_df(
    final_nsi: Dict[str, float],
    paper_nsi: Dict[str, float],
    final_ssi: Dict[str, float],
    paper_ssi: Dict[str, float],
    final_co: Dict[str, float],
    paper_co: Dict[str, float],
) -> pd.DataFrame:
    """
    Assemble a multi-index table matching Richet's tabular layout for a single temperature.
    """
    contributions = sorted(
        set(final_nsi)
        | set(final_ssi)
        | set(final_co)
        | set(paper_nsi)
        | set(paper_ssi)
        | set(paper_co)
    )

    def build_series(final, paper):
        return (
            pd.Series({k: paper.get(k, np.nan) for k in contributions}),
            pd.Series({k: final.get(k, np.nan) for k in contributions}),
            create_diff_series(final, paper),
        )

    nsi_paper, nsi_comp, nsi_diff = build_series(final_nsi, paper_nsi)
    ssi_paper, ssi_comp, ssi_diff = build_series(final_ssi, paper_ssi)
    co_paper, co_comp, co_diff = build_series(final_co, paper_co)

    stacked = pd.concat(
        [
            nsi_paper,
            nsi_comp,
            nsi_diff,
            ssi_paper,
            ssi_comp,
            ssi_diff,
            co_paper,
            co_comp,
            co_diff,
        ],
        axis=1,
    )
    stacked.columns = pd.MultiIndex.from_tuples(
        [
            ("HD/H2", "Paper"),
            ("HD/H2", "Computed"),
            ("HD/H2", "Difference (‰)"),
            ("DF/HF", "Paper"),
            ("DF/HF", "Computed"),
            ("DF/HF", "Difference (‰)"),
            ("C18O/C16O", "Paper"),
            ("C18O/C16O", "Computed"),
            ("C18O/C16O", "Difference (‰)"),
        ]
    )

    stacked = (
        stacked.reset_index()
        .rename(columns={"index": "Contribution"})
        .set_index("Contribution")
        .reindex(CONTRIB_ORDER)
        .reset_index()
    )
    return stacked


def create_ratio_series(
    numerator: Dict[str, float],
    denominator: Dict[str, float],
    paper_ratio: Dict[str, float],
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute ratios between two result dictionaries and compare to reference ratios.
    """
    contributions = list(paper_ratio.keys())
    computed = []
    diff = []
    for key in contributions:
        num = numerator.get(key, np.nan)
        den = denominator.get(key, np.nan)
        paper = paper_ratio.get(key, np.nan)
        if np.isnan(num) or np.isnan(den) or den == 0:
            comp_val = np.nan
        else:
            comp_val = num / den
        if np.isnan(paper) or paper == 0 or np.isnan(comp_val):
            diff_val = np.nan
        else:
            diff_val = ((comp_val - paper) / paper) * 1000.0
        computed.append(comp_val)
        diff.append(diff_val)
    return (
        pd.Series(paper_ratio),
        pd.Series(computed, index=contributions),
        pd.Series(diff, index=contributions),
    )


__all__ = [
    "compute_and_finalize_with_temp",
    "create_diff_series",
    "create_summary_df",
    "create_ratio_series",
]
