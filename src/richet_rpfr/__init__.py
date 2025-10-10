"""
richet_rpfr
===========

Utilities for replicating Richet et al. (1977) reduced partition function ratios,
generating Gaussian input decks, and comparing experimental and computed
molecular constants.
"""

from . import isotopes
from .molecular_constants import (
    MolecularConstants,
    load_molecular_constants_from_excel,
    load_molecular_constants_from_fit_output,
    canonicalize_isotopologue,
    is_most_common_isotopologue,
    generate_isotopic_variants,
)
from .partition_functions import PartitionFunctionCalculator
from .richet_tables import (
    compute_and_finalize_with_temp,
    create_summary_df,
    create_diff_series,
)
from .comparison import compare_experimental_and_computed

__all__ = [
    "isotopes",
    "MolecularConstants",
    "load_molecular_constants_from_excel",
    "load_molecular_constants_from_fit_output",
    "canonicalize_isotopologue",
    "is_most_common_isotopologue",
    "generate_isotopic_variants",
    "PartitionFunctionCalculator",
    "compute_and_finalize_with_temp",
    "create_summary_df",
    "create_diff_series",
    "compare_experimental_and_computed",
]
