"""Tests for the comparison pipeline — Amila vs Experimental RPFR."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from richet_rpfr.molecular_constants import (
    load_experimental_constants,
    load_amila_rpfr,
    generate_all_variants,
)
from richet_rpfr.comparison import (
    compute_experimental_rpfr,
    compare_amila_to_experimental,
    AMILA_CONTRIB_COLS,
)

DATA_FILE = Path(__file__).resolve().parent.parent / "RPFR-CCSD(T)_DTQ (1).xlsx"
HAVE_DATA = DATA_FILE.exists()


@pytest.mark.skipif(not HAVE_DATA, reason="Data file not found")
class TestComputeExperimentalRPFR:
    @pytest.fixture(scope="class")
    def data(self):
        amila = load_amila_rpfr(DATA_FILE)
        exp = load_experimental_constants(DATA_FILE)
        lookup = generate_all_variants(exp)
        exp_rpfr = compute_experimental_rpfr(amila, lookup)
        return amila, exp_rpfr

    def test_same_length(self, data):
        amila, exp_rpfr = data
        assert len(exp_rpfr) == len(amila)

    def test_no_nan_rpfr(self, data):
        _, exp_rpfr = data
        assert exp_rpfr["RPFR 273.15K"].isna().sum() == 0

    def test_all_contributions_present(self, data):
        _, exp_rpfr = data
        for col in AMILA_CONTRIB_COLS:
            assert col in exp_rpfr.columns

    def test_translational_positive(self, data):
        _, exp_rpfr = data
        assert (exp_rpfr["Translational"] > 0).all()

    def test_rpfr_positive(self, data):
        """All RPFR values should be positive."""
        _, exp_rpfr = data
        assert (exp_rpfr["RPFR 273.15K"] > 0).all()

    def test_molecule_column_preserved(self, data):
        amila, exp_rpfr = data
        assert list(exp_rpfr["Molecule"]) == list(amila["Molecule"])


@pytest.mark.skipif(not HAVE_DATA, reason="Data file not found")
class TestCompareAmilaToExperimental:
    @pytest.fixture(scope="class")
    def comparison(self):
        amila = load_amila_rpfr(DATA_FILE)
        exp = load_experimental_constants(DATA_FILE)
        lookup = generate_all_variants(exp)
        exp_rpfr = compute_experimental_rpfr(amila, lookup)
        return compare_amila_to_experimental(amila, exp_rpfr)

    def test_has_diff_columns(self, comparison):
        diff_cols = [c for c in comparison.columns if "Diff" in c]
        assert len(diff_cols) == len(AMILA_CONTRIB_COLS)

    def test_has_amila_columns(self, comparison):
        amila_cols = [c for c in comparison.columns if "Amila" in c]
        assert len(amila_cols) == len(AMILA_CONTRIB_COLS)

    def test_has_exp_columns(self, comparison):
        exp_cols = [c for c in comparison.columns if "Exp" in c]
        assert len(exp_cols) == len(AMILA_CONTRIB_COLS)

    def test_diff_is_permil(self, comparison):
        """Differences should be in per-mil (reasonable range for most pairs)."""
        # Exclude outliers (e.g. T35Cl data issue) — check median is reasonable
        rpfr_diff = comparison["RPFR 273.15K (Diff \u2030)"]
        median_abs_diff = rpfr_diff.abs().median()
        assert median_abs_diff < 50, f"Median |diff| = {median_abs_diff} ‰ seems too large"

    def test_known_pair_diff_small(self, comparison):
        """For well-characterised pairs like C18O/C16O, diff should be small."""
        row = comparison[comparison["Molecule"] == "12C18O/12C16O"]
        if not row.empty:
            rpfr_diff = abs(row["RPFR 273.15K (Diff \u2030)"].values[0])
            assert rpfr_diff < 10, f"C18O/C16O RPFR diff = {rpfr_diff} ‰ seems too large"
