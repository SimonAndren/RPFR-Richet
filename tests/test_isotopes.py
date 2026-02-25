"""Tests for isotope data, label parsing, and reduced mass computation."""

import math

import pytest

from richet_rpfr.isotopes import (
    ISOTOPE_MASSES,
    MOST_ABUNDANT,
    parse_isotopologue_label,
    isotopologue_to_formula,
    compute_reduced_mass,
    degeneracy_clusters,
)


# ---------------------------------------------------------------------------
# parse_isotopologue_label
# ---------------------------------------------------------------------------
class TestParseIsotopologueLabel:
    def test_simple_diatomic(self):
        assert parse_isotopologue_label("1H1H") == [(1, "H"), (1, "H")]

    def test_heteronuclear(self):
        assert parse_isotopologue_label("12C16O") == [(12, "C"), (16, "O")]

    def test_deuterium_substituted(self):
        assert parse_isotopologue_label("1H2H") == [(1, "H"), (2, "H")]

    def test_heavy_atoms(self):
        result = parse_isotopologue_label("40Ca32S")
        assert result == [(40, "Ca"), (32, "S")]

    def test_empty_string(self):
        assert parse_isotopologue_label("") == []


# ---------------------------------------------------------------------------
# isotopologue_to_formula
# ---------------------------------------------------------------------------
class TestIsotopologueToFormula:
    def test_homonuclear(self):
        assert isotopologue_to_formula("1H1H") == "H2"

    def test_heteronuclear(self):
        assert isotopologue_to_formula("12C16O") == "CO"

    def test_different_isotopes_same_element(self):
        # 1H2H is HD — atoms differ so should not collapse to H2
        assert isotopologue_to_formula("1H2H") == "H2"


# ---------------------------------------------------------------------------
# compute_reduced_mass
# ---------------------------------------------------------------------------
class TestComputeReducedMass:
    def test_h2(self):
        m_H = ISOTOPE_MASSES[("H", 1)]
        expected = (m_H * m_H) / (m_H + m_H)
        result = compute_reduced_mass([(1, "H"), (1, "H")])
        assert result == pytest.approx(expected, rel=1e-10)

    def test_hd(self):
        m_H = ISOTOPE_MASSES[("H", 1)]
        m_D = ISOTOPE_MASSES[("H", 2)]
        expected = (m_H * m_D) / (m_H + m_D)
        result = compute_reduced_mass([(1, "H"), (2, "H")])
        assert result == pytest.approx(expected, rel=1e-10)

    def test_co(self):
        m_C = ISOTOPE_MASSES[("C", 12)]
        m_O = ISOTOPE_MASSES[("O", 16)]
        expected = (m_C * m_O) / (m_C + m_O)
        result = compute_reduced_mass([(12, "C"), (16, "O")])
        assert result == pytest.approx(expected, rel=1e-10)

    def test_not_diatomic_raises(self):
        with pytest.raises(ValueError, match="diatomics"):
            compute_reduced_mass([(1, "H")])


# ---------------------------------------------------------------------------
# degeneracy_clusters
# ---------------------------------------------------------------------------
class TestDegeneracyClusters:
    def test_single_frequency(self):
        assert degeneracy_clusters([1000.0]) == [[1000.0]]

    def test_degenerate_pair(self):
        result = degeneracy_clusters([1000.0, 1000.5])
        assert len(result) == 1
        assert len(result[0]) == 2

    def test_separated_frequencies(self):
        result = degeneracy_clusters([500.0, 1000.0, 1500.0])
        assert len(result) == 3

    def test_empty(self):
        assert degeneracy_clusters([]) == []

    def test_nan_values_filtered(self):
        result = degeneracy_clusters([float("nan"), 1000.0, float("nan")])
        assert len(result) == 1
        assert result[0] == [1000.0]


# ---------------------------------------------------------------------------
# ISOTOPE_MASSES completeness
# ---------------------------------------------------------------------------
class TestIsotopeMassesCompleteness:
    @pytest.mark.parametrize("element,mass_number", [
        ("Li", 6), ("Li", 7),
        ("Na", 23),
        ("Mg", 24), ("Mg", 25), ("Mg", 26),
        ("Ca", 40), ("Ca", 42), ("Ca", 43), ("Ca", 44), ("Ca", 46), ("Ca", 48),
        ("K", 39), ("K", 40), ("K", 41),
        ("H", 1), ("H", 2), ("H", 3),
        ("C", 12), ("C", 13), ("C", 14),
        ("O", 16), ("O", 17), ("O", 18),
        ("S", 32), ("S", 33), ("S", 34), ("S", 36),
    ])
    def test_isotope_exists(self, element, mass_number):
        assert (element, mass_number) in ISOTOPE_MASSES

    @pytest.mark.parametrize("element", ["Li", "Na", "Mg", "Ca", "K"])
    def test_most_abundant_entry(self, element):
        assert element in MOST_ABUNDANT
