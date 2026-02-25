"""Tests for molecular constants loading, variant generation, and data parsing."""

import math
from pathlib import Path

import numpy as np
import pytest

from richet_rpfr.molecular_constants import (
    MolecularConstants,
    load_experimental_constants,
    load_amila_rpfr,
    generate_all_variants,
    generate_isotopic_variants,
    canonicalize_isotopologue,
    is_most_common_isotopologue,
    _parse_numeric,
    _parse_exp_name,
)
from richet_rpfr.isotopes import ISOTOPE_MASSES, compute_reduced_mass

DATA_FILE = Path(__file__).resolve().parent.parent / "RPFR-CCSD(T)_DTQ (1).xlsx"
HAVE_DATA = DATA_FILE.exists()


# ---------------------------------------------------------------------------
# MolecularConstants dataclass
# ---------------------------------------------------------------------------
class TestMolecularConstants:
    def test_mass_auto_computed(self):
        mc = MolecularConstants(isotopologue="1H19F", harmonic=[4138.0])
        m_H = ISOTOPE_MASSES[("H", 1)]
        m_F = ISOTOPE_MASSES[("F", 19)]
        assert mc.mass == pytest.approx(m_H + m_F, rel=1e-6)

    def test_mass_h2(self):
        mc = MolecularConstants(isotopologue="1H1H", harmonic=[4401.0])
        m_H = ISOTOPE_MASSES[("H", 1)]
        assert mc.mass == pytest.approx(2 * m_H, rel=1e-6)

    def test_moments_of_inertia_computed(self):
        mc = MolecularConstants(
            isotopologue="12C16O",
            harmonic=[2170.0],
            rotational_constants={"B0 (cm-1)": 1.9225},
        )
        assert "I_B" in mc.moments_of_inertia
        assert mc.moments_of_inertia["I_B"] > 0

    def test_delta_appended_from_alpha(self):
        mc = MolecularConstants(
            isotopologue="12C16O",
            harmonic=[2170.0],
            rotational_constants={"B0 (cm-1)": 1.9225},
            alpha=[0.0175],
        )
        assert len(mc.delta) == 1
        assert mc.delta[0] == pytest.approx(0.0175 / 1.9225, rel=1e-4)

    def test_unknown_isotope_raises(self):
        with pytest.raises(ValueError, match="No isotopic mass"):
            MolecularConstants(isotopologue="99X1H")


# ---------------------------------------------------------------------------
# _parse_numeric
# ---------------------------------------------------------------------------
class TestParseNumeric:
    def test_float_passthrough(self):
        assert _parse_numeric(3.14) == pytest.approx(3.14)

    def test_int_passthrough(self):
        assert _parse_numeric(42) == pytest.approx(42.0)

    def test_scientific_star_notation(self):
        assert _parse_numeric("4.71*10-2") == pytest.approx(0.0471, rel=1e-3)

    def test_scientific_times_notation(self):
        assert _parse_numeric("2.335*10-3") == pytest.approx(0.002335, rel=1e-3)

    def test_dash_returns_none(self):
        assert _parse_numeric("-") is None

    def test_empty_returns_none(self):
        assert _parse_numeric("") is None

    def test_nan_returns_none(self):
        assert _parse_numeric(float("nan")) is None


# ---------------------------------------------------------------------------
# _parse_exp_name
# ---------------------------------------------------------------------------
class TestParseExpName:
    def test_shorthand_h2(self):
        assert _parse_exp_name("H2") == "1H1H"

    def test_shorthand_hd(self):
        assert _parse_exp_name("HD") == "1H2H"

    def test_shorthand_hf(self):
        assert _parse_exp_name("HF") == "1H19F"

    def test_shorthand_df(self):
        assert _parse_exp_name("DF") == "2H19F"

    def test_shorthand_naf(self):
        assert _parse_exp_name("NaF") == "23Na19F"

    def test_shorthand_na35cl(self):
        assert _parse_exp_name("Na35Cl") == "23Na35Cl"

    def test_shorthand_7lif(self):
        assert _parse_exp_name("7LiF") == "7Li19F"

    def test_shorthand_39kf(self):
        assert _parse_exp_name("39KF") == "39K19F"

    def test_already_labelled(self):
        assert _parse_exp_name("14N14N") == "14N14N"

    def test_already_labelled_multi(self):
        assert _parse_exp_name("24Mg16O") == "24Mg16O"


# ---------------------------------------------------------------------------
# canonicalize / is_most_common
# ---------------------------------------------------------------------------
class TestCanonicalize:
    def test_already_canonical(self):
        assert canonicalize_isotopologue("12C16O") == "12C16O"

    def test_reordering(self):
        # O before C alphabetically? No, C < O, so 12C16O is canonical
        # But 16O12C should reorder to 12C16O
        assert canonicalize_isotopologue("16O12C") == "12C16O"


class TestIsMostCommon:
    def test_h2_most_common(self):
        assert is_most_common_isotopologue("1H1H") is True

    def test_hd_not_most_common(self):
        assert is_most_common_isotopologue("1H2H") is False

    def test_co_most_common(self):
        assert is_most_common_isotopologue("12C16O") is True

    def test_c13o_not_most_common(self):
        assert is_most_common_isotopologue("13C16O") is False


# ---------------------------------------------------------------------------
# generate_isotopic_variants — scaling rules
# ---------------------------------------------------------------------------
class TestGenerateIsotopicVariants:
    @pytest.fixture
    def h2_mc(self):
        return MolecularConstants(
            isotopologue="1H1H",
            harmonic=[4401.21],
            anharmonic_dia=[121.336],
            rotational_constants={"B0 (cm-1)": 59.322},
            alpha=[3.062],
            symmetry=2.0,
        )

    def test_generates_variants(self, h2_mc):
        variants = list(generate_isotopic_variants(h2_mc))
        # H has isotopes 1,2,3 → 3×3=9 combinations minus 1 original = 8
        assert len(variants) == 8

    def test_variant_labels(self, h2_mc):
        labels = {v["Molecule"] for v in generate_isotopic_variants(h2_mc)}
        assert "1H2H" in labels
        assert "2H2H" in labels
        assert "1H3H" in labels

    def test_include_original(self, h2_mc):
        variants = list(generate_isotopic_variants(h2_mc, include_original=True))
        labels = {v["Molecule"] for v in variants}
        assert "1H1H" in labels
        assert len(variants) == 9

    def test_frequency_scaling(self, h2_mc):
        """Scaled frequency should follow ω_new = ω_ref × ρ where ρ = √(μ_ref/μ_new)."""
        variants = {v["Molecule"]: v for v in generate_isotopic_variants(h2_mc)}
        hd = variants["1H2H"]

        mu_ref = compute_reduced_mass([(1, "H"), (1, "H")])
        mu_hd = compute_reduced_mass([(1, "H"), (2, "H")])
        rho = np.sqrt(mu_ref / mu_hd)

        assert hd["w1 (cm-1)"] == pytest.approx(4401.21 * rho, rel=1e-8)

    def test_anharmonic_scaling(self, h2_mc):
        """Anharmonic constant scales as ρ²."""
        variants = {v["Molecule"]: v for v in generate_isotopic_variants(h2_mc)}
        hd = variants["1H2H"]

        mu_ref = compute_reduced_mass([(1, "H"), (1, "H")])
        mu_hd = compute_reduced_mass([(1, "H"), (2, "H")])
        rho = np.sqrt(mu_ref / mu_hd)

        assert hd["w1x1 (cm-1)"] == pytest.approx(121.336 * rho**2, rel=1e-8)

    def test_rotational_scaling(self, h2_mc):
        """Rotational constant scales as ρ²."""
        variants = {v["Molecule"]: v for v in generate_isotopic_variants(h2_mc)}
        hd = variants["1H2H"]

        mu_ref = compute_reduced_mass([(1, "H"), (1, "H")])
        mu_hd = compute_reduced_mass([(1, "H"), (2, "H")])
        rho = np.sqrt(mu_ref / mu_hd)

        assert hd["B0 (cm-1)"] == pytest.approx(59.322 * rho**2, rel=1e-8)

    def test_alpha_scaling(self, h2_mc):
        """Vibration-rotation constant scales as ρ³."""
        variants = {v["Molecule"]: v for v in generate_isotopic_variants(h2_mc)}
        hd = variants["1H2H"]

        mu_ref = compute_reduced_mass([(1, "H"), (1, "H")])
        mu_hd = compute_reduced_mass([(1, "H"), (2, "H")])
        rho = np.sqrt(mu_ref / mu_hd)

        assert hd["a1 (cm-1)"] == pytest.approx(3.062 * rho**3, rel=1e-8)


# ---------------------------------------------------------------------------
# Excel loaders (require data file)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not HAVE_DATA, reason="Data file not found")
class TestLoadExperimentalConstants:
    @pytest.fixture(scope="class")
    def exp(self):
        return load_experimental_constants(DATA_FILE)

    def test_returns_dict(self, exp):
        assert isinstance(exp, dict)
        assert len(exp) >= 30

    def test_h2_loaded(self, exp):
        assert "1H1H" in exp
        mc = exp["1H1H"]
        assert mc.harmonic[0] == pytest.approx(4401.21, rel=1e-4)
        assert mc.symmetry == 2.0

    def test_co_loaded(self, exp):
        assert "12C16O" in exp
        mc = exp["12C16O"]
        assert mc.harmonic[0] == pytest.approx(2169.81, rel=1e-4)
        assert mc.symmetry is None

    def test_b0_from_be(self, exp):
        """B0 = Be - αe/2."""
        mc = exp["1H1H"]
        # Be=60.853, αe=3.062 → B0=59.322
        assert mc.rotational_constants["B0 (cm-1)"] == pytest.approx(59.322, rel=1e-3)

    def test_kf_loaded(self, exp):
        """39KF should be loaded despite αe having *10- notation."""
        assert "39K19F" in exp
        mc = exp["39K19F"]
        assert mc.harmonic[0] == pytest.approx(428.0, rel=1e-3)

    def test_nacl_loaded(self, exp):
        assert "23Na35Cl" in exp

    def test_lif_loaded(self, exp):
        assert "7Li19F" in exp

    def test_homonuclear_symmetry(self, exp):
        for label in ["1H1H", "2H2H", "14N14N", "16O16O", "32S32S", "35Cl35Cl"]:
            if label in exp:
                assert exp[label].symmetry == 2.0, f"{label} should have symmetry=2"

    def test_heteronuclear_no_symmetry(self, exp):
        for label in ["1H19F", "12C16O", "14N16O", "32S16O"]:
            if label in exp:
                assert exp[label].symmetry is None, f"{label} should have symmetry=None"


@pytest.mark.skipif(not HAVE_DATA, reason="Data file not found")
class TestLoadAmilaRPFR:
    @pytest.fixture(scope="class")
    def amila(self):
        return load_amila_rpfr(DATA_FILE)

    def test_returns_dataframe(self, amila):
        assert len(amila) > 100

    def test_required_columns(self, amila):
        for col in ["Molecule", "SSI", "NSI", "Method"]:
            assert col in amila.columns

    def test_methods_present(self, amila):
        methods = amila["Method"].unique()
        assert "CCSD(T)_Q5" in methods
        assert "CCSD(T)_DTQ" in methods

    def test_ssi_nsi_split(self, amila):
        row = amila[amila["Molecule"] == "1H2H/1H1H"].iloc[0]
        assert row["SSI"] == "1H2H"
        assert row["NSI"] == "1H1H"

    def test_no_section_labels_in_data(self, amila):
        """Section labels like CCSD(T)_Q5 should not appear as Molecule values."""
        assert not amila["Molecule"].str.startswith("CCSD(T)").any()


@pytest.mark.skipif(not HAVE_DATA, reason="Data file not found")
class TestGenerateAllVariants:
    @pytest.fixture(scope="class")
    def lookup(self):
        exp = load_experimental_constants(DATA_FILE)
        return generate_all_variants(exp)

    def test_parents_included(self, lookup):
        assert "1H1H" in lookup
        assert "12C16O" in lookup

    def test_variants_generated(self, lookup):
        assert "1H2H" in lookup
        assert "13C16O" in lookup
        assert "12C18O" in lookup

    def test_all_amila_pairs_covered(self, lookup):
        amila = load_amila_rpfr(DATA_FILE)
        missing = set()
        for _, row in amila.iterrows():
            if row["NSI"] not in lookup:
                missing.add(row["NSI"])
            if row["SSI"] not in lookup:
                missing.add(row["SSI"])
        assert missing == set(), f"Missing isotopologues: {missing}"

    def test_variant_has_correct_type(self, lookup):
        mc = lookup["1H2H"]
        assert isinstance(mc, MolecularConstants)
        assert len(mc.harmonic) == 1
        assert mc.mass > 0
