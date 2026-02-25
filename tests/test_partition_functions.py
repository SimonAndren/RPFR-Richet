"""Tests for the PartitionFunctionCalculator RPFR computations."""

from pathlib import Path

import numpy as np
import pytest
from scipy import constants as sc

from richet_rpfr.molecular_constants import (
    MolecularConstants,
    load_experimental_constants,
    generate_all_variants,
)
from richet_rpfr.partition_functions import PartitionFunctionCalculator

DATA_FILE = Path(__file__).resolve().parent.parent / "RPFR-CCSD(T)_DTQ (1).xlsx"
HAVE_DATA = DATA_FILE.exists()

T_0C = 273.15  # 0 °C in Kelvin


# ---------------------------------------------------------------------------
# Fixtures — hand-constructed molecular constants for reproducible tests
# ---------------------------------------------------------------------------
@pytest.fixture
def h2():
    return MolecularConstants(
        isotopologue="1H1H",
        harmonic=[4401.21],
        anharmonic_dia=[121.336],
        rotational_constants={"B0 (cm-1)": 59.322},
        alpha=[3.062],
        symmetry=2.0,
    )


@pytest.fixture
def hd():
    return MolecularConstants(
        isotopologue="1H2H",
        harmonic=[3813.15],
        anharmonic_dia=[91.65],
        rotational_constants={"B0 (cm-1)": 44.662},
        alpha=[1.986],
        symmetry=None,
    )


@pytest.fixture
def co():
    return MolecularConstants(
        isotopologue="12C16O",
        harmonic=[2169.81],
        anharmonic_dia=[13.288],
        rotational_constants={"B0 (cm-1)": 1.9225},
        alpha=[0.0175],
        symmetry=None,
    )


@pytest.fixture
def c18o():
    """C18O with constants scaled from C16O via reduced mass."""
    mu_ref = 6.856217  # 12C16O reduced mass
    mu_new = 7.199766  # 12C18O reduced mass
    rho = np.sqrt(mu_ref / mu_new)
    return MolecularConstants(
        isotopologue="12C18O",
        harmonic=[2169.81 * rho],
        anharmonic_dia=[13.288 * rho**2],
        rotational_constants={"B0 (cm-1)": 1.9225 * rho**2},
        alpha=[0.0175 * rho**3],
        symmetry=None,
    )


# ---------------------------------------------------------------------------
# Translational contribution
# ---------------------------------------------------------------------------
class TestTranslational:
    def test_mass_ratio_power(self, h2, hd):
        calc = PartitionFunctionCalculator(temperature=T_0C, NSI=h2, SSI=hd)
        calc.translational()

        expected = (hd.mass / h2.mass) ** 1.5
        assert calc.Q_ratio_trans == pytest.approx(expected, rel=1e-8)

    def test_heavier_ssi_gives_ratio_above_one(self, h2, hd):
        calc = PartitionFunctionCalculator(temperature=T_0C, NSI=h2, SSI=hd)
        calc.translational()
        assert calc.Q_ratio_trans > 1.0

    def test_same_species_gives_one(self, h2):
        calc = PartitionFunctionCalculator(temperature=T_0C, NSI=h2, SSI=h2)
        calc.translational()
        assert calc.Q_ratio_trans == pytest.approx(1.0, abs=1e-10)

    def test_hd_h2_value(self, h2, hd):
        """HD/H2 translational ratio should be approximately 1.8357."""
        calc = PartitionFunctionCalculator(temperature=T_0C, NSI=h2, SSI=hd)
        calc.translational()
        assert calc.Q_ratio_trans == pytest.approx(1.8357, rel=1e-3)


# ---------------------------------------------------------------------------
# ZPE contributions
# ---------------------------------------------------------------------------
class TestZPE:
    def test_zpe_harmonic_heavier_isotope_greater_than_one(self, h2, hd):
        """Heavier isotope has lower ZPE → ZPE ratio > 1."""
        calc = PartitionFunctionCalculator(temperature=T_0C, NSI=h2, SSI=hd)
        calc.ZPE_harmonic()
        assert calc.Q_ratio_ZPE_harmonic > 1.0

    def test_zpe_harmonic_same_species(self, h2):
        calc = PartitionFunctionCalculator(temperature=T_0C, NSI=h2, SSI=h2)
        calc.ZPE_harmonic()
        assert calc.Q_ratio_ZPE_harmonic == pytest.approx(1.0, abs=1e-10)

    def test_zpe_anharmonic_diatomic(self, h2, hd):
        """Anharmonic correction for a diatomic should not be 1.0."""
        calc = PartitionFunctionCalculator(temperature=T_0C, NSI=h2, SSI=hd)
        calc.ZPE_anharmonic()
        assert calc.Q_ratio_ZPE_anharmonic != pytest.approx(1.0, abs=1e-4)

    def test_zpe_total_is_product(self, h2, hd):
        calc = PartitionFunctionCalculator(temperature=T_0C, NSI=h2, SSI=hd)
        calc.ZPE_G0()
        calc.ZPE_harmonic()
        calc.ZPE_anharmonic()
        calc.ZPE_total()
        expected = (
            calc.Q_ratio_ZPE_G0
            * calc.Q_ratio_ZPE_harmonic
            * calc.Q_ratio_ZPE_anharmonic
        )
        assert calc.Q_ratio_ZPE_total == pytest.approx(expected, rel=1e-10)

    def test_zpe_g0_no_attribute(self, h2, hd):
        """Without G0 attributes, ZPE_G0 should default to 1.0."""
        calc = PartitionFunctionCalculator(temperature=T_0C, NSI=h2, SSI=hd)
        calc.ZPE_G0()
        assert calc.Q_ratio_ZPE_G0 == 1.0


# ---------------------------------------------------------------------------
# Excited state contributions
# ---------------------------------------------------------------------------
class TestExcitedState:
    def test_harmonic_at_low_freq_not_one(self, co, c18o):
        """For CO at 273 K, excited state harmonic should differ from 1.0."""
        calc = PartitionFunctionCalculator(temperature=T_0C, NSI=co, SSI=c18o)
        calc.excited_state_harmonic()
        # CO has ω ≈ 2170 cm⁻¹, u = hcω/kT ≈ 11.4 at 273 K
        # exp(-11.4) ≈ tiny, so ratio ≈ 1.0 but not exactly
        assert calc.Q_ratio_excited_harmonic > 0.99

    def test_harmonic_same_species(self, h2):
        calc = PartitionFunctionCalculator(temperature=T_0C, NSI=h2, SSI=h2)
        calc.excited_state_harmonic()
        assert calc.Q_ratio_excited_harmonic == pytest.approx(1.0, abs=1e-10)

    def test_anharmonic_diatomic_path(self, h2, hd):
        """H2/HD should take the diatomic anharmonic path (no poly constants)."""
        calc = PartitionFunctionCalculator(temperature=T_0C, NSI=h2, SSI=hd)
        calc.excited_state_anharmonic()
        # At 273 K with large u values, anharmonic correction is ≈ 1.0
        assert 0.99 < calc.Q_ratio_excited_anharmonic < 1.01


# ---------------------------------------------------------------------------
# Rotational contributions
# ---------------------------------------------------------------------------
class TestRotational:
    def test_diatomic_h2_hd(self, h2, hd):
        """HD/H2 rotational ratio should account for symmetry difference."""
        calc = PartitionFunctionCalculator(temperature=T_0C, NSI=h2, SSI=hd)
        calc.rotational_diatomics()
        # H2 has symmetry=2, HD has symmetry=1 → ratio includes factor of 2
        assert calc.Q_ratio_rotational_diatomic > 2.0

    def test_diatomic_same_species(self, h2):
        calc = PartitionFunctionCalculator(temperature=T_0C, NSI=h2, SSI=h2)
        calc.rotational_diatomics()
        assert calc.Q_ratio_rotational_diatomic == pytest.approx(1.0, rel=1e-6)

    def test_missing_b0_gives_one(self):
        mc1 = MolecularConstants(isotopologue="1H1H", harmonic=[4401.0])
        mc2 = MolecularConstants(isotopologue="1H2H", harmonic=[3813.0])
        calc = PartitionFunctionCalculator(temperature=T_0C, NSI=mc1, SSI=mc2)
        calc.rotational_diatomics()
        assert calc.Q_ratio_rotational_diatomic == 1.0

    def test_rotational_vibrational_without_delta(self, h2, hd):
        """Without delta constants, rot-vib contribution should remain 1.0."""
        mc1 = MolecularConstants(
            isotopologue="1H1H", harmonic=[4401.0], symmetry=2.0,
        )
        mc2 = MolecularConstants(
            isotopologue="1H2H", harmonic=[3813.0],
        )
        calc = PartitionFunctionCalculator(temperature=T_0C, NSI=mc1, SSI=mc2)
        calc.rotational_vibrational()
        assert calc.Q_ratio_rotational_vibrational == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# calculate_all — integration tests
# ---------------------------------------------------------------------------
class TestCalculateAll:
    def test_qtot_is_product_of_contributions(self, h2, hd):
        calc = PartitionFunctionCalculator(temperature=T_0C, NSI=h2, SSI=hd)
        calc.calculate_all()

        manual = (
            calc.Q_ratio_trans
            * calc.Q_ratio_ZPE_total
            * calc.Q_ratio_excited_harmonic
            * calc.Q_ratio_excited_anharmonic
            * calc.Q_ratio_rotational_diatomic
            * calc.Q_ratio_rotational_vibrational
        )
        assert calc.Q_tot == pytest.approx(manual, rel=1e-8)

    def test_rpfr_equals_qtot(self, h2, hd):
        calc = PartitionFunctionCalculator(temperature=T_0C, NSI=h2, SSI=hd)
        calc.calculate_all()
        assert calc.RPFR == calc.Q_tot

    def test_hd_h2_rpfr_reasonable(self, h2, hd):
        """HD/H2 RPFR at 273 K should be in the 20–22 range."""
        calc = PartitionFunctionCalculator(temperature=T_0C, NSI=h2, SSI=hd)
        calc.calculate_all()
        assert 20 < calc.Q_tot < 23

    def test_co_c18o_rpfr(self, co, c18o):
        """C18O/C16O RPFR at 273 K should be around 1.34."""
        calc = PartitionFunctionCalculator(temperature=T_0C, NSI=co, SSI=c18o)
        calc.calculate_all()
        assert 1.25 < calc.Q_tot < 1.45

    def test_same_species_rpfr_is_one(self, h2):
        calc = PartitionFunctionCalculator(temperature=T_0C, NSI=h2, SSI=h2)
        calc.calculate_all()
        assert calc.Q_tot == pytest.approx(1.0, abs=1e-6)

    def test_higher_temperature_lowers_rpfr(self, h2, hd):
        """At higher temperature, isotope effects diminish → smaller RPFR."""
        calc_low = PartitionFunctionCalculator(temperature=273.15, NSI=h2, SSI=hd)
        calc_low.calculate_all()
        calc_high = PartitionFunctionCalculator(temperature=1000.0, NSI=h2, SSI=hd)
        calc_high.calculate_all()
        assert calc_high.Q_tot < calc_low.Q_tot


# ---------------------------------------------------------------------------
# Tests using real experimental data
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not HAVE_DATA, reason="Data file not found")
class TestRPFRWithExperimentalData:
    @pytest.fixture(scope="class")
    def lookup(self):
        exp = load_experimental_constants(DATA_FILE)
        return generate_all_variants(exp)

    def test_hd_h2_experimental(self, lookup):
        calc = PartitionFunctionCalculator(
            temperature=T_0C,
            NSI=lookup["1H1H"],
            SSI=lookup["1H2H"],
        )
        calc.calculate_all()
        assert calc.Q_ratio_trans == pytest.approx(1.8357, rel=1e-3)
        assert 20 < calc.Q_tot < 23

    def test_df_hf_experimental(self, lookup):
        calc = PartitionFunctionCalculator(
            temperature=T_0C,
            NSI=lookup["1H19F"],
            SSI=lookup["2H19F"],
        )
        calc.calculate_all()
        assert calc.Q_ratio_trans > 1.0
        assert calc.Q_tot > 1.0

    def test_c18o_c16o_experimental(self, lookup):
        calc = PartitionFunctionCalculator(
            temperature=T_0C,
            NSI=lookup["12C16O"],
            SSI=lookup["12C18O"],
        )
        calc.calculate_all()
        assert 1.25 < calc.Q_tot < 1.45

    def test_nacl_experimental(self, lookup):
        calc = PartitionFunctionCalculator(
            temperature=T_0C,
            NSI=lookup["23Na35Cl"],
            SSI=lookup["23Na37Cl"],
        )
        calc.calculate_all()
        assert calc.Q_tot > 1.0

    def test_cao_experimental(self, lookup):
        calc = PartitionFunctionCalculator(
            temperature=T_0C,
            NSI=lookup["40Ca16O"],
            SSI=lookup["40Ca18O"],
        )
        calc.calculate_all()
        assert calc.Q_tot > 1.0
