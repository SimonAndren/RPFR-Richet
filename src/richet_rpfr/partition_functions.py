"""
Partition function contributions and RPFR calculator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy import constants as sc

from . import isotopes


@dataclass
class PartitionFunctionCalculator:
    """
    Encapsulate the contribution-by-contribution evaluation of the reduced partition
    function ratio Q'/Q between a substituted (SSI) and nominal (NSI) isotopologue.
    """

    temperature: float
    NSI: object
    SSI: object
    beta_scale: float = 1.0

    Q_ratio_trans: float = 1.0
    Q_ratio_ZPE_G0: float = 1.0
    Q_ratio_ZPE_harmonic: float = 1.0
    Q_ratio_ZPE_anharmonic: float = 1.0
    Q_ratio_ZPE_total: float = 1.0
    Q_ratio_excited_harmonic: float = 1.0
    Q_ratio_excited_anharmonic: float = 1.0
    Q_ratio_rotational_linear: float = 1.0
    Q_ratio_rotational_linear_w_corr: float = 1.0
    Q_ratio_rotational_nonlinear_corr: float = 1.0
    Q_ratio_rotational_diatomic: float = 1.0
    Q_ratio_rotational_vibrational: float = 1.0
    RPFR: float = 1.0
    Q_tot: float = 1.0
    missing_data_notes: List[str] = field(default_factory=list)

    def translational(self) -> None:
        self.Q_ratio_trans = (self._ssi_mass / self._nsi_mass) ** (3.0 / 2.0)

    def ZPE_G0(self) -> None:
        G0_NSI = getattr(self.NSI, "G0", None)
        G0_SSI = getattr(self.SSI, "G0", None)
        if G0_NSI is None or G0_SSI is None:
            self.Q_ratio_ZPE_G0 = 1.0
            return
        energy_diff = (G0_SSI - G0_NSI) * (sc.h * sc.c * 100)
        self.Q_ratio_ZPE_G0 = np.exp(-energy_diff / (sc.k * self.temperature))

    def ZPE_harmonic(self) -> None:
        NSI = np.array(self.NSI.harmonic, dtype=float)
        SSI = np.array(self.SSI.harmonic, dtype=float)
        NSI_ZPE = 0.5 * sc.h * sc.c * 100 * NSI.sum() * self.beta_scale
        SSI_ZPE = 0.5 * sc.h * sc.c * 100 * SSI.sum() * self.beta_scale
        energy_diff = SSI_ZPE - NSI_ZPE
        self.Q_ratio_ZPE_harmonic = np.exp(-energy_diff / (sc.k * self.temperature))

    def ZPE_anharmonic(self) -> None:
        clusters = isotopes.degeneracy_clusters(self.NSI.harmonic, tolerance=1.0)
        degeneracies = [len(cluster) for cluster in clusters]
        n_modes = len(degeneracies)

        if n_modes == 0:
            self.Q_ratio_ZPE_anharmonic = 1.0
            return

        if n_modes == 1:
            factor = (sc.h * sc.c * 100) / (4.0 * sc.k * self.temperature)
            self.Q_ratio_ZPE_anharmonic = np.exp(
                factor * (self.SSI.anharmonic_dia[0] - self.NSI.anharmonic_dia[0])
            )
            return

        d = np.array(degeneracies, dtype=float)
        X_NSI = self._build_anharmonic_matrix(self.NSI.anharmonic_poly, n_modes)
        X_SSI = self._build_anharmonic_matrix(self.SSI.anharmonic_poly, n_modes)
        anh_sum = float(
            np.sum(
                [
                    d[i] * d[j] * (X_SSI[i, j] - X_NSI[i, j])
                    for i in range(n_modes)
                    for j in range(n_modes)
                ]
            )
        )
        factor = (sc.h * sc.c * 100) / (4.0 * sc.k * self.temperature)
        self.Q_ratio_ZPE_anharmonic = np.exp(-factor * anh_sum)

    def ZPE_total(self) -> None:
        self.Q_ratio_ZPE_total = (
            self.Q_ratio_ZPE_G0 * self.Q_ratio_ZPE_harmonic * self.Q_ratio_ZPE_anharmonic
        )

    def excited_state_harmonic(self) -> None:
        beta = (sc.h * sc.c * 100) / (sc.k * self.temperature)
        NSI_ui = beta * np.array(self.NSI.harmonic, dtype=float)
        SSI_ui = beta * np.array(self.SSI.harmonic, dtype=float)
        NSI_product = np.prod(1 - np.exp(-NSI_ui))
        SSI_product = np.prod(1 - np.exp(-SSI_ui))
        if NSI_product == 0 or SSI_product == 0:
            self.Q_ratio_excited_harmonic = 1.0
        else:
            self.Q_ratio_excited_harmonic = NSI_product / SSI_product

    def excited_state_anharmonic(self) -> None:
        if len(self.NSI.anharmonic_poly) == 0:
            self.Q_ratio_excited_anharmonic = self._excited_state_anharmonic_diatomic()
        else:
            self.Q_ratio_excited_anharmonic = self._excited_state_anharmonic_poly()

    def rotational(self) -> None:
        if len(self.NSI.harmonic) == 1:
            self.rotational_diatomics()
            self.rotational_linear_corr()
        elif len(self.NSI.harmonic) % 3 == 0:
            self.rotational_nonlinear_corr()
        else:
            self.rotational_linear_corr()

    def rotational_diatomics(self) -> None:
        beta = sc.h * sc.c * 100 / (sc.k * self.temperature)
        NSI_B = self._get_rotational_constant(self.NSI, "B0 (cm-1)", "rotational diatomic contribution")
        SSI_B = self._get_rotational_constant(self.SSI, "B0 (cm-1)", "rotational diatomic contribution")
        if NSI_B == 0.0 or SSI_B == 0.0:
            self.Q_ratio_rotational_diatomic = 1.0
            return
        symmetry_NSI = getattr(self.NSI, "symmetry", None) or 1
        symmetry_SSI = getattr(self.SSI, "symmetry", None) or 1
        Jmax = 200
        Q_rot_NSI = sum((2 * j + 1) * np.exp(-NSI_B * j * (j + 1) * beta) for j in range(Jmax + 1))
        Q_rot_SSI = sum((2 * j + 1) * np.exp(-SSI_B * j * (j + 1) * beta) for j in range(Jmax + 1))
        self.Q_ratio_rotational_diatomic = (
            Q_rot_SSI * symmetry_NSI / (Q_rot_NSI * symmetry_SSI)
        )

    def rotational_linear_corr(self) -> None:
        sigma_NSI = self._linear_sigma(self.NSI)
        sigma_SSI = self._linear_sigma(self.SSI)

        if sigma_NSI is None or sigma_SSI is None:
            self.Q_ratio_rotational_linear = 1.0
            self.Q_ratio_rotational_linear_w_corr = 1.0
            return

        correction_NSI = self._linear_correction(sigma_NSI)
        correction_SSI = self._linear_correction(sigma_SSI)

        symmetry_ratio = (self.NSI.symmetry or 1) / (self.SSI.symmetry or 1)
        sigma_ratio = sigma_NSI / sigma_SSI
        self.Q_ratio_rotational_linear = symmetry_ratio * sigma_ratio
        self.Q_ratio_rotational_linear_w_corr = (
            self.Q_ratio_rotational_linear * (correction_SSI / correction_NSI)
        )

    def rotational_nonlinear_corr(self) -> None:
        symmetry_ratio = (self.NSI.symmetry or 1) / (self.SSI.symmetry or 1)

        inertia_NSI = self._principal_inertia(self.NSI)
        inertia_SSI = self._principal_inertia(self.SSI)
        if inertia_NSI is None or inertia_SSI is None:
            self.Q_ratio_rotational_nonlinear_corr = 1.0
            return

        ratio_base = symmetry_ratio * np.sqrt(
            (inertia_SSI[0] * inertia_SSI[1] * inertia_SSI[2])
            / (inertia_NSI[0] * inertia_NSI[1] * inertia_NSI[2])
        )

        sigma_NSI = [self._nonlinear_sigma(I) for I in inertia_NSI]
        sigma_SSI = [self._nonlinear_sigma(I) for I in inertia_SSI]

        correction_NSI = self._nonlinear_correction(*sigma_NSI)
        correction_SSI = self._nonlinear_correction(*sigma_SSI)

        self.Q_ratio_rotational_nonlinear_corr = ratio_base * (correction_SSI / correction_NSI)

    def rotational_vibrational(self) -> None:
        degeneracies_NSI, modes_NSI = self._mode_degeneracies(self.NSI.harmonic)
        degeneracies_SSI, modes_SSI = self._mode_degeneracies(self.SSI.harmonic)

        context = "rotational-vibrational contribution"

        if not degeneracies_NSI:
            self._record_missing(self.NSI, "vibrational mode data", context)
        if not degeneracies_SSI:
            self._record_missing(self.SSI, "vibrational mode data", context)
        if not degeneracies_NSI or not degeneracies_SSI:
            self.Q_ratio_rotational_vibrational = 1.0
            return

        freqs_NSI = [float(np.mean(m)) if len(m) else 0.0 for m in modes_NSI]
        freqs_SSI = [float(np.mean(m)) if len(m) else 0.0 for m in modes_SSI]

        max_len = max(len(degeneracies_NSI), len(degeneracies_SSI))
        degeneracies_NSI = self._pad_sequence(
            degeneracies_NSI, max_len, self.NSI, "vibrational degeneracies", context
        )
        degeneracies_SSI = self._pad_sequence(
            degeneracies_SSI, max_len, self.SSI, "vibrational degeneracies", context
        )
        freqs_NSI = self._pad_sequence(freqs_NSI, max_len, self.NSI, "vibrational frequencies", context)
        freqs_SSI = self._pad_sequence(freqs_SSI, max_len, self.SSI, "vibrational frequencies", context)

        NSI_ui = self._compute_u_values(freqs_NSI)
        SSI_ui = self._compute_u_values(freqs_SSI)

        NSI_delta = self._pad_sequence(
            self.NSI.delta[:max_len], max_len, self.NSI, "delta constants", context
        )
        SSI_delta = self._pad_sequence(
            self.SSI.delta[:max_len], max_len, self.SSI, "delta constants", context
        )

        def contribution(degeneracies, deltas, ui):
            result = 1.0
            for d, delta_i, u in zip(degeneracies, deltas, ui):
                denom = np.exp(u) - 1.0
                if denom == 0:
                    continue
                factor = 1 + 0.5 * (delta_i / denom)
                result *= factor**d
            return result

        Q_rot_vib_NSI = contribution(degeneracies_NSI, NSI_delta, NSI_ui)
        Q_rot_vib_SSI = contribution(degeneracies_SSI, SSI_delta, SSI_ui)
        if Q_rot_vib_NSI == 0:
            raise ValueError("Q_rot_vib for NSI is zero; cannot compute ratio.")
        self.Q_ratio_rotational_vibrational = Q_rot_vib_SSI / Q_rot_vib_NSI

    def calculate_all(self) -> None:
        self.missing_data_notes.clear()
        self.translational()
        self.ZPE_G0()
        self.ZPE_harmonic()
        self.ZPE_anharmonic()
        self.ZPE_total()
        self.excited_state_harmonic()
        self.excited_state_anharmonic()
        self.rotational()
        self.rotational_vibrational()

        cumulative = [
            self.Q_ratio_trans,
            self.Q_ratio_ZPE_total,
            self.Q_ratio_excited_harmonic,
            self.Q_ratio_excited_anharmonic,
            self.Q_ratio_rotational_vibrational,
        ]
        if self.Q_ratio_rotational_diatomic != 1.0:
            cumulative.append(self.Q_ratio_rotational_diatomic)
        elif self.Q_ratio_rotational_linear_w_corr != 1.0:
            cumulative.append(self.Q_ratio_rotational_linear_w_corr)
        elif self.Q_ratio_rotational_nonlinear_corr != 1.0:
            cumulative.append(self.Q_ratio_rotational_nonlinear_corr)
        self.Q_tot = float(np.prod(cumulative))
        self.RPFR = self.Q_tot

    def _record_missing(self, species, constant: str, context: str) -> None:
        label = getattr(species, "isotopologue", repr(species))
        note = (
            f"{label}: missing {constant} for {context}; assumed 0.0 and contribution set to 1.0."
        )
        if note not in self.missing_data_notes:
            self.missing_data_notes.append(note)

    def _get_rotational_constant(self, species, key: str, context: str) -> float:
        value = species.rotational_constants.get(key)
        if value is None:
            self._record_missing(species, key, context)
            return 0.0
        value = float(value)
        if np.isnan(value):
            self._record_missing(species, key, context)
            return 0.0
        return value

    def _pad_sequence(
        self,
        values: Sequence[float],
        target_len: int,
        species,
        label: str,
        context: str,
        fill_value: float = 0.0,
    ) -> List[float]:
        padded = list(values)
        if len(padded) < target_len:
            self._record_missing(species, label, context)
            padded.extend([fill_value] * (target_len - len(padded)))
        return padded

    def print_contribution_table(self) -> None:
        """
        Print a table of the individual contribution ratios currently stored on this instance.
        Call `calculate_all` beforehand to ensure the values reflect the latest inputs.
        """
        entries = [
            ("Translational", self.Q_ratio_trans),
            ("ZPE (G0)", self.Q_ratio_ZPE_G0),
            ("ZPE (harmonic)", self.Q_ratio_ZPE_harmonic),
            ("ZPE (anharmonic)", self.Q_ratio_ZPE_anharmonic),
            ("ZPE (total)", self.Q_ratio_ZPE_total),
            ("Excited (harmonic)", self.Q_ratio_excited_harmonic),
            ("Excited (anharmonic)", self.Q_ratio_excited_anharmonic),
            ("Rotational (linear)", self.Q_ratio_rotational_linear),
            ("Rotational (linear, corr.)", self.Q_ratio_rotational_linear_w_corr),
            ("Rotational (nonlinear, corr.)", self.Q_ratio_rotational_nonlinear_corr),
            ("Rotational (diatomic)", self.Q_ratio_rotational_diatomic),
            ("Rotational-vibrational", self.Q_ratio_rotational_vibrational),
            ("Q total", self.Q_tot),
            ("RPFR", self.RPFR),
        ]

        label_width = max(len(label) for label, _ in entries)
        value_header = "Value"
        header = f"{'Contribution'.ljust(label_width)} | {value_header}"
        separator = f"{'-' * label_width}-+-{'-' * len(value_header)}"
        print(header)
        print(separator)
        for label, value in entries:
            value_str = f"{value:.6e}"
            print(f"{label.ljust(label_width)} | {value_str}")
        if self.missing_data_notes:
            print("\nNotes:")
            for note in self.missing_data_notes:
                print(f"- {note}")

    # Helper properties -------------------------------------------------
    @property
    def _nsi_mass(self) -> float:
        mass = getattr(self.NSI, "mass", None)
        if mass is None:
            raise ValueError("Missing NSI mass.")
        return mass

    @property
    def _ssi_mass(self) -> float:
        mass = getattr(self.SSI, "mass", None)
        if mass is None:
            raise ValueError("Missing SSI mass.")
        return mass

    # Helper methods ----------------------------------------------------
    def _build_anharmonic_matrix(self, constants: Sequence[float], n_modes: int) -> np.ndarray:
        expected_len = n_modes * (n_modes + 1) // 2
        if len(constants) < expected_len:
            raise ValueError(
                f"Insufficient anharmonic constants. Expected {expected_len}, got {len(constants)}."
            )
        array = np.zeros((n_modes, n_modes))
        index = 0
        for i in range(n_modes):
            for j in range(i, n_modes):
                value = constants[index]
                array[i, j] = value
                array[j, i] = value
                index += 1
        return array

    def _excited_state_anharmonic_diatomic(self) -> float:
        beta = (sc.h * sc.c * 100) / (sc.k * self.temperature)
        NSI_freq = self.NSI.harmonic[0]
        SSI_freq = self.SSI.harmonic[0]
        x_NSI = self.NSI.anharmonic_dia[0]
        x_SSI = self.SSI.anharmonic_dia[0]

        def contribution(freq, x):
            u = beta * freq
            x_term = beta * x
            exp_neg_u = np.exp(-u)
            denom = (1 - exp_neg_u) ** 2
            if denom == 0:
                return 1.0
            return 1 - 2 * x_term * exp_neg_u / denom

        return contribution(SSI_freq, x_SSI) / contribution(NSI_freq, x_NSI)

    def _excited_state_anharmonic_poly(self) -> float:
        degeneracies, _ = self._mode_degeneracies(self.NSI.harmonic)
        if not degeneracies:
            return 1.0
        beta = (sc.h * sc.c * 100) / (sc.k * self.temperature)
        NSI_freqs = np.array(self.NSI.harmonic)
        SSI_freqs = np.array(self.SSI.harmonic)
        NSI_freqs = NSI_freqs[~np.isnan(NSI_freqs)]
        SSI_freqs = SSI_freqs[~np.isnan(SSI_freqs)]

        def series(freqs, anharm):
            values = 0.0
            for freq, x in zip(freqs, anharm):
                u = beta * freq
                exp_term = np.exp(-u)
                denom = (1 - exp_term) ** 2
                if denom == 0:
                    continue
                values += 2 * x * exp_term / denom
            return values

        numerator = 1 - series(SSI_freqs, self.SSI.anharmonic_dia)
        denominator = 1 - series(NSI_freqs, self.NSI.anharmonic_dia)
        if denominator == 0:
            return 1.0
        return numerator / denominator

    def _mode_degeneracies(self, frequencies: Sequence[float]) -> Tuple[List[int], List[List[float]]]:
        clusters = isotopes.degeneracy_clusters(frequencies, tolerance=1.0)
        return [len(cluster) for cluster in clusters], clusters

    def _compute_u_values(self, freqs: Iterable[float]) -> np.ndarray:
        beta = (sc.h * sc.c * 100) / (sc.k * self.temperature)
        return beta * np.array(list(freqs))

    def _linear_sigma(self, species) -> float | None:
        B0 = self._get_rotational_constant(
            species, "B0 (cm-1)", "linear rotational contribution"
        )
        if B0 == 0.0:
            return None
        return B0 * sc.h * sc.c * 100 / (sc.k * self.temperature)

    @staticmethod
    def _linear_correction(sigma: float) -> float:
        return 1 + sigma / 3 + sigma**2 / 15 + sigma**3 / 105 + sigma**4 / 945 + sigma**5 / 10395

    def _principal_inertia(self, species) -> Tuple[float, float, float] | None:
        moments = species.moments_of_inertia
        axes = ("I_A", "I_B", "I_C")
        values: List[float] = []
        for axis in axes:
            value = moments.get(axis)
            if value is None or (isinstance(value, float) and np.isnan(value)):
                self._record_missing(species, axis, "nonlinear rotational contribution")
                return None
            values.append(value)
        return tuple(values)

    def _nonlinear_sigma(self, inertia: float) -> float:
        return sc.h**2 / (8 * np.pi**2 * inertia * sc.k * self.temperature)

    def _nonlinear_correction(self, sigma_A: float, sigma_B: float, sigma_C: float) -> float:
        numerator = 2 * (sigma_A + sigma_B + sigma_C) - (
            sigma_A * sigma_B / sigma_C
            + sigma_B * sigma_C / sigma_A
            + sigma_A * sigma_C / sigma_B
        )
        return 1 + numerator / 12


__all__ = ["PartitionFunctionCalculator"]
