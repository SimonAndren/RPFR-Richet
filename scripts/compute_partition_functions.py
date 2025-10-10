#!/usr/bin/env python
"""
Example workflow: compute RPFR contributions for the HD/H2 pair.
"""

from pathlib import Path

from richet_rpfr import (
    PartitionFunctionCalculator,
    load_molecular_constants_from_excel,
)


def main() -> None:
    excel_path = Path("data/raw/excel/Richet - RPFR & mol constans.xlsx")
    constants = load_molecular_constants_from_excel(
        excel_path, sheet_name="Diatoms updated (Table 5.2)"
    )
    NSI = constants["1H1H"]
    SSI = constants["1H2H"]

    calculator = PartitionFunctionCalculator(temperature=273.15, NSI=NSI, SSI=SSI)
    calculator.calculate_all()
    print("Q_tot =", calculator.Q_tot)
    print("Breakdown:")
    for attribute in [
        "Q_ratio_trans",
        "Q_ratio_ZPE_total",
        "Q_ratio_excited_harmonic",
        "Q_ratio_excited_anharmonic",
        "Q_ratio_rotational_vibrational",
        "Q_ratio_rotational_diatomic",
    ]:
        print(f"  {attribute}: {getattr(calculator, attribute)}")


if __name__ == "__main__":
    main()
