#!/usr/bin/env python
"""
Example workflow: load Table 5 molecular constants from the Richet spreadsheet.
"""

from pathlib import Path

from richet_rpfr import (
    generate_isotopic_variants,
    is_most_common_isotopologue,
    load_molecular_constants_from_excel,
)


def main() -> None:
    excel_path = Path("data/raw/excel/Richet - RPFR & mol constans.xlsx")
    constants = load_molecular_constants_from_excel(
        excel_path, sheet_name="Diatoms updated (Table 5.2)"
    )
    print(f"Loaded {len(constants)} isotopologues from {excel_path.name}")

    # Demonstrate generating isotopic variants for the most abundant species.
    variants = []
    for name, mc in constants.items():
        if is_most_common_isotopologue(name):
            variants.extend(generate_isotopic_variants(mc))
    print(f"Generated {len(variants)} isotopic variants.")


if __name__ == "__main__":
    main()
