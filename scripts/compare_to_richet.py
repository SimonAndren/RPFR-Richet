#!/usr/bin/env python
"""
Example workflow: recreate Richet Table 5 contributions at 0 °C.
"""

from pathlib import Path

import pandas as pd

from richet_rpfr import (
    compute_and_finalize_with_temp,
    create_summary_df,
    load_molecular_constants_from_excel,
)


def main() -> None:
    excel_path = Path("data/raw/excel/Richet - RPFR & mol constans.xlsx")
    constants = load_molecular_constants_from_excel(
        excel_path, sheet_name="Diatoms updated (Table 5.2)"
    )

    NSI_H2 = constants["1H1H"]
    SSI_H2 = constants["1H2H"]
    NSI_HF = constants["1H19F"]
    SSI_HF = constants["2H19F"]
    NSI_CO = constants["12C16O"]
    SSI_CO = constants["12C18O"]

    T_0C = 273.15

    paper_HD_H2 = {
        "Translational": 1.83561,
        "ZPE_G0": 1.00910,
        "ZPE_harmonic": 4.71546,
        "ZPE_anharmonic": 0.96079,
        "ZPE_total": 4.57179,
        "Rotational Diatomic": 2.59246,
        "Total": 21.75600,
    }
    paper_DF_HF = {
        "Translational": 1.07638,
        "ZPE_G0": 1.01192,
        "ZPE_harmonic": 20.1428,
        "ZPE_anharmonic": 0.94356,
        "ZPE_total": 19.2325,
        "Rotational Diatomic": 1.86165,
        "Total": 38.5389,
    }
    paper_C18O_C16O = {
        "Translational": 1.10929,
        "ZPE_G0": 1.00005,
        "ZPE_harmonic": 1.14804,
        "ZPE_anharmonic": 0.99917,
        "ZPE_total": 1.14714,
        "Rotational Diatomic": 1.04985,
        "Total": 1.33595,
    }

    final_H2 = compute_and_finalize_with_temp(T_0C, NSI_H2, SSI_H2, paper_HD_H2)
    final_HF = compute_and_finalize_with_temp(T_0C, NSI_HF, SSI_HF, paper_DF_HF)
    final_CO = compute_and_finalize_with_temp(T_0C, NSI_CO, SSI_CO, paper_C18O_C16O)

    summary = create_summary_df(final_H2, paper_HD_H2, final_HF, paper_DF_HF, final_CO, paper_C18O_C16O)
    pd.set_option("display.width", 0)
    print(summary)


if __name__ == "__main__":
    main()
