# Richet RPFR Replication

This repository reorganises notebooks and helper scripts used to reproduce the reduced partition
function ratios (RPFR) reported by Richet *et al.* (1977). It is structured for collaboration and
ready to be pushed to GitHub.

## Repository layout

```
.
├── data
│   ├── raw
│   │   └── excel/                     # Primary experimental constants (Table 5)
│   └── processed
│       └── spreadsheets/              # Derived isotopologue tables & comparisons
├── notebooks
│   ├── 01_rpfr_h2_example.ipynb
│   ├── 02_table4_replication.ipynb
│   └── 03_rpfr_ccsdt_example.ipynb
├── scripts
│   ├── compare_to_richet.py
│   ├── compute_partition_functions.py
│   └── load_molecular_constants.py
└── src
    └── richet_rpfr/                   # Shareable Python package
        ├── isotopes.py                # Isotope masses & label helpers
        ├── molecular_constants.py     # Excel loaders & scaling utilities
        ├── partition_functions.py     # RPFR contribution calculator
        ├── richet_tables.py           # Richet table replication helpers
        └── comparison.py              # Experimental vs. computed comparisons
```

## How to use

1. **Install dependencies**

   ```bash
   pip install pandas numpy scipy
   ```

2. **Load molecular constants**

   ```python
   from pathlib import Path
   from richet_rpfr import load_molecular_constants_from_excel

   constants = load_molecular_constants_from_excel(
       Path("data/raw/excel/Richet - RPFR & mol constans.xlsx"),
       sheet_name="Diatoms updated (Table 5.2)",
   )
   co = constants["12C16O"]
   ```

3. **Generate isotopic variants**

   ```python
   from richet_rpfr import generate_isotopic_variants, is_most_common_isotopologue

   variants = []
   for name, mc in constants.items():
       if is_most_common_isotopologue(name):
           variants.extend(generate_isotopic_variants(mc))
   ```

4. **Compute RPFRs**

   ```python
   from richet_rpfr import PartitionFunctionCalculator

   nsi = constants["1H1H"]
   ssi = constants["1H2H"]
   calc = PartitionFunctionCalculator(temperature=273.15, NSI=nsi, SSI=ssi)
   calc.calculate_all()
   print(calc.Q_tot)
   ```

5. **Compare against Richet (1977)**

   ```python
   from richet_rpfr import (
       compute_and_finalize_with_temp,
       create_summary_df,
       compare_experimental_and_computed,
   )
   ```

   See `notebooks/02_table4_replication.ipynb` for a worked example that recreates Table 4 and
   benchmarks the experimental data once the `richet_rpfr` package is on the Python path.

## Next steps

- Add tests or scripts under `scripts/` for automated verification.

## Licensing

Add license information here before publishing the repository.
