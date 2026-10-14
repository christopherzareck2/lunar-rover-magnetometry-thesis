# Regolith study (thesis Section 5.5 and Table 5.4)

Question: the thesis models everything below the ground as non-magnetic. If the regolith around an outcrop carries
an induced magnetization, the 4 x 4 m hole that the outcrop makes in it acts as a buried plug of magnetization
-M_reg along the site field. How much direction error does that add to the recovered outcrop magnetization, and
which fits or measurements remove it?

| File | Purpose |
|---|---|
| `y10_verify.py` | Checks that a regolith slab of +-1.2 km with a hole equals the plug formulation, and that the prism kernel agrees with an independent point-dipole sum. Writes `y10_verify.json` |
| `y10_regolith.py` | The study. Regolith thickness 0.5 to 5 m, susceptibility 0.01 to 0.04, site field 100 and 300 nT, three hole modes, five mitigations, 100 draws per case. Writes `y10_results.json` (a rerun with `Y10_SEEDTAG` set gave `y10_results_seed2.json`, the second-seed check). Settings come from `Y10_*` environment variables, read at the top of the script |
| `y10_scan.json` | Finer scan over susceptibility of the noise-free bias, same script and settings otherwise |
| `y10_report_tables.py`, `y10_tables.py` | Build the tables from the JSON files. `y10_report_tables.py` writes `y10_tables.md` and `y10_mitigation_table.csv`. Both read the JSON in the current directory, so run them from this folder |

```bash
conda activate lunar-mag
cd studies/regolith
nice python y10_verify.py        # a minute or two
nice python y10_regolith.py      # about 2 minutes
python y10_report_tables.py
```

The study imports `analysis/a09_systematics.py` and `inversion_helpers` read-only. Susceptibility and bulk-density
inputs are assumptions taken from the literature; the bulk-density citation has not been checked against its
source. The thesis (Section 5.5, Table 5.4) states the results and their limits.
