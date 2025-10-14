# Pipeline Scripts

Core production pipeline scripts for tax unit construction and calibration.

## Execution Order

1. **01_construct_tax_units.py** - Construct tax units from PUMS data
2. **02_apply_soi_calibration.py** - Apply SOI weight calibration
3. **03_validate_results.py** - Validate results against benchmarks

## Usage

Run the full pipeline:
```bash
python pipeline/01_construct_tax_units.py
python pipeline/02_apply_soi_calibration.py
python pipeline/03_validate_results.py
```
