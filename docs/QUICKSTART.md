# Quick Start Guide - Hawaii Tax Estimation

## Basic Usage

### 1. Apply IRS SOI Calibration (Recommended)

The simplest way to get accurate tax unit weights:

```python
import pandas as pd
from src.tax.validation.irs_soi_calibration import apply_irs_soi_calibration

# Load your tax units
tax_units = pd.read_parquet('data/processed/tax_units.parquet')

# Apply IPF calibration (automatic, <0.1% error)
tax_units_calibrated = apply_irs_soi_calibration(tax_units)

# Use calibrated weights for analysis
total_returns = tax_units_calibrated['weight_irs_calibrated'].sum()
print(f"Total returns: {total_returns:,.0f}")
```

### 2. Separate Capital Gains Income

```python
from src.tax.income import apply_capital_gains_separation

# Separate income into regular income and capital gains
tax_units_calibrated = apply_capital_gains_separation(
    tax_units_calibrated,
    agi_col='agi',
    income_col='income'
)

# Now you have:
# - regular_income (wages, business, etc.)
# - capital_gains_income (preferential tax treatment)
```

### 3. Calculate Hawaii State Taxes

```python
from src.tax.hawaii_calculator import HawaiiTaxCalculator

# Initialize calculator
calculator = HawaiiTaxCalculator()

# Calculate taxes for all tax units
tax_units_with_taxes = calculator.calculate_tax_units_batch(
    tax_units_calibrated,
    year=2024
)

# Analyze results
total_revenue = (
    tax_units_with_taxes['hi_tax_tax_liability'] * 
    tax_units_with_taxes['weight_irs_calibrated']
).sum()

print(f"Total state revenue: ${total_revenue/1e9:.2f}B")
```

### 4. Validate Results

```python
from src.tax.validation.irs_soi_calibration import validate_irs_soi_calibration

# Check calibration accuracy
validation = validate_irs_soi_calibration(
    tax_units_calibrated,
    weight_col='weight_irs_calibrated'
)

# View results
print(validation)
```

## Complete Pipeline Example

```python
import pandas as pd
from src.tax.validation.irs_soi_calibration import apply_irs_soi_calibration
from src.tax.income import apply_capital_gains_separation
from src.tax.hawaii_calculator import HawaiiTaxCalculator

# Step 1: Load tax units
tax_units = pd.read_parquet('data/processed/tax_units.parquet')
print(f"Loaded {len(tax_units):,} tax units")

# Step 2: Apply IPF calibration
tax_units = apply_irs_soi_calibration(tax_units)
print(f"Total returns: {tax_units['weight_irs_calibrated'].sum():,.0f}")

# Step 3: Separate capital gains from regular income
tax_units = apply_capital_gains_separation(tax_units)
print(f"Capital gains: {(tax_units['capital_gains_income'] * tax_units['weight_irs_calibrated']).sum()/1e9:.2f}B")

# Step 4: Calculate taxes
calculator = HawaiiTaxCalculator()
tax_units = calculator.calculate_tax_units_batch(tax_units, year=2024)

# Step 5: Analyze by filing status
analysis = tax_units.groupby('filing_status').agg({
    'weight_irs_calibrated': 'sum',
    'hi_tax_tax_liability': lambda x: (x * tax_units.loc[x.index, 'weight_irs_calibrated']).sum()
}).round(0)

analysis.columns = ['Returns', 'Total Tax']
analysis['Avg Tax'] = analysis['Total Tax'] / analysis['Returns']

print("\nAnalysis by Filing Status:")
print(analysis)

# Step 6: Save results
tax_units.to_parquet('data/processed/tax_units_final.parquet')
print("\n✅ Complete! Results saved.")
```

## Key Features

### IPF Calibration (Default)
- **Accuracy**: <0.1% error on filing status and AGI brackets
- **Method**: Iterative Proportional Fitting (industry standard)
- **Speed**: Converges in ~30-50 iterations
- **Automatic**: No manual tuning required

### Capital Gains Separation
- **Data Source**: DOTAX SOI 2022 Table 21
- **Method**: AGI bracket-specific percentages
- **Output**: Separate regular_income and capital_gains_income columns
- **Use Case**: Apply preferential tax rates to capital gains

### Hawaii Tax Calculator
- **Years**: 2017-2031 tax brackets supported
- **Filing Statuses**: Single, Joint, HoH, MFS
- **Features**: Progressive brackets, standard deductions
- **Output**: Tax liability, effective rate, marginal rate

### Validation Tools
- Compare to DOTAX SOI benchmarks
- Check filing status distribution
- Verify AGI bracket accuracy
- Generate detailed reports

## Common Tasks

### Analyze by Income Bracket

```python
# Create income brackets
def income_bracket(income):
    if income < 25000: return '$0-25k'
    elif income < 50000: return '$25-50k'
    elif income < 75000: return '$50-75k'
    elif income < 100000: return '$75-100k'
    elif income < 200000: return '$100-200k'
    else: return '$200k+'

tax_units['income_bracket'] = tax_units['income'].apply(income_bracket)

# Aggregate by bracket
bracket_analysis = tax_units.groupby('income_bracket').agg({
    'weight_irs_calibrated': 'sum',
    'hi_tax_tax_liability': lambda x: (x * tax_units.loc[x.index, 'weight_irs_calibrated']).sum()
})

print(bracket_analysis)
```

### Compare Tax Years

```python
# Calculate for multiple years
for year in [2024, 2026, 2028]:
    tax_units[f'tax_{year}'] = calculator.calculate_tax_units_batch(
        tax_units, year=year
    )['hi_tax_tax_liability']

# Compare
tax_units['change_2024_2028'] = tax_units['tax_2028'] - tax_units['tax_2024']
avg_change = (tax_units['change_2024_2028'] * tax_units['weight_irs_calibrated']).sum() / tax_units['weight_irs_calibrated'].sum()

print(f"Average tax change (2024→2028): ${avg_change:,.2f}")
```

### Export for Analysis

```python
# Select key columns
export_df = tax_units[[
    'filing_status',
    'income',
    'num_dependents',
    'hi_tax_tax_liability',
    'hi_tax_effective_rate',
    'weight_irs_calibrated'
]].copy()

# Save to CSV for Excel/Tableau
export_df.to_csv('data/processed/tax_analysis.csv', index=False)

# Or save to Parquet for Python/R
export_df.to_parquet('data/processed/tax_analysis.parquet')
```

## Troubleshooting

### Missing 'agi' or 'income' column
The calibration needs an income column. It will automatically use:
1. `agi` (if present)
2. `total_income` (if present)
3. `income` (if present)

### Calibration not converging
The IPF algorithm may not fully converge in 50 iterations, but typically achieves <0.1% error. To adjust:

```python
# Not currently exposed, but can be modified in source if needed
# Default: max_iterations=50, tolerance=0.0001
```

### Performance issues
For large datasets (>100k records), consider:
- Using batch processing
- Filtering to relevant subset first
- Running on more powerful hardware

## Next Steps

- See `docs/IPF_CALIBRATION_UPDATE.md` for technical details
- See `CALIBRATION_OPTIONS.md` for methodology comparison
- See `docs/TAX_CALCULATION_GUIDE.md` for tax calculation details
- Run `scripts/test_ipf_pipeline.py` for a complete example

## Support

For questions or issues:
1. Check existing documentation in `docs/`
2. Review example scripts in `scripts/`
3. Examine test scripts for working examples
