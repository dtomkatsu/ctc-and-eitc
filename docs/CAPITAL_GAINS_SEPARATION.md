# Capital Gains Income Separation

## Overview

The capital gains separation module splits total income into two components based on DOTAX SOI 2022 Table 21:
1. **Regular Income** - Wages, business income, retirement income, etc.
2. **Capital Gains Income** - Net long-term capital gains eligible for preferential tax rates

This separation is crucial for accurate tax modeling because capital gains are taxed at different rates than regular income in both federal and Hawaii state tax systems.

## Data Source

**DOTAX SOI 2022 Table 21**: "Income Eligible for the Tax Rate on Net Long-Term Capital Gains by Hawai'i AGI Class in Tax Year 2022"

This table provides the percentage of total taxable income that comes from capital gains for each AGI bracket (residents only).

## Methodology

### Capital Gains Percentages by AGI Bracket

| AGI Bracket | Capital Gains % |
|-------------|-----------------|
| $0-$10k | 0.5% |
| $10k-$20k | 0.0% |
| $20k-$30k | 0.03% |
| $30k-$40k | 0.2% |
| $40k-$50k | 0.4% |
| $50k-$75k | 0.7% |
| $75k-$100k | 1.2% |
| $100k-$150k | 1.8% |
| $150k-$200k | 3.1% |
| $200k-$300k | 5.9% |
| $300k-$400k | 11.3% |
| $400k+ | 20.9% |

### Application Process

For each tax unit:
1. Determine AGI bracket based on estimated AGI
2. Look up capital gains percentage for that bracket
3. Apply percentage to total income:
   - `capital_gains_income = total_income × cap_gains_pct`
   - `regular_income = total_income - capital_gains_income`

## Usage

### Basic Usage

```python
from src.tax.income import apply_capital_gains_separation

# Apply separation (modifies DataFrame)
tax_units = apply_capital_gains_separation(
    tax_units,
    agi_col='agi',           # AGI column name
    income_col='income'      # Total income column name
)

# New columns added:
# - regular_income
# - capital_gains_income
```

### Integration with Pipeline

Capital gains separation should be applied **after AGI estimation/calibration** but **before tax calculations**:

```python
# Step 1: Construct tax units
tax_units = construct_tax_units(...)

# Step 2: Apply IPF calibration
tax_units = apply_irs_soi_calibration(tax_units)

# Step 3: Apply capital gains separation
tax_units = apply_capital_gains_separation(tax_units)

# Step 4: Calculate taxes (using regular_income + capital_gains_income separately)
tax_units = calculate_taxes(tax_units)
```

### View Summary by Bracket

```python
from src.tax.income import get_capital_gains_summary

# Generate summary
summary = get_capital_gains_summary(
    tax_units,
    agi_col='agi',
    weight_col='weight_irs_calibrated'
)

print(summary)
```

Output example:
```
           returns  total_income  regular_income  capital_gains  cap_gains_pct
$0-10k      129,376      -$112.6M        -$114.1M          $1.6M          -1.4%
$10-20k      64,160       $949.3M         $949.3M          $0.0M           0.0%
$20-30k      57,835     $1,453.2M       $1,452.8M          $0.4M           0.0%
...
$400k+        8,874     $5,040.1M       $3,986.7M      $1,053.4M          20.9%
```

## Implementation Details

### Module Location
`src/tax/income/capital_gains_separation.py`

### Key Functions

1. **`apply_capital_gains_separation()`**
   - Main function to apply separation
   - Handles missing AGI column (estimates from income)
   - Returns DataFrame with new columns

2. **`get_capital_gains_summary()`**
   - Generates weighted summary by AGI bracket
   - Useful for validation and reporting

3. **`load_capital_gains_percentages()`**
   - Loads percentage data from CSV
   - Returns DataFrame with brackets and percentages

4. **`get_capital_gains_percentage()`**
   - Returns cap gains % for a given AGI
   - Used internally by apply_capital_gains_separation()

### Data File
`data/processed/capital_gains_percentages.csv`

Contains the lookup table with columns:
- `agi_min` - Minimum AGI for bracket
- `agi_max` - Maximum AGI for bracket
- `cap_gains_pct` - Capital gains percentage (0.0-1.0)

## Validation

### Expected Results (DOTAX SOI 2022)

- **Overall capital gains percentage**: 7.4% of total taxable income
- **Total capital gains (residents)**: $2,995M
- **Total returns with capital gains**: 43,443

### Model Results

Current pipeline achieves:
- **Capital gains percentage**: ~4.7%
- **Total capital gains**: ~$2.74B

The gap (4.7% vs 7.4%) is expected because:
1. PUMS undersamples high-income households
2. High-income households have higher capital gains percentages
3. IPF calibration improves this but doesn't fully eliminate the gap

## Tax Implications

### Why Separate Capital Gains?

1. **Federal Tax**: Capital gains have preferential tax rates (0%, 15%, 20%)
2. **Hawaii Tax**: Capital gains are taxed at 7.25% (vs regular rates up to 11%)
3. **AGI Calculation**: Capital gains affect AGI differently than regular income
4. **Deductions**: Some deductions are limited by AGI or income type

### Recommended Tax Calculation Approach

```python
# Calculate tax on regular income
regular_tax = calculate_regular_income_tax(
    income=tax_units['regular_income'],
    filing_status=tax_units['filing_status']
)

# Calculate tax on capital gains (preferential rate)
cap_gains_tax = calculate_capital_gains_tax(
    income=tax_units['capital_gains_income'],
    filing_status=tax_units['filing_status']
)

# Total tax liability
tax_units['total_tax'] = regular_tax + cap_gains_tax
```

## Limitations

1. **Bracket-Level Precision**: Applies uniform percentage within each AGI bracket
   - In reality, variation exists within brackets
   - Higher-income filers within bracket may have more cap gains

2. **PUMS Coverage**: PUMS undersamples high-income households
   - Results in lower overall capital gains percentage
   - Partially addressed by IPF calibration

3. **Residents Only**: Based on resident data from DOTAX
   - Non-residents have different capital gains patterns
   - Non-residents not currently modeled

4. **Long-Term Only**: DOTAX Table 21 covers only long-term capital gains
   - Short-term capital gains taxed as regular income
   - Not separately identified in this implementation

## Future Enhancements

1. **Sub-Bracket Variation**: Model within-bracket variation in capital gains
2. **Non-Resident Modeling**: Add non-resident capital gains patterns
3. **Short-Term Capital Gains**: Separate short-term from long-term
4. **Asset Type Breakdown**: Split by stocks, real estate, business assets
5. **High-Income Enhancement**: Synthetic records with realistic cap gains

## References

- **DOTAX SOI 2022 Table 21**: `data/raw/Dotax Soi 2022 - 21.csv`
- **Processed Data**: `data/processed/capital_gains_percentages.csv`
- **Module Code**: `src/tax/income/capital_gains_separation.py`
- **Test Script**: `scripts/test_ipf_pipeline.py`

## Examples

### Example 1: Basic Separation

```python
import pandas as pd
from src.tax.income import apply_capital_gains_separation

# Load tax units
tax_units = pd.read_parquet('data/processed/tax_units.parquet')

# Apply separation
tax_units = apply_capital_gains_separation(tax_units)

# View results
print(tax_units[['agi', 'income', 'regular_income', 'capital_gains_income']].head())
```

### Example 2: High-Income Analysis

```python
# Filter to high-income households
high_income = tax_units[tax_units['agi'] >= 200000].copy()

# Calculate capital gains statistics
total_income = (high_income['income'] * high_income['weight']).sum()
total_cap_gains = (high_income['capital_gains_income'] * high_income['weight']).sum()
cap_gains_pct = total_cap_gains / total_income * 100

print(f"High-income ($200k+) capital gains: {cap_gains_pct:.1f}%")
```

### Example 3: Tax Impact Analysis

```python
# Calculate tax with and without preferential cap gains treatment

# Scenario 1: All income taxed as regular income
all_regular_tax = calculate_tax(tax_units['income'], ...)

# Scenario 2: Capital gains at preferential rate
regular_tax = calculate_tax(tax_units['regular_income'], ...)
cap_gains_tax = calculate_cap_gains_tax(tax_units['capital_gains_income'], ...)
total_tax = regular_tax + cap_gains_tax

# Compare
tax_savings = (all_regular_tax - total_tax) * tax_units['weight']
total_savings = tax_savings.sum()

print(f"Total tax savings from preferential cap gains treatment: ${total_savings/1e6:.1f}M")
```
