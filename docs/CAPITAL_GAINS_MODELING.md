# Capital Gains Modeling Guide

## Overview

Capital gains have been added to the Hawaii tax model to enable apples-to-apples comparison with DOTax benchmarks. This document explains how capital gains are estimated and how to toggle them for scenario modeling.

## Data Source

Capital gains estimates are based on **DOTax SOI 2022 Table 21**, which shows:
- Net long-term capital gains by AGI bracket
- Percentage of total taxable income from capital gains
- Number of filers with capital gains by bracket

### Table 21 Summary (Residents)
- **Total capital gains**: $2,995M
- **Filers with cap gains**: 43,443 (7.4% of all filers)
- **Average cap gains**: $68,945 per filer

## How It Works

### 1. Capital Gains Rates by Bracket

Capital gains are estimated as a percentage of taxable income based on the filer's AGI bracket:

| AGI Bracket | Cap Gains % of Taxable Income |
|-------------|------------------------------|
| $0k-$10k | 0.5% |
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

### 2. Participation Rates

Not all filers have capital gains. Participation rates increase with income:

| AGI Bracket | Participation Rate |
|-------------|-------------------|
| $0k-$10k | 0.03% |
| $10k-$20k | 0.02% |
| $20k-$30k | 0.4% |
| $30k-$40k | 2.7% |
| $40k-$50k | 3.8% |
| $50k-$75k | 6.6% |
| $75k-$100k | 11.0% |
| $100k-$150k | 14.7% |
| $150k-$200k | 21.8% |
| $200k-$300k | 29.5% |
| $300k-$400k | 39.2% |
| $400k+ | 47.2% |

### 3. Calculation Method

For each tax unit:
1. Determine AGI bracket
2. Check if filer has capital gains (random draw based on participation rate)
3. If yes, estimate capital gains as: `taxable_income × cap_gains_rate`
4. Add capital gains to AGI

## Usage

### Running with Capital Gains (Default)

```python
python scripts/regenerate_tax_units.py
```

This includes capital gains by default for apples-to-apples comparison with DOTax.

### Running without Capital Gains (Policy Scenarios)

To model scenarios without capital gains (e.g., policy changes):

```python
# Modify the script call
if __name__ == '__main__':
    main(include_capital_gains=False)
```

Or create a wrapper script:

```python
from scripts.regenerate_tax_units import main

# Run without capital gains
main(include_capital_gains=False)
```

## Output Fields

When capital gains are included, the following fields are added to the tax units DataFrame:

- **`capital_gains`**: Estimated capital gains amount ($)
- **`agi_with_cap_gains`**: AGI including capital gains (used for tax calculation)
- **`agi_without_cap_gains`**: Original AGI from PUMS (for scenario comparisons)

## Example: Policy Scenario Analysis

### Scenario 1: Current Law (with capital gains)
```python
# Default run
main(include_capital_gains=True)
# Result: Model tax = $2.657B vs DOTax = $3.029B
```

### Scenario 2: Hypothetical Tax Change (without capital gains)
```python
# Run without capital gains
main(include_capital_gains=False)
# Result: Model tax = $2.598B vs DOTax = $3.029B
# Difference shows impact of capital gains on revenue
```

### Scenario 3: Compare Revenue Impact
```python
import pandas as pd

# Run both scenarios
df_with_cg = pd.read_parquet('data/processed/tax_units_calibrated_with_cg.parquet')
df_without_cg = pd.read_parquet('data/processed/tax_units_calibrated_without_cg.parquet')

# Calculate difference
cg_revenue_impact = (df_with_cg['hi_state_tax'] * df_with_cg['weight']).sum() - \
                    (df_without_cg['hi_state_tax'] * df_without_cg['weight']).sum()

print(f"Capital gains revenue impact: ${cg_revenue_impact/1e6:.1f}M")
```

## Model Accuracy

### With Capital Gains
- **Total revenue**: $2.657B vs $3.029B benchmark (-12.3%)
- **Filers with cap gains**: 46,761 (7.4%)
- **Total cap gains**: $558M (vs $2,995M benchmark)

### Underestimation Reasons
1. **Concentration effect**: Capital gains highly concentrated in top earners
2. **PUMS limitations**: Survey doesn't capture full capital gains distribution
3. **Participation undercount**: Model estimates fewer high-income filers with cap gains

### Validation by Bracket
- **$30k-$40k**: +7.7% ✅
- **$100k-$150k**: +10.2% ⚠️
- **$150k-$200k**: -5.5% ✅
- **$200k-$300k**: -9.3% ✅
- **$300k-$400k**: -8.9% ✅
- **$400k+**: -57.6% ❌ (capital gains concentration)

## Technical Implementation

### Module: `src/tax/adjustments/capital_gains.py`

**Key Classes**:
- `CapitalGainsEstimator`: Estimates capital gains based on Table 21

**Key Functions**:
- `estimate_capital_gains()`: Estimate cap gains for single tax unit
- `apply_capital_gains_to_dataframe()`: Batch apply to DataFrame

**Example Usage**:
```python
from src.tax.adjustments.capital_gains import CapitalGainsEstimator

estimator = CapitalGainsEstimator()

# For a $100k AGI filer
agi = 100000
taxable_income = 85000  # After deductions

cap_gains = estimator.estimate_capital_gains(agi, taxable_income)
print(f"Estimated capital gains: ${cap_gains:,.0f}")
```

## Future Enhancements

1. **Improved concentration modeling**: Better capture top 1% capital gains
2. **Asset-based estimation**: Use homeownership/wealth proxies
3. **Year-over-year variation**: Model capital gains volatility
4. **Alternative scenarios**: Different cap gains tax rates

## References

- **Data Source**: Hawaii Department of Taxation, SOI 2022, Table 21
- **Implementation**: `/src/tax/adjustments/capital_gains.py`
- **Pipeline**: `/scripts/regenerate_tax_units.py` (lines 92-124)
- **Documentation**: This file and `INVESTIGATION_COMPLETE_SUMMARY.md`
