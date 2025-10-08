# Hawaii Income Tax Calculation Guide

## Overview

This project now includes comprehensive Hawaii state income tax calculation capabilities using historical and projected tax brackets and standard deductions.

## Data Files

### Tax Brackets
**File:** `data/raw/hawaii_tax_brackets_master_all.csv`

Contains Hawaii state income tax brackets for multiple years (2017, 2024, 2026, 2028) and all filing statuses:
- **Joint_Surviving_Spouse**: Married filing jointly or qualifying widow(er)
- **Head_of_Household**: Head of household filers
- **Single_Married_Separate**: Single filers or married filing separately

**Structure:**
- `income_min`: Lower bound of bracket
- `income_max`: Upper bound of bracket (inf for highest bracket)
- `rate`: Tax rate as percentage (e.g., 1.4 = 1.4%)
- `base_tax`: Tax owed on income up to bracket minimum
- `base_income`: Income at start of bracket
- `year`: Tax year
- `filing_status`: Filing status category

### Standard Deductions
**File:** `data/raw/hawaii_standard_deductions_by_year.csv`

Contains standard deductions by year and filing status for years 2017-2031.

## Usage

### Basic Tax Calculation

```python
from src.tax.brackets import load_tax_data

# Load calculator
calculator = load_tax_data()

# Calculate tax for a single filer with $50,000 income in 2024
result = calculator.calculate_tax(50000, 2024, 'single')

print(f"Tax Liability: ${result['tax_liability']:,.2f}")
print(f"Effective Rate: {result['effective_rate']:.2f}%")
```

### Calculate for Tax Units Dataset

```python
import pandas as pd
from src.tax.brackets import load_tax_data

# Load your tax units
tax_units = pd.read_parquet('data/processed/tax_units_rule_based.parquet')

# Calculate taxes
calculator = load_tax_data()
tax_units_with_tax = calculator.calculate_tax_for_dataframe(
    tax_units,
    income_col='income',
    filing_status_col='filing_status',
    year=2024
)

# Results are added with 'hi_tax_' prefix
print(tax_units_with_tax[['income', 'filing_status', 'hi_tax_tax_liability', 'hi_tax_effective_rate']])
```

### Compare Tax Scenarios Across Years

```python
from src.tax.brackets import load_tax_data

calculator = load_tax_data()

# Compare a $75,000 joint filer across different years
comparison = calculator.compare_scenarios(
    income=75000,
    filing_status='married_filing_jointly',
    years=[2017, 2024, 2026, 2028]
)

print(comparison[['year', 'tax_liability', 'effective_rate']])
```

## Tax Calculation Details

### Filing Status Mapping

The calculator automatically maps PUMS filing statuses to Hawaii tax statuses:

| PUMS Status | Hawaii Status |
|-------------|---------------|
| `single` | `Single_Married_Separate` |
| `married_filing_jointly` | `Joint_Surviving_Spouse` |
| `married_filing_separate` | `Single_Married_Separate` |
| `head_of_household` | `Head_of_Household` |
| `qualifying_widow` | `Joint_Surviving_Spouse` |

### Calculation Process

1. **Determine Taxable Income**
   - Start with gross income
   - Subtract standard deduction (or itemized if greater)
   - Result is taxable income

2. **Apply Progressive Tax Brackets**
   - Income is taxed at different rates for different portions
   - Lower income is taxed at lower rates
   - Only income above each threshold is taxed at the higher rate

3. **Calculate Tax Liability**
   - Sum of tax from each bracket
   - Uses base_tax + (income_in_bracket × rate)

4. **Calculate Rates**
   - **Marginal Rate**: Rate applied to last dollar earned
   - **Effective Rate**: Total tax / gross income

### Example Calculation

For a single filer with $50,000 income in 2024:

1. **Gross Income:** $50,000
2. **Standard Deduction:** $4,400
3. **Taxable Income:** $45,600

Tax calculation (2024 Single brackets):
- $0 - $9,600 at 1.4%: $134.40
- $9,600 - $14,400 at 3.2%: $153.60
- $14,400 - $19,200 at 5.5%: $264.00
- $19,200 - $24,000 at 6.4%: $307.20
- $24,000 - $36,000 at 6.8%: $816.00
- $36,000 - $45,600 at 7.2%: $691.20

**Total Tax:** $2,366.40
**Effective Rate:** 4.73%
**Marginal Rate:** 7.2%

## Scripts

### Test Calculator
```bash
python scripts/test_tax_calculator.py
```
Quick test to verify calculator is working correctly.

### Calculate Taxes for All Tax Units
```bash
python scripts/calculate_hawaii_taxes.py
```
Comprehensive script that:
- Shows example calculations
- Compares scenarios across years
- Applies calculations to full tax units dataset
- Generates summary statistics

## Output Columns

When using `calculate_tax_for_dataframe()`, the following columns are added:

- `hi_tax_gross_income`: Original income
- `hi_tax_standard_deduction`: Standard deduction applied
- `hi_tax_taxable_income`: Income after deductions
- `hi_tax_tax_liability`: Total tax owed
- `hi_tax_effective_rate`: Effective tax rate (%)
- `hi_tax_marginal_rate`: Marginal tax rate (%)
- `hi_tax_bracket`: Description of top tax bracket

## Analyzing Tax Bracket Shifts

### Scenario Analysis

To analyze the impact of tax bracket changes:

```python
from src.tax.brackets import load_tax_data

calculator = load_tax_data()

# Define income levels to analyze
income_levels = [25000, 50000, 75000, 100000, 150000, 200000]

# Compare across years
for income in income_levels:
    comparison = calculator.compare_scenarios(
        income=income,
        filing_status='married_filing_jointly',
        years=[2024, 2026, 2028]
    )
    
    # Calculate change from baseline
    baseline_tax = comparison[comparison['year'] == 2024]['tax_liability'].iloc[0]
    
    for _, row in comparison.iterrows():
        if row['year'] != 2024:
            change = row['tax_liability'] - baseline_tax
            pct_change = (change / baseline_tax) * 100
            print(f"Income ${income:,}, Year {int(row['year'])}: "
                  f"${change:+,.2f} ({pct_change:+.1f}%)")
```

### Distributional Impact

To analyze impact across the income distribution:

```python
import pandas as pd
from src.tax.brackets import load_tax_data

# Load tax units
tax_units = pd.read_parquet('data/processed/tax_units_rule_based.parquet')

calculator = load_tax_data()

# Calculate for baseline year
baseline = calculator.calculate_tax_for_dataframe(tax_units, year=2024)

# Calculate for comparison year
comparison = calculator.calculate_tax_for_dataframe(tax_units, year=2028)

# Calculate change
tax_units['tax_change'] = (comparison['hi_tax_tax_liability'] - 
                           baseline['hi_tax_tax_liability'])

# Analyze by income quintile
tax_units['income_quintile'] = pd.qcut(tax_units['income'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

summary = tax_units.groupby('income_quintile').agg({
    'tax_change': ['mean', 'median', 'sum'],
    'income': 'mean'
})

print(summary)
```

## Future Enhancements

Potential additions to the tax calculation system:

1. **Federal Tax Integration**: Add federal income tax calculations
2. **Tax Credits**: Incorporate Hawaii state tax credits
3. **Itemized Deductions**: Support for itemized deductions beyond standard
4. **Alternative Minimum Tax**: Add AMT calculations if applicable
5. **Multi-Year Projections**: Automated scenario generation for policy analysis
6. **Visualization Tools**: Charts and graphs for tax impact analysis
