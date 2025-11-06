# Quick Start: Running Tax Scenarios

## Run Pre-Configured Scenarios

```bash
python scripts/analysis/run_tax_scenario.py
```

This runs:
1. 2017 vs Act 46 (2025) comparison
2. Act 46 vs Act 46 Rollback comparison  
3. Future 2027 projections
4. Bracket-by-bracket comparison

## Create a Custom Scenario (5 minutes)

```python
#!/usr/bin/env python3
import pandas as pd
from src.tax.config import TaxSystemConfig, TaxCalculator

# 1. Load tax units
tax_units = pd.read_parquet('data/processed/projections/tax_units_2026_baseline.parquet')

# 2. Initialize calculator
calculator = TaxCalculator()

# 3. Define your scenario
my_scenario = TaxSystemConfig(
    name="my_test",
    year=2025,
    bracket_year=2025,
    standard_deduction_year=2025,
    personal_exemption=1200,
    description="Test scenario"
)

# 4. Calculate revenue
revenue = calculator.calculate_revenue(tax_units, my_scenario)

# 5. Display results
print(f"Revenue: ${revenue['total_revenue_millions']:,.1f}M")
print(f"Avg tax: ${revenue['average_tax_per_filer']:,.0f}")
print(f"Effective rate: {revenue['effective_rate']:.2f}%")
```

## Available Pre-Configured Systems

```python
from src.tax.config import TaxSystemRegistry

# 2017 tax law
system_2017 = TaxSystemRegistry.get_2017_system()

# Act 46 (2025 tax year) 
system_act46 = TaxSystemRegistry.get_act46_2025_system()

# Act 46 with rollback increases
system_rollback = TaxSystemRegistry.get_act46_rollback_targeted()

# Future projections
system_2027 = TaxSystemRegistry.get_act46_2027_system()
```

## Common Operations

### Compare Two Systems
```python
from src.tax.config import compare_systems

comparison = compare_systems(
    tax_units,
    baseline_config=system_2017,
    scenario_config=system_act46,
    calculator=calculator
)
print(comparison)
```

### Get Brackets for a Year
```python
brackets = calculator.get_brackets(
    year=2025,
    filing_status='married_filing_jointly'
)
print(brackets[['income_min', 'income_max', 'rate']])
```

### Get Standard Deduction
```python
deduction = calculator.get_standard_deduction(
    year=2025,
    filing_status='single'
)
print(f"Deduction: ${deduction:,.0f}")
```

## Year Convention (IMPORTANT)

CSV files use "taxable year beginning after" convention:

| CSV Year | Law Enacted | Applies To Tax Year |
|----------|-------------|---------------------|
| 2018 | 2017 law | 2018 (income earned in 2018) |
| 2025 | Act 46 (2024 law) | 2025 (income earned in 2025) |

## Full Documentation

See `docs/TAX_SCENARIO_CONFIGURATION_GUIDE.md` for complete details.
