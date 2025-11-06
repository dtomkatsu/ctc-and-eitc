# Tax Scenario Configuration Guide

## Overview

The centralized tax configuration system (`src/tax/config/`) eliminates the need to write new scripts for every tax scenario. Instead, you configure tax systems and run analyses using a unified API.

## Quick Start

```python
from src.tax.config import TaxSystemRegistry, TaxCalculator, compare_systems

# Load tax units
tax_units = pd.read_parquet('data/processed/projections/tax_units_2026_baseline.parquet')

# Initialize calculator
calculator = TaxCalculator()

# Get pre-configured tax systems
system_2017 = TaxSystemRegistry.get_2017_system()
system_act46 = TaxSystemRegistry.get_act46_2025_system()

# Compare systems
comparison = compare_systems(tax_units, system_2017, system_act46, calculator)
print(comparison)
```

## Core Components

### 1. TaxSystemConfig

Defines a complete tax system configuration:

```python
@dataclass
class TaxSystemConfig:
    name: str                           # Unique identifier
    year: int                           # Tax year (what year's income)
    bracket_year: int                   # Which bracket schedule to use
    standard_deduction_year: int        # Which deduction schedule to use
    personal_exemption: float           # Per-person exemption amount
    description: str                    # Human-readable description
    bracket_adjustments: Optional[Dict] # For rollback scenarios
```

### 2. TaxSystemRegistry

Pre-configured tax systems for common scenarios:

#### Available Systems

**2017 Tax System (Pre-Act 46)**
```python
system = TaxSystemRegistry.get_2017_system()
```
- Uses 2018 brackets (2017 law, applies to 2018 tax year)
- Uses 2018 standard deductions
- Personal exemption: $1,144

**Act 46 (2025 Tax Year)**
```python
system = TaxSystemRegistry.get_act46_2025_system()
```
- Uses 2025 brackets (Act 46, applies to 2025+ tax years)
- Uses 2025 standard deductions
- Personal exemption: $1,200

**Act 46 with Targeted Rollback**
```python
system = TaxSystemRegistry.get_act46_rollback_targeted()
```
- Starts with Act 46 baseline
- Adds targeted rate increases to top 5 brackets:
  - 5th highest: +0.25pp
  - 4th highest: +0.25pp
  - 3rd highest: +0.25pp
  - 2nd highest: +0.50pp
  - Highest: +1.00pp

### 3. TaxCalculator

Centralized calculation engine:

```python
calculator = TaxCalculator()

# Calculate tax for a single unit
result = calculator.calculate_tax(
    income=100000,
    config=system_config,
    filing_status='married_filing_jointly',
    num_exemptions=3
)
# Returns: {
#   'gross_income': 100000,
#   'standard_deduction': 8800,
#   'personal_exemptions': 3600,
#   'taxable_income': 87600,
#   'tax_liability': 5234.56,
#   'marginal_rate': 7.6,
#   'effective_rate': 5.23
# }

# Calculate revenue for all tax units
revenue = calculator.calculate_revenue(tax_units, config)
# Returns: {
#   'total_revenue_millions': 3023.4,
#   'average_tax_per_filer': 4729,
#   'average_income': 78500,
#   'effective_rate': 6.03,
#   'total_filers': 639309
# }
```

## Creating Custom Scenarios

### Example 1: Simple Custom Year

```python
from src.tax.config import TaxSystemConfig

custom_system = TaxSystemConfig(
    name="future_2029",
    year=2029,
    bracket_year=2029,
    standard_deduction_year=2029,
    personal_exemption=1250,
    description="Future projection for 2029"
)

revenue = calculator.calculate_revenue(tax_units, custom_system)
```

### Example 2: Custom Bracket Adjustments

```python
# Create a custom rollback scenario
custom_rollback = TaxSystemConfig(
    name="aggressive_rollback",
    year=2025,
    bracket_year=2025,
    standard_deduction_year=2025,
    personal_exemption=1200,
    description="Aggressive rollback: +2.0pp on top 2 brackets",
    bracket_adjustments={
        'top_5_adjustments': [0.0, 0.0, 0.0, 2.0, 2.0]
    }
)

revenue = calculator.calculate_revenue(tax_units, custom_rollback)
```

### Example 3: Mix-and-Match Years

```python
# Use 2025 brackets with 2027 deductions
hybrid_system = TaxSystemConfig(
    name="hybrid_2025_2027",
    year=2026,
    bracket_year=2025,
    standard_deduction_year=2027,
    personal_exemption=1200,
    description="2025 brackets with 2027 deductions"
)
```

## Common Use Cases

### 1. Compare Two Systems

```python
from src.tax.config import compare_systems

baseline = TaxSystemRegistry.get_2017_system()
scenario = TaxSystemRegistry.get_act46_2025_system()

comparison = compare_systems(tax_units, baseline, scenario, calculator)
print(comparison)
```

Output:
```
      system                    description  revenue_millions  avg_tax  effective_rate
  2017_system  Pre-Act 46 baseline (2017)            3538.2     5535            7.05
   act46_2025  Act 46 (2024 law, ...)                3023.4     4729            6.03
   Difference  act46_2025 vs 2017_system             -514.8     -806           -1.02
```

### 2. Test Multiple Scenarios

```python
scenarios = [
    TaxSystemRegistry.get_2017_system(),
    TaxSystemRegistry.get_act46_2025_system(),
    TaxSystemRegistry.get_act46_rollback_targeted(),
]

results = []
for scenario in scenarios:
    revenue = calculator.calculate_revenue(tax_units, scenario)
    results.append({
        'scenario': scenario.name,
        'revenue': revenue['total_revenue_millions']
    })

df = pd.DataFrame(results)
print(df)
```

### 3. Analyze Bracket Structure

```python
# Get brackets for any year/status
brackets = calculator.get_brackets(
    year=2025,
    filing_status='married_filing_jointly'
)

print(brackets[['income_min', 'income_max', 'rate']])
```

### 4. Get Standard Deductions

```python
# Get deduction for any year/status
deduction = calculator.get_standard_deduction(
    year=2025,
    filing_status='single'
)
print(f"Single filer deduction: ${deduction:,.0f}")
```

## Running Scenarios

### Quick Run
```bash
python scripts/analysis/run_tax_scenario.py
```

This runs pre-configured comparisons:
1. 2017 vs Act 46 (2025)
2. Act 46 vs Act 46 Rollback
3. Future projections (2027)
4. Bracket-by-bracket comparison

### Custom Script Template

```python
#!/usr/bin/env python3
import pandas as pd
from src.tax.config import TaxSystemConfig, TaxCalculator

# Load data
tax_units = pd.read_parquet('data/processed/projections/tax_units_2026_baseline.parquet')
calculator = TaxCalculator()

# Define your scenario
my_scenario = TaxSystemConfig(
    name="my_custom_scenario",
    year=2025,
    bracket_year=2025,
    standard_deduction_year=2025,
    personal_exemption=1200,
    description="Your description here"
)

# Calculate
revenue = calculator.calculate_revenue(tax_units, my_scenario)

# Display results
print(f"Revenue: ${revenue['total_revenue_millions']:,.1f}M")
```

## Data Files

The system automatically loads from:
- **Brackets**: `data/raw/hawaii_tax_brackets_master_all.csv`
- **Deductions**: `data/raw/hawaii_standard_deductions_by_year.csv`

### Year Convention

**Important**: Years in CSV files follow the "taxable year beginning after" convention:
- Year `2025` in CSV = 2024 law (applies to taxable years beginning after 12/31/2024)
- This means it applies to 2025 tax year (income earned in 2025, filed in 2026)

### Adding New Years

To add future projections, simply add rows to the CSV files:

**hawaii_tax_brackets_master_all.csv**:
```csv
income_min,income_max,rate,base_tax,base_income,year,filing_status
0,20000,1.5,0,0,2030,Joint_Surviving_Spouse
...
```

**hawaii_standard_deductions_by_year.csv**:
```csv
Year,Joint_Surviving_Spouse,Head_of_Household,Single_Married_Separate
2030,20000,15000,10000
```

## Best Practices

### 1. Use Registry for Common Scenarios
```python
# Good
system = TaxSystemRegistry.get_act46_2025_system()

# Avoid (unless truly custom)
system = TaxSystemConfig(name="act46", year=2025, ...)
```

### 2. Reuse Calculator Instance
```python
# Good - initialize once
calculator = TaxCalculator()
for scenario in scenarios:
    revenue = calculator.calculate_revenue(tax_units, scenario)

# Avoid - creates overhead
for scenario in scenarios:
    calculator = TaxCalculator()  # Don't do this
    revenue = calculator.calculate_revenue(tax_units, scenario)
```

### 3. Name Scenarios Clearly
```python
# Good
TaxSystemConfig(name="act46_2025_rollback_conservative", ...)

# Avoid
TaxSystemConfig(name="scenario1", ...)
```

### 4. Document Custom Adjustments
```python
custom = TaxSystemConfig(
    name="custom_rollback",
    year=2025,
    bracket_year=2025,
    standard_deduction_year=2025,
    personal_exemption=1200,
    description="Top 3 brackets +1.0pp to recover $150M",  # Clear description
    bracket_adjustments={
        'top_5_adjustments': [0.0, 0.0, 1.0, 1.0, 1.0]
    }
)
```

## Extending the System

### Add New Pre-Configured System

Edit `src/tax/config/tax_system_config.py`:

```python
@classmethod
def get_my_new_scenario(cls) -> TaxSystemConfig:
    """My new scenario description."""
    return TaxSystemConfig(
        name="my_scenario",
        year=2026,
        bracket_year=2026,
        standard_deduction_year=2026,
        personal_exemption=cls.PERSONAL_EXEMPTIONS.get(2026, 1200),
        description="Detailed description"
    )
```

### Add Custom Bracket Adjustment Logic

Extend `TaxCalculator.apply_bracket_adjustments()` for more complex scenarios.

## Troubleshooting

### "No brackets found for year X"
- Check that the year exists in `hawaii_tax_brackets_master_all.csv`
- Remember: year in CSV uses "taxable year beginning after" convention

### "No standard deduction found for year X"
- Check that the year exists in `hawaii_standard_deductions_by_year.csv`

### Revenue seems wrong
- Verify `bracket_year` and `standard_deduction_year` are correct
- Check that tax units have required columns: `filing_status`, `income`, `weight`
- Ensure `num_exemptions` column exists or defaults correctly

## Examples

See `scripts/analysis/run_tax_scenario.py` for complete working examples.
