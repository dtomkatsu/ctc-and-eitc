# Model Configuration Usage Guide

## Overview

All model parameters are now centralized in `config/model_config.py`. This provides a single source of truth for:
- Ensemble weights
- Growth rates
- Revenue targets
- File paths
- Calibration parameters
- Tax calculation settings

## Quick Start

### Basic Usage

```python
from config import ModelConfig

# Create config instance
config = ModelConfig()

# Access parameters
weights = config.ENSEMBLE_WEIGHTS
growth_rate = config.MODERATE_PARAMS['growth_rate']
target = config.FY_2026_RESIDENT_TARGET
```

### Scenario-Specific Parameters

```python
# Get parameters for specific scenario
params = ModelConfig.get_scenario_params('moderate')
print(params['target_resident'])  # 3085
print(params['income_scaling'])   # 0.935

# Get ensemble weights for scenario
weights = ModelConfig.get_ensemble_weights('conservative')
```

### Calculate Weighted Growth

```python
# Calculate weighted growth rate for scenario
growth = ModelConfig.calculate_weighted_growth('moderate')
print(f"Weighted growth: {growth:.1%}")  # 3.1%
```

### Get Target Revenues

```python
# Get all target revenues for scenario
targets = ModelConfig.get_target_revenue('moderate')
print(targets['resident'])     # 3085
print(targets['total'])        # 3383
print(targets['nonresident'])  # 298
```

## Available Scenarios

### Conservative
- **Base**: FY 2024 actual only
- **Growth**: 1.5% over 2 years
- **Target**: $3,082M residents
- **Use case**: Most cautious projections

### Moderate (RECOMMENDED)
- **Base**: Blend FY 2024 + FY 2025 estimate
- **Growth**: 2.0% annual
- **Target**: $3,085M residents
- **Use case**: Balanced, defensible projections

### Aggressive
- **Base**: FY 2025 estimate
- **Growth**: 2.5% annual
- **Target**: $3,074M residents
- **Use case**: Optimistic projections

## Changing Active Scenario

To switch the default scenario, edit `config/model_config.py`:

```python
class ModelConfig:
    ACTIVE_SCENARIO = 'moderate'  # Change to 'conservative' or 'aggressive'
```

Or specify scenario when calling methods:

```python
# Use specific scenario without changing default
params = ModelConfig.get_scenario_params('aggressive')
```

## Key Parameters Reference

### Fiscal Year Data
```python
config.FY_2024_TOTAL           # 3280 (last confirmed actual)
config.FY_2025_ESTIMATE        # 3288 (DOT projection)
config.FY_2024_RESIDENT        # 2991
config.FY_2025_RESIDENT_ESTIMATE  # 2999
```

### Revenue Shares
```python
config.NONRESIDENT_SHARE       # 0.088 (8.8%)
config.RESIDENT_SHARE          # 0.912 (91.2%)
```

### Act 46 Parameters
```python
config.ACT46_IMPACT_RATE       # -0.199 (-19.9%)
config.ACT46_IMPACT_TOTAL      # -597 (Million)
config.FY_2026_POST_ACT46_OFFICIAL  # 2691 (Million)
```

### Filing Status Targets (SOI)
```python
config.SOI_FILING_STATUS_TARGETS
# {
#     'single': 0.510,  # 51.0%
#     'joint': 0.360,   # 36.0%
#     'hoh': 0.096,     # 9.6%
#     'mfs': 0.034      # 3.4%
# }
```

### File Paths
```python
config.DATA_DIR                # data/
config.PUMS_DIR               # data/raw/pums/
config.CALIBRATION_OUTPUT_DIR # data/processed/calibration/
config.PROJECTION_OUTPUT_DIR  # data/processed/projections/
```

## Validation

### Validate Scenario Configuration
```python
# Check if scenario is properly configured
is_valid = ModelConfig.validate_scenario('moderate')

if is_valid:
    print("✅ Scenario valid")
else:
    print("❌ Scenario has issues")
```

### Print Configuration Summary
```python
# Print full configuration for scenario
ModelConfig.print_summary('moderate')
```

Output:
```
================================================================================
HAWAII TAX MODEL CONFIGURATION - MODERATE SCENARIO
================================================================================

Scenario Parameters:
  base_year: 2024.5
  base_resident: 2995.008
  growth_rate: 0.02
  target_resident: 3085
  target_total: 3383
  income_scaling: 0.935

Ensemble Weights:
  fy_actual_2022_2024: 30.0%
  fy_2025_estimate: 10.0%
  dotax_2018_2021: 20.0%
  bls_wage: 25.0%
  acs_income: 10.0%
  demographics: 5.0%

Weighted Growth Rate: 3.1%

Target Revenues (2026):
  resident: $3,085M
  total: $3,383M
  nonresident: $298M
```

## Integration Examples

### In Calibration Scripts

```python
from config import ModelConfig

class MyCalibrator:
    def __init__(self, scenario='moderate'):
        self.config = ModelConfig()
        self.scenario = scenario
        
        # Load all parameters from config
        params = self.config.get_scenario_params(scenario)
        self.target_revenue = params['target_resident']
        self.income_scaling = params['income_scaling']
        
        # Load ensemble weights
        self.weights = self.config.get_ensemble_weights(scenario)
```

### In Projection Scripts

```python
from config import ModelConfig

def project_revenue(scenario='moderate'):
    config = ModelConfig()
    
    # Get target and parameters
    targets = config.get_target_revenue(scenario)
    params = config.get_scenario_params(scenario)
    
    # Calculate projection
    base = params['base_resident']
    growth = params['growth_rate']
    years = params['years_forward']
    
    projection = base * (1 + growth) ** years
    
    return {
        'projection': projection,
        'target': targets['resident'],
        'error': projection - targets['resident']
    }
```

### In Tax Calculation Scripts

```python
from config import ModelConfig

def calculate_taxes(tax_units):
    config = ModelConfig()
    
    # Use itemized deduction parameters
    rates = config.ITEMIZED_DEDUCTION_RATES
    
    # Apply SALT cap
    salt_deduction = min(state_tax + property_tax, rates['salt_cap'])
    
    # Calculate mortgage interest
    mortgage_interest = mortgage_payment * rates['mortgage_interest_rate']
    
    return total_deductions
```

## Best Practices

### 1. Always Import from Config
❌ **Don't do this:**
```python
GROWTH_RATE = 0.02  # Hardcoded
TARGET_REVENUE = 3085
```

✅ **Do this:**
```python
from config import ModelConfig
config = ModelConfig()
GROWTH_RATE = config.MODERATE_PARAMS['growth_rate']
TARGET_REVENUE = config.MODERATE_PARAMS['target_resident']
```

### 2. Use Scenario Methods
❌ **Don't do this:**
```python
if scenario == 'moderate':
    weights = {'fy_actual': 0.30, 'dotax': 0.20, ...}
elif scenario == 'conservative':
    weights = {'fy_actual': 0.30, 'dotax': 0.20, ...}
```

✅ **Do this:**
```python
weights = ModelConfig.get_ensemble_weights(scenario)
```

### 3. Validate Before Using
```python
# Always validate scenario before running
if not ModelConfig.validate_scenario(scenario):
    raise ValueError(f"Invalid scenario: {scenario}")

# Then proceed with calculations
params = ModelConfig.get_scenario_params(scenario)
```

### 4. Document Config Changes
When updating `config/model_config.py`, add a comment:
```python
# Updated 2025-11-03: Increased growth rate based on Q3 actuals
MODERATE_PARAMS = {
    'growth_rate': 0.022,  # Was 0.020
    ...
}
```

## Updating Parameters

### To Update a Single Parameter
1. Edit `config/model_config.py`
2. Find the parameter (e.g., `GROWTH_RATE`)
3. Update the value
4. Save the file
5. All scripts automatically use new value

### To Add a New Scenario
1. Add scenario parameters:
```python
OPTIMISTIC_PARAMS = {
    'base_year': 2025,
    'growth_rate': 0.030,
    'target_resident': 3100,
    ...
}
```

2. Add ensemble weights:
```python
ENSEMBLE_WEIGHTS_OPTIMISTIC = {
    'fy_actual_2022_2024': 0.20,
    'dotax_2018_2021': 0.30,
    ...
}
```

3. Update `get_scenario_params()` method:
```python
scenarios = {
    'conservative': cls.CONSERVATIVE_PARAMS,
    'moderate': cls.MODERATE_PARAMS,
    'aggressive': cls.AGGRESSIVE_PARAMS,
    'optimistic': cls.OPTIMISTIC_PARAMS  # Add new
}
```

## Testing Configuration

Run the config module directly to test:
```bash
python config/model_config.py
```

This will:
- Print configuration summary
- Validate all scenarios
- Check that weights sum to 1.0
- Verify required parameters exist

## Troubleshooting

### Issue: Import Error
```python
ModuleNotFoundError: No module named 'config'
```

**Solution**: Add project root to path:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ModelConfig
```

### Issue: Weights Don't Sum to 1.0
```python
ValueError: Weights sum to 0.95, not 1.0
```

**Solution**: Config automatically normalizes weights, but if you see this warning, check that all weights are defined correctly in `config/model_config.py`.

### Issue: Scenario Not Found
```python
ValueError: Unknown scenario: moderete
```

**Solution**: Check spelling. Valid scenarios are: 'conservative', 'moderate', 'aggressive'

## Summary

The centralized configuration provides:
- ✅ **Single source of truth** for all parameters
- ✅ **Easy scenario switching** without code changes
- ✅ **Automatic validation** of configuration
- ✅ **Clear documentation** of all settings
- ✅ **Consistent parameters** across all scripts

**Key takeaway**: Always import from `config.ModelConfig` rather than hardcoding values!
