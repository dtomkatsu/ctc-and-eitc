#!/usr/bin/env python3
"""
Capital Gains Assignment Options - Complete Guide

This script demonstrates all available assignment methods for capital gains
and provides recommendations for different use cases.
"""

print("="*80)
print("CAPITAL GAINS ASSIGNMENT METHODS - COMPLETE GUIDE")
print("="*80)

print("""
🎯 AVAILABLE ASSIGNMENT METHODS:

1. 'random' (Default)
   - Uses random assignment based on participation rates
   - Different results each run due to randomness
   - Good for: Monte Carlo simulations, sensitivity analysis
   
2. 'deterministic_top' (Recommended)
   - Assigns to highest income filers within each bracket
   - Completely reproducible results
   - Most realistic (wealthy filers more likely to have capital gains)
   - Good for: Production models, validation, debugging
   
3. 'deterministic_income'
   - Assigns based on income percentile within bracket
   - Uses participation rate to determine percentile threshold
   - Reproducible and spreads assignment across income range
   - Good for: Sensitivity analysis with income distribution
   
4. 'deterministic_hash'
   - Assigns based on deterministic hash of filer data
   - Reproducible but appears random
   - Good for: Pseudo-random assignment that's reproducible

""")

print("="*80)
print("HOW TO USE DIFFERENT METHODS")
print("="*80)

print("""
📝 METHOD 1: Update regenerate_tax_units.py

In scripts/regenerate_tax_units.py, modify the capital gains section:

# For deterministic assignment (recommended):
tax_units = apply_capital_gains_to_dataframe(
    tax_units,
    agi_col='agi',
    taxable_income_col=None,
    include_random=True,
    assignment_method='deterministic_top'  # Change this line
)

# Available options:
# - 'random'              (default, varies each run)
# - 'deterministic_top'   (highest income filers, reproducible)
# - 'deterministic_income' (income percentile, reproducible)
# - 'deterministic_hash'  (hash-based, reproducible)

""")

print("📝 METHOD 2: Direct function call")
print("""
from src.tax.adjustments.capital_gains import apply_capital_gains_to_dataframe

# Apply with specific method
result_df = apply_capital_gains_to_dataframe(
    df,
    agi_col='agi',
    assignment_method='deterministic_top'
)
""")

print("="*80)
print("COMPARISON OF METHODS")
print("="*80)

print("""
📊 CONSISTENCY:
✅ deterministic_top:    Always same results
✅ deterministic_income: Always same results  
✅ deterministic_hash:   Always same results
❌ random:              Different results each run

📊 REALISM:
✅ deterministic_top:    Most realistic (wealthy get cap gains)
⚠️  deterministic_income: Somewhat realistic
⚠️  deterministic_hash:   Pseudo-random distribution
⚠️  random:              Random distribution

📊 USE CASES:
🎯 Production models:     deterministic_top
🎯 Model validation:      deterministic_top
🎯 Debugging:            deterministic_top
🎯 Sensitivity analysis:  deterministic_income or random
🎯 Monte Carlo sims:     random
""")

print("="*80)
print("CURRENT CONFIGURATION")
print("="*80)

# Check current configuration
try:
    with open('scripts/regenerate_tax_units.py', 'r') as f:
        content = f.read()
        if 'assignment_method=' in content:
            # Extract the current method
            lines = content.split('\n')
            for line in lines:
                if 'assignment_method=' in line:
                    method = line.split('assignment_method=')[1].split('#')[0].strip().strip("'\"")
                    print(f"✅ Current method: {method}")
                    break
        else:
            print("⚠️  Using default method: 'random'")
except:
    print("❌ Could not read current configuration")

print("""
🔧 TO CHANGE METHOD:
1. Edit scripts/regenerate_tax_units.py
2. Find the apply_capital_gains_to_dataframe call
3. Change assignment_method parameter
4. Run: python scripts/regenerate_tax_units.py

💡 RECOMMENDATION:
Use 'deterministic_top' for consistent, realistic results.
This assigns capital gains to the highest income filers within
each bracket, which matches real-world patterns and provides
reproducible results for model validation.
""")

print("="*80)
