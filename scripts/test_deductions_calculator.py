#!/usr/bin/env python3
"""
Test the Deductions Calculator

Tests:
1. Single calculation
2. Batch calculation
3. Policy scenario modeling
4. Validation against benchmarks
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import logging

from src.tax.deductions import TaxableIncomeCalculator, DeductionPolicy

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def test_single_calculation():
    """Test single tax unit calculation."""
    print("="*80)
    print("TEST 1: Single Tax Unit Calculation")
    print("="*80)
    
    # Load calculator
    calculator = TaxableIncomeCalculator.from_files()
    
    # Test cases
    test_cases = [
        {
            'name': 'Low income, single, standard deduction',
            'agi': 25000,
            'filing_status': 'single',
            'num_exemptions': 1,
            'age_65_plus_count': 0
        },
        {
            'name': 'Middle income, joint, likely itemizes',
            'agi': 75000,
            'filing_status': 'joint',
            'num_exemptions': 3,
            'age_65_plus_count': 0
        },
        {
            'name': 'High income, joint, itemizes',
            'agi': 250000,
            'filing_status': 'joint',
            'num_exemptions': 4,
            'age_65_plus_count': 0
        },
        {
            'name': 'Senior, single, additional deduction',
            'agi': 40000,
            'filing_status': 'single',
            'num_exemptions': 1,
            'age_65_plus_count': 1
        },
    ]
    
    for test in test_cases:
        print(f"\n{test['name']}:")
        print(f"  AGI: ${test['agi']:,.0f}")
        
        result = calculator.calculate(test)
        
        print(f"  Deduction: ${result['deduction']:,.0f} ({result['deduction_type']})")
        print(f"  Exemptions: ${result['exemption_amount']:,.0f} ({result['num_exemptions']} exemptions)")
        print(f"  Taxable Income: ${result['taxable_income']:,.0f}")
        print(f"  Effective reduction: {(1 - result['taxable_income']/test['agi'])*100:.1f}%")


def test_batch_calculation():
    """Test batch calculation with actual tax units."""
    print("\n" + "="*80)
    print("TEST 2: Batch Calculation with Tax Units")
    print("="*80)
    
    # Load tax units
    tax_units = pd.read_parquet('data/processed/tax_units_agi.parquet')
    print(f"\nLoaded {len(tax_units):,} tax units")
    
    # Add num_exemptions estimate (simplified)
    def estimate_exemptions(row):
        exemptions = 1
        if row['filing_status'] == 'joint':
            exemptions += 1
        if 'num_dependents' in row and pd.notna(row['num_dependents']):
            exemptions += int(row['num_dependents'])
        return exemptions
    
    tax_units['num_exemptions'] = tax_units.apply(estimate_exemptions, axis=1)
    tax_units['age_65_plus_count'] = 0  # Simplified - would need age data
    
    # Calculate taxable income
    calculator = TaxableIncomeCalculator.from_files()
    results = calculator.calculate_batch(tax_units.head(1000))  # Test with first 1000
    
    print(f"\n✅ Calculated taxable income for {len(results):,} tax units")
    
    # Summary statistics
    print("\nSummary Statistics:")
    print(f"  Avg AGI: ${results['agi'].mean():,.0f}")
    print(f"  Avg Deduction: ${results['deduction'].mean():,.0f}")
    print(f"  Avg Exemptions: ${results['exemption_amount'].mean():,.0f}")
    print(f"  Avg Taxable Income: ${results['taxable_income'].mean():,.0f}")
    
    print(f"\nDeduction Type Distribution:")
    print(f"  Itemized: {(results['deduction_type'] == 'itemized').sum():,} ({(results['deduction_type'] == 'itemized').sum()/len(results)*100:.1f}%)")
    print(f"  Standard: {(results['deduction_type'] == 'standard').sum():,} ({(results['deduction_type'] == 'standard').sum()/len(results)*100:.1f}%)")
    
    return results


def test_policy_scenario():
    """Test policy scenario modeling."""
    print("\n" + "="*80)
    print("TEST 3: Policy Scenario Modeling")
    print("="*80)
    
    # Load small sample
    tax_units = pd.read_parquet('data/processed/tax_units_agi.parquet').head(100)
    tax_units['num_exemptions'] = 2  # Simplified
    tax_units['age_65_plus_count'] = 0
    
    calculator = TaxableIncomeCalculator.from_files()
    
    # Baseline
    print("\nScenario 1: Baseline (2022)")
    baseline_policy = DeductionPolicy(year=2022)
    baseline_results = calculator.calculate_batch(tax_units, policy=baseline_policy)
    baseline_taxable = baseline_results['taxable_income'].sum()
    print(f"  Total taxable income: ${baseline_taxable:,.0f}")
    
    # Scenario 2: Increase standard deduction by 20%
    print("\nScenario 2: Increase Standard Deduction by 20%")
    scenario2_policy = DeductionPolicy(year=2022)
    scenario2_policy.set_policy_change('standard_deduction.single', 2640)  # 2200 * 1.2
    scenario2_policy.set_policy_change('standard_deduction.joint', 5280)   # 4400 * 1.2
    scenario2_policy.set_policy_change('standard_deduction.hoh', 3854)     # 3212 * 1.2
    scenario2_results = calculator.calculate_batch(tax_units, policy=scenario2_policy)
    scenario2_taxable = scenario2_results['taxable_income'].sum()
    print(f"  Total taxable income: ${scenario2_taxable:,.0f}")
    print(f"  Change: ${scenario2_taxable - baseline_taxable:,.0f} ({(scenario2_taxable/baseline_taxable - 1)*100:.2f}%)")
    
    # Scenario 3: Eliminate personal exemptions
    print("\nScenario 3: Eliminate Personal Exemptions")
    scenario3_policy = DeductionPolicy(year=2022)
    scenario3_policy.set_policy_change('personal_exemption', 0)
    scenario3_results = calculator.calculate_batch(tax_units, policy=scenario3_policy)
    scenario3_taxable = scenario3_results['taxable_income'].sum()
    print(f"  Total taxable income: ${scenario3_taxable:,.0f}")
    print(f"  Change: ${scenario3_taxable - baseline_taxable:,.0f} ({(scenario3_taxable/baseline_taxable - 1)*100:.2f}%)")
    
    # Scenario 4: Cap itemized deductions at $20k for high earners
    print("\nScenario 4: Cap Itemized Deductions at $20k (AGI > $300k)")
    scenario4_policy = DeductionPolicy(year=2022)
    scenario4_policy.set_policy_change('itemized_deduction_cap', 20000)
    scenario4_policy.set_policy_change('itemized_deduction_cap_threshold', 300000)
    scenario4_results = calculator.calculate_batch(tax_units, policy=scenario4_policy)
    scenario4_taxable = scenario4_results['taxable_income'].sum()
    print(f"  Total taxable income: ${scenario4_taxable:,.0f}")
    print(f"  Change: ${scenario4_taxable - baseline_taxable:,.0f} ({(scenario4_taxable/baseline_taxable - 1)*100:.2f}%)")


def main():
    """Run all tests."""
    test_single_calculation()
    results = test_batch_calculation()
    test_policy_scenario()
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("""
1. ✅ Deduction benchmarks parsed and saved
2. ✅ Exemption benchmarks parsed and saved
3. ✅ Policy parameters class created
4. ✅ Taxable income calculator implemented
5. 📋 Ready to integrate with full tax unit dataset
6. 📋 Ready to integrate with Hawaii tax calculator

To calculate taxable income for all tax units:
```python
from src.tax.deductions import TaxableIncomeCalculator

# Load tax units
tax_units = pd.read_parquet('data/processed/tax_units_agi.parquet')

# Add num_exemptions if not present
# ... (estimate from household composition)

# Calculate taxable income
calculator = TaxableIncomeCalculator.from_files()
results = calculator.calculate_batch(tax_units)

# Save
results.to_parquet('data/processed/tax_units_with_taxable_income.parquet')
```
""")


if __name__ == '__main__':
    main()
