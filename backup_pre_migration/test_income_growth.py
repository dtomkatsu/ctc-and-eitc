"""
Test script to validate income growth adjustments.

This script demonstrates the income growth factors and validates
that they are applied correctly throughout the codebase.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import logging
import pandas as pd
import numpy as np
from config.income_growth import (
    RESIDENT_GROWTH,
    NONRESIDENT_GROWTH,
    apply_income_growth,
    log_growth_factors,
    validate_growth_factors
)
from tax.units.income import calculate_person_income

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_growth_factor_calculations():
    """Test that growth factors are calculated correctly."""
    print("\n" + "="*80)
    print("TEST 1: Growth Factor Calculations")
    print("="*80)
    
    # Test resident growth
    assert abs(RESIDENT_GROWTH.real_growth - 1.056) < 0.001, \
        f"Resident real growth should be 1.056, got {RESIDENT_GROWTH.real_growth}"
    
    assert abs(RESIDENT_GROWTH.nominal_growth / RESIDENT_GROWTH.inflation - 
               RESIDENT_GROWTH.real_growth) < 0.001, \
        "Real growth should equal nominal / inflation"
    
    # Test nonresident growth
    assert abs(NONRESIDENT_GROWTH.real_growth - 1.053) < 0.001, \
        f"Nonresident real growth should be 1.053, got {NONRESIDENT_GROWTH.real_growth}"
    
    assert abs(NONRESIDENT_GROWTH.nominal_growth / NONRESIDENT_GROWTH.inflation - 
               NONRESIDENT_GROWTH.real_growth) < 0.001, \
        "Real growth should equal nominal / inflation"
    
    print("✅ All growth factor calculations are correct")


def test_apply_income_growth():
    """Test the apply_income_growth function."""
    print("\n" + "="*80)
    print("TEST 2: Apply Income Growth Function")
    print("="*80)
    
    # Test resident
    resident_income = 75000
    projected_resident = apply_income_growth(resident_income, is_resident=True)
    expected_resident = 75000 * 1.056
    
    assert abs(projected_resident - expected_resident) < 0.01, \
        f"Expected {expected_resident}, got {projected_resident}"
    
    print(f"Resident: ${resident_income:,} (2023) → ${projected_resident:,.0f} (2026)")
    print(f"  Growth: {(projected_resident / resident_income - 1):.2%}")
    
    # Test nonresident
    nonresident_income = 100000
    projected_nonresident = apply_income_growth(nonresident_income, is_resident=False)
    expected_nonresident = 100000 * 1.053
    
    assert abs(projected_nonresident - expected_nonresident) < 0.01, \
        f"Expected {expected_nonresident}, got {projected_nonresident}"
    
    print(f"\nNonresident: ${nonresident_income:,} (2022) → ${projected_nonresident:,.0f} (2026)")
    print(f"  Growth: {(projected_nonresident / nonresident_income - 1):.2%}")
    
    print("\n✅ Income growth function works correctly")


def test_calculate_person_income():
    """Test that calculate_person_income applies growth correctly."""
    print("\n" + "="*80)
    print("TEST 3: Person Income Calculation with Growth")
    print("="*80)
    
    # Create mock person data
    person = pd.Series({
        'WAGP': 50000,
        'SEMP': 10000,
        'INTP': 1000,
        'DIV': 500,
        'RETP': 0,
        'SSP': 0,
        'OIP': 0,
        'ADJINC': 1.0  # No PUMS adjustment for simplicity
    })
    
    # Calculate without growth
    income_base = calculate_person_income(person, apply_2026_growth=False, is_resident=True)
    expected_base = 50000 + 10000 + 1000 + 500
    
    assert abs(income_base - expected_base) < 0.01, \
        f"Base income should be {expected_base}, got {income_base}"
    
    print(f"Base income (2023): ${income_base:,.0f}")
    
    # Calculate with growth (resident)
    income_2026 = calculate_person_income(person, apply_2026_growth=True, is_resident=True)
    expected_2026 = expected_base * 1.056
    
    assert abs(income_2026 - expected_2026) < 0.01, \
        f"2026 income should be {expected_2026}, got {income_2026}"
    
    print(f"Projected income (2026): ${income_2026:,.0f}")
    print(f"Growth: {(income_2026 / income_base - 1):.2%}")
    
    print("\n✅ Person income calculation with growth works correctly")


def test_income_distribution_shift():
    """Test how income distribution shifts with growth."""
    print("\n" + "="*80)
    print("TEST 4: Income Distribution Shift")
    print("="*80)
    
    # Create sample income distribution
    np.random.seed(42)
    n_samples = 10000
    
    # Log-normal distribution (realistic income distribution)
    base_incomes = np.random.lognormal(mean=10.8, sigma=0.8, size=n_samples)
    
    # Apply growth
    projected_incomes = np.array([
        apply_income_growth(income, is_resident=True) 
        for income in base_incomes
    ])
    
    # Compare distributions
    print("\nBase Year (2023) Income Distribution:")
    print(f"  Mean: ${base_incomes.mean():,.0f}")
    print(f"  Median: ${np.median(base_incomes):,.0f}")
    print(f"  P10: ${np.percentile(base_incomes, 10):,.0f}")
    print(f"  P90: ${np.percentile(base_incomes, 90):,.0f}")
    
    print("\nProjected (2026) Income Distribution:")
    print(f"  Mean: ${projected_incomes.mean():,.0f}")
    print(f"  Median: ${np.median(projected_incomes):,.0f}")
    print(f"  P10: ${np.percentile(projected_incomes, 10):,.0f}")
    print(f"  P90: ${np.percentile(projected_incomes, 90):,.0f}")
    
    # Verify uniform scaling
    mean_growth = projected_incomes.mean() / base_incomes.mean()
    median_growth = np.median(projected_incomes) / np.median(base_incomes)
    
    print(f"\nGrowth Factors:")
    print(f"  Mean: {mean_growth:.4f} (expected: 1.056)")
    print(f"  Median: {median_growth:.4f} (expected: 1.056)")
    
    assert abs(mean_growth - 1.056) < 0.001, "Mean should grow by 1.056"
    assert abs(median_growth - 1.056) < 0.001, "Median should grow by 1.056"
    
    print("\n✅ Income distribution shifts correctly")


def test_edge_cases():
    """Test edge cases."""
    print("\n" + "="*80)
    print("TEST 5: Edge Cases")
    print("="*80)
    
    # Zero income
    zero_income = apply_income_growth(0, is_resident=True)
    assert zero_income == 0, "Zero income should remain zero"
    print("✅ Zero income handled correctly")
    
    # Negative income (possible with losses)
    negative_income = apply_income_growth(-10000, is_resident=True)
    expected_negative = -10000 * 1.056
    assert abs(negative_income - expected_negative) < 0.01, "Negative income should scale"
    print("✅ Negative income handled correctly")
    
    # Very large income
    large_income = apply_income_growth(10_000_000, is_resident=True)
    expected_large = 10_000_000 * 1.056
    assert abs(large_income - expected_large) < 1, "Large income should scale"
    print("✅ Large income handled correctly")
    
    print("\n✅ All edge cases handled correctly")


def compare_resident_vs_nonresident():
    """Compare resident vs nonresident growth."""
    print("\n" + "="*80)
    print("COMPARISON: Resident vs Nonresident Growth")
    print("="*80)
    
    test_incomes = [25000, 50000, 75000, 100000, 150000, 200000]
    
    print(f"\n{'Base Income':<15} {'Resident 2026':<20} {'Nonresident 2026':<20} {'Difference':<15}")
    print("-" * 75)
    
    for income in test_incomes:
        resident_2026 = apply_income_growth(income, is_resident=True)
        nonresident_2026 = apply_income_growth(income, is_resident=False)
        diff = resident_2026 - nonresident_2026
        
        print(f"${income:>12,}   ${resident_2026:>15,.0f}   ${nonresident_2026:>15,.0f}   ${diff:>12,.0f}")
    
    print("\nKey Insight:")
    print(f"  Residents grow slightly faster (5.6%) than nonresidents (5.3%)")
    print(f"  Difference: {(RESIDENT_GROWTH.real_growth - NONRESIDENT_GROWTH.real_growth):.4f}")
    print(f"  For $100k income, difference is ${100000 * (RESIDENT_GROWTH.real_growth - NONRESIDENT_GROWTH.real_growth):,.0f}")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("INCOME GROWTH ADJUSTMENT TESTS")
    print("="*80)
    
    # Log growth factors
    log_growth_factors()
    
    # Validate growth factors
    print("\n")
    validate_growth_factors()
    
    # Run tests
    test_growth_factor_calculations()
    test_apply_income_growth()
    test_calculate_person_income()
    test_income_distribution_shift()
    test_edge_cases()
    compare_resident_vs_nonresident()
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED ✅")
    print("="*80)
    print("\nIncome growth adjustments are working correctly!")
    print("Ready to use in full population modeling.")


if __name__ == '__main__':
    main()
