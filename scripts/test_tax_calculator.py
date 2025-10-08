"""
Quick test of Hawaii tax calculator to verify it works correctly
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tax.brackets import load_tax_data


def test_basic_calculation():
    """Test basic tax calculation"""
    print("Testing Hawaii Tax Calculator...")
    print("="*60)
    
    try:
        # Load calculator
        calculator = load_tax_data()
        print("✓ Successfully loaded tax data")
        
        # Test calculation for single filer with $50,000 income in 2024
        result = calculator.calculate_tax(50000, 2024, 'single')
        
        print("\nTest Case: Single filer, $50,000 income, 2024")
        print("-"*60)
        print(f"Gross Income:        ${result['gross_income']:,.2f}")
        print(f"Standard Deduction:  ${result['standard_deduction']:,.2f}")
        print(f"Taxable Income:      ${result['taxable_income']:,.2f}")
        print(f"Tax Liability:       ${result['tax_liability']:,.2f}")
        print(f"Effective Rate:      {result['effective_rate']:.2f}%")
        print(f"Marginal Rate:       {result['marginal_rate']:.2f}%")
        
        # Verify results are reasonable
        assert result['gross_income'] == 50000, "Gross income mismatch"
        assert result['standard_deduction'] > 0, "Standard deduction should be positive"
        assert result['taxable_income'] > 0, "Taxable income should be positive"
        assert result['tax_liability'] > 0, "Tax liability should be positive"
        assert 0 < result['effective_rate'] < 100, "Effective rate out of range"
        assert 0 < result['marginal_rate'] < 100, "Marginal rate out of range"
        
        print("\n✓ All assertions passed!")
        print("="*60)
        print("Tax calculator is working correctly!\n")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_basic_calculation()
    sys.exit(0 if success else 1)
