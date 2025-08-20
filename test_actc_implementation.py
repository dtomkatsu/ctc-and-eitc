#!/usr/bin/env python3
"""
Test the ACTC (Additional Child Tax Credit) implementation.
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from src.tax.credits.ctc import calculate_ctc, _calculate_actc, CTCParameters

def test_actc_scenarios():
    """Test various ACTC scenarios."""
    
    print("Testing ACTC Implementation")
    print("=" * 50)
    
    # Test cases with different earned income levels
    test_cases = [
        {
            'name': 'Low income - no ACTC',
            'earned_income': 2000,
            'total_income': 2000,
            'num_children': 1,
            'expected_actc': 0
        },
        {
            'name': 'At threshold - no ACTC',
            'earned_income': 2500,
            'total_income': 2500,
            'num_children': 1,
            'expected_actc': 0
        },
        {
            'name': 'Just above threshold',
            'earned_income': 3000,
            'total_income': 3000,
            'num_children': 1,
            'expected_actc': 75  # 15% of $500
        },
        {
            'name': 'Moderate income - 1 child',
            'earned_income': 15000,
            'total_income': 15000,
            'num_children': 1,
            'expected_actc': 1600  # Max $1,600 per child
        },
        {
            'name': 'Moderate income - 2 children',
            'earned_income': 15000,
            'total_income': 15000,
            'num_children': 2,
            'expected_actc': 1875  # 15% of $12,500 = $1,875
        },
        {
            'name': 'High income - 2 children at max',
            'earned_income': 25000,
            'total_income': 25000,
            'num_children': 2,
            'expected_actc': 3200  # Max $1,600 * 2 children
        }
    ]
    
    params = CTCParameters()
    
    for case in test_cases:
        print(f"\nTest: {case['name']}")
        print(f"  Earned Income: ${case['earned_income']:,}")
        print(f"  Children: {case['num_children']}")
        
        # Test ACTC calculation directly
        actc_amount = _calculate_actc(case['earned_income'], case['num_children'], params)
        print(f"  ACTC Amount: ${actc_amount:,.0f}")
        print(f"  Expected: ${case['expected_actc']:,.0f}")
        
        # Create a tax unit for full CTC calculation
        tax_unit = {
            'filing_status': 'single',
            'income': case['total_income'],
            'earned_income': case['earned_income'],
            'dependents': [
                {'age': 10, 'relationship': '22', 'citizenship': '1'}
                for _ in range(case['num_children'])
            ]
        }
        
        # Calculate full CTC
        ctc_result = calculate_ctc(tax_unit)
        print(f"  Total CTC: ${ctc_result['ctc_total']:,.0f}")
        print(f"  Refundable: ${ctc_result['ctc_refundable']:,.0f}")
        print(f"  Non-refundable: ${ctc_result['ctc_nonrefundable']:,.0f}")
        
        # Check if ACTC matches expected
        if abs(actc_amount - case['expected_actc']) < 1:
            print("  ✓ ACTC calculation correct")
        else:
            print(f"  ✗ ACTC calculation incorrect (got {actc_amount}, expected {case['expected_actc']})")

def test_high_income_phaseout():
    """Test CTC with high income phaseout."""
    
    print("\n" + "=" * 50)
    print("Testing High Income Phaseout")
    print("=" * 50)
    
    # Test high income that triggers phaseout
    tax_unit = {
        'filing_status': 'single',
        'income': 220000,  # $20k over $200k threshold
        'earned_income': 220000,
        'dependents': [
            {'age': 10, 'relationship': '22', 'citizenship': '1'}
        ]
    }
    
    ctc_result = calculate_ctc(tax_unit)
    
    print(f"Income: ${tax_unit['income']:,}")
    print(f"Base CTC: $2,000")
    print(f"Phaseout: ${20 * 50} (20 increments of $50)")
    print(f"Expected CTC after phaseout: ${2000 - 1000}")
    print(f"Actual CTC: ${ctc_result['ctc_total']:,.0f}")
    print(f"Refundable portion: ${ctc_result['ctc_refundable']:,.0f}")

if __name__ == "__main__":
    test_actc_scenarios()
    test_high_income_phaseout()
