"""
Calculate Taxable Income for comparison with DOTAX SOI tables.

SOI tables use TAXABLE INCOME (after standard deduction), not total income.
This module converts PUMS total income → AGI → Taxable Income to match SOI definitions.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

# 2022 Standard Deduction amounts (matching SOI tax year)
STANDARD_DEDUCTION_2022 = {
    'single': 12950,
    'married_filing_jointly': 25900,
    'married_filing_separately': 12950,
    'head_of_household': 19400,
}

# 2024 Standard Deduction amounts
STANDARD_DEDUCTION_2024 = {
    'single': 14600,
    'married_filing_jointly': 29200,
    'married_filing_separately': 14600,
    'head_of_household': 21900,
}


class TaxableIncomeCalculator:
    """
    Calculate taxable income from PUMS total income to match SOI definitions.
    
    Process:
    1. Start with PUMS total income (all sources)
    2. Subtract AGI adjustments (IRA, student loans, etc.) → AGI
    3. Subtract standard deduction → Taxable Income
    
    This matches the income brackets used in SOI Tables 13A, 13B, 13C.
    """
    
    def __init__(self, tax_year: int = 2022):
        """
        Initialize calculator for a specific tax year.
        
        Args:
            tax_year: Tax year for standard deduction amounts (2022 or 2024)
        """
        self.tax_year = tax_year
        
        if tax_year == 2022:
            self.standard_deductions = STANDARD_DEDUCTION_2022
        elif tax_year == 2024:
            self.standard_deductions = STANDARD_DEDUCTION_2024
        else:
            logger.warning(f"Unknown tax year {tax_year}, using 2022 deductions")
            self.standard_deductions = STANDARD_DEDUCTION_2022
    
    def get_standard_deduction(self, filing_status: str) -> float:
        """Get standard deduction for filing status."""
        status = filing_status.lower().strip()
        return self.standard_deductions.get(status, self.standard_deductions['single'])
    
    def calculate_agi(self, total_income: float, 
                     age: Optional[int] = None,
                     filing_status: str = 'single',
                     has_self_employment: bool = False) -> tuple[float, float]:
        """
        Calculate AGI from total income.
        
        Args:
            total_income: PUMS total income
            age: Age of primary filer
            filing_status: Filing status
            has_self_employment: Whether filer has self-employment income
            
        Returns:
            Tuple of (AGI, total_adjustments)
        """
        # Import here to avoid circular dependency
        from tax.adjustments.agi_adjustments import AGIAdjustmentEstimator
        
        estimator = AGIAdjustmentEstimator()
        adjustments = estimator.estimate_total_adjustments(
            total_income, age, filing_status, has_self_employment
        )
        
        agi = total_income - adjustments['total']
        return max(0, agi), adjustments['total']
    
    def calculate_taxable_income(self, 
                                total_income: float,
                                filing_status: str,
                                age: Optional[int] = None,
                                has_self_employment: bool = False,
                                apply_agi_adjustments: bool = True) -> Dict[str, float]:
        """
        Calculate taxable income from PUMS total income.
        
        Args:
            total_income: PUMS total income (all sources)
            filing_status: Filing status
            age: Age of primary filer
            has_self_employment: Whether filer has SE income
            apply_agi_adjustments: Whether to apply AGI adjustments
            
        Returns:
            Dictionary with breakdown:
            - total_income: Starting PUMS income
            - agi_adjustments: Adjustments to income (IRA, student loans, etc.)
            - agi: Adjusted Gross Income
            - standard_deduction: Standard deduction amount
            - taxable_income: Final taxable income (for SOI comparison)
        """
        result = {
            'total_income': total_income,
            'agi_adjustments': 0.0,
            'agi': total_income,
            'standard_deduction': self.get_standard_deduction(filing_status),
            'taxable_income': 0.0
        }
        
        # Step 1: Calculate AGI (if requested)
        if apply_agi_adjustments:
            agi, adjustments = self.calculate_agi(
                total_income, age, filing_status, has_self_employment
            )
            result['agi'] = agi
            result['agi_adjustments'] = adjustments
        
        # Step 2: Subtract standard deduction to get taxable income
        taxable = result['agi'] - result['standard_deduction']
        result['taxable_income'] = max(0, taxable)
        
        return result
    
    def apply_to_tax_units(self, 
                          tax_units: pd.DataFrame,
                          income_col: str = 'income',
                          filing_status_col: str = 'filing_status',
                          age_col: Optional[str] = None,
                          apply_agi_adjustments: bool = True) -> pd.DataFrame:
        """
        Apply taxable income calculation to a DataFrame of tax units.
        
        Args:
            tax_units: DataFrame with tax units
            income_col: Column name for total income
            filing_status_col: Column name for filing status
            age_col: Column name for age (optional)
            apply_agi_adjustments: Whether to apply AGI adjustments
            
        Returns:
            DataFrame with added columns:
            - agi: Adjusted Gross Income
            - agi_adjustments: Total AGI adjustments
            - standard_deduction: Standard deduction amount
            - taxable_income: Taxable income (for SOI comparison)
        """
        result_df = tax_units.copy()
        
        logger.info(f"Calculating taxable income for {len(tax_units)} tax units")
        logger.info(f"Applying AGI adjustments: {apply_agi_adjustments}")
        
        # Calculate for each tax unit
        results = []
        for idx, row in tax_units.iterrows():
            total_income = row[income_col]
            filing_status = row[filing_status_col]
            age = row[age_col] if age_col and age_col in tax_units.columns else None
            
            # Detect self-employment (simple heuristic)
            has_se = False
            if 'has_self_employment' in row:
                has_se = row['has_self_employment']
            
            calc_result = self.calculate_taxable_income(
                total_income, filing_status, age, has_se, apply_agi_adjustments
            )
            results.append(calc_result)
        
        # Add columns to dataframe
        result_df['agi'] = [r['agi'] for r in results]
        result_df['agi_adjustments'] = [r['agi_adjustments'] for r in results]
        result_df['standard_deduction'] = [r['standard_deduction'] for r in results]
        result_df['taxable_income'] = [r['taxable_income'] for r in results]
        
        # Log statistics
        avg_total = result_df[income_col].mean()
        avg_agi = result_df['agi'].mean()
        avg_taxable = result_df['taxable_income'].mean()
        
        logger.info(f"Average total income: ${avg_total:,.0f}")
        logger.info(f"Average AGI: ${avg_agi:,.0f} ({(avg_agi/avg_total)*100:.1f}%)")
        logger.info(f"Average taxable income: ${avg_taxable:,.0f} ({(avg_taxable/avg_total)*100:.1f}%)")
        logger.info(f"Average reduction: ${avg_total - avg_taxable:,.0f} ({((avg_total-avg_taxable)/avg_total)*100:.1f}%)")
        
        return result_df


def calculate_taxable_income_simple(total_income: float, 
                                   filing_status: str,
                                   tax_year: int = 2022,
                                   apply_agi_adjustments: bool = True) -> float:
    """
    Simple function to calculate taxable income.
    
    Args:
        total_income: PUMS total income
        filing_status: Filing status
        tax_year: Tax year (2022 or 2024)
        apply_agi_adjustments: Whether to apply AGI adjustments
        
    Returns:
        Taxable income for SOI comparison
    """
    calculator = TaxableIncomeCalculator(tax_year)
    result = calculator.calculate_taxable_income(
        total_income, filing_status, apply_agi_adjustments=apply_agi_adjustments
    )
    return result['taxable_income']


if __name__ == '__main__':
    # Test examples
    calculator = TaxableIncomeCalculator(tax_year=2022)
    
    test_cases = [
        (30000, 'single', None),
        (75000, 'married_filing_jointly', 35),
        (150000, 'head_of_household', 40),
        (10000, 'single', 25),  # Low income - should be $0 taxable
    ]
    
    print("="*80)
    print("TAXABLE INCOME CALCULATION EXAMPLES (2022 Tax Year)")
    print("="*80)
    
    for total_income, status, age in test_cases:
        result = calculator.calculate_taxable_income(
            total_income, status, age, apply_agi_adjustments=True
        )
        
        print(f"\nFiling Status: {status}")
        print(f"Total Income:        ${result['total_income']:>12,.0f}")
        print(f"AGI Adjustments:    -${result['agi_adjustments']:>12,.0f}")
        print(f"AGI:                 ${result['agi']:>12,.0f}")
        print(f"Standard Deduction: -${result['standard_deduction']:>12,.0f}")
        print(f"Taxable Income:      ${result['taxable_income']:>12,.0f}")
        print(f"Total Reduction:     ${total_income - result['taxable_income']:>12,.0f} ({((total_income - result['taxable_income'])/total_income)*100:.1f}%)")
