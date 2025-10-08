"""
Hawaii State Tax Credits

Based on Hawaii Department of Taxation data and common state credits.
Major credits include:
- Food/Excise Tax Credit (refundable)
- Renewable Energy Technologies Credit
- Child and Dependent Care Credit
- Low-Income Household Renters Credit
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


class HawaiiTaxCredits:
    """
    Calculate Hawaii state tax credits.
    
    Note: These are STATE credits, separate from federal credits.
    """
    
    def __init__(self, year: int = 2024):
        """Initialize with tax year"""
        self.year = year
    
    def food_excise_tax_credit(self, agi: float, filing_status: str,
                               num_dependents: int = 0) -> float:
        """
        Hawaii Food/Excise Tax Credit (refundable).
        
        This is the largest state credit, providing relief for low-income families
        from the general excise tax burden.
        
        2024 amounts (approximate):
        - Base: $110 per exemption
        - Phase out starts at higher income levels
        """
        # Income thresholds by filing status (2024)
        phase_out_start = {
            'single': 30000,
            'married_filing_jointly': 50000,
            'head_of_household': 40000,
            'married_filing_separate': 25000,
        }
        
        phase_out_end = {
            'single': 50000,
            'married_filing_jointly': 70000,
            'head_of_household': 60000,
            'married_filing_separate': 35000,
        }
        
        # Get thresholds for this filing status
        start = phase_out_start.get(filing_status, 30000)
        end = phase_out_end.get(filing_status, 50000)
        
        # Base credit per exemption
        base_credit_per_exemption = 110
        
        # Number of exemptions (filer + dependents)
        num_exemptions = 1 + num_dependents
        if filing_status == 'married_filing_jointly':
            num_exemptions += 1  # Add spouse
        
        # Calculate base credit
        base_credit = base_credit_per_exemption * num_exemptions
        
        # Apply phase out
        if agi <= start:
            return base_credit
        elif agi >= end:
            return 0
        else:
            # Linear phase out
            phase_out_pct = (agi - start) / (end - start)
            return base_credit * (1 - phase_out_pct)
    
    def renewable_energy_credit(self, agi: float, filing_status: str) -> float:
        """
        Renewable Energy Technologies Income Tax Credit.
        
        For solar, wind, and other renewable energy installations.
        This is highly variable based on actual installations.
        
        Estimate: ~2% of middle-to-high income filers claim this
        Average credit: ~$2,000
        """
        # Only middle-to-high income homeowners typically claim
        if agi < 50000 or agi > 300000:
            return 0
        
        # Random chance of claiming (2% of eligible filers)
        if np.random.random() < 0.02:
            # Average credit amount varies by income
            if agi < 100000:
                return 1500
            elif agi < 200000:
                return 2500
            else:
                return 3500
        
        return 0
    
    def child_dependent_care_credit(self, agi: float, filing_status: str,
                                    num_dependents: int = 0) -> float:
        """
        Hawaii Child and Dependent Care Tax Credit.
        
        Based on federal credit but with Hawaii-specific rules.
        Typically 20-35% of federal credit amount.
        """
        if num_dependents == 0:
            return 0
        
        # Income thresholds
        if agi > 100000:
            return 0
        
        # Estimate based on number of dependents and income
        if num_dependents >= 1:
            # Average credit ranges from $200-$800
            if agi < 30000:
                base = 800
            elif agi < 60000:
                base = 500
            else:
                base = 300
            
            # Additional for multiple dependents
            if num_dependents >= 2:
                base *= 1.5
            
            return base
        
        return 0
    
    def low_income_renters_credit(self, agi: float, filing_status: str) -> float:
        """
        Low-Income Household Renters Credit (refundable).
        
        For renters with low income.
        """
        # Income thresholds
        max_income = {
            'single': 30000,
            'married_filing_jointly': 40000,
            'head_of_household': 35000,
            'married_filing_separate': 20000,
        }
        
        threshold = max_income.get(filing_status, 30000)
        
        if agi > threshold:
            return 0
        
        # Assume 30% of low-income filers are renters
        if np.random.random() < 0.30:
            # Credit amount based on income
            if agi < 15000:
                return 150
            elif agi < 25000:
                return 100
            else:
                return 50
        
        return 0
    
    def calculate_total_credits(self, agi: float, filing_status: str,
                               num_dependents: int = 0,
                               tax_before_credits: float = 0) -> Dict[str, float]:
        """
        Calculate all Hawaii state tax credits.
        
        Args:
            agi: Adjusted Gross Income
            filing_status: Filing status
            num_dependents: Number of dependents
            tax_before_credits: Tax liability before credits
            
        Returns:
            Dictionary with individual credits and totals
        """
        credits = {
            'food_excise': self.food_excise_tax_credit(agi, filing_status, num_dependents),
            'renewable_energy': self.renewable_energy_credit(agi, filing_status),
            'child_care': self.child_dependent_care_credit(agi, filing_status, num_dependents),
            'renters': self.low_income_renters_credit(agi, filing_status),
        }
        
        # Separate refundable and non-refundable
        refundable_credits = credits['food_excise'] + credits['renters']
        nonrefundable_credits = credits['renewable_energy'] + credits['child_care']
        
        # Non-refundable credits limited to tax liability
        nonrefundable_credits = min(nonrefundable_credits, tax_before_credits)
        
        credits['total_refundable'] = refundable_credits
        credits['total_nonrefundable'] = nonrefundable_credits
        credits['total'] = refundable_credits + nonrefundable_credits
        
        return credits


def calculate_hawaii_credits(agi: float, filing_status: str,
                             num_dependents: int = 0,
                             tax_before_credits: float = 0,
                             year: int = 2024) -> Dict[str, float]:
    """
    Convenience function to calculate Hawaii state tax credits.
    
    Args:
        agi: Adjusted Gross Income
        filing_status: Filing status
        num_dependents: Number of dependents
        tax_before_credits: Tax liability before credits
        year: Tax year
        
    Returns:
        Dictionary with credit amounts
    """
    calculator = HawaiiTaxCredits(year)
    return calculator.calculate_total_credits(agi, filing_status, num_dependents, tax_before_credits)


def apply_credits_to_dataframe(df: pd.DataFrame,
                               agi_col: str = 'agi',
                               filing_status_col: str = 'filing_status',
                               dependents_col: str = 'num_dependents',
                               tax_col: str = 'tax_liability') -> pd.DataFrame:
    """
    Apply Hawaii tax credits to a DataFrame of tax units.
    
    Args:
        df: DataFrame with tax units
        agi_col: Name of AGI column
        filing_status_col: Name of filing status column
        dependents_col: Name of dependents column
        tax_col: Name of tax liability column
        
    Returns:
        DataFrame with added credit columns
    """
    result_df = df.copy()
    calculator = HawaiiTaxCredits()
    
    credits_list = []
    for _, row in df.iterrows():
        agi = row[agi_col]
        filing_status = row[filing_status_col]
        num_dependents = row.get(dependents_col, 0)
        tax_before_credits = row[tax_col]
        
        credits = calculator.calculate_total_credits(
            agi, filing_status, num_dependents, tax_before_credits
        )
        credits_list.append(credits)
    
    # Add credit columns
    credits_df = pd.DataFrame(credits_list)
    for col in credits_df.columns:
        result_df[f'hi_credit_{col}'] = credits_df[col]
    
    # Calculate net tax after credits
    result_df['hi_tax_after_credits'] = (
        result_df[tax_col] - 
        result_df['hi_credit_total_nonrefundable'] - 
        result_df['hi_credit_total_refundable']
    )
    
    return result_df


if __name__ == '__main__':
    # Example usage
    calculator = HawaiiTaxCredits(2024)
    
    # Test cases
    test_cases = [
        (20000, 'single', 0, 500, "Low-income single, no dependents"),
        (35000, 'head_of_household', 2, 1500, "HoH with 2 kids"),
        (75000, 'married_filing_jointly', 1, 3000, "Joint filers, 1 child"),
        (150000, 'married_filing_jointly', 0, 8000, "High-income joint, no kids"),
    ]
    
    print("Hawaii State Tax Credits Examples:")
    print("="*80)
    
    for agi, status, deps, tax, desc in test_cases:
        credits = calculator.calculate_total_credits(agi, status, deps, tax)
        
        print(f"\n{desc}")
        print(f"  AGI: ${agi:,} | Status: {status} | Dependents: {deps} | Tax: ${tax:,}")
        print(f"  Food/Excise:      ${credits['food_excise']:>8,.2f}")
        print(f"  Renewable Energy: ${credits['renewable_energy']:>8,.2f}")
        print(f"  Child Care:       ${credits['child_care']:>8,.2f}")
        print(f"  Renters:          ${credits['renters']:>8,.2f}")
        print(f"  Total Credits:    ${credits['total']:>8,.2f}")
        print(f"  Net Tax:          ${max(0, tax - credits['total']):>8,.2f}")
