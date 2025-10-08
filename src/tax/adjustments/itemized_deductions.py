"""
Itemized Deductions Estimation for Hawaii

Based on IRS SOI 2022 data:
- 12.1% of filers itemize (81,360 out of 674,660)
- Average itemized deduction: $34,592
- Total itemized deductions: $2.81B

Major categories:
- State & Local Taxes (SALT): $1.23B (85.7% of itemizers)
- Home Mortgage Interest: $1.09B (81.9% of itemizers)
- Charitable Contributions: $502M (72.7% of itemizers)
- Medical & Dental: $339M (17.1% of itemizers)
- Real Estate Taxes: $254M (85.7% of itemizers)
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


class ItemizedDeductionEstimator:
    """
    Estimate itemized deductions based on SOI patterns.
    
    From SOI 2022 Hawaii:
    - 12.1% of filers itemize
    - Itemizers are concentrated in higher income brackets
    - Average itemized deduction: $34,592
    """
    
    # SOI-based itemization rates by income
    ITEMIZATION_RATES = {
        'under_50k': 0.02,      # 2% itemize
        '50k_to_100k': 0.08,    # 8% itemize
        '100k_to_200k': 0.25,   # 25% itemize
        '200k_plus': 0.45,      # 45% itemize
    }
    
    # Average deductions by income level (from SOI patterns)
    AVG_DEDUCTIONS = {
        'under_50k': 15000,
        '50k_to_100k': 22000,
        '100k_to_200k': 35000,
        '200k_plus': 55000,
    }
    
    def __init__(self):
        """Initialize with SOI-based rates"""
        self.overall_itemization_rate = 0.121  # 12.1% from SOI
        self.avg_itemized_deduction = 34592
    
    def get_income_bracket(self, agi: float) -> str:
        """Determine income bracket for itemization estimation"""
        if agi < 50000:
            return 'under_50k'
        elif agi < 100000:
            return '50k_to_100k'
        elif agi < 200000:
            return '100k_to_200k'
        else:
            return '200k_plus'
    
    def will_itemize(self, agi: float, filing_status: str) -> bool:
        """
        Determine if filer will itemize based on income.
        
        Higher income filers are more likely to itemize.
        """
        bracket = self.get_income_bracket(agi)
        itemization_rate = self.ITEMIZATION_RATES[bracket]
        
        # Random selection based on probability
        return np.random.random() < itemization_rate
    
    def estimate_salt_deduction(self, agi: float, filing_status: str) -> float:
        """
        Estimate State and Local Tax (SALT) deduction.
        
        From SOI: 85.7% of itemizers claim SALT, avg $17,642
        SALT cap: $10,000 (federal limit applies)
        """
        # Base SALT as % of income
        if agi < 100000:
            salt = agi * 0.05  # ~5% of income
        elif agi < 200000:
            salt = agi * 0.06  # ~6% of income
        else:
            salt = agi * 0.07  # ~7% of income
        
        # Apply $10,000 federal cap
        return min(salt, 10000)
    
    def estimate_mortgage_interest(self, agi: float, filing_status: str) -> float:
        """
        Estimate home mortgage interest deduction.
        
        From SOI: 81.9% of itemizers claim mortgage interest, avg $16,381
        """
        # Only homeowners (estimate ~65% of itemizers)
        if np.random.random() > 0.65:
            return 0
        
        # Mortgage interest based on income (proxy for home value)
        if agi < 75000:
            return 8000
        elif agi < 150000:
            return 15000
        elif agi < 300000:
            return 22000
        else:
            return 30000
    
    def estimate_charitable_contributions(self, agi: float, filing_status: str) -> float:
        """
        Estimate charitable contribution deduction.
        
        From SOI: 72.7% of itemizers claim charitable, avg $8,488
        """
        # Charitable giving as % of income
        if agi < 75000:
            giving_rate = 0.015  # 1.5%
        elif agi < 150000:
            giving_rate = 0.025  # 2.5%
        elif agi < 300000:
            giving_rate = 0.035  # 3.5%
        else:
            giving_rate = 0.045  # 4.5%
        
        # 72.7% of itemizers give
        if np.random.random() < 0.727:
            return agi * giving_rate
        return 0
    
    def estimate_medical_deduction(self, agi: float, age: Optional[int] = None) -> float:
        """
        Estimate medical and dental expense deduction.
        
        From SOI: 17.1% of itemizers claim medical, avg $24,371
        Only expenses > 7.5% of AGI are deductible
        """
        # Medical deductions more common for elderly and lower income
        if age and age >= 65:
            claim_rate = 0.30
        else:
            claim_rate = 0.10
        
        if np.random.random() < claim_rate:
            # Medical expenses need to exceed 7.5% of AGI
            threshold = agi * 0.075
            
            # Estimate total medical expenses
            if age and age >= 65:
                medical_expenses = agi * 0.15  # 15% for elderly
            else:
                medical_expenses = agi * 0.10  # 10% for others
            
            # Only amount above threshold is deductible
            return max(0, medical_expenses - threshold)
        
        return 0
    
    def estimate_real_estate_taxes(self, agi: float, filing_status: str) -> float:
        """
        Estimate real estate tax deduction.
        
        From SOI: 85.7% of itemizers claim real estate taxes, avg $3,638
        Note: This is separate from SALT income taxes
        """
        # Property tax based on income (proxy for home value)
        if agi < 75000:
            return 2500
        elif agi < 150000:
            return 4000
        elif agi < 300000:
            return 6000
        else:
            return 8000
    
    def estimate_total_itemized_deductions(self, agi: float, 
                                          filing_status: str,
                                          age: Optional[int] = None) -> Dict[str, float]:
        """
        Estimate total itemized deductions.
        
        Returns:
            Dictionary with individual deductions and total
        """
        deductions = {
            'salt': self.estimate_salt_deduction(agi, filing_status),
            'mortgage_interest': self.estimate_mortgage_interest(agi, filing_status),
            'charitable': self.estimate_charitable_contributions(agi, filing_status),
            'medical': self.estimate_medical_deduction(agi, age),
            'real_estate_taxes': self.estimate_real_estate_taxes(agi, filing_status),
        }
        
        # Note: SALT cap of $10,000 includes both income and property taxes
        # Adjust if combined exceeds cap
        combined_salt = deductions['salt'] + deductions['real_estate_taxes']
        if combined_salt > 10000:
            # Proportionally reduce both
            reduction_factor = 10000 / combined_salt
            deductions['salt'] *= reduction_factor
            deductions['real_estate_taxes'] *= reduction_factor
        
        deductions['total'] = sum(deductions.values())
        
        return deductions
    
    def get_deduction(self, agi: float, filing_status: str,
                     standard_deduction: float,
                     age: Optional[int] = None) -> Dict[str, float]:
        """
        Determine whether to use standard or itemized deduction.
        
        Args:
            agi: Adjusted Gross Income
            filing_status: Filing status
            standard_deduction: Standard deduction amount
            age: Age of primary filer
            
        Returns:
            Dictionary with deduction info
        """
        # Determine if filer will itemize
        will_itemize = self.will_itemize(agi, filing_status)
        
        if will_itemize:
            # Calculate itemized deductions
            itemized = self.estimate_total_itemized_deductions(agi, filing_status, age)
            
            # Only itemize if it exceeds standard deduction
            if itemized['total'] > standard_deduction:
                return {
                    'type': 'itemized',
                    'amount': itemized['total'],
                    'standard_deduction': standard_deduction,
                    'benefit_over_standard': itemized['total'] - standard_deduction,
                    'details': itemized
                }
        
        # Use standard deduction
        return {
            'type': 'standard',
            'amount': standard_deduction,
            'standard_deduction': standard_deduction,
            'benefit_over_standard': 0,
            'details': {}
        }


def estimate_deduction(agi: float, filing_status: str,
                      standard_deduction: float,
                      age: Optional[int] = None) -> Dict[str, float]:
    """
    Convenience function to estimate deduction (standard or itemized).
    
    Args:
        agi: Adjusted Gross Income
        filing_status: Filing status
        standard_deduction: Standard deduction amount
        age: Age of primary filer
        
    Returns:
        Dictionary with deduction information
    """
    estimator = ItemizedDeductionEstimator()
    return estimator.get_deduction(agi, filing_status, standard_deduction, age)


if __name__ == '__main__':
    # Example usage
    estimator = ItemizedDeductionEstimator()
    
    # Test cases
    test_cases = [
        (40000, 'single', 4400, 30, "Low income single"),
        (85000, 'married_filing_jointly', 8800, 45, "Middle income joint"),
        (150000, 'married_filing_jointly', 8800, 50, "High income joint"),
        (300000, 'married_filing_jointly', 8800, 55, "Very high income joint"),
    ]
    
    print("Itemized Deduction Examples:")
    print("="*80)
    
    for agi, status, std_ded, age, desc in test_cases:
        print(f"\n{desc}")
        print(f"  AGI: ${agi:,} | Status: {status} | Std Deduction: ${std_ded:,}")
        
        # Run multiple times to show probability
        itemized_count = 0
        total_deduction = 0
        
        for _ in range(100):
            result = estimator.get_deduction(agi, status, std_ded, age)
            if result['type'] == 'itemized':
                itemized_count += 1
                total_deduction += result['amount']
        
        print(f"  Itemization Rate: {itemized_count}%")
        if itemized_count > 0:
            avg_itemized = total_deduction / itemized_count
            print(f"  Avg Itemized Deduction: ${avg_itemized:,.2f}")
            print(f"  Benefit over Standard: ${avg_itemized - std_ded:,.2f}")
