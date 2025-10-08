"""
Adjust PUMS total income to approximate Adjusted Gross Income (AGI)

Based on IRS SOI data for Hawaii (2022), this module estimates the adjustments
that convert total income to AGI.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


class AGIAdjustmentEstimator:
    """
    Estimates AGI adjustments based on SOI data patterns.
    
    From SOI 2022 Hawaii data:
    - Total Income: $57.07B
    - AGI: $56.50B
    - Total Adjustments: $563M (0.99% of total income)
    
    Major adjustments:
    - IRA contributions: $59.9M (10.6% of adjustments)
    - Self-employed health insurance: $127.8M (22.7%)
    - Self-employed retirement: $42.7M (7.6%)
    - Student loan interest: $10.0M (1.8%)
    - Educator expenses: $4.2M (0.7%)
    """
    
    # SOI-based adjustment rates (as % of income)
    ADJUSTMENT_RATES = {
        'ira_contribution': {
            'base_rate': 0.001,  # 0.1% of income on average
            'income_threshold': 50000,
            'max_rate': 0.015,  # Up to 1.5% for middle income
        },
        'self_employed_health': {
            'base_rate': 0.002,  # 0.2% of income
            'se_multiplier': 3.0,  # 3x for self-employed
        },
        'self_employed_retirement': {
            'base_rate': 0.0007,  # 0.07% of income
            'se_multiplier': 5.0,  # 5x for self-employed
        },
        'student_loan_interest': {
            'base_rate': 0.0002,  # 0.02% of income
            'age_factor': True,  # Higher for younger filers
            'income_cap': 145000,  # Phase out above this
        },
        'educator_expenses': {
            'flat_amount': 300,  # Average per educator
            'educator_rate': 0.05,  # 5% of filers are educators
        }
    }
    
    def __init__(self):
        """Initialize with SOI-based rates"""
        self.total_adjustment_rate = 0.0099  # 0.99% from SOI data
    
    def estimate_ira_contribution(self, income: float, age: Optional[int] = None,
                                  filing_status: str = 'single') -> float:
        """
        Estimate IRA contribution deduction.
        
        Higher for middle-income filers, phases out at high income.
        """
        if income < 20000:
            return 0
        
        rates = self.ADJUSTMENT_RATES['ira_contribution']
        
        # Peak contribution rate for middle income
        if 50000 <= income <= 120000:
            rate = rates['max_rate']
        elif income < 50000:
            # Scale up from base to max
            rate = rates['base_rate'] + (rates['max_rate'] - rates['base_rate']) * (income - 20000) / 30000
        else:
            # Phase out for high income
            rate = rates['max_rate'] * max(0, (200000 - income) / 80000)
        
        return income * rate
    
    def estimate_se_health_insurance(self, income: float, 
                                     is_self_employed: bool = False) -> float:
        """Estimate self-employed health insurance deduction"""
        rates = self.ADJUSTMENT_RATES['self_employed_health']
        base = income * rates['base_rate']
        
        if is_self_employed:
            return base * rates['se_multiplier']
        return base
    
    def estimate_se_retirement(self, income: float,
                               is_self_employed: bool = False) -> float:
        """Estimate self-employed retirement plan deduction"""
        rates = self.ADJUSTMENT_RATES['self_employed_retirement']
        base = income * rates['base_rate']
        
        if is_self_employed:
            return base * rates['se_multiplier']
        return base
    
    def estimate_student_loan_interest(self, income: float,
                                       age: Optional[int] = None) -> float:
        """Estimate student loan interest deduction"""
        rates = self.ADJUSTMENT_RATES['student_loan_interest']
        
        # Phase out above income cap
        if income > rates['income_cap']:
            return 0
        
        # Higher for younger filers
        base = income * rates['base_rate']
        
        if age and age < 35:
            base *= 2.0  # Double for younger filers
        elif age and age < 45:
            base *= 1.5
        
        return min(base, 2500)  # Cap at $2,500
    
    def estimate_educator_expenses(self, income: float) -> float:
        """Estimate educator expenses deduction"""
        rates = self.ADJUSTMENT_RATES['educator_expenses']
        
        # Assume 5% of filers are educators
        if np.random.random() < rates['educator_rate']:
            return rates['flat_amount']
        return 0
    
    def estimate_total_adjustments(self, income: float, 
                                   age: Optional[int] = None,
                                   filing_status: str = 'single',
                                   is_self_employed: bool = False) -> Dict[str, float]:
        """
        Estimate all adjustments to income.
        
        Returns:
            Dictionary with individual adjustments and total
        """
        adjustments = {
            'ira_contribution': self.estimate_ira_contribution(income, age, filing_status),
            'se_health_insurance': self.estimate_se_health_insurance(income, is_self_employed),
            'se_retirement': self.estimate_se_retirement(income, is_self_employed),
            'student_loan_interest': self.estimate_student_loan_interest(income, age),
            'educator_expenses': self.estimate_educator_expenses(income),
        }
        
        adjustments['total'] = sum(adjustments.values())
        
        return adjustments
    
    def calculate_agi(self, total_income: float,
                     age: Optional[int] = None,
                     filing_status: str = 'single',
                     is_self_employed: bool = False) -> float:
        """
        Calculate AGI from total income.
        
        Args:
            total_income: Total income from all sources
            age: Age of primary filer
            filing_status: Filing status
            is_self_employed: Whether filer is self-employed
            
        Returns:
            Adjusted Gross Income (AGI)
        """
        adjustments = self.estimate_total_adjustments(
            total_income, age, filing_status, is_self_employed
        )
        
        agi = total_income - adjustments['total']
        return max(0, agi)


def estimate_agi_from_total_income(total_income: float,
                                   age: Optional[int] = None,
                                   filing_status: str = 'single',
                                   is_self_employed: bool = False) -> float:
    """
    Convenience function to estimate AGI from total income.
    
    Args:
        total_income: Total income from all sources
        age: Age of primary filer (optional)
        filing_status: Filing status
        is_self_employed: Whether filer is self-employed
        
    Returns:
        Estimated Adjusted Gross Income (AGI)
    """
    estimator = AGIAdjustmentEstimator()
    return estimator.calculate_agi(total_income, age, filing_status, is_self_employed)


def apply_agi_adjustments_to_dataframe(df: pd.DataFrame,
                                       income_col: str = 'income',
                                       age_col: Optional[str] = None,
                                       filing_status_col: str = 'filing_status') -> pd.DataFrame:
    """
    Apply AGI adjustments to a DataFrame of tax units.
    
    Args:
        df: DataFrame with tax units
        income_col: Name of income column
        age_col: Name of age column (optional)
        filing_status_col: Name of filing status column
        
    Returns:
        DataFrame with added AGI column
    """
    result_df = df.copy()
    estimator = AGIAdjustmentEstimator()
    
    agis = []
    for _, row in df.iterrows():
        income = row[income_col]
        age = row[age_col] if age_col and age_col in df.columns else None
        filing_status = row[filing_status_col]
        
        # Simple heuristic: assume 10% of filers are self-employed
        is_self_employed = income > 50000 and np.random.random() < 0.10
        
        agi = estimator.calculate_agi(income, age, filing_status, is_self_employed)
        agis.append(agi)
    
    result_df['agi'] = agis
    result_df['agi_adjustments'] = result_df[income_col] - result_df['agi']
    
    return result_df


if __name__ == '__main__':
    # Example usage
    estimator = AGIAdjustmentEstimator()
    
    # Test cases
    test_cases = [
        (30000, 25, 'single', False),
        (75000, 35, 'married_filing_jointly', False),
        (100000, 45, 'single', True),  # Self-employed
        (150000, 50, 'married_filing_jointly', False),
    ]
    
    print("AGI Adjustment Examples:")
    print("="*80)
    
    for income, age, status, se in test_cases:
        adjustments = estimator.estimate_total_adjustments(income, age, status, se)
        agi = estimator.calculate_agi(income, age, status, se)
        
        print(f"\nIncome: ${income:,} | Age: {age} | Status: {status} | SE: {se}")
        print(f"  IRA:              ${adjustments['ira_contribution']:>8,.2f}")
        print(f"  SE Health Ins:    ${adjustments['se_health_insurance']:>8,.2f}")
        print(f"  SE Retirement:    ${adjustments['se_retirement']:>8,.2f}")
        print(f"  Student Loan:     ${adjustments['student_loan_interest']:>8,.2f}")
        print(f"  Educator:         ${adjustments['educator_expenses']:>8,.2f}")
        print(f"  Total Adj:        ${adjustments['total']:>8,.2f}")
        print(f"  AGI:              ${agi:>8,.2f}")
        print(f"  Reduction:        {(1 - agi/income)*100:>7.2f}%")
