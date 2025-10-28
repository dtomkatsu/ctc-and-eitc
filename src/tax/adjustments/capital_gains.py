"""
Capital Gains Estimation for Hawaii

Based on DOTax SOI 2022 Table 21:
- Capital gains percentages by AGI bracket (residents)
- These represent the % of total taxable income from capital gains
- Used to create apples-to-apples comparison with DOTax benchmarks

Total capital gains: $2,995M from 43,443 filers (7.4% of total taxable income)
"""

import pandas as pd
import numpy as np
from typing import Optional


class CapitalGainsEstimator:
    """
    Estimate capital gains income based on DOTax Table 21.
    
    Table 21 shows the percentage of total taxable income that comes from
    net long-term capital gains by AGI bracket for Hawaii residents.
    """
    
    # Capital gains as % of taxable income by AGI bracket (from Table 21)
    CAPITAL_GAINS_RATES = {
        (0, 10000): 0.005,      # 0.5%
        (10000, 20000): 0.000,  # 0.0%
        (20000, 30000): 0.0003, # 0.03%
        (30000, 40000): 0.002,  # 0.2%
        (40000, 50000): 0.004,  # 0.4%
        (50000, 75000): 0.007,  # 0.7%
        (75000, 100000): 0.012, # 1.2%
        (100000, 150000): 0.018, # 1.8%
        (150000, 200000): 0.031, # 3.1%
        (200000, 300000): 0.059, # 5.9%
        (300000, 400000): 0.113, # 11.3%
        (400000, float('inf')): 0.209  # 20.9%
    }
    
    # Filer participation rates by bracket (from Table 21)
    # Precisely calibrated to match EXACT filer counts from Table 21
    PARTICIPATION_RATES = {
        (0, 10000): 0.000436,      # 34 filers
        (10000, 20000): 0.000226,  # 13 filers
        (20000, 30000): 0.003315,  # 234 filers
        (30000, 40000): 0.027265,  # 1,616 filers
        (40000, 50000): 0.030951,  # 2,045 filers
        (50000, 75000): 0.053789,  # 6,044 filers
        (75000, 100000): 0.095893, # 6,067 filers
        (100000, 150000): 0.134976, # 9,146 filers
        (150000, 200000): 0.226516, # 6,095 filers
        (200000, 300000): 0.289652, # 5,577 filers
        (300000, 400000): 0.405963, # 2,384 filers
        (400000, float('inf')): 0.514454  # 4,188 filers
    }
    
    def __init__(self):
        """Initialize capital gains estimator"""
        self.total_cap_gains_millions = 2995  # From Table 21
        self.total_filers_with_cap_gains = 43443  # From Table 21
        self.avg_cap_gains_per_filer = (self.total_cap_gains_millions * 1_000_000) / self.total_filers_with_cap_gains
        
        # Bracket-specific scaling factors to match Table 21 amounts per bracket
        # Redistributed to fix bracket-level errors while maintaining total accuracy
        self.bracket_scaling = {
            (0, 10000): 1900.0,     # $0.5M target (already accurate)
            (10000, 20000): 1.0,    # $0.0M target (already accurate)
            (20000, 30000): 1330.0, # $0.3M target (reduced from 1900)
            (30000, 40000): 52.5,   # $3.6M target (reduced 30% from 75.0)
            (40000, 50000): 34.3,   # $7.7M target (reduced 30% from 49.0)
            (50000, 75000): 23.8,   # $30.1M target (reduced 30% from 34.0)
            (75000, 100000): 14.7,  # $46.0M target (reduced 30% from 21.0)
            (100000, 150000): 9.4,  # $112.0M target (reduced 30% from 13.5)
            (150000, 200000): 5.6,  # $126.0M target (reduced 30% from 8.0)
            (200000, 300000): 4.5,  # $242.0M target (reduced 30% from 6.5)
            (300000, 400000): 3.7,  # $216.9M target (reduced 30% from 5.3)
            (400000, float('inf')): 5.3   # $2,210.3M target (increased 40% from 3.8)
        }
    
    def get_capital_gains_rate(self, agi: float) -> float:
        """
        Get capital gains rate for a given AGI.
        
        Args:
            agi: Adjusted Gross Income
            
        Returns:
            Capital gains as % of taxable income
        """
        for (min_agi, max_agi), rate in self.CAPITAL_GAINS_RATES.items():
            if min_agi <= agi < max_agi:
                return rate
        
        # Default to highest bracket if above all thresholds
        return 0.209
    
    def get_participation_rate(self, agi: float) -> float:
        """
        Get probability that a filer in this AGI bracket has capital gains.
        
        Args:
            agi: Adjusted Gross Income
            
        Returns:
            Probability of having capital gains (0-1)
        """
        for (min_agi, max_agi), rate in self.PARTICIPATION_RATES.items():
            if min_agi <= agi < max_agi:
                return rate
        
        # Default to highest bracket
        return 0.514454
    
    def estimate_capital_gains(self, agi: float, taxable_income: float,
                              include_random: bool = True) -> float:
        """
        Estimate capital gains for a tax unit.
        
        Args:
            agi: Adjusted Gross Income
            taxable_income: Taxable income after deductions
            include_random: If True, use participation rate to randomly assign cap gains
            
        Returns:
            Estimated capital gains amount
        """
        if taxable_income <= 0:
            return 0.0
        
        # Get capital gains rate for this bracket
        cap_gains_rate = self.get_capital_gains_rate(agi)
        
        # Determine if this filer has capital gains
        if include_random:
            participation_rate = self.get_participation_rate(agi)
            # Cap participation rate at 100%
            participation_rate = min(participation_rate, 1.0)
            if np.random.random() > participation_rate:
                return 0.0
        
        # Get bracket-specific scaling factor
        scaling_factor = 1.0
        for (min_agi, max_agi), factor in self.bracket_scaling.items():
            if min_agi <= agi < max_agi:
                scaling_factor = factor
                break
        
        # Estimate capital gains as % of taxable income with bracket-specific scaling
        cap_gains = taxable_income * cap_gains_rate * scaling_factor
        
        return cap_gains
    
    def estimate_agi_with_cap_gains(self, agi_without_cap_gains: float,
                                    include_random: bool = True) -> tuple[float, float]:
        """
        Estimate AGI including capital gains.
        
        Since Table 21 shows cap gains as % of taxable income, we need to
        work backwards to estimate the AGI that includes capital gains.
        
        Args:
            agi_without_cap_gains: AGI without capital gains (from PUMS)
            include_random: If True, use participation rate
            
        Returns:
            Tuple of (estimated_cap_gains, agi_with_cap_gains)
        """
        # Get capital gains rate for this bracket
        cap_gains_rate = self.get_capital_gains_rate(agi_without_cap_gains)
        
        # Determine if this filer has capital gains
        if include_random:
            participation_rate = self.get_participation_rate(agi_without_cap_gains)
            # Cap participation rate at 100%
            participation_rate = min(participation_rate, 1.0)
            if np.random.random() > participation_rate:
                return (0.0, agi_without_cap_gains)
        
        # Get bracket-specific scaling factor
        scaling_factor = 1.0
        for (min_agi, max_agi), factor in self.bracket_scaling.items():
            if min_agi <= agi_without_cap_gains < max_agi:
                scaling_factor = factor
                break
        
        # Estimate capital gains
        # Assume taxable income ≈ AGI * 0.85 (rough estimate after deductions)
        estimated_taxable_income = agi_without_cap_gains * 0.85
        cap_gains = estimated_taxable_income * cap_gains_rate * scaling_factor
        
        # Add capital gains to AGI
        agi_with_cap_gains = agi_without_cap_gains + cap_gains
        
        return (cap_gains, agi_with_cap_gains)


def apply_capital_gains_to_dataframe(df: pd.DataFrame,
                                     agi_col: str = 'agi',
                                     taxable_income_col: Optional[str] = None,
                                     include_random: bool = True) -> pd.DataFrame:
    """
    Apply capital gains estimation to a DataFrame of tax units.
    
    Args:
        df: DataFrame with tax units
        agi_col: Name of AGI column
        taxable_income_col: Name of taxable income column (optional)
        include_random: If True, randomly assign based on participation rates
        
    Returns:
        DataFrame with added columns:
        - capital_gains: Estimated capital gains amount
        - agi_with_cap_gains: AGI including capital gains
    """
    result_df = df.copy()
    estimator = CapitalGainsEstimator()
    
    capital_gains = []
    agi_with_cap_gains_list = []
    
    for _, row in df.iterrows():
        agi = row[agi_col]
        
        if taxable_income_col and taxable_income_col in df.columns:
            # Use actual taxable income if available
            taxable_income = row[taxable_income_col]
            cap_gains = estimator.estimate_capital_gains(agi, taxable_income, include_random)
            agi_with_cg = agi + cap_gains
        else:
            # Estimate both cap gains and new AGI
            cap_gains, agi_with_cg = estimator.estimate_agi_with_cap_gains(agi, include_random)
        
        capital_gains.append(cap_gains)
        agi_with_cap_gains_list.append(agi_with_cg)
    
    result_df['capital_gains'] = capital_gains
    result_df['agi_with_cap_gains'] = agi_with_cap_gains_list
    
    return result_df


if __name__ == '__main__':
    # Example usage
    estimator = CapitalGainsEstimator()
    
    # Test cases
    test_cases = [
        (30000, "Low income"),
        (75000, "Middle income"),
        (150000, "Upper-middle income"),
        (500000, "High income"),
    ]
    
    print("Capital Gains Estimation Examples:")
    print("="*80)
    
    for agi, desc in test_cases:
        cap_gains_rate = estimator.get_capital_gains_rate(agi)
        participation_rate = estimator.get_participation_rate(agi)
        
        # Estimate with 100% participation to show max amount
        estimated_taxable_income = agi * 0.85
        max_cap_gains = estimated_taxable_income * cap_gains_rate
        
        print(f"\n{desc} (AGI: ${agi:,})")
        print(f"  Cap gains rate: {cap_gains_rate*100:.2f}% of taxable income")
        print(f"  Participation rate: {participation_rate*100:.2f}%")
        print(f"  Est. cap gains (if has): ${max_cap_gains:,.0f}")
        print(f"  AGI with cap gains: ${agi + max_cap_gains:,.0f}")
