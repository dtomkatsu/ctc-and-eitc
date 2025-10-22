"""
Taxable Income Calculator

Calculates taxable income from AGI using benchmark assignment approach:
2. Assign exemptions based on family size and AGI bracket
3. Calculate: Taxable Income = AGI - Deductions - Exemptions
"""

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from .policy import DeductionPolicy

logger = logging.getLogger(__name__)


class TaxableIncomeCalculator:
    """
    Calculates taxable income using benchmark assignment approach.
    """
    
    @classmethod
    def from_files(
        cls,
        deduction_benchmarks_path: Optional[str] = None,
        policy: Optional[DeductionPolicy] = None
    ) -> 'TaxableIncomeCalculator':
        """
        Create a TaxableIncomeCalculator from benchmark files.
        
        Args:
            deduction_benchmarks_path: Path to deduction benchmarks CSV file
            policy: Optional DeductionPolicy instance
            
        Returns:
            TaxableIncomeCalculator instance
        """
        import os
        import pandas as pd
        
        # Define default benchmark data
        default_benchmarks = pd.DataFrame({
            'agi_min': [0, 10000, 20000, 30000, 40000, 50000, 75000, 100000, 150000, 200000, 300000, 400000],
            'agi_max': [10000, 20000, 30000, 40000, 50000, 75000, 100000, 150000, 200000, 300000, 400000, float('inf')],
            'avg_itemized': [5000, 8000, 12000, 15000, 18000, 22000, 28000, 35000, 45000, 65000, 90000, 150000],
            'itemization_rate': [0.07, 0.19, 0.28, 0.50, 0.74, 0.85, 0.93, 0.88, 0.91, 0.78, 0.74, 0.67]
        })
        
        # If no path provided, use default benchmarks
        if deduction_benchmarks_path is None:
            logger.warning("No deduction benchmarks path provided. Using default benchmarks.")
            return cls(default_benchmarks, policy)
            
        try:
            # Try to load the benchmarks from file
            if not os.path.exists(deduction_benchmarks_path):
                logger.warning(f"Deduction benchmarks file not found at {deduction_benchmarks_path}")
                logger.warning("Using default deduction benchmarks")
                return cls(default_benchmarks, policy)
                
            # Read the CSV file
            df = pd.read_csv(deduction_benchmarks_path)
            
            # Check if we have the expected columns
            if 'agi_min' in df.columns and 'agi_max' in df.columns and 'avg_itemized' in df.columns:
                return cls(df, policy)
            else:
                # If the file doesn't have the expected columns, use default benchmarks
                logger.warning("Deduction benchmarks file does not have the expected columns.")
                logger.warning("Using default deduction benchmarks")
                return cls(default_benchmarks, policy)
                
        except Exception as e:
            logger.warning(f"Error loading deduction benchmarks: {str(e)}")
            logger.warning("Using default deduction benchmarks")
            return cls(default_benchmarks, policy)
    
    def __init__(self, deduction_benchmarks: pd.DataFrame, policy: Optional[DeductionPolicy] = None):
        """
        Initialize with deduction benchmarks and optional policy.
        
        Args:
            deduction_benchmarks: DataFrame with deduction benchmarks by AGI bracket
            policy: DeductionPolicy instance (creates default if None)
        """
        self.deduction_benchmarks = deduction_benchmarks
        self.policy = policy or DeductionPolicy()
    
    def _normalize_filing_status(self, status: str) -> str:
        """Normalize filing status to standard format."""
        status = str(status).lower()
        if status in ['single', 's']:
            return 'single'
        elif status in ['joint', 'married_filing_jointly', 'mfj', 'j']:
            return 'joint'
        elif status in ['head_of_household', 'hoh', 'h']:
            return 'hoh'
        elif status in ['married_filing_separately', 'mfs', 'separate']:
            return 'mfs'
        return 'single'  # Default to single
        
    def find_agi_bracket(self, agi: float, benchmarks: pd.DataFrame) -> Optional[Dict]:
        """Find the AGI bracket for a given income."""
        if benchmarks is None or len(benchmarks) == 0:
            return None
            
        for _, row in benchmarks.iterrows():
            if row['agi_min'] <= agi < row['agi_max']:
                return row.to_dict()
        return None
    
    def _get_itemization_rate(self, agi: float, filing_status: str) -> float:
        """
        Get itemization probability based on AGI bracket from Table A4-2.
        
        Args:
            agi: Adjusted Gross Income
            
        Returns:
            Probability of itemizing (0-1)
        """
        # Itemization rates from Table A4-2 (2022)
        brackets = [
            (0, 10000, 0.07),        # 2,588 / 36,836
            (10000, 20000, 0.19),    # 10,690 / 56,049
            (20000, 30000, 0.28),    # 15,222 / 54,966
            (30000, 40000, 0.50),    # 29,552 / 58,598
            (40000, 50000, 0.74),    # 39,198 / 52,983
            (50000, 75000, 0.85),    # 77,709 / 90,877
            (75000, 100000, 0.93),   # 51,126 / 54,756
            (100000, 150000, 0.88),  # 54,341 / 61,913
            (150000, 200000, 0.91),  # 25,446 / 27,926
            (200000, 300000, 0.78),  # 14,723 / 18,911
            (300000, 400000, 0.74),  # 4,459 / 6,065
            (400000, float('inf'), 0.67)  # 5,972 / 8,867
        ]
        
        base_rate = 0.5
        for min_agi, max_agi, itemize_pct in brackets:
            if min_agi <= agi < max_agi:
                base_rate = itemize_pct
                break
        else:
            base_rate = 0.5

        status = self._normalize_filing_status(filing_status)
        if status == 'hoh':
            # More aggressive HoH reduction to align with DOTAX patterns
            if agi < 30000:
                base_rate *= 0.45  # Very low itemization for low-income HoH
            elif agi < 50000:
                base_rate *= 0.50  # Moderate reduction for mid-low income
            elif agi < 75000:
                base_rate *= 0.65  # Moderate reduction for mid income
            elif agi < 100000:
                base_rate *= 0.75  # Smaller reduction for higher income
            else:
                base_rate *= 0.85  # Minimal reduction for high income
        elif status == 'single':
            if agi < 50000:
                base_rate *= 0.9
        elif status == 'mfs':
            base_rate = min(base_rate * 1.1, 0.99)

        return max(0.05, min(base_rate, 0.99))

    def calculate_deduction(
        self,
        agi: float,
        filing_status: str,
        age_65_plus_count: int = 0,
        policy: Optional[DeductionPolicy] = None,
        forced_type: Optional[str] = None
    ) -> Tuple[float, str]:
        """
        Calculate deduction amount (itemized vs standard).
        
        Uses AGI-bracket specific itemization rates from Table A4-2.
        
        Args:
            agi: Adjusted Gross Income
            filing_status: Filing status ('single', 'joint', 'hoh', 'mfs')
            age_65_plus_count: Number of taxpayers aged 65+
            policy: DeductionPolicy (uses self.policy if None)
        
        Returns:
            Tuple of (deduction_amount, deduction_type)
        """
        policy = policy or self.policy
        filing_status = self._normalize_filing_status(filing_status)
        
        # Find AGI bracket for benchmark data
        bracket = self.find_agi_bracket(agi, self.deduction_benchmarks)
        
        if bracket is None:
            # Fallback to standard deduction if no bracket found
            standard = policy.get_standard_deduction(filing_status, age_65_plus_count)
            return standard, 'standard'
        
        # Get standard and itemized deduction amounts
        standard_deduction = policy.get_standard_deduction(filing_status, age_65_plus_count)
        itemized_deduction = bracket['avg_itemized']
        itemized_deduction = policy.apply_itemized_deduction_cap(itemized_deduction, agi)

        if forced_type:
            forced_type = forced_type.lower()
            if forced_type == 'itemized' and itemized_deduction > 0:
                return itemized_deduction, 'itemized'
            if forced_type == 'standard':
                return standard_deduction, 'standard'
            # Fallback to choosing the larger deduction if forced to itemize but amount is zero
            if forced_type == 'itemized':
                return max(standard_deduction, itemized_deduction), 'itemized' if itemized_deduction >= standard_deduction else 'standard'

        # Get itemization probability for this AGI and filing status
        itemize_prob = self._get_itemization_rate(agi, filing_status)
        
        # Add small random variation (5%) to prevent artificial thresholds
        itemize_prob *= (0.95 + 0.1 * np.random.random())
        itemize_prob = max(0.01, min(0.99, itemize_prob))
        
        # Decide based on probability
        if np.random.random() < itemize_prob and itemized_deduction > 0:
            return itemized_deduction, 'itemized'
        else:
            return standard_deduction, 'standard'
    
    def calculate_exemptions(
        self,
        agi: float,
        num_exemptions: int,
        policy: Optional[DeductionPolicy] = None
    ) -> float:
        """
        Calculate personal exemption amount.
        
        Args:
            agi: Adjusted Gross Income
            num_exemptions: Number of exemptions claimed
            policy: DeductionPolicy (uses self.policy if None)
        
        Returns:
            Total exemption amount
        """
        policy = policy or self.policy
        
        # Use policy to calculate exemptions (handles phaseouts)
        return policy.get_total_exemptions(num_exemptions, agi)
    
    def calculate(
        self,
        tax_unit: Dict,
        policy: Optional[DeductionPolicy] = None
    ) -> Dict:
        """
        Calculate taxable income for a single tax unit.
        
        Args:
            tax_unit: Dict or DataFrame row with:
                - agi: Adjusted Gross Income
                - filing_status: Filing status
                - num_exemptions: Number of exemptions (optional, defaults to 1)
                - age_65_plus_count: Number aged 65+ (optional, defaults to 0)
            policy: DeductionPolicy (uses self.policy if None)
        
        Returns:
            Dict with taxable_income and breakdown
        """
        policy = policy or self.policy
        
        agi = tax_unit.get('agi', 0)
        filing_status = tax_unit.get('filing_status', 'single')
        num_exemptions = tax_unit.get('num_exemptions', 1)
        age_65_plus_count = tax_unit.get('age_65_plus_count', 0)
        
        # Calculate deductions
        deduction, deduction_type = self.calculate_deduction(
            agi, filing_status, age_65_plus_count, policy,
            forced_type=tax_unit.get('forced_deduction_type')
        )
        
        # Calculate exemptions
        exemption_amount = self.calculate_exemptions(agi, num_exemptions, policy)
        
        # Calculate taxable income (cannot be negative)
        taxable_income = max(0, agi - deduction - exemption_amount)
        
        return {
            'taxable_income': taxable_income,
            'agi': agi,
            'deduction': deduction,
            'deduction_type': deduction_type,
            'exemption_amount': exemption_amount,
            'num_exemptions': num_exemptions,
            'filing_status': filing_status
        }
    
    def calculate_batch(
        self,
        tax_units_df: pd.DataFrame,
        policy: Optional[DeductionPolicy] = None
    ) -> pd.DataFrame:
        """
        Calculate taxable income for all tax units.
        
        Args:
            tax_units_df: DataFrame with tax units (must have 'agi' and 'filing_status')
            policy: DeductionPolicy (uses self.policy if None)
        
        Returns:
            DataFrame with original columns plus:
            - taxable_income
            - deduction
            - deduction_type
            - exemption_amount
        """
        policy = policy or self.policy
        
        logger.info(f"Calculating taxable income for {len(tax_units_df):,} tax units")
        logger.info(f"Policy: {policy.get_modifications_summary()}")
        
        results = []
        
        for idx, tax_unit in tax_units_df.iterrows():
            result = self.calculate(tax_unit.to_dict(), policy)
            results.append(result)
            
            # Progress logging
            if (idx + 1) % 10000 == 0:
                logger.info(f"  Processed {idx + 1:,} / {len(tax_units_df):,} tax units")
        
        results_df = pd.DataFrame(results)
        
        # Merge with original data (keep all original columns)
        output_df = tax_units_df.copy()
        output_df['taxable_income'] = results_df['taxable_income']
        output_df['deduction'] = results_df['deduction']
        output_df['deduction_type'] = results_df['deduction_type']
        output_df['exemption_amount'] = results_df['exemption_amount']
        
        logger.info(f"✅ Calculated taxable income for {len(output_df):,} tax units")
        
        return output_df
    
    def validate_against_benchmarks(self, results_df: pd.DataFrame) -> Dict:
        """
        Validate calculated deductions/exemptions against benchmarks.
        
        Args:
            results_df: DataFrame from calculate_batch()
        
        Returns:
            Dict with validation metrics
        """
        # Group by AGI bracket and compare to benchmarks
        validation = {}
        
        # Total deductions
        total_deductions = results_df['deduction'].sum()
        benchmark_total = (
            self.deduction_benchmarks['itemized_amount_millions'].sum() +
            self.deduction_benchmarks['standard_amount_millions'].sum()
        ) * 1e6
        
        validation['total_deductions'] = {
            'calculated': total_deductions,
            'benchmark': benchmark_total,
            'error_pct': abs(total_deductions - benchmark_total) / benchmark_total * 100
        }
        
        # Total exemptions
        total_exemptions = results_df['exemption_amount'].sum()
        benchmark_exemptions = self.exemption_benchmarks['exemption_amount_millions'].sum() * 1e6
        
        validation['total_exemptions'] = {
            'calculated': total_exemptions,
            'benchmark': benchmark_exemptions,
            'error_pct': abs(total_exemptions - benchmark_exemptions) / benchmark_exemptions * 100
        }
        
        # Itemization rate
        itemize_rate = (results_df['deduction_type'] == 'itemized').sum() / len(results_df) * 100
        benchmark_itemize_rate = (
            self.deduction_benchmarks['itemized_returns'].sum() /
            self.deduction_benchmarks['total_returns'].sum() * 100
        )
        
        validation['itemization_rate'] = {
            'calculated': itemize_rate,
            'benchmark': benchmark_itemize_rate,
            'error_pct': abs(itemize_rate - benchmark_itemize_rate)
        }
        
        return validation


def estimate_num_exemptions(tax_unit: pd.Series, exemption_benchmarks: pd.DataFrame) -> int:
    """
    Estimate number of exemptions for a tax unit based on household composition.
    
    Args:
        tax_unit: Tax unit row with household data
        exemption_benchmarks: Exemption benchmarks for AGI bracket averages
    
    Returns:
        Estimated number of exemptions
    """
    # Start with self
    num_exemptions = 1
    
    # Add spouse if joint filing
    if tax_unit.get('filing_status') == 'joint':
        num_exemptions += 1
    
    # Add dependents if available
    if 'num_dependents' in tax_unit and pd.notna(tax_unit['num_dependents']):
        num_exemptions += int(tax_unit['num_dependents'])
    
    # If no dependent info, use AGI bracket average
    elif 'agi' in tax_unit:
        agi = tax_unit['agi']
        matching = exemption_benchmarks[
            (exemption_benchmarks['agi_min'] <= agi) & 
            (exemption_benchmarks['agi_max'] > agi)
        ]
        
        if len(matching) > 0:
            avg_exemptions = matching.iloc[0]['avg_exemptions_per_return']
            num_exemptions = max(1, round(avg_exemptions))
    
    return num_exemptions


if __name__ == '__main__':
    # Test the calculator
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    # Load calculator
    calculator = TaxableIncomeCalculator.from_files()
    
    # Test single calculation
    test_unit = {
        'agi': 75000,
        'filing_status': 'joint',
        'num_exemptions': 3,
        'age_65_plus_count': 0
    }
    
    result = calculator.calculate(test_unit)
    
    print("\nTest Calculation:")
    print(f"  AGI: ${result['agi']:,.0f}")
    print(f"  Deduction: ${result['deduction']:,.0f} ({result['deduction_type']})")
    print(f"  Exemptions: ${result['exemption_amount']:,.0f} ({result['num_exemptions']} exemptions)")
    print(f"  Taxable Income: ${result['taxable_income']:,.0f}")
