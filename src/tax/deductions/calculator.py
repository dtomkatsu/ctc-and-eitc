"""
Taxable Income Calculator

Calculates taxable income from AGI using benchmark assignment approach:
1. Assign deductions based on AGI bracket averages
2. Assign exemptions based on family size and AGI bracket
3. Calculate: Taxable Income = AGI - Deductions - Exemptions
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

from .policy import DeductionPolicy

logger = logging.getLogger(__name__)


class TaxableIncomeCalculator:
    """
    Calculate taxable income from AGI using benchmark assignment.
    
    Formula: Taxable Income = max(0, AGI - Deductions - Personal Exemptions)
    
    Example:
        # Load benchmarks
        calculator = TaxableIncomeCalculator.from_files()
        
        # Calculate for all tax units
        results = calculator.calculate_batch(tax_units_df)
        
        # With custom policy
        policy = DeductionPolicy(year=2022)
        policy.set_policy_change('personal_exemption', 1500)
        results = calculator.calculate_batch(tax_units_df, policy=policy)
    """
    
    def __init__(
        self,
        deduction_benchmarks: pd.DataFrame,
        exemption_benchmarks: pd.DataFrame,
        policy: Optional[DeductionPolicy] = None
    ):
        """
        Initialize calculator with benchmark data.
        
        Args:
            deduction_benchmarks: DataFrame from parse_deduction_benchmarks()
            exemption_benchmarks: DataFrame from parse_exemption_benchmarks()
            policy: DeductionPolicy instance (defaults to 2022 baseline)
        """
        self.deduction_benchmarks = deduction_benchmarks.sort_values('agi_min').reset_index(drop=True)
        self.exemption_benchmarks = exemption_benchmarks.sort_values('agi_min').reset_index(drop=True)
        self.policy = policy or DeductionPolicy()
        
        logger.info(f"Initialized TaxableIncomeCalculator with {len(deduction_benchmarks)} deduction brackets")
    
    @classmethod
    def from_files(
        cls,
        deduction_path: str = 'data/processed/deduction_benchmarks.csv',
        exemption_path: str = 'data/processed/exemption_benchmarks.csv',
        policy: Optional[DeductionPolicy] = None
    ) -> 'TaxableIncomeCalculator':
        """
        Load calculator from saved benchmark files.
        
        Args:
            deduction_path: Path to deduction benchmarks CSV
            exemption_path: Path to exemption benchmarks CSV
            policy: DeductionPolicy instance
        
        Returns:
            TaxableIncomeCalculator instance
        """
        deduction_benchmarks = pd.read_csv(deduction_path)
        exemption_benchmarks = pd.read_csv(exemption_path)
        
        return cls(deduction_benchmarks, exemption_benchmarks, policy)
    
    def find_agi_bracket(self, agi: float, benchmarks: pd.DataFrame) -> Optional[pd.Series]:
        """
        Find the AGI bracket for a given AGI value.
        
        Args:
            agi: Adjusted Gross Income
            benchmarks: Benchmark DataFrame with agi_min and agi_max columns
        
        Returns:
            Bracket row as Series, or None if not found
        """
        matching = benchmarks[
            (benchmarks['agi_min'] <= agi) & (benchmarks['agi_max'] > agi)
        ]
        
        if len(matching) > 0:
            return matching.iloc[0]
        
        # If AGI is above highest bracket, use highest bracket
        if agi >= benchmarks['agi_max'].max():
            return benchmarks.iloc[-1]
        
        # If AGI is below lowest bracket, use lowest bracket
        if agi < benchmarks['agi_min'].min():
            return benchmarks.iloc[0]
        
        return None
    
    def _normalize_filing_status(self, filing_status: str) -> str:
        """Normalize filing status to policy format."""
        status_map = {
            'single': 'single',
            'married_filing_jointly': 'joint',
            'joint': 'joint',
            'head_of_household': 'hoh',
            'hoh': 'hoh',
            'married_filing_separately': 'mfs',
            'mfs': 'mfs'
        }
        return status_map.get(filing_status, filing_status)
    
    def calculate_deduction(
        self,
        agi: float,
        filing_status: str,
        age_65_plus_count: int = 0,
        policy: Optional[DeductionPolicy] = None
    ) -> Tuple[float, str]:
        """
        Calculate deduction amount (itemized vs standard).
        
        Uses benchmark assignment: assigns based on AGI bracket propensity to itemize.
        
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
        
        # Find AGI bracket
        bracket = self.find_agi_bracket(agi, self.deduction_benchmarks)
        
        if bracket is None:
            # Fallback to standard deduction
            standard = policy.get_standard_deduction(filing_status, age_65_plus_count)
            return standard, 'standard'
        
        # Get standard deduction from policy
        standard_deduction = policy.get_standard_deduction(filing_status, age_65_plus_count)
        
        # Get itemized deduction from bracket average
        itemized_deduction = bracket['avg_itemized']
        
        # Apply itemized deduction cap if applicable
        itemized_deduction = policy.apply_itemized_deduction_cap(itemized_deduction, agi)
        
        # Use propensity to itemize to decide
        # For deterministic assignment: use itemize if avg_itemized > standard
        # This matches aggregate patterns while being deterministic
        if itemized_deduction > standard_deduction:
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
            agi, filing_status, age_65_plus_count, policy
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
