#!/usr/bin/env python3
"""
Centralized tax system configuration for Hawaii state income tax projections.
Handles loading brackets, deductions, exemptions, and tax calculations for any year/scenario.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class TaxSystemConfig:
    """Configuration for a specific tax system (year, brackets, deductions, exemptions)."""
    
    name: str
    year: int  # Tax year (what year's income this applies to)
    bracket_year: int  # Which bracket schedule to use
    standard_deduction_year: int  # Which standard deduction schedule to use
    personal_exemption: float  # Personal exemption amount per person
    description: str = ""
    
    # Optional: bracket adjustments (for rollback scenarios)
    bracket_adjustments: Optional[Dict[str, List[Tuple[float, float]]]] = None
    # Format: {'filing_status': [(income_threshold, rate_adjustment_pp), ...]}


class TaxSystemRegistry:
    """Registry of pre-configured tax systems for common scenarios."""
    
    # Personal exemption amounts by year
    PERSONAL_EXEMPTIONS = {
        2018: 1144,  # 2017 law (applies to 2018 tax year)
        2022: 1144,
        2025: 1200,  # Indexed
        2026: 1200,
        2027: 1200,
    }
    
    @classmethod
    def get_2017_system(cls) -> TaxSystemConfig:
        """2017 tax law (pre-Act 46)."""
        return TaxSystemConfig(
            name="2017_system",
            year=2018,  # Applies to 2018 tax year
            bracket_year=2018,
            standard_deduction_year=2018,
            personal_exemption=cls.PERSONAL_EXEMPTIONS[2018],
            description="Pre-Act 46 baseline (2017 law)"
        )
    
    @classmethod
    def get_act46_2025_system(cls) -> TaxSystemConfig:
        """Act 46 tax system for 2025 tax year."""
        return TaxSystemConfig(
            name="act46_2025",
            year=2025,
            bracket_year=2025,  # Act 46 brackets (taxable years beginning after 12/31/2024)
            standard_deduction_year=2025,  # Act 46 standard deductions
            personal_exemption=cls.PERSONAL_EXEMPTIONS[2025],
            description="Act 46 (2024 law, applies to 2025+ tax years)"
        )
    
    @classmethod
    def get_act46_2027_system(cls) -> TaxSystemConfig:
        """Act 46 tax system for 2027 tax year."""
        return TaxSystemConfig(
            name="act46_2027",
            year=2027,
            bracket_year=2027,  # Future Act 46 brackets
            standard_deduction_year=2027,  # Future standard deductions
            personal_exemption=cls.PERSONAL_EXEMPTIONS.get(2027, 1200),
            description="Act 46 (future projections for 2027 tax year)"
        )
    
    @classmethod
    def get_act46_rollback_targeted(cls, base_year: int = 2025) -> TaxSystemConfig:
        """Act 46 with targeted rollback increases (+0.25pp, +0.5pp, +1.0pp on top 5 brackets)."""
        return TaxSystemConfig(
            name="act46_rollback_targeted",
            year=base_year,
            bracket_year=base_year,
            standard_deduction_year=base_year,
            personal_exemption=cls.PERSONAL_EXEMPTIONS.get(base_year, 1200),
            description="Act 46 with targeted rate increases (top 5 brackets)",
            bracket_adjustments={
                # Adjustments will be applied to top 5 brackets of each filing status
                'top_5_adjustments': [0.25, 0.25, 0.25, 0.50, 1.00]  # 5th, 4th, 3rd, 2nd, 1st highest
            }
        )


class TaxCalculator:
    """Centralized tax calculation engine."""
    
    def __init__(self, project_root: Optional[Path] = None):
        """Initialize calculator with data file paths."""
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent.parent
        
        self.project_root = project_root
        self.brackets_path = project_root / "data/raw/hawaii_tax_brackets_master_all.csv"
        self.deductions_path = project_root / "data/raw/hawaii_standard_deductions_by_year.csv"
        
        # Load data
        self.all_brackets = pd.read_csv(self.brackets_path)
        self.all_deductions = pd.read_csv(self.deductions_path)
        
        logger.info(f"Loaded brackets for years: {sorted(self.all_brackets['year'].unique())}")
        logger.info(f"Loaded deductions for years: {sorted(self.all_deductions['Year'].unique())}")
    
    def get_brackets(self, year: int, filing_status: str) -> pd.DataFrame:
        """Get tax brackets for a specific year and filing status."""
        # Standardize filing status names
        status_map = {
            'single': 'Single_Married_Separate',
            'married_filing_jointly': 'Joint_Surviving_Spouse',
            'married_filing_separately': 'Single_Married_Separate',
            'head_of_household': 'Head_of_Household',
            'qualifying_widow': 'Joint_Surviving_Spouse'
        }
        
        mapped_status = status_map.get(filing_status, filing_status)
        
        brackets = self.all_brackets[
            (self.all_brackets['year'] == year) & 
            (self.all_brackets['filing_status'] == mapped_status)
        ].copy()
        
        if brackets.empty:
            raise ValueError(f"No brackets found for year {year}, status {mapped_status}")
        
        return brackets.sort_values('income_min').reset_index(drop=True)
    
    def get_standard_deduction(self, year: int, filing_status: str) -> float:
        """Get standard deduction for a specific year and filing status."""
        # Standardize filing status names
        status_map = {
            'single': 'Single_Married_Separate',
            'married_filing_jointly': 'Joint_Surviving_Spouse',
            'married_filing_separately': 'Single_Married_Separate',
            'head_of_household': 'Head_of_Household',
            'qualifying_widow': 'Joint_Surviving_Spouse'
        }
        
        mapped_status = status_map.get(filing_status, filing_status)
        
        deduction_row = self.all_deductions[self.all_deductions['Year'] == year]
        
        if deduction_row.empty:
            raise ValueError(f"No standard deduction found for year {year}")
        
        return float(deduction_row[mapped_status].iloc[0])
    
    def apply_bracket_adjustments(
        self, 
        brackets: pd.DataFrame, 
        adjustments: Dict[str, List[float]]
    ) -> pd.DataFrame:
        """Apply rate adjustments to brackets (for rollback scenarios)."""
        adjusted = brackets.copy()
        
        if 'top_5_adjustments' in adjustments:
            # Apply to top 5 brackets
            top_5_increases = adjustments['top_5_adjustments']
            
            # Get top 5 brackets by income_min
            top_5_indices = adjusted.nlargest(5, 'income_min').index
            
            for i, idx in enumerate(sorted(top_5_indices, reverse=True)):
                if i < len(top_5_increases):
                    adjusted.loc[idx, 'rate'] = adjusted.loc[idx, 'rate'] + top_5_increases[i] / 100
        
        return adjusted
    
    def calculate_tax(
        self,
        income: float,
        config: TaxSystemConfig,
        filing_status: str,
        num_exemptions: int = 1
    ) -> Dict[str, float]:
        """
        Calculate tax liability for a single tax unit.
        
        Args:
            income: Gross income (AGI)
            config: Tax system configuration
            filing_status: Filing status
            num_exemptions: Number of personal exemptions
            
        Returns:
            Dict with tax calculation details
        """
        # Get standard deduction
        std_deduction = self.get_standard_deduction(config.standard_deduction_year, filing_status)
        
        # Calculate personal exemptions
        personal_exemptions = num_exemptions * config.personal_exemption
        
        # Calculate taxable income
        taxable_income = max(0, income - std_deduction - personal_exemptions)
        
        # Get brackets
        brackets = self.get_brackets(config.bracket_year, filing_status)
        
        # Apply adjustments if specified
        if config.bracket_adjustments:
            brackets = self.apply_bracket_adjustments(brackets, config.bracket_adjustments)
        
        # Calculate tax using brackets
        tax = 0.0
        marginal_rate = 0.0
        
        for _, bracket in brackets.iterrows():
            bracket_min = bracket['income_min']
            bracket_max = bracket['income_max'] if pd.notna(bracket['income_max']) else float('inf')
            rate = bracket['rate']
            
            # Convert rate to decimal if stored as percentage
            if rate > 1:
                rate = rate / 100
            
            if taxable_income > bracket_min:
                taxable_in_bracket = min(taxable_income, bracket_max) - bracket_min
                tax += taxable_in_bracket * rate
                marginal_rate = rate
                
                if taxable_income <= bracket_max:
                    break
        
        return {
            'gross_income': income,
            'standard_deduction': std_deduction,
            'personal_exemptions': personal_exemptions,
            'taxable_income': taxable_income,
            'tax_liability': tax,
            'marginal_rate': marginal_rate * 100,  # Convert to percentage
            'effective_rate': (tax / income * 100) if income > 0 else 0
        }
    
    def calculate_revenue(
        self,
        tax_units: pd.DataFrame,
        config: TaxSystemConfig,
        filing_status_col: str = 'filing_status',
        income_col: str = 'income',
        weight_col: str = 'weight',
        num_exemptions_col: str = 'num_exemptions'
    ) -> Dict[str, float]:
        """
        Calculate total revenue for a dataframe of tax units.
        
        Args:
            tax_units: DataFrame with tax units
            config: Tax system configuration
            filing_status_col: Column name for filing status
            income_col: Column name for income
            weight_col: Column name for weights
            num_exemptions_col: Column name for number of exemptions
            
        Returns:
            Dict with revenue statistics
        """
        liabilities = []
        weights = tax_units[weight_col].values
        
        # Set default exemptions if column doesn't exist
        if num_exemptions_col not in tax_units.columns:
            num_exemptions = np.ones(len(tax_units))
        else:
            num_exemptions = tax_units[num_exemptions_col].values
        
        for income, status, num_ex in zip(
            tax_units[income_col], 
            tax_units[filing_status_col],
            num_exemptions
        ):
            try:
                result = self.calculate_tax(income, config, status, int(num_ex))
                liabilities.append(result['tax_liability'])
            except Exception as e:
                logger.warning(f"Error calculating tax for income ${income:,.0f}, status {status}: {e}")
                liabilities.append(0)
        
        liabilities = np.array(liabilities)
        total_revenue = float(np.sum(liabilities * weights)) / 1e6  # In millions
        avg_tax = float(np.average(liabilities, weights=weights))
        avg_income = float(np.average(tax_units[income_col], weights=weights))
        
        return {
            'total_revenue_millions': total_revenue,
            'average_tax_per_filer': avg_tax,
            'average_income': avg_income,
            'effective_rate': (avg_tax / avg_income * 100) if avg_income > 0 else 0,
            'total_filers': float(weights.sum())
        }


def compare_systems(
    tax_units: pd.DataFrame,
    baseline_config: TaxSystemConfig,
    scenario_config: TaxSystemConfig,
    calculator: Optional[TaxCalculator] = None
) -> pd.DataFrame:
    """
    Compare two tax systems and return detailed comparison.
    
    Args:
        tax_units: DataFrame with tax units
        baseline_config: Baseline tax system
        scenario_config: Scenario tax system to compare
        calculator: Optional pre-initialized calculator
        
    Returns:
        DataFrame with comparison results
    """
    if calculator is None:
        calculator = TaxCalculator()
    
    baseline_revenue = calculator.calculate_revenue(tax_units, baseline_config)
    scenario_revenue = calculator.calculate_revenue(tax_units, scenario_config)
    
    results = []
    results.append({
        'system': baseline_config.name,
        'description': baseline_config.description,
        'revenue_millions': baseline_revenue['total_revenue_millions'],
        'avg_tax': baseline_revenue['average_tax_per_filer'],
        'effective_rate': baseline_revenue['effective_rate']
    })
    
    results.append({
        'system': scenario_config.name,
        'description': scenario_config.description,
        'revenue_millions': scenario_revenue['total_revenue_millions'],
        'avg_tax': scenario_revenue['average_tax_per_filer'],
        'effective_rate': scenario_revenue['effective_rate']
    })
    
    # Add comparison row
    revenue_diff = scenario_revenue['total_revenue_millions'] - baseline_revenue['total_revenue_millions']
    pct_diff = (revenue_diff / baseline_revenue['total_revenue_millions']) * 100
    
    results.append({
        'system': 'Difference',
        'description': f"{scenario_config.name} vs {baseline_config.name}",
        'revenue_millions': revenue_diff,
        'avg_tax': scenario_revenue['average_tax_per_filer'] - baseline_revenue['average_tax_per_filer'],
        'effective_rate': scenario_revenue['effective_rate'] - baseline_revenue['effective_rate']
    })
    
    return pd.DataFrame(results)
