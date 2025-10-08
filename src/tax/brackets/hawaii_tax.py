"""
Hawaii State Income Tax Calculator

This module handles Hawaii state income tax calculations using historical
and projected tax brackets and standard deductions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TaxBracket:
    """Represents a single tax bracket"""
    income_min: float
    income_max: float
    rate: float  # As percentage (e.g., 1.4 for 1.4%)
    base_tax: float
    base_income: float
    
    def calculate_tax_in_bracket(self, income: float) -> float:
        """Calculate tax for income within this bracket"""
        if income <= self.income_min:
            return 0.0
        
        taxable_in_bracket = min(income, self.income_max) - self.income_min
        return taxable_in_bracket * (self.rate / 100.0)


class HawaiiTaxCalculator:
    """
    Calculator for Hawaii state income tax.
    
    Handles multiple years and filing statuses with progressive tax brackets.
    """
    
    # Mapping from PUMS filing status to Hawaii tax filing status
    FILING_STATUS_MAP = {
        'single': 'Single_Married_Separate',
        'married_filing_jointly': 'Joint_Surviving_Spouse',
        'married_filing_separate': 'Single_Married_Separate',
        'head_of_household': 'Head_of_Household',
        'qualifying_widow': 'Joint_Surviving_Spouse'
    }
    
    def __init__(self, brackets_df: pd.DataFrame, deductions_df: pd.DataFrame):
        """
        Initialize calculator with tax brackets and standard deductions.
        
        Args:
            brackets_df: DataFrame with columns: income_min, income_max, rate, 
                        base_tax, base_income, year, filing_status
            deductions_df: DataFrame with columns: Year, Joint_Surviving_Spouse,
                          Head_of_Household, Single_Married_Separate
        """
        self.brackets_df = brackets_df
        self.deductions_df = deductions_df
        
        # Organize brackets by year and filing status
        self.brackets_by_year = {}
        for year in brackets_df['year'].unique():
            self.brackets_by_year[year] = {}
            year_data = brackets_df[brackets_df['year'] == year]
            
            for status in year_data['filing_status'].unique():
                status_data = year_data[year_data['filing_status'] == status]
                brackets = []
                
                for _, row in status_data.iterrows():
                    brackets.append(TaxBracket(
                        income_min=row['income_min'],
                        income_max=row['income_max'],
                        rate=row['rate'],
                        base_tax=row['base_tax'],
                        base_income=row['base_income']
                    ))
                
                # Sort brackets by income_min
                brackets.sort(key=lambda b: b.income_min)
                self.brackets_by_year[year][status] = brackets
        
        # Organize deductions by year
        self.deductions_by_year = {}
        for _, row in deductions_df.iterrows():
            year = int(row['Year'])
            self.deductions_by_year[year] = {
                'Joint_Surviving_Spouse': row['Joint_Surviving_Spouse'],
                'Head_of_Household': row['Head_of_Household'],
                'Single_Married_Separate': row['Single_Married_Separate']
            }
    
    def get_standard_deduction(self, year: int, filing_status: str) -> float:
        """
        Get standard deduction for a given year and filing status.
        
        Args:
            year: Tax year
            filing_status: Hawaii filing status
            
        Returns:
            Standard deduction amount
        """
        # Use closest available year if exact year not found
        available_years = sorted(self.deductions_by_year.keys())
        if year not in available_years:
            # Find closest year
            year = min(available_years, key=lambda y: abs(y - year))
        
        return self.deductions_by_year[year][filing_status]
    
    def calculate_taxable_income(self, gross_income: float, year: int, 
                                 filing_status: str, 
                                 itemized_deductions: Optional[float] = None) -> float:
        """
        Calculate taxable income after standard or itemized deductions.
        
        Args:
            gross_income: Gross income before deductions
            year: Tax year
            filing_status: Hawaii filing status
            itemized_deductions: Optional itemized deductions (if greater than standard)
            
        Returns:
            Taxable income
        """
        standard_deduction = self.get_standard_deduction(year, filing_status)
        
        # Use the greater of standard or itemized deductions
        if itemized_deductions is not None:
            deduction = max(standard_deduction, itemized_deductions)
        else:
            deduction = standard_deduction
        
        return max(0, gross_income - deduction)
    
    def calculate_tax(self, income: float, year: int, filing_status: str,
                     use_taxable_income: bool = False) -> Dict[str, float]:
        """
        Calculate Hawaii state income tax.
        
        Args:
            income: Income amount (gross or taxable depending on use_taxable_income)
            year: Tax year
            filing_status: Hawaii filing status (or PUMS status, will be mapped)
            use_taxable_income: If True, income is already taxable income.
                               If False, will apply standard deduction first.
            
        Returns:
            Dictionary with:
                - gross_income: Original income
                - standard_deduction: Standard deduction applied
                - taxable_income: Income after deductions
                - tax_liability: Total tax owed
                - effective_rate: Effective tax rate (%)
                - marginal_rate: Marginal tax rate (%)
                - bracket: Description of top bracket
        """
        # Map PUMS filing status to Hawaii status if needed
        if filing_status in self.FILING_STATUS_MAP:
            filing_status = self.FILING_STATUS_MAP[filing_status]
        
        # Use closest available year if exact year not found
        available_years = sorted(self.brackets_by_year.keys())
        if year not in available_years:
            year = min(available_years, key=lambda y: abs(y - year))
        
        # Get brackets for this year and status
        if filing_status not in self.brackets_by_year[year]:
            raise ValueError(f"Filing status {filing_status} not found for year {year}")
        
        brackets = self.brackets_by_year[year][filing_status]
        
        # Calculate taxable income if needed
        gross_income = income
        if use_taxable_income:
            taxable_income = income
            standard_deduction = 0
        else:
            standard_deduction = self.get_standard_deduction(year, filing_status)
            taxable_income = max(0, income - standard_deduction)
        
        # Calculate tax using bracket structure
        if taxable_income <= 0:
            return {
                'gross_income': gross_income,
                'standard_deduction': standard_deduction,
                'taxable_income': 0,
                'tax_liability': 0,
                'effective_rate': 0,
                'marginal_rate': 0,
                'bracket': 'No tax liability'
            }
        
        # Find the applicable bracket and calculate tax
        tax_liability = 0
        marginal_rate = 0
        bracket_desc = ""
        
        for bracket in brackets:
            if taxable_income > bracket.income_min:
                # Income falls in or above this bracket
                if taxable_income <= bracket.income_max:
                    # This is the final bracket
                    tax_in_bracket = (taxable_income - bracket.income_min) * (bracket.rate / 100.0)
                    tax_liability = bracket.base_tax + tax_in_bracket
                    marginal_rate = bracket.rate
                    bracket_desc = f"${bracket.income_min:,.0f} - ${bracket.income_max:,.0f} at {bracket.rate}%"
                    break
                else:
                    # Income exceeds this bracket, continue to next
                    continue
        else:
            # Income is in the highest bracket (inf upper bound)
            last_bracket = brackets[-1]
            tax_in_bracket = (taxable_income - last_bracket.income_min) * (last_bracket.rate / 100.0)
            tax_liability = last_bracket.base_tax + tax_in_bracket
            marginal_rate = last_bracket.rate
            bracket_desc = f"${last_bracket.income_min:,.0f}+ at {last_bracket.rate}%"
        
        # Calculate effective rate
        effective_rate = (tax_liability / gross_income * 100) if gross_income > 0 else 0
        
        return {
            'gross_income': gross_income,
            'standard_deduction': standard_deduction,
            'taxable_income': taxable_income,
            'tax_liability': tax_liability,
            'effective_rate': effective_rate,
            'marginal_rate': marginal_rate,
            'bracket': bracket_desc
        }
    
    def calculate_tax_for_dataframe(self, df: pd.DataFrame, 
                                   income_col: str = 'income',
                                   filing_status_col: str = 'filing_status',
                                   year: int = 2024) -> pd.DataFrame:
        """
        Calculate taxes for a DataFrame of tax units.
        
        Args:
            df: DataFrame with tax units
            income_col: Name of income column
            filing_status_col: Name of filing status column
            year: Tax year to use
            
        Returns:
            DataFrame with added tax calculation columns
        """
        result_df = df.copy()
        
        # Calculate tax for each row
        tax_results = []
        for _, row in df.iterrows():
            income = row[income_col]
            filing_status = row[filing_status_col]
            
            tax_info = self.calculate_tax(income, year, filing_status)
            tax_results.append(tax_info)
        
        # Add results as new columns
        tax_df = pd.DataFrame(tax_results)
        for col in tax_df.columns:
            result_df[f'hi_tax_{col}'] = tax_df[col]
        
        return result_df
    
    def compare_scenarios(self, income: float, filing_status: str,
                         years: List[int]) -> pd.DataFrame:
        """
        Compare tax liability across different years/scenarios.
        
        Args:
            income: Income amount
            filing_status: Filing status
            years: List of years to compare
            
        Returns:
            DataFrame with comparison across years
        """
        results = []
        
        for year in years:
            tax_info = self.calculate_tax(income, year, filing_status)
            tax_info['year'] = year
            results.append(tax_info)
        
        return pd.DataFrame(results)


def load_tax_data(data_dir: Optional[Path] = None) -> HawaiiTaxCalculator:
    """
    Load tax brackets and deductions data and create calculator.
    
    Args:
        data_dir: Directory containing the CSV files. If None, uses default location.
        
    Returns:
        Initialized HawaiiTaxCalculator
    """
    if data_dir is None:
        # Default to project data directory
        data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'raw'
    
    brackets_file = data_dir / 'hawaii_tax_brackets_master_all.csv'
    deductions_file = data_dir / 'hawaii_standard_deductions_by_year.csv'
    
    # Load data
    brackets_df = pd.read_csv(brackets_file)
    deductions_df = pd.read_csv(deductions_file)
    
    return HawaiiTaxCalculator(brackets_df, deductions_df)


if __name__ == '__main__':
    # Example usage
    calculator = load_tax_data()
    
    # Calculate tax for a single filer with $50,000 income in 2024
    result = calculator.calculate_tax(50000, 2024, 'single')
    
    print("Hawaii State Income Tax Calculation")
    print("=" * 50)
    print(f"Gross Income: ${result['gross_income']:,.2f}")
    print(f"Standard Deduction: ${result['standard_deduction']:,.2f}")
    print(f"Taxable Income: ${result['taxable_income']:,.2f}")
    print(f"Tax Liability: ${result['tax_liability']:,.2f}")
    print(f"Effective Rate: {result['effective_rate']:.2f}%")
    print(f"Marginal Rate: {result['marginal_rate']:.2f}%")
    print(f"Tax Bracket: {result['bracket']}")
