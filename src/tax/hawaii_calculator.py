"""
Hawaii State Tax Calculator for 2022 Tax Year

This module calculates Hawaii state income tax based on 2022 tax tables.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class HawaiiTaxCalculator:
    """Calculate Hawaii state income tax for 2022."""
    
    def __init__(self, tax_tables_dir: Optional[str] = None):
        """
        Initialize calculator with Hawaii 2022 tax tables.
        
        Args:
            tax_tables_dir: Directory containing Hawaii tax table CSVs
        """
        if tax_tables_dir is None:
            tax_tables_dir = Path(__file__).parent.parent.parent / 'data' / 'tax_tables' / 'hawaii_2022'
        else:
            tax_tables_dir = Path(tax_tables_dir)
            
        self.tax_tables_dir = tax_tables_dir
        
        # Load tax tables
        self._load_tax_tables()
        
        logger.info(f"Loaded Hawaii 2022 tax tables from {tax_tables_dir}")
        
    def _load_tax_tables(self):
        """Load all tax table CSVs."""
        # Load standard deductions and exemptions
        deductions_df = pd.read_csv(self.tax_tables_dir / 'hawaii_2022_standard_deductions_and_exemptions.csv')
        
        # Parse standard deductions (first 5 rows)
        self.standard_deductions = {}
        for _, row in deductions_df.head(5).iterrows():
            status = row['Filing_Status'].lower().replace(' ', '_').replace('(', '').replace(')', '')
            if status == 'qualifying_widower':
                status = 'qualifying_widow'
            self.standard_deductions[status] = float(row['Standard_Deduction'])
        
        # Personal exemption (same for all)
        self.personal_exemption = 1144
        
        # Load tax brackets for each filing status
        self.tax_brackets = {}
        
        bracket_files = {
            'single': 'hawaii_2022_tax_brackets_single.csv',
            'married_filing_jointly': 'hawaii_2022_tax_brackets_married_filing_jointly.csv',
            'married_filing_separately': 'hawaii_2022_tax_brackets_married_filing_separately.csv',
            'head_of_household': 'hawaii_2022_tax_brackets_head_of_household.csv'
        }
        
        for status, filename in bracket_files.items():
            df = pd.read_csv(self.tax_tables_dir / filename)
            
            # Process brackets
            brackets = []
            for _, row in df.iterrows():
                min_income = float(row['Min_Income'])
                max_income = float('inf') if row['Max_Income'] == 'No Limit' else float(row['Max_Income'])
                rate = float(row['Tax_Rate_Percent']) / 100.0
                brackets.append({
                    'min': min_income,
                    'max': max_income,
                    'rate': rate
                })
            
            self.tax_brackets[status] = brackets
            
    def calculate_taxable_income(self, 
                                 agi: float, 
                                 filing_status: str,
                                 num_dependents: int = 0,
                                 num_adults: int = 1,
                                 deduction_amount: float = None) -> float:
        """
        Calculate Hawaii taxable income from AGI.
        
        Args:
            agi: Adjusted Gross Income
            filing_status: Filing status (single, married_filing_jointly, etc.)
            num_dependents: Number of dependents
            num_adults: Number of adults (1 for single/HoH, 2 for MFJ/MFS)
            deduction_amount: Actual deduction amount (itemized or standard)
                            If None, uses standard deduction
            
        Returns:
            Hawaii taxable income
        """
        # Use provided deduction amount, or default to standard deduction
        if deduction_amount is not None:
            deduction = deduction_amount
        else:
            deduction = self.standard_deductions.get(filing_status, 0)
        
        # Calculate personal exemptions
        total_people = num_adults + num_dependents
        total_exemptions = total_people * self.personal_exemption
        
        # Taxable income = AGI - deduction - exemptions
        taxable_income = max(0, agi - deduction - total_exemptions)
        
        return taxable_income
        
    def calculate_tax(self, taxable_income: float, filing_status: str) -> float:
        """
        Calculate Hawaii state tax using marginal brackets.
        
        Args:
            taxable_income: Hawaii taxable income
            filing_status: Filing status
            
        Returns:
            Hawaii state income tax
        """
        if taxable_income <= 0:
            return 0.0
            
        brackets = self.tax_brackets.get(filing_status, self.tax_brackets['single'])
        
        tax = 0.0
        previous_max = 0.0
        
        for bracket in brackets:
            if taxable_income <= bracket['min']:
                break
                
            # Calculate tax for this bracket
            bracket_income = min(taxable_income, bracket['max']) - bracket['min']
            if bracket_income > 0:
                tax += bracket_income * bracket['rate']
                
            if taxable_income <= bracket['max']:
                break
                
        return tax
    
    def calculate_tax_unit(self, tax_unit: pd.Series) -> Dict[str, float]:
        """
        Calculate Hawaii tax for a single tax unit.
        
        Args:
            tax_unit: Series containing tax unit data with columns:
                - agi or adjusted_gross_income
                - filing_status
                - num_dependents
                - num_adults (optional)
                
        Returns:
            Dictionary with tax calculation details
        """
        # Extract data
        agi = tax_unit.get('agi', tax_unit.get('adjusted_gross_income', 0))
        filing_status = tax_unit.get('filing_status', 'single')
        num_dependents = tax_unit.get('num_dependents', 0)
        
        # Determine num_adults based on filing status
        if filing_status == 'married_filing_jointly':
            num_adults = 2
        else:
            num_adults = 1
            
        # Calculate taxable income
        taxable_income = self.calculate_taxable_income(
            agi, filing_status, num_dependents, num_adults
        )
        
        # Calculate tax
        tax = self.calculate_tax(taxable_income, filing_status)
        
        return {
            'agi': agi,
            'filing_status': filing_status,
            'num_dependents': num_dependents,
            'num_adults': num_adults,
            'standard_deduction': self.standard_deductions.get(filing_status, 0),
            'personal_exemptions': (num_adults + num_dependents) * self.personal_exemption,
            'taxable_income': taxable_income,
            'hawaii_tax': tax,
            'effective_rate': (tax / agi * 100) if agi > 0 else 0
        }
    
    def calculate_tax_units_batch(self, tax_units: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Hawaii tax for a DataFrame of tax units.
        
        Args:
            tax_units: DataFrame with tax unit data
            
        Returns:
            DataFrame with added tax calculation columns
        """
        logger.info(f"Calculating Hawaii taxes for {len(tax_units):,} tax units...")
        
        # Make a copy to avoid modifying original
        result = tax_units.copy()
        
        if 'is_synthetic_ultra_high' in result.columns:
            synthetic_mask = result['is_synthetic_ultra_high'] == True
        else:
            synthetic_mask = pd.Series(False, index=result.index)
        
        work_mask = ~synthetic_mask
        
        preserved_taxable = None
        preserved_tax = None
        if synthetic_mask.any():
            logger.info(f"Preserving existing tax values for {synthetic_mask.sum()} synthetic ultra-high filers")
            if 'hi_taxable_income' in result.columns:
                preserved_taxable = result.loc[synthetic_mask, 'hi_taxable_income'].copy()
            if 'hi_state_tax' in result.columns:
                preserved_tax = result.loc[synthetic_mask, 'hi_state_tax'].copy()
        
        # Ensure AGI column exists
        if 'agi' not in result.columns and 'adjusted_gross_income' in result.columns:
            result['agi'] = result['adjusted_gross_income']
        elif 'agi' not in result.columns:
            logger.warning("No AGI column found, calculating from income columns")
            income_cols = [col for col in result.columns if 'income' in col.lower()]
            if income_cols:
                result['agi'] = result[income_cols[0]]
            else:
                raise ValueError("Cannot find or construct AGI column")
        
        # Ensure num_dependents exists
        if 'num_dependents' not in result.columns:
            result['num_dependents'] = 0
            
        # Determine num_adults
        result['num_adults'] = result['filing_status'].apply(
            lambda x: 2 if x == 'married_filing_jointly' else 1
        )
        
        if work_mask.any():
            logger.info("Calculating taxable income...")
            result.loc[work_mask, 'hi_taxable_income'] = result.loc[work_mask].apply(
                lambda row: self.calculate_taxable_income(
                    row['agi'],
                    row['filing_status'],
                    row['num_dependents'],
                    row['num_adults'],
                    row.get('total_deductions', None)
                ),
                axis=1
            )
        
        if work_mask.any():
            logger.info("Calculating Hawaii state tax...")
            result.loc[work_mask, 'hi_state_tax'] = result.loc[work_mask].apply(
                lambda row: self.calculate_tax(
                    row['hi_taxable_income'],
                    row['filing_status']
                ),
                axis=1
            )
        
        if synthetic_mask.any():
            if preserved_taxable is not None:
                result.loc[synthetic_mask, 'hi_taxable_income'] = preserved_taxable
            else:
                result.loc[synthetic_mask, 'hi_taxable_income'] = result.loc[synthetic_mask].apply(
                    lambda row: self.calculate_taxable_income(
                        row['agi'],
                        row['filing_status'],
                        row['num_dependents'],
                        row['num_adults'],
                        row.get('total_deductions', None)
                    ),
                    axis=1
                )
            # Always recalculate tax for synthetic filers to ensure valid values
            logger.info(f"   Recalculating taxes for {synthetic_mask.sum()} synthetic filers...")
            for idx in result[synthetic_mask].index:
                row = result.loc[idx]
                taxable_income = row['hi_taxable_income']
                filing_status = row['filing_status']
                
                if pd.notna(taxable_income):
                    tax = self.calculate_tax(taxable_income, filing_status)
                    result.loc[idx, 'hi_state_tax'] = tax
                    logger.debug(f"     Synthetic filer {idx}: taxable=${taxable_income:,.0f} → tax=${tax:,.2f}")
                else:
                    logger.warning(f"     Synthetic filer {idx}: taxable income is NaN")
            
            # Verify all synthetic filers now have valid taxes
            synthetic_nan_count = result.loc[synthetic_mask, 'hi_state_tax'].isna().sum()
            if synthetic_nan_count > 0:
                logger.warning(f"   ⚠️  {synthetic_nan_count} synthetic filers still have NaN tax")
            else:
                logger.info(f"   ✅ All synthetic filers now have valid tax values")
        
        # Calculate effective rate
        result['hi_effective_rate'] = np.where(
            result['agi'] > 0,
            (result['hi_state_tax'] / result['agi'] * 100),
            0
        )
        
        logger.info(f"✅ Hawaii tax calculation complete")
        logger.info(f"   Total tax liability: ${result['hi_state_tax'].sum():,.2f}")
        logger.info(f"   Average tax per unit: ${result['hi_state_tax'].mean():,.2f}")
        logger.info(f"   Average effective rate: {result['hi_effective_rate'].mean():.2f}%")
        
        return result


def calculate_taxes(tax_units: pd.DataFrame, 
                   tax_tables_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Convenience function to calculate Hawaii taxes on tax units.
    
    Args:
        tax_units: DataFrame with tax unit data
        tax_tables_dir: Optional path to tax tables directory
        
    Returns:
        DataFrame with tax calculations added
    """
    calculator = HawaiiTaxCalculator(tax_tables_dir)
    return calculator.calculate_tax_units_batch(tax_units)
