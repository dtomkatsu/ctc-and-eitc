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

    # Optional: bracket scenario tag (matches 'scenario' column in bracket CSV)
    # Leave None to use baseline rows (scenario == '')
    bracket_scenario: Optional[str] = None

    # Optional: credit scenario tag (e.g. 'hb2306_hd1' for enhanced CDCC)
    # Leave None to use current-law credit calculations
    credit_scenario: Optional[str] = None

    # Optional: income surcharges
    surcharges: Optional[Dict[str, Dict[str, float]]] = None
    # Format: {'filing_status': {'threshold': 1000000, 'rate': 0.01}}


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
        """Act 46 tax system for 2027 tax year.

        Act 46 (2024) includes a scheduled bracket phase-in for 2027+.
        Brackets expand (e.g., first bracket widens from 0-9,600 to 0-14,400) to match
        the increased standard deduction ($4,400/$8,800 in 2025 → $8,000/$16,000 in 2027).
        """
        return TaxSystemConfig(
            name="act46_2027",
            year=2027,
            bracket_year=2027,  # Act 46 2027 bracket schedule (includes phase-in)
            standard_deduction_year=2027,  # Act 46 scheduled phase-in: $8k single / $16k joint
            personal_exemption=cls.PERSONAL_EXEMPTIONS.get(2027, 1200),
            description="Act 46 (2024) — 2027 bracket phase-in and standard deduction phase-in"
        )
    
    @classmethod
    def get_millionaire_tax_2027(cls, surcharge_rate: float = 0.02) -> TaxSystemConfig:
        """Act 46 2027 + millionaire's tax surcharge on taxable income above $1M."""
        return TaxSystemConfig(
            name=f"millionaire_tax_2027_{surcharge_rate:.0%}",
            year=2027,
            bracket_year=2027,
            standard_deduction_year=2027,
            personal_exemption=cls.PERSONAL_EXEMPTIONS.get(2027, 1200),
            description=f"Act 46 2027 + {surcharge_rate:.0%} surcharge on taxable income > $1M (all statuses)",
            surcharges={
                'Joint_Surviving_Spouse': {'threshold': 1_000_000, 'rate': surcharge_rate},
                'Single_Married_Separate': {'threshold': 1_000_000, 'rate': surcharge_rate},
                'Head_of_Household': {'threshold': 1_000_000, 'rate': surcharge_rate},
            }
        )

    @classmethod
    def get_sb3125_original_2027_system(cls) -> TaxSystemConfig:
        """SB3125 original proposal (2026 session, before SD1 amendments).

        Alternative to Act 46 with bracket expansion for TY2027+.
        Wider low/middle brackets compared to Act 46.
        Top brackets: $350k-$450k at 7.90%, etc.
        """
        return TaxSystemConfig(
            name="sb3125_original_2027",
            year=2027,
            bracket_year=2027,
            standard_deduction_year=2027,
            personal_exemption=cls.PERSONAL_EXEMPTIONS.get(2027, 1200),
            description="SB3125 original (2026) — bracket expansion for TY2027",
            bracket_scenario='',  # Use baseline (original SB3125) rows
        )

    @classmethod
    def get_sb3125_sd1_2027_system(cls) -> TaxSystemConfig:
        """SB3125 SD1 (2026 session) bracket schedule effective TY2027.

        Changes vs existing 2027 baseline (SB3125 original):
          - Joint/Surviving Spouse: top brackets above $350k shift from 7.9%→8.25% at $350k,
            collapsing the $650k-$800k tier (top rate now at $650k instead of $800k)
          - Head of Household: top brackets above $262.5k shift one level up,
            collapsing the $487.5k-$600k tier (top rate at $487.5k instead of $600k)
          - Single/MFS: top brackets above $175k shift one level up,
            collapsing the $325k-$400k tier (top rate at $325k instead of $400k)
        Lower brackets are unchanged from the SB3125 original schedule.
        """
        return TaxSystemConfig(
            name="sb3125_sd1_2027",
            year=2027,
            bracket_year=2027,
            standard_deduction_year=2027,
            personal_exemption=cls.PERSONAL_EXEMPTIONS.get(2027, 1200),
            description="SB3125 SD1 (2026 session) — higher top bracket rates effective TY2027",
            bracket_scenario='sb3125_sd1',
            credit_scenario='sb3125_sd1',
        )

    @classmethod
    def get_hb2306_hd1_2027_system(cls) -> TaxSystemConfig:
        """HB2306 HD1 (2025 session) bracket schedule effective TY2027.

        Changes vs Act 46 2025 baseline:
          - Keeps all 12 brackets from Act 46 2025 schedule (same boundaries)
          - Top 3 rates increase by 1pp: 9%→10%, 10%→11%, 11%→12%
          - Repeals previously scheduled 2027 and 2029 bracket phase-ins
        Also includes enhanced CDCC and sunset extension (Act 163 to 2032),
        modeled separately via credit_scenario.
        """
        return TaxSystemConfig(
            name="hb2306_hd1_2027",
            year=2027,
            bracket_year=2027,
            standard_deduction_year=2027,  # Same Act 46 phase-in as baseline
            personal_exemption=cls.PERSONAL_EXEMPTIONS.get(2027, 1200),
            description="HB2306 HD1 — top 3 rates +1pp, enhanced CDCC, sunset extension to 2032",
            bracket_scenario='hb2306_hd1',
            credit_scenario='hb2306_hd1',
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
    
    def get_brackets(self, year: int, filing_status: str,
                     scenario: Optional[str] = None) -> pd.DataFrame:
        """Get tax brackets for a specific year, filing status, and optional scenario.

        Args:
            scenario: If None, loads baseline rows (scenario column is blank/NaN).
                      If set (e.g. 'sb3125_sd1'), loads rows with that scenario tag.
        """
        status_map = {
            'single': 'Single_Married_Separate',
            'married_filing_jointly': 'Joint_Surviving_Spouse',
            'married_filing_separately': 'Single_Married_Separate',
            'head_of_household': 'Head_of_Household',
            'qualifying_widow': 'Joint_Surviving_Spouse'
        }

        mapped_status = status_map.get(filing_status, filing_status)

        df = self.all_brackets
        mask = (df['year'] == year) & (df['filing_status'] == mapped_status)

        if scenario:
            mask &= (df['scenario'].fillna('') == scenario)
        else:
            mask &= (df['scenario'].fillna('') == '')

        brackets = df[mask].copy()

        if brackets.empty:
            raise ValueError(f"No brackets found for year={year}, status={mapped_status}, "
                             f"scenario={scenario!r}")

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
        num_exemptions: int = 1,
        deduction_override: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate tax liability for a single tax unit.

        Args:
            income: Gross income (AGI)
            config: Tax system configuration
            filing_status: Filing status
            num_exemptions: Number of personal exemptions
            deduction_override: If provided, use this instead of the standard deduction
                (e.g., for itemized deductions that exceed the standard deduction)

        Returns:
            Dict with tax calculation details
        """
        # Get standard deduction (or use override if provided)
        std_deduction = self.get_standard_deduction(config.standard_deduction_year, filing_status)
        if deduction_override is not None:
            std_deduction = deduction_override
        
        # Calculate personal exemptions
        personal_exemptions = num_exemptions * config.personal_exemption
        
        # Calculate taxable income
        taxable_income = max(0, income - std_deduction - personal_exemptions)
        
        # Get brackets (with optional scenario tag)
        brackets = self.get_brackets(config.bracket_year, filing_status,
                                     scenario=config.bracket_scenario)
        
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
        
        # Apply surcharges if configured
        surcharge_amount = 0.0
        if config.surcharges:
            # Handle mapped status for config lookup
            status_map = {
                'single': 'Single_Married_Separate',
                'married_filing_jointly': 'Joint_Surviving_Spouse',
                'married_filing_separately': 'Single_Married_Separate',
                'head_of_household': 'Head_of_Household',
                'qualifying_widow': 'Joint_Surviving_Spouse'
            }
            # Try both original and mapped status
            status_keys = [filing_status, status_map.get(filing_status, filing_status)]
            
            for status_key in status_keys:
                if status_key in config.surcharges:
                    surcharge_config = config.surcharges[status_key]
                    threshold = surcharge_config['threshold']
                    rate = surcharge_config['rate']
                    
                    # Surcharge applies to taxable income above threshold
                    if taxable_income > threshold:
                        surcharge = (taxable_income - threshold) * rate
                        surcharge_amount += surcharge
                        tax += surcharge
                        marginal_rate += rate
                    break  # Apply only one matching surcharge config

        return {
            'gross_income': income,
            'standard_deduction': std_deduction,
            'personal_exemptions': personal_exemptions,
            'taxable_income': taxable_income,
            'tax_liability': tax,
            'surcharge_amount': surcharge_amount,
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
        num_exemptions_col: str = 'num_exemptions',
        deduction_col: Optional[str] = None,
        num_dependents_col: str = 'num_dependents',
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
            deduction_col: Optional column with per-unit deduction overrides
            num_dependents_col: Column name for number of dependents

        Returns:
            Dict with revenue statistics (includes credit breakdown)
        """
        from src.tax.adjustments.hawaii_credits import HawaiiTaxCredits

        liabilities = []
        weights = tax_units[weight_col].values

        # Set default exemptions if column doesn't exist
        if num_exemptions_col not in tax_units.columns:
            num_exemptions = np.ones(len(tax_units))
        else:
            num_exemptions = tax_units[num_exemptions_col].values

        # Set default dependents if column doesn't exist
        if num_dependents_col not in tax_units.columns:
            num_dependents = np.zeros(len(tax_units))
        else:
            num_dependents = tax_units[num_dependents_col].values

        # Deduction overrides (None if column not specified)
        deductions = tax_units[deduction_col].values if deduction_col and deduction_col in tax_units.columns else None

        for i, (income, status, num_ex) in enumerate(zip(
            tax_units[income_col],
            tax_units[filing_status_col],
            num_exemptions
        )):
            try:
                ded = float(deductions[i]) if deductions is not None else None
                result = self.calculate_tax(income, config, status, int(num_ex), deduction_override=ded)
                liabilities.append(result['tax_liability'])
            except Exception as e:
                logger.warning(f"Error calculating tax for income ${income:,.0f}, status {status}: {e}")
                liabilities.append(0)

        liabilities = np.array(liabilities)

        # Calculate credits
        credit_calc = HawaiiTaxCredits(year=config.year)
        total_credits = np.zeros(len(tax_units))
        total_cdcc = np.zeros(len(tax_units))

        for i, (income, status, tax_li, n_dep) in enumerate(zip(
            tax_units[income_col],
            tax_units[filing_status_col],
            liabilities,
            num_dependents
        )):
            try:
                cr = credit_calc.calculate_total_credits(
                    agi=income, filing_status=status,
                    num_dependents=int(n_dep),
                    tax_before_credits=tax_li,
                    credit_scenario=config.credit_scenario,
                )
                total_credits[i] = cr['total']
                total_cdcc[i] = cr['child_care']
            except Exception:
                pass

        net_liabilities = liabilities - total_credits

        total_revenue_before = float(np.sum(liabilities * weights)) / 1e6
        total_credits_m = float(np.sum(total_credits * weights)) / 1e6
        total_cdcc_m = float(np.sum(total_cdcc * weights)) / 1e6
        total_revenue = float(np.sum(net_liabilities * weights)) / 1e6
        avg_tax = float(np.average(net_liabilities, weights=weights))
        avg_income = float(np.average(tax_units[income_col], weights=weights))

        return {
            'total_revenue_millions': total_revenue,
            'total_revenue_before_credits_millions': total_revenue_before,
            'total_credits_millions': total_credits_m,
            'total_cdcc_millions': total_cdcc_m,
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
    
    def _row(rev, cfg):
        return {
            'system': cfg.name,
            'description': cfg.description,
            'revenue_millions': rev['total_revenue_millions'],
            'revenue_before_credits_millions': rev['total_revenue_before_credits_millions'],
            'total_credits_millions': rev['total_credits_millions'],
            'total_cdcc_millions': rev['total_cdcc_millions'],
            'avg_tax': rev['average_tax_per_filer'],
            'effective_rate': rev['effective_rate'],
        }

    results = [_row(baseline_revenue, baseline_config),
               _row(scenario_revenue, scenario_config)]

    # Add comparison row
    revenue_diff = scenario_revenue['total_revenue_millions'] - baseline_revenue['total_revenue_millions']

    results.append({
        'system': 'Difference',
        'description': f"{scenario_config.name} vs {baseline_config.name}",
        'revenue_millions': revenue_diff,
        'revenue_before_credits_millions': (scenario_revenue['total_revenue_before_credits_millions']
                                            - baseline_revenue['total_revenue_before_credits_millions']),
        'total_credits_millions': (scenario_revenue['total_credits_millions']
                                   - baseline_revenue['total_credits_millions']),
        'total_cdcc_millions': (scenario_revenue['total_cdcc_millions']
                                - baseline_revenue['total_cdcc_millions']),
        'avg_tax': scenario_revenue['average_tax_per_filer'] - baseline_revenue['average_tax_per_filer'],
        'effective_rate': scenario_revenue['effective_rate'] - baseline_revenue['effective_rate'],
    })

    return pd.DataFrame(results)
