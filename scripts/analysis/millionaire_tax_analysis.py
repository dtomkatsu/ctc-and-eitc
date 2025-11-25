#!/usr/bin/env python3
"""
Analysis of "Millionaire's Tax" Scenario for Tax Year 2026.
Surcharge of 1% on income over:
- $1,000,000 for Married Filing Jointly / Surviving Spouse
- $500,000 for Single / Married Filing Separately / Head of Household
"""

import sys
from pathlib import Path
import pandas as pd
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tax.config import TaxSystemConfig, TaxSystemRegistry, TaxCalculator, compare_systems

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_tax_units(year: int = 2026) -> pd.DataFrame:
    """Load projected tax units for analysis."""
    # Using the same loading logic as the main runner
    tax_units_path = PROJECT_ROOT / f"data/processed/projections/tax_units_{year}_baseline.parquet"
    
    if not tax_units_path.exists():
        raise FileNotFoundError(f"Tax units file not found: {tax_units_path}")
    
    df = pd.read_parquet(tax_units_path)
    
    # Ensure consistent column names
    if 'filing_status_clean' in df.columns:
        df['filing_status'] = df['filing_status_clean']
    
    if 'income_projected' in df.columns and 'income' not in df.columns:
        df['income'] = df['income_projected']
    
    # Add num_exemptions if not present (1 filer + dependents)
    if 'num_exemptions' not in df.columns:
        df['num_exemptions'] = 1 + df.get('num_dependents', 0)
    
    logger.info(f"Loaded {len(df):,} tax units, {df['weight'].sum():,.0f} weighted filers")
    return df

def main():
    print("\n" + "="*100)
    print("MILLIONAIRE'S TAX SCENARIO ANALYSIS (Tax Year 2026)")
    print("="*100)
    
    # 1. Load Data
    tax_units = load_tax_units(2026)
    calculator = TaxCalculator(PROJECT_ROOT)
    
    # 2. Define Baseline (Act 46)
    # Note: Act 46 (2025) system applies to tax years 2025+
    baseline_system = TaxSystemRegistry.get_act46_2025_system()
    baseline_system.name = "Act 46 (Baseline)"
    
    # 3. Define Millionaire Tax Scenario
    millionaire_system = TaxSystemConfig(
        name="Millionaire Tax",
        year=2026,
        bracket_year=2025, # Use Act 46 brackets
        standard_deduction_year=2025, # Use Act 46 deductions
        personal_exemption=TaxSystemRegistry.PERSONAL_EXEMPTIONS[2026],
        description="Act 46 + 1% surcharge on income > $500k (Single/HoH) or > $1M (Joint)",
        surcharges={
            'Joint_Surviving_Spouse': {'threshold': 1_000_000, 'rate': 0.01},
            'Single_Married_Separate': {'threshold': 500_000, 'rate': 0.01},
            'Head_of_Household': {'threshold': 500_000, 'rate': 0.01}
        }
    )
    
    # 4. Run Comparison
    print("\nCalculating revenue...")
    comparison = compare_systems(tax_units, baseline_system, millionaire_system, calculator)
    
    print("\nRESULTS:")
    print("-" * 100)
    print(comparison.to_string(index=False))
    print("-" * 100)
    
    # 5. Detailed Impact Analysis
    print("\nDETAILED REVENUE GAIN ANALYSIS:")
    
    # Calculate specifically how much comes from surcharge
    # We can inspect individual liabilities by running calculation manually for affected units
    # Or just infer from the total difference (since that's the only change)
    
    revenue_gain = comparison.loc[comparison['system'] == 'Difference', 'revenue_millions'].values[0]
    print(f"\nTotal Revenue Gain: ${revenue_gain:,.1f} Million")
    
    # Let's count how many filers are affected
    print("\nAffected Filers Estimate:")
    
    # Re-run calculation for affected check
    # We'll check who has income above thresholds
    
    affected_count = 0
    affected_tax_gain = 0.0
    
    # Threshold mapping
    thresholds = {
        'Joint_Surviving_Spouse': 1_000_000,
        'Single_Married_Separate': 500_000,
        'Head_of_Household': 500_000,
        # Map raw statuses just in case
        'married_filing_jointly': 1_000_000,
        'qualifying_widow': 1_000_000,
        'single': 500_000,
        'married_filing_separately': 500_000,
        'head_of_household': 500_000
    }
    
    status_col = 'filing_status'
    if status_col not in tax_units.columns and 'filing_status_clean' in tax_units.columns:
        status_col = 'filing_status_clean'
        
    # Helper to get deduction for taxable income calc
    # We need to roughly estimate taxable income to see who pays the surcharge
    # The surcharge applies to TAXABLE income, right?
    # "surcharge of 1% on income over $500,000" usually means taxable income in tax code contexts, 
    # but sometimes AGI. The TaxCalculator implementation applies it to TAXABLE income.
    # Given the prompt "on income over...", applying to taxable income is standard for surcharges on top of tax.
    
    print("(Based on Taxable Income > Threshold)")
    
    for idx, row in tax_units.iterrows():
        status = row[status_col]
        income = row['income'] # AGI
        weight = row['weight']
        
        # Approximate threshold check using AGI first to filter
        # (Taxable is always <= AGI)
        # If AGI < 500k, definitely not affected
        if income < 500_000:
            continue
            
        threshold = thresholds.get(status, 500_000) # Default to lower to be safe
        
        # Calculate actual tax to see surcharge
        # We can use the calculator's internal logic for this unit
        res = calculator.calculate_tax(
            income, 
            millionaire_system, 
            status, 
            row.get('num_exemptions', 1)
        )
        
        if res.get('surcharge_amount', 0) > 0:
            affected_count += weight
            affected_tax_gain += res['surcharge_amount'] * weight
            
    print(f"  Number of Filers Affected: {affected_count:,.0f}")
    print(f"  Average Surcharge per Affected Filer: ${affected_tax_gain / affected_count:,.0f}" if affected_count > 0 else "  Average Surcharge: $0")
    
    print("\n" + "="*100)

if __name__ == "__main__":
    main()
