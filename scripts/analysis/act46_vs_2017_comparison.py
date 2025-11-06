#!/usr/bin/env python3
"""Compare Act 46 (2026) revenue to what 2017 tax system would generate in 2026."""

import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tax.brackets.hawaii_tax import HawaiiTaxCalculator, load_tax_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_tax_units():
    """Load 2026 projected tax units."""
    tax_units_path = Path("data/processed/projections/tax_units_2026_baseline.parquet")
    df = pd.read_parquet(tax_units_path)
    
    # Clean filing status for Hawaii calculator
    status_mapping = {
        'single': 'single',
        'married_filing_jointly': 'married_filing_jointly',
        'married_filing_separately': 'married_filing_separately', 
        'head_of_household': 'head_of_household',
        'qualifying_widow': 'married_filing_jointly'
    }
    df['filing_status_clean'] = df['filing_status'].map(status_mapping)
    
    logger.info(f"Loaded {len(df):,} tax units, {df['weight'].sum():,.0f} weighted filers")
    return df


def calculate_revenue_by_year(tax_units: pd.DataFrame, year: int, use_corrected_act46: bool = False) -> float:
    """Calculate total revenue using specified year's tax system."""
    
    if use_corrected_act46:
        # Use corrected Act 46 with proper deductions and exemptions
        from pathlib import Path
        act46_path = Path("data/processed/projections/tax_units_2026_act46_corrected.parquet")
        if act46_path.exists():
            df_act46 = pd.read_parquet(act46_path)
            revenue = (df_act46['tax_liability_act46'] * df_act46['weight']).sum() / 1e6
            return revenue
    
    # Load tax calculator for the specified year
    calculator = load_tax_data()
    
    liabilities = []
    weights = tax_units["weight"].values
    
    # Use taxable income if available (accounts for itemized deductions)
    if "taxable_income" in tax_units.columns:
        incomes = tax_units["taxable_income"].values
        use_taxable = True
    else:
        incomes = tax_units["income"].values
        use_taxable = False

    for income, status in zip(incomes, tax_units["filing_status_clean"]):
        try:
            tax_info = calculator.calculate_tax(income, year, status, use_taxable_income=use_taxable)
            liabilities.append(tax_info["tax_liability"])
        except Exception as e:
            logger.warning(f"Error calculating tax for income ${income:,.0f}, status {status}: {e}")
            liabilities.append(0)

    liabilities = np.array(liabilities)
    revenue = float(np.sum(liabilities * weights)) / 1e6  # Convert to millions
    return revenue


def main():
    """Compare 2017 vs 2026 tax systems applied to 2026 income distribution."""
    
    logger.info("Act 46 vs 2017 Tax System Comparison")
    logger.info("="*80)
    
    # Load 2026 tax units
    tax_units = load_tax_units()
    
    # Calculate revenue under different tax systems
    logger.info("\nCalculating revenue under different tax systems...")
    
    # 2017 tax system (pre-Act 46 baseline)
    logger.info("  Using 2017 tax brackets and deductions...")
    revenue_2017_system = calculate_revenue_by_year(tax_units, 2017)
    
    # 2024 Act 46 system (corrected with proper deductions and exemptions)
    logger.info("  Using Act 46 (2024 brackets, 2024 deductions, 2025 personal exemptions)...")
    revenue_2026_system = calculate_revenue_by_year(tax_units, 2026, use_corrected_act46=True)
    
    # Results
    logger.info("\n" + "="*80)
    logger.info("REVENUE COMPARISON RESULTS")
    logger.info("="*80)
    
    logger.info(f"\n2026 Income Distribution Applied To:")
    logger.info(f"  2017 Tax System: ${revenue_2017_system:,.1f}M")
    logger.info(f"  2026 Tax System (Act 46): ${revenue_2026_system:,.1f}M")
    
    revenue_loss = revenue_2017_system - revenue_2026_system
    loss_percentage = (revenue_loss / revenue_2017_system) * 100
    
    logger.info(f"\nAct 46 Impact:")
    logger.info(f"  Revenue Loss: ${revenue_loss:,.1f}M")
    logger.info(f"  Percentage Loss: {loss_percentage:.1f}%")
    
    # Key differences between systems
    logger.info(f"\nKey Tax System Differences (2017 → Act 46):")
    logger.info(f"  Standard Deductions (2017 → 2024):")
    logger.info(f"    Joint: $4,400 → $8,800 (+$4,400)")
    logger.info(f"    Single: $2,200 → $4,400 (+$2,200)")
    logger.info(f"    HoH: $3,212 → $6,424 (+$3,212)")
    
    logger.info(f"\n  Personal Exemptions:")
    logger.info(f"    2017: $1,144 per person")
    logger.info(f"    2025: $1,200 per person (+$56)")
    
    logger.info(f"\n  Top Bracket Changes:")
    logger.info(f"    2017: 11.0% on $400k+ (Joint)")
    logger.info(f"    Act 46 (2024): 11.0% on $650k+ (Joint)")
    logger.info(f"    → Top rate threshold increased by $250k")
    
    logger.info(f"\n  Bracket Structure:")
    logger.info(f"    2017: More brackets, lower thresholds")
    logger.info(f"    2026: Fewer brackets, higher thresholds")
    
    # Save results
    results = {
        'tax_system': ['2017_system_2026_income', '2026_system_act46'],
        'revenue_millions': [revenue_2017_system, revenue_2026_system],
        'description': [
            '2017 tax brackets/deductions applied to 2026 income distribution',
            '2026 Act 46 tax brackets/deductions applied to 2026 income distribution'
        ]
    }
    
    results_df = pd.DataFrame(results)
    output_path = Path("data/processed/projections/act46_vs_2017_comparison.csv")
    results_df.to_csv(output_path, index=False)
    
    logger.info(f"\nResults saved to: {output_path}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
