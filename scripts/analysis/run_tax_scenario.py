#!/usr/bin/env python3
"""
General-purpose tax scenario runner using centralized configuration.
No need to write new scripts - just configure and run!
"""

import sys
from pathlib import Path
import pandas as pd
import logging

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tax.config import TaxSystemConfig, TaxSystemRegistry, TaxCalculator, compare_systems

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_tax_units(year: int = 2026) -> pd.DataFrame:
    """Load projected tax units for analysis."""
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
    """Run tax scenario analysis."""
    
    print("\n" + "="*100)
    print("TAX SCENARIO ANALYSIS - CENTRALIZED CONFIGURATION")
    print("="*100)
    
    # Load tax units
    logger.info("\nLoading tax units...")
    tax_units = load_tax_units(2026)
    
    # Initialize calculator
    logger.info("Initializing tax calculator...")
    calculator = TaxCalculator(PROJECT_ROOT)
    
    # ===================================================================
    # SCENARIO 1: 2017 vs Act 46 (2025)
    # ===================================================================
    print("\n" + "="*100)
    print("SCENARIO 1: 2017 Tax System vs Act 46 (2025)")
    print("="*100)
    
    system_2017 = TaxSystemRegistry.get_2017_system()
    system_act46 = TaxSystemRegistry.get_act46_2025_system()
    
    comparison = compare_systems(tax_units, system_2017, system_act46, calculator)
    print("\n" + comparison.to_string(index=False))
    
    # ===================================================================
    # SCENARIO 2: Act 46 (2025) vs Act 46 with Targeted Rollback
    # ===================================================================
    print("\n" + "="*100)
    print("SCENARIO 2: Act 46 (2025) vs Act 46 with Targeted Rollback")
    print("="*100)
    
    system_rollback = TaxSystemRegistry.get_act46_rollback_targeted(2025)
    
    comparison = compare_systems(tax_units, system_act46, system_rollback, calculator)
    print("\n" + comparison.to_string(index=False))
    
    # ===================================================================
    # SCENARIO 3: Custom Scenario Example
    # ===================================================================
    print("\n" + "="*100)
    print("SCENARIO 3: Custom Scenario - Future Projections (2027)")
    print("="*100)
    
    system_2027 = TaxSystemRegistry.get_act46_2027_system()
    
    revenue_2027 = calculator.calculate_revenue(tax_units, system_2027)
    
    print(f"\n2027 Projected Revenue (Act 46 system):")
    print(f"  Total revenue: ${revenue_2027['total_revenue_millions']:,.1f}M")
    print(f"  Average tax per filer: ${revenue_2027['average_tax_per_filer']:,.0f}")
    print(f"  Effective rate: {revenue_2027['effective_rate']:.2f}%")
    print(f"  Total filers: {revenue_2027['total_filers']:,.0f}")
    
    # ===================================================================
    # SCENARIO 4: Detailed Bracket-by-Bracket Analysis
    # ===================================================================
    print("\n" + "="*100)
    print("SCENARIO 4: Bracket Schedule Comparison")
    print("="*100)
    
    print("\n2017 System Brackets (Joint):")
    brackets_2017 = calculator.get_brackets(2018, 'married_filing_jointly')
    print(brackets_2017[['income_min', 'income_max', 'rate']].head(10).to_string(index=False))
    
    print("\n\nAct 46 (2025) Brackets (Joint):")
    brackets_2025 = calculator.get_brackets(2025, 'married_filing_jointly')
    print(brackets_2025[['income_min', 'income_max', 'rate']].head(10).to_string(index=False))
    
    print("\n" + "="*100)
    print("Analysis complete!")
    print("="*100 + "\n")


if __name__ == "__main__":
    main()
