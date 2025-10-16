"""
Calculate Hawaii State Taxes on Calibrated Tax Units

This script:
1. Loads calibrated tax units
2. Calculates Hawaii state income tax for 2022
3. Generates reports by AGI bracket and filing status
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd
import numpy as np
import glob
import logging
from datetime import datetime

from src.tax.hawaii_calculator import HawaiiTaxCalculator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


def load_calibrated_tax_units():
    """Load the most recent calibrated tax units."""
    files = glob.glob('data/processed/tax_units_calibrated_*.parquet')
    if not files:
        raise FileNotFoundError("No calibrated tax units found. Run regenerate_tax_units.py first.")
    
    latest_file = max(files)
    logger.info(f"📂 Loading calibrated tax units from: {latest_file}")
    
    tax_units = pd.read_parquet(latest_file)
    logger.info(f"   Total tax units: {len(tax_units):,}")
    
    # Use calibrated weight if available
    if 'weight_calibrated' in tax_units.columns:
        logger.info(f"   Using calibrated weights")
        tax_units['weight'] = tax_units['weight_calibrated']
    
    weighted_total = tax_units['weight'].sum()
    logger.info(f"   Weighted total: {weighted_total:,.0f}")
    
    return tax_units


def define_agi_brackets():
    """Define AGI brackets for reporting (aligned with SOI)."""
    return [
        (0, 2400, "Under $2.4k"),
        (2400, 4800, "$2.4k - $4.8k"),
        (4800, 9600, "$4.8k - $9.6k"),
        (9600, 14400, "$9.6k - $14.4k"),
        (14400, 19200, "$14.4k - $19.2k"),
        (19200, 24000, "$19.2k - $24k"),
        (24000, 36000, "$24k - $36k"),
        (36000, 48000, "$36k - $48k"),
        (48000, 96000, "$48k - $96k"),
        (96000, 150000, "$96k - $150k"),
        (150000, 200000, "$150k - $200k"),
        (200000, 300000, "$200k - $300k"),
        (300000, float('inf'), "$300k+")
    ]


def assign_agi_bracket(agi, brackets):
    """Assign an AGI to a bracket."""
    for min_agi, max_agi, label in brackets:
        if min_agi <= agi < max_agi:
            return label
    return brackets[-1][2]  # Return highest bracket if above all


def generate_summary_by_filing_status(tax_units):
    """Generate summary statistics by filing status."""
    logger.info("\n" + "="*80)
    logger.info("HAWAII TAX SUMMARY BY FILING STATUS")
    logger.info("="*80)
    
    statuses = ['single', 'married_filing_jointly', 'head_of_household', 'married_filing_separately']
    status_labels = {
        'single': 'Single',
        'married_filing_jointly': 'Married Filing Jointly',
        'head_of_household': 'Head of Household',
        'married_filing_separately': 'Married Filing Separately'
    }
    
    summary_data = []
    
    for status in statuses:
        df = tax_units[tax_units['filing_status'] == status].copy()
        if len(df) == 0:
            continue
            
        weighted_count = df['weight'].sum()
        total_agi = (df['agi'] * df['weight']).sum()
        total_tax = (df['hi_state_tax'] * df['weight']).sum()
        
        summary_data.append({
            'Filing Status': status_labels[status],
            'Returns': f"{weighted_count:,.0f}",
            'Total AGI': f"${total_agi:,.0f}",
            'Total Tax': f"${total_tax:,.0f}",
            'Avg AGI': f"${total_agi/weighted_count:,.0f}",
            'Avg Tax': f"${total_tax/weighted_count:,.0f}",
            'Eff. Rate': f"{(total_tax/total_agi*100):.2f}%"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))
    
    # Overall totals
    total_returns = tax_units['weight'].sum()
    total_agi = (tax_units['agi'] * tax_units['weight']).sum()
    total_tax = (tax_units['hi_state_tax'] * tax_units['weight']).sum()
    
    logger.info("\n" + "-"*80)
    logger.info(f"OVERALL TOTALS:")
    logger.info(f"  Total Returns:     {total_returns:>15,.0f}")
    logger.info(f"  Total AGI:         ${total_agi:>15,.0f}")
    logger.info(f"  Total State Tax:   ${total_tax:>15,.0f}")
    logger.info(f"  Average AGI:       ${total_agi/total_returns:>15,.0f}")
    logger.info(f"  Average State Tax: ${total_tax/total_returns:>15,.0f}")
    logger.info(f"  Effective Rate:    {total_tax/total_agi*100:>15.2f}%")
    logger.info("="*80)
    
    return summary_df


def generate_agi_bracket_report(tax_units):
    """Generate detailed report by AGI bracket and filing status."""
    logger.info("\n" + "="*80)
    logger.info("RETURNS BY AGI BRACKET AND FILING STATUS")
    logger.info("="*80)
    
    brackets = define_agi_brackets()
    
    # Assign brackets
    tax_units['agi_bracket'] = tax_units['agi'].apply(
        lambda x: assign_agi_bracket(x, brackets)
    )
    
    statuses = ['single', 'married_filing_jointly', 'head_of_household', 'married_filing_separately']
    status_labels = {
        'single': 'Single',
        'married_filing_jointly': 'MFJ',
        'head_of_household': 'HoH',
        'married_filing_separately': 'MFS'
    }
    
    # Create pivot table
    report_data = []
    
    for _, max_val, label in brackets:
        row = {'AGI Bracket': label}
        
        for status in statuses:
            mask = (tax_units['filing_status'] == status) & (tax_units['agi_bracket'] == label)
            count = tax_units[mask]['weight'].sum()
            row[status_labels[status]] = f"{count:,.0f}"
        
        # Total across statuses
        mask = tax_units['agi_bracket'] == label
        total = tax_units[mask]['weight'].sum()
        row['Total'] = f"{total:,.0f}"
        
        report_data.append(row)
    
    # Add totals row
    totals_row = {'AGI Bracket': 'TOTAL'}
    for status in statuses:
        count = tax_units[tax_units['filing_status'] == status]['weight'].sum()
        totals_row[status_labels[status]] = f"{count:,.0f}"
    totals_row['Total'] = f"{tax_units['weight'].sum():,.0f}"
    report_data.append(totals_row)
    
    report_df = pd.DataFrame(report_data)
    print("\n" + report_df.to_string(index=False))
    
    return report_df


def generate_tax_by_bracket_report(tax_units):
    """Generate total tax liability by AGI bracket."""
    logger.info("\n" + "="*80)
    logger.info("TOTAL HAWAII STATE TAX BY AGI BRACKET")
    logger.info("="*80)
    
    brackets = define_agi_brackets()
    
    report_data = []
    
    for _, max_val, label in brackets:
        mask = tax_units['agi_bracket'] == label
        df = tax_units[mask]
        
        if len(df) == 0:
            continue
        
        returns = df['weight'].sum()
        total_agi = (df['agi'] * df['weight']).sum()
        total_tax = (df['hi_state_tax'] * df['weight']).sum()
        avg_agi = total_agi / returns if returns > 0 else 0
        avg_tax = total_tax / returns if returns > 0 else 0
        eff_rate = (total_tax / total_agi * 100) if total_agi > 0 else 0
        
        report_data.append({
            'AGI Bracket': label,
            'Returns': f"{returns:,.0f}",
            'Total AGI': f"${total_agi:,.0f}",
            'Total Tax': f"${total_tax:,.0f}",
            'Avg AGI': f"${avg_agi:,.0f}",
            'Avg Tax': f"${avg_tax:,.0f}",
            'Eff. Rate': f"{eff_rate:.2f}%"
        })
    
    report_df = pd.DataFrame(report_data)
    print("\n" + report_df.to_string(index=False))
    
    return report_df


def save_results(tax_units, summary_df, agi_bracket_df, tax_bracket_df):
    """Save results to files."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save detailed tax units
    output_file = f'data/processed/tax_units_with_hi_tax_{timestamp}.parquet'
    tax_units.to_parquet(output_file, index=False)
    logger.info(f"\n💾 Saved detailed results to: {output_file}")
    
    # Save CSV reports
    reports_dir = Path('analysis_results/hawaii_taxes')
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    summary_df.to_csv(reports_dir / f'summary_by_status_{timestamp}.csv', index=False)
    agi_bracket_df.to_csv(reports_dir / f'returns_by_agi_bracket_{timestamp}.csv', index=False)
    tax_bracket_df.to_csv(reports_dir / f'tax_by_agi_bracket_{timestamp}.csv', index=False)
    
    logger.info(f"💾 Saved CSV reports to: {reports_dir}")


def main():
    """Main execution function."""
    logger.info("\n" + "="*80)
    logger.info("HAWAII STATE TAX CALCULATION")
    logger.info("="*80)
    
    # Load calibrated tax units
    tax_units = load_calibrated_tax_units()
    
    # Initialize calculator
    logger.info(f"\n🧮 Initializing Hawaii tax calculator...")
    calculator = HawaiiTaxCalculator()
    
    # Calculate taxes
    logger.info(f"\n💵 Calculating Hawaii state taxes for 2022...")
    tax_units = calculator.calculate_tax_units_batch(tax_units)
    
    # Generate reports
    summary_df = generate_summary_by_filing_status(tax_units)
    agi_bracket_df = generate_agi_bracket_report(tax_units)
    tax_bracket_df = generate_tax_by_bracket_report(tax_units)
    
    # Save results
    save_results(tax_units, summary_df, agi_bracket_df, tax_bracket_df)
    
    logger.info(f"\n✅ Hawaii tax calculation complete!")


if __name__ == '__main__':
    main()
