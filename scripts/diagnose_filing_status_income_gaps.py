"""
Diagnose income distribution gaps by filing status.

Compares model AGI and tax liability per filing status against DOTAX SOI Table 5A
to identify why revenue gaps persist even after filing status calibration.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# DOTAX SOI Table 5A - Hawaii AGI and Tax Liability by Filing Status (2022)
DOTAX_BENCHMARKS = {
    'Joint': {
        'returns': 216358,
        'agi_millions': 26551,
        'tax_millions': 1674,
    },
    'Single': {
        'returns': 335198,
        'agi_millions': 14297,
        'tax_millions': 864,
    },
    'MFS': {
        'returns': 16007,
        'agi_millions': 3149,
        'tax_millions': 289,
    },
    'Head of Household': {
        'returns': 67393,
        'agi_millions': 3744,
        'tax_millions': 202,
    },
}

def standardize_filing_status(df):
    """Standardize filing status names."""
    status_map = {
        'single': 'Single',
        'Single': 'Single',
        'married_filing_jointly': 'Joint',
        'Married Filing Jointly': 'Joint',
        'married_filing_separately': 'MFS',
        'Married Filing Separately': 'MFS',
        'head_of_household': 'Head of Household',
        'Head of Household': 'Head of Household',
        'Head Of Household': 'Head of Household',
        'qualifying_widow': 'Joint',
    }
    
    return df['filing_status'].map(status_map).fillna('Other')

def main():
    print("=" * 100)
    print("FILING STATUS INCOME DISTRIBUTION DIAGNOSTIC")
    print("=" * 100)
    print()
    
    # Find most recent calibrated file
    processed_dir = Path('data/processed')
    calibrated_files = sorted(processed_dir.glob('tax_units_filing_status_calibrated_*.parquet'))
    
    if not calibrated_files:
        print("Error: No calibrated tax units file found")
        return
    
    input_file = calibrated_files[-1]
    print(f"Input file: {input_file}")
    print()
    
    # Load data
    df = pd.read_parquet(input_file)
    print(f"Loaded {len(df):,} tax units")
    print()
    
    # Standardize filing status
    df['filing_status_clean'] = standardize_filing_status(df)
    
    # Calculate weighted statistics by filing status
    print("=" * 100)
    print("MODEL vs DOTAX COMPARISON BY FILING STATUS")
    print("=" * 100)
    print()
    
    results = []
    
    for status in ['Single', 'Joint', 'Head of Household', 'MFS']:
        status_df = df[df['filing_status_clean'] == status].copy()
        
        if len(status_df) == 0:
            print(f"⚠️  No units found for {status}")
            continue
        
        # Model statistics
        model_returns = (status_df['weight'].sum())
        model_agi_total = (status_df['agi'] * status_df['weight']).sum() / 1_000_000
        model_tax_total = (status_df['hi_state_tax'] * status_df['weight']).sum() / 1_000_000
        
        model_avg_agi = model_agi_total * 1_000_000 / model_returns if model_returns > 0 else 0
        model_avg_tax = model_tax_total * 1_000_000 / model_returns if model_returns > 0 else 0
        
        # DOTAX benchmarks
        dotax = DOTAX_BENCHMARKS[status]
        dotax_returns = dotax['returns']
        dotax_agi_total = dotax['agi_millions']
        dotax_tax_total = dotax['tax_millions']
        
        dotax_avg_agi = dotax_agi_total * 1_000_000 / dotax_returns
        dotax_avg_tax = dotax_tax_total * 1_000_000 / dotax_returns
        
        # Calculate gaps
        returns_gap = model_returns - dotax_returns
        returns_gap_pct = (returns_gap / dotax_returns * 100) if dotax_returns > 0 else 0
        
        agi_total_gap = model_agi_total - dotax_agi_total
        agi_total_gap_pct = (agi_total_gap / dotax_agi_total * 100) if dotax_agi_total > 0 else 0
        
        agi_avg_gap = model_avg_agi - dotax_avg_agi
        agi_avg_gap_pct = (agi_avg_gap / dotax_avg_agi * 100) if dotax_avg_agi > 0 else 0
        
        tax_total_gap = model_tax_total - dotax_tax_total
        tax_total_gap_pct = (tax_total_gap / dotax_tax_total * 100) if dotax_tax_total > 0 else 0
        
        tax_avg_gap = model_avg_tax - dotax_avg_tax
        tax_avg_gap_pct = (tax_avg_gap / dotax_avg_tax * 100) if dotax_avg_tax > 0 else 0
        
        # Effective tax rate
        model_eff_rate = (model_avg_tax / model_avg_agi * 100) if model_avg_agi > 0 else 0
        dotax_eff_rate = (dotax_avg_tax / dotax_avg_agi * 100) if dotax_avg_agi > 0 else 0
        eff_rate_gap = model_eff_rate - dotax_eff_rate
        
        results.append({
            'status': status,
            'model_returns': model_returns,
            'dotax_returns': dotax_returns,
            'returns_gap_pct': returns_gap_pct,
            'model_agi_total': model_agi_total,
            'dotax_agi_total': dotax_agi_total,
            'agi_total_gap_pct': agi_total_gap_pct,
            'model_avg_agi': model_avg_agi,
            'dotax_avg_agi': dotax_avg_agi,
            'agi_avg_gap_pct': agi_avg_gap_pct,
            'model_tax_total': model_tax_total,
            'dotax_tax_total': dotax_tax_total,
            'tax_total_gap_pct': tax_total_gap_pct,
            'model_avg_tax': model_avg_tax,
            'dotax_avg_tax': dotax_avg_tax,
            'tax_avg_gap_pct': tax_avg_gap_pct,
            'model_eff_rate': model_eff_rate,
            'dotax_eff_rate': dotax_eff_rate,
            'eff_rate_gap': eff_rate_gap,
        })
        
        print(f"{'='*100}")
        print(f"{status.upper()}")
        print(f"{'='*100}")
        print()
        
        print(f"{'Metric':<30} {'Model':>15} {'DOTAX':>15} {'Gap':>15} {'Gap %':>10}")
        print("-" * 100)
        
        print(f"{'Returns':<30} {model_returns:>15,.0f} {dotax_returns:>15,} {returns_gap:>+15,.0f} {returns_gap_pct:>+9.1f}%")
        print()
        
        print(f"{'Total AGI ($M)':<30} {model_agi_total:>15,.1f} {dotax_agi_total:>15,.0f} {agi_total_gap:>+15,.1f} {agi_total_gap_pct:>+9.1f}%")
        print(f"{'Avg AGI per return':<30} ${model_avg_agi:>14,.0f} ${dotax_avg_agi:>14,.0f} ${agi_avg_gap:>+14,.0f} {agi_avg_gap_pct:>+9.1f}%")
        print()
        
        print(f"{'Total Tax ($M)':<30} {model_tax_total:>15,.1f} {dotax_tax_total:>15,.0f} {tax_total_gap:>+15,.1f} {tax_total_gap_pct:>+9.1f}%")
        print(f"{'Avg Tax per return':<30} ${model_avg_tax:>14,.0f} ${dotax_avg_tax:>14,.0f} ${tax_avg_gap:>+14,.0f} {tax_avg_gap_pct:>+9.1f}%")
        print()
        
        print(f"{'Effective Tax Rate':<30} {model_eff_rate:>14.2f}% {dotax_eff_rate:>14.2f}% {eff_rate_gap:>+14.2f}pp")
        print()
    
    # Summary table
    print("=" * 100)
    print("SUMMARY: KEY GAPS BY FILING STATUS")
    print("=" * 100)
    print()
    
    summary_df = pd.DataFrame(results)
    
    print(f"{'Status':<20} {'Returns Gap %':>15} {'Avg AGI Gap %':>15} {'Avg Tax Gap %':>15} {'Eff Rate Gap':>15}")
    print("-" * 100)
    
    for _, row in summary_df.iterrows():
        status_emoji = "✅" if abs(row['tax_avg_gap_pct']) < 10 else "⚠️" if abs(row['tax_avg_gap_pct']) < 20 else "❌"
        print(f"{row['status']:<20} {row['returns_gap_pct']:>+14.1f}% {row['agi_avg_gap_pct']:>+14.1f}% "
              f"{row['tax_avg_gap_pct']:>+14.1f}% {row['eff_rate_gap']:>+14.2f}pp {status_emoji}")
    
    print()
    print("=" * 100)
    print("KEY FINDINGS")
    print("=" * 100)
    print()
    
    # Identify key issues
    for _, row in summary_df.iterrows():
        status = row['status']
        
        print(f"{status}:")
        
        if abs(row['returns_gap_pct']) > 1:
            print(f"  • Returns count: {row['returns_gap_pct']:+.1f}% vs target (should be ~0% after calibration)")
        
        if abs(row['agi_avg_gap_pct']) > 5:
            direction = "lower" if row['agi_avg_gap_pct'] < 0 else "higher"
            print(f"  • Average AGI is {abs(row['agi_avg_gap_pct']):.1f}% {direction} than DOTAX")
        
        if abs(row['tax_avg_gap_pct']) > 5:
            direction = "lower" if row['tax_avg_gap_pct'] < 0 else "higher"
            print(f"  • Average tax is {abs(row['tax_avg_gap_pct']):.1f}% {direction} than DOTAX")
        
        if abs(row['eff_rate_gap']) > 0.5:
            direction = "lower" if row['eff_rate_gap'] < 0 else "higher"
            print(f"  • Effective tax rate is {abs(row['eff_rate_gap']):.2f}pp {direction} than DOTAX")
        
        print()
    
    print("=" * 100)
    print("RECOMMENDATIONS")
    print("=" * 100)
    print()
    
    for _, row in summary_df.iterrows():
        status = row['status']
        
        if abs(row['tax_total_gap_pct']) > 10:
            print(f"{status}:")
            
            if row['agi_avg_gap_pct'] < -5:
                print(f"  → Incomes are too low ({row['agi_avg_gap_pct']:+.1f}%)")
                print(f"     Consider: Income distribution calibration for {status} filers")
            
            if row['eff_rate_gap'] < -0.5:
                print(f"  → Effective tax rate is too low ({row['eff_rate_gap']:+.2f}pp)")
                print(f"     Consider: Review deductions/credits applied to {status} filers")
            
            if row['returns_gap_pct'] < -5:
                print(f"  → Too few returns ({row['returns_gap_pct']:+.1f}%)")
                print(f"     Consider: Verify filing status calibration is working correctly")
            
            print()

if __name__ == '__main__':
    main()
