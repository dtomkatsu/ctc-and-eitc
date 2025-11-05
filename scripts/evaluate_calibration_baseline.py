#!/usr/bin/env python3
"""
Evaluate Current Calibration Baseline

Runs the full pipeline once with current parameters and shows 
detailed breakdown of errors vs DOTAX targets for all brackets.

This gives us the starting point before optimization.
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load the latest calibrated tax units
def find_latest_tax_units():
    """Find the most recent calibrated tax units file."""
    processed_dir = project_root / 'data' / 'processed'
    files = list(processed_dir.glob('tax_units_filing_status_calibrated_*.parquet'))
    if not files:
        raise FileNotFoundError("No calibrated tax units found. Run regenerate_tax_units.py first.")
    return max(files, key=lambda p: p.stat().st_mtime)

def load_dotax_targets():
    """Load DOTAX Table A8 targets."""
    targets = [
        {'bracket': '$0-$10k', 'min': 0, 'max': 10_000, 
         'filers': 129_257, 'agi_m': 0.618, 'tax_m': 3.0},
        {'bracket': '$10-$20k', 'min': 10_000, 'max': 20_000, 
         'filers': 64_037, 'agi_m': 0.945, 'tax_m': 21.0},
        {'bracket': '$20-$30k', 'min': 20_000, 'max': 30_000, 
         'filers': 57_821, 'agi_m': 1.444, 'tax_m': 51.0},
        {'bracket': '$30-$40k', 'min': 30_000, 'max': 40_000, 
         'filers': 59_992, 'agi_m': 2.093, 'tax_m': 92.0},
        {'bracket': '$40-$50k', 'min': 40_000, 'max': 50_000, 
         'filers': 53_533, 'agi_m': 2.388, 'tax_m': 116.0},
        {'bracket': '$50-$75k', 'min': 50_000, 'max': 75_000, 
         'filers': 91_537, 'agi_m': 5.666, 'tax_m': 293.0},
        {'bracket': '$75-$100k', 'min': 75_000, 'max': 100_000, 
         'filers': 54_945, 'agi_m': 4.729, 'tax_m': 261.0},
        {'bracket': '$100-$150k', 'min': 100_000, 'max': 150_000, 
         'filers': 62_066, 'agi_m': 7.474, 'tax_m': 438.0},
        {'bracket': '$150-$200k', 'min': 150_000, 'max': 200_000, 
         'filers': 28_013, 'agi_m': 4.823, 'tax_m': 294.0},
        {'bracket': '$200-$300k', 'min': 200_000, 'max': 300_000, 
         'filers': 18_915, 'agi_m': 4.566, 'tax_m': 310.0},
        {'bracket': '$300-$400k', 'min': 300_000, 'max': 400_000, 
         'filers': 6_076, 'agi_m': 2.085, 'tax_m': 153.0},
        {'bracket': '$400k+', 'min': 400_000, 'max': float('inf'), 
         'filers': 8_875, 'agi_m': 8.800, 'tax_m': 998.0},
    ]
    return targets

def evaluate_baseline():
    """Evaluate current calibration against targets."""
    # Load latest tax units
    tax_units_path = find_latest_tax_units()
    print(f"Loading: {tax_units_path.name}")
    print()
    
    tax_units = pd.read_parquet(tax_units_path)
    targets = load_dotax_targets()
    
    # Evaluate each bracket
    print("="*100)
    print("BASELINE CALIBRATION EVALUATION")
    print("="*100)
    print()
    print(f"{'Bracket':<15} {'Metric':<8} {'Target':>12} {'Model':>12} {'Diff':>12} {'% Error':>10} {'Status':>8}")
    print("-"*100)
    
    total_errors = {'filers': 0, 'agi': 0, 'tax': 0}
    bracket_count = 0
    within_10pct_tax = 0
    within_5pct_filers = 0
    
    for target in targets:
        bracket_name = target['bracket']
        
        # Create bracket mask
        if target['max'] == float('inf'):
            mask = tax_units['agi'] >= target['min']
        else:
            mask = (tax_units['agi'] >= target['min']) & (tax_units['agi'] < target['max'])
        
        # Calculate model values
        model_filers = tax_units.loc[mask, 'weight'].sum()
        model_agi_m = (tax_units.loc[mask, 'agi'] * tax_units.loc[mask, 'weight']).sum() / 1_000_000
        model_tax_m = (tax_units.loc[mask, 'hi_state_tax'] * tax_units.loc[mask, 'weight']).sum() / 1_000_000
        
        # Calculate errors
        filer_error_pct = (model_filers - target['filers']) / target['filers'] * 100
        agi_error_pct = (model_agi_m - target['agi_m']) / target['agi_m'] * 100 if target['agi_m'] > 0 else 0
        tax_error_pct = (model_tax_m - target['tax_m']) / target['tax_m'] * 100 if target['tax_m'] > 0 else 0
        
        # Track metrics
        total_errors['filers'] += abs(filer_error_pct)
        total_errors['agi'] += abs(agi_error_pct)
        total_errors['tax'] += abs(tax_error_pct)
        bracket_count += 1
        
        if abs(tax_error_pct) <= 10:
            within_10pct_tax += 1
        if abs(filer_error_pct) <= 5:
            within_5pct_filers += 1
        
        # Status indicators
        def get_status(error_pct):
            if abs(error_pct) <= 5:
                return "✅"
            elif abs(error_pct) <= 10:
                return "⚠️"
            else:
                return "❌"
        
        # Print filer count
        print(f"{bracket_name:<15} {'Filers':<8} {target['filers']:>12,.0f} {model_filers:>12,.0f} "
              f"{model_filers-target['filers']:>+12,.0f} {filer_error_pct:>+9.1f}% {get_status(filer_error_pct):>8}")
        
        # Print AGI
        print(f"{'':<15} {'AGI ($M)':<8} {target['agi_m']:>12,.1f} {model_agi_m:>12,.1f} "
              f"{model_agi_m-target['agi_m']:>+12,.1f} {agi_error_pct:>+9.1f}% {get_status(agi_error_pct):>8}")
        
        # Print tax
        print(f"{'':<15} {'Tax ($M)':<8} {target['tax_m']:>12,.1f} {model_tax_m:>12,.1f} "
              f"{model_tax_m-target['tax_m']:>+12,.1f} {tax_error_pct:>+9.1f}% {get_status(tax_error_pct):>8}")
        
        print()
    
    # Summary statistics
    print("="*100)
    print("SUMMARY STATISTICS")
    print("="*100)
    print(f"Average absolute error:")
    print(f"  Filer counts: {total_errors['filers']/bracket_count:.1f}%")
    print(f"  AGI:          {total_errors['agi']/bracket_count:.1f}%")
    print(f"  Tax liability: {total_errors['tax']/bracket_count:.1f}%")
    print()
    print(f"Brackets within thresholds:")
    print(f"  Tax liability ±10%:  {within_10pct_tax}/{bracket_count} ({within_10pct_tax/bracket_count*100:.1f}%)")
    print(f"  Filer counts ±5%:    {within_5pct_filers}/{bracket_count} ({within_5pct_filers/bracket_count*100:.1f}%)")
    print()
    
    # Calculate total errors
    total_filers_target = sum(t['filers'] for t in targets)
    total_filers_model = tax_units['weight'].sum()
    total_agi_target = sum(t['agi_m'] for t in targets)
    total_agi_model = (tax_units['agi'] * tax_units['weight']).sum() / 1_000_000
    total_tax_target = sum(t['tax_m'] for t in targets)
    total_tax_model = (tax_units['hi_state_tax'] * tax_units['weight']).sum() / 1_000_000
    
    print(f"Total accuracy:")
    print(f"  Filers: {total_filers_model:,.0f} vs {total_filers_target:,.0f} ({(total_filers_model-total_filers_target)/total_filers_target*100:+.2f}%)")
    print(f"  AGI:    ${total_agi_model:,.0f}M vs ${total_agi_target:,.0f}M ({(total_agi_model-total_agi_target)/total_agi_target*100:+.2f}%)")
    print(f"  Tax:    ${total_tax_model:,.0f}M vs ${total_tax_target:,.0f}M ({(total_tax_model-total_tax_target)/total_tax_target*100:+.2f}%)")
    print("="*100)
    
    return {
        'avg_filer_error': total_errors['filers']/bracket_count,
        'avg_agi_error': total_errors['agi']/bracket_count,
        'avg_tax_error': total_errors['tax']/bracket_count,
        'pct_within_10pct_tax': within_10pct_tax/bracket_count*100,
        'pct_within_5pct_filers': within_5pct_filers/bracket_count*100,
    }


if __name__ == '__main__':
    evaluate_baseline()
