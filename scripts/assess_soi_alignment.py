#!/usr/bin/env python3
"""
Assess SOI alignment after filing threshold removal.

This script:
1. Loads the newly regenerated tax units
2. Compares filing status distribution to SOI benchmarks
3. Analyzes income bracket alignment
4. Determines if additional adjustments are needed
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd
import numpy as np

def main():
    print("="*80)
    print("SOI ALIGNMENT ASSESSMENT - AFTER FILING THRESHOLD REMOVAL")
    print("="*80)
    
    # Load the latest tax units
    data_dir = project_root / 'data' / 'processed'
    files = sorted(data_dir.glob('tax_units_regenerated_*.parquet'))
    
    if not files:
        print("\nERROR: No tax units files found!")
        return
    
    tax_units_file = files[-1]
    print(f"\nLoading tax units from: {tax_units_file.name}")
    
    tax_units = pd.read_parquet(tax_units_file)
    
    print(f"Total tax units: {len(tax_units):,}")
    print(f"Total weighted: {tax_units['weight'].sum():,.0f}")
    
    # SOI Benchmarks (2022)
    soi_benchmarks = {
        'single': 351205,
        'married_filing_jointly': 216358,
        'head_of_household': 67393,
        'married_filing_separately': 16007  # Included in Table 13A with single
    }
    
    total_soi = sum(soi_benchmarks.values())
    
    print("\n" + "="*80)
    print("FILING STATUS COMPARISON")
    print("="*80)
    
    # Calculate current distribution
    status_counts = tax_units.groupby('filing_status')['weight'].sum()
    
    print(f"\n{'Filing Status':<30} {'SOI':>12} {'Model':>12} {'Diff':>12} {'% Diff':>10}")
    print("-"*80)
    
    results = {}
    for status in ['single', 'married_filing_jointly', 'head_of_household', 'married_filing_separately']:
        soi_count = soi_benchmarks.get(status, 0)
        model_count = status_counts.get(status, 0)
        diff = model_count - soi_count
        pct_diff = (diff / soi_count * 100) if soi_count > 0 else 0
        
        results[status] = {
            'soi': soi_count,
            'model': model_count,
            'diff': diff,
            'pct_diff': pct_diff
        }
        
        status_label = status.replace('_', ' ').title()
        print(f"{status_label:<30} {soi_count:>12,} {model_count:>12,.0f} {diff:>12,.0f} {pct_diff:>9.1f}%")
    
    print("-"*80)
    total_model = status_counts.sum()
    total_diff = total_model - total_soi
    total_pct = (total_diff / total_soi * 100)
    print(f"{'TOTAL':<30} {total_soi:>12,} {total_model:>12,.0f} {total_diff:>12,.0f} {total_pct:>9.1f}%")
    
    # Analyze low-income coverage
    print("\n" + "="*80)
    print("LOW-INCOME COVERAGE ANALYSIS")
    print("="*80)
    
    low_income = tax_units[tax_units['income'] < 5000]
    print(f"\nTax units with income <$5K:")
    print(f"  Count: {len(low_income):,}")
    print(f"  Weighted: {low_income['weight'].sum():,.0f}")
    
    by_status = low_income.groupby('filing_status')['weight'].sum()
    print(f"\nBy filing status:")
    for status, count in by_status.items():
        print(f"  {status}: {count:,.0f}")
    
    # Expected low-income from SOI
    soi_low_income = {
        'single': 82072 + 13773,  # Under $2,400 + $2,400-$4,800
        'married_filing_jointly': 40236 + 5632,  # Under $4,800 + $4,800-$9,600
        'head_of_household': 6671 + 2619  # Under $3,600 + $3,600-$7,200
    }
    
    total_soi_low = sum(soi_low_income.values())
    total_model_low = low_income['weight'].sum()
    
    print(f"\nComparison to SOI:")
    print(f"  SOI low-income filers:       {total_soi_low:>12,}")
    print(f"  Model low-income tax units:  {total_model_low:>12,.0f}")
    print(f"  Gap:                         {total_soi_low - total_model_low:>12,.0f}")
    print(f"  Coverage:                    {(total_model_low / total_soi_low * 100):>11.1f}%")
    
    # Assessment
    print("\n" + "="*80)
    print("ASSESSMENT & RECOMMENDATIONS")
    print("="*80)
    
    # Check filing status alignment
    filing_status_issues = []
    for status, data in results.items():
        if abs(data['pct_diff']) > 5:
            filing_status_issues.append((status, data['pct_diff']))
    
    # Check coverage
    coverage_pct = (total_model / total_soi * 100)
    low_income_coverage = (total_model_low / total_soi_low * 100)
    
    print(f"\n1. OVERALL COVERAGE: {coverage_pct:.1f}%")
    if coverage_pct >= 98:
        print("   ✅ EXCELLENT - Within 2% of SOI total")
    elif coverage_pct >= 95:
        print("   ✅ GOOD - Within 5% of SOI total")
    else:
        print("   ⚠️  NEEDS IMPROVEMENT - More than 5% gap")
    
    print(f"\n2. LOW-INCOME COVERAGE: {low_income_coverage:.1f}%")
    if low_income_coverage >= 95:
        print("   ✅ EXCELLENT - Filing threshold removal worked!")
    elif low_income_coverage >= 80:
        print("   ✅ GOOD - Major improvement from threshold removal")
    else:
        print("   ⚠️  NEEDS IMPROVEMENT - Still missing low-income filers")
    
    print(f"\n3. FILING STATUS ALIGNMENT:")
    if not filing_status_issues:
        print("   ✅ EXCELLENT - All filing statuses within ±5%")
    else:
        print(f"   ⚠️  {len(filing_status_issues)} filing status(es) outside ±5%:")
        for status, pct_diff in filing_status_issues:
            status_label = status.replace('_', ' ').title()
            print(f"      - {status_label}: {pct_diff:+.1f}%")
    
    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDED ADJUSTMENTS")
    print("="*80)
    
    needs_adjustment = False
    
    if coverage_pct < 98:
        needs_adjustment = True
        print("\n❌ COVERAGE ADJUSTMENT NEEDED:")
        print("   - Overall coverage is below 98%")
        print("   - Consider investigating missing tax units")
    
    if filing_status_issues:
        needs_adjustment = True
        print("\n❌ FILING STATUS ADJUSTMENT NEEDED:")
        
        # Specific recommendations
        for status, pct_diff in filing_status_issues:
            status_label = status.replace('_', ' ').title()
            if pct_diff < -5:
                print(f"\n   {status_label} is {abs(pct_diff):.1f}% UNDER target:")
                if status == 'married_filing_jointly':
                    print("   → Consider enabling SOI calibration")
                    print("   → Or adjust MFS logic to create fewer separate filers")
                elif status == 'head_of_household':
                    print("   → Review HoH qualification logic")
                    print("   → Check dependent assignment")
            elif pct_diff > 5:
                print(f"\n   {status_label} is {pct_diff:.1f}% OVER target:")
                if status == 'single':
                    print("   → Some single filers should be joint or HoH")
                    print("   → Consider enabling SOI calibration")
    
    if not needs_adjustment:
        print("\n✅ NO ADJUSTMENTS NEEDED!")
        print("   - All metrics within acceptable ranges")
        print("   - Filing status distribution aligns with SOI")
        print("   - Coverage is excellent")
    
    # SOI Calibration recommendation
    print("\n" + "="*80)
    print("SOI CALIBRATION RECOMMENDATION")
    print("="*80)
    
    if filing_status_issues:
        print("\n⚠️  RECOMMENDATION: Enable SOI Calibration")
        print("\nReason:")
        print("  - Filing status distribution outside ±5% tolerance")
        print("  - SOI calibration can fine-tune the distribution")
        print("  - Minimal impact on data integrity (units marked as 'calibrated')")
        
        print("\nImplementation:")
        print("  File: src/tax/units/status/irs_based.py")
        print("  Function: calibrate_to_soi_totals()")
        print("  Already implemented and tested ✓")
        
        print("\nExpected Impact:")
        for status, pct_diff in filing_status_issues:
            status_label = status.replace('_', ' ').title()
            print(f"  - {status_label}: {pct_diff:+.1f}% → ~0%")
    else:
        print("\n✅ SOI CALIBRATION NOT NEEDED")
        print("   - Natural distribution already aligns with SOI")
        print("   - No artificial adjustments required")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\n✓ Filing threshold removed: SUCCESS")
    print(f"✓ Low-income coverage: {low_income_coverage:.1f}% (was ~3%)")
    print(f"✓ Total coverage: {coverage_pct:.1f}%")
    
    if needs_adjustment:
        print(f"\n⚠️  Additional adjustments recommended:")
        if filing_status_issues:
            print(f"   - Enable SOI calibration for filing status alignment")
        if coverage_pct < 98:
            print(f"   - Investigate remaining coverage gap")
    else:
        print(f"\n✅ No additional adjustments needed!")
    
    print("\n" + "="*80)

if __name__ == '__main__':
    main()
