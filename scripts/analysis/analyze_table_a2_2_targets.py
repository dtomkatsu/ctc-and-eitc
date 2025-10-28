#!/usr/bin/env python3
"""
Analyze Table A2-2 Targets for Income-Sensitive Adjustments

This script analyzes the DOTAX Table A2-2 data to determine:
1. Tax liability targets by filing status and AGI bracket
2. Required adjustment factors to match these targets
3. Income-sensitive adjustment recommendations
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def load_table_a2_2():
    """Load and parse Table A2-2 data"""
    
    # Manual data entry from Table A2-2 (Tax Liability Before Credits)
    data = {
        'agi_bracket': [
            '$0k-$10k', '$10k-$20k', '$20k-$30k', '$30k-$40k', '$40k-$50k',
            '$50k-$75k', '$75k-$100k', '$100k-$150k', '$150k-$200k', 
            '$200k-$300k', '$300k-$400k', '$400k+'
        ],
        'joint_tax_liability': [0.01, 0.9, 4, 10, 17, 69, 109, 269, 230, 251, 118, 596],
        'single_tax_liability': [3, 19, 41, 67, 80, 182, 122, 135, 52, 50, 29, 375],
        'hoh_tax_liability': [0.1, 2, 6, 14, 19, 42, 31, 34, 12, 10, 5, 26]
    }
    
    return pd.DataFrame(data)

def analyze_current_vs_targets():
    """Compare current model results to Table A2-2 targets"""
    
    print("="*80)
    print("TABLE A2-2 ANALYSIS: Tax Liability by Filing Status and AGI Bracket")
    print("="*80)
    
    # Load Table A2-2 targets
    targets = load_table_a2_2()
    
    # Load latest validation results
    validation_files = list(Path('data/processed').glob('validation_table_12a_*.csv'))
    if not validation_files:
        print("❌ No validation files found. Run regenerate_tax_units.py first.")
        return
    
    latest_file = max(validation_files, key=lambda x: x.stat().st_mtime)
    print(f"📊 Using validation file: {latest_file.name}")
    
    validation = pd.read_csv(latest_file)
    
    print("\nTABLE A2-2 TARGETS (Tax Liability Before Credits, $M):")
    print("-" * 80)
    print(f"{'AGI Bracket':<15} {'Joint ($M)':<12} {'Single ($M)':<12} {'HoH ($M)':<12}")
    print("-" * 80)
    
    total_joint = 0
    total_single = 0  
    total_hoh = 0
    
    for _, row in targets.iterrows():
        bracket = row['agi_bracket']
        joint = row['joint_tax_liability']
        single = row['single_tax_liability']
        hoh = row['hoh_tax_liability']
        
        print(f"{bracket:<15} ${joint:<11.1f} ${single:<11.1f} ${hoh:<11.1f}")
        
        total_joint += joint
        total_single += single
        total_hoh += hoh
    
    print("-" * 80)
    print(f"{'TOTAL':<15} ${total_joint:<11.1f} ${total_single:<11.1f} ${total_hoh:<11.1f}")
    
    # Compare to current model (from validation table)
    print(f"\nCURRENT MODEL vs TABLE A2-2 TARGETS:")
    print("-" * 80)
    
    # Map validation brackets to A2-2 brackets
    bracket_mapping = {
        '$0k-$10k': '$0k-$10k',
        '$10k-$20k': '$10k-$20k', 
        '$20k-$30k': '$20k-$30k',
        '$30k-$40k': '$30k-$40k',
        '$40k-$50k': '$40k-$50k',
        '$50k-$75k': '$50k-$75k',
        '$75k-$100k': '$75k-$100k',
        '$100k-$150k': '$100k-$150k',
        '$150k-$200k': '$150k-$200k',
        '$200k-$300k': '$200k-$300k',
        '$300k-$400k': '$300k-$400k',
        '$400k+': '$400k+'
    }
    
    print(f"{'Bracket':<15} {'Status':<8} {'Target':<10} {'Model':<10} {'Gap':<10} {'% Diff':<8}")
    print("-" * 80)
    
    # Calculate gaps for each bracket and filing status
    adjustment_recommendations = []
    
    for _, target_row in targets.iterrows():
        bracket = target_row['agi_bracket']
        
        # Find corresponding validation row
        val_row = validation[validation['agi_bracket'] == bracket]
        if len(val_row) == 0:
            continue
        val_row = val_row.iloc[0]
        
        model_total = val_row['model_tax_millions']
        
        # Estimate filing status breakdown (rough approximation)
        # Joint: ~55%, Single: ~35%, HoH: ~10% of total tax
        joint_target = target_row['joint_tax_liability']
        single_target = target_row['single_tax_liability'] 
        hoh_target = target_row['hoh_tax_liability']
        total_target = joint_target + single_target + hoh_target
        
        if total_target > 0:
            joint_model_est = model_total * (joint_target / total_target)
            single_model_est = model_total * (single_target / total_target)
            hoh_model_est = model_total * (hoh_target / total_target)
            
            # Calculate gaps
            joint_gap = joint_model_est - joint_target
            single_gap = single_model_est - single_target
            hoh_gap = hoh_model_est - hoh_target
            
            joint_pct = (joint_gap / joint_target * 100) if joint_target > 0 else 0
            single_pct = (single_gap / single_target * 100) if single_target > 0 else 0
            hoh_pct = (hoh_gap / hoh_target * 100) if hoh_target > 0 else 0
            
            print(f"{bracket:<15} {'Joint':<8} ${joint_target:<9.1f} ${joint_model_est:<9.1f} ${joint_gap:<9.1f} {joint_pct:<7.1f}%")
            print(f"{'':<15} {'Single':<8} ${single_target:<9.1f} ${single_model_est:<9.1f} ${single_gap:<9.1f} {single_pct:<7.1f}%")
            print(f"{'':<15} {'HoH':<8} ${hoh_target:<9.1f} ${hoh_model_est:<9.1f} ${hoh_gap:<9.1f} {hoh_pct:<7.1f}%")
            print()
            
            # Store recommendations
            if abs(joint_pct) > 10:
                adjustment_recommendations.append({
                    'bracket': bracket,
                    'filing_status': 'joint',
                    'gap_pct': joint_pct,
                    'recommended_adjustment': -joint_pct / 100 * 0.5  # 50% of gap
                })
            
            if abs(single_pct) > 10:
                adjustment_recommendations.append({
                    'bracket': bracket,
                    'filing_status': 'single',
                    'gap_pct': single_pct,
                    'recommended_adjustment': -single_pct / 100 * 0.5
                })
            
            if abs(hoh_pct) > 10:
                adjustment_recommendations.append({
                    'bracket': bracket,
                    'filing_status': 'hoh',
                    'gap_pct': hoh_pct,
                    'recommended_adjustment': -hoh_pct / 100 * 0.5
                })
    
    # Generate adjustment recommendations
    print("\n" + "="*80)
    print("RECOMMENDED INCOME-SENSITIVE ADJUSTMENT FACTORS")
    print("="*80)
    
    if adjustment_recommendations:
        print("\nBrackets needing >10% adjustment:")
        print("-" * 60)
        print(f"{'Bracket':<15} {'Status':<8} {'Gap %':<8} {'Adj Factor':<12}")
        print("-" * 60)
        
        for rec in adjustment_recommendations:
            print(f"{rec['bracket']:<15} {rec['filing_status']:<8} {rec['gap_pct']:<7.1f}% {rec['recommended_adjustment']:<11.3f}")
        
        print("\nImplementation Strategy:")
        print("-" * 60)
        print("1. Increase itemized deduction rates for middle-income brackets ($50k-$200k)")
        print("2. Boost AGI adjustment factors for joint filers (higher contribution limits)")
        print("3. Apply filing-status-specific multipliers in deduction calculations")
        print("4. Focus on brackets with largest gaps first")
        
        # Calculate average adjustments needed by income range
        low_income_adj = np.mean([r['recommended_adjustment'] for r in adjustment_recommendations 
                                 if r['bracket'] in ['$0k-$10k', '$10k-$20k', '$20k-$30k']])
        mid_income_adj = np.mean([r['recommended_adjustment'] for r in adjustment_recommendations 
                                 if r['bracket'] in ['$30k-$40k', '$40k-$50k', '$50k-$75k', '$75k-$100k']])
        high_income_adj = np.mean([r['recommended_adjustment'] for r in adjustment_recommendations 
                                  if r['bracket'] in ['$100k-$150k', '$150k-$200k', '$200k-$300k']])
        
        print(f"\nAverage adjustment factors needed:")
        print(f"  Low income (<$30k):   {low_income_adj:.3f}")
        print(f"  Middle income ($30k-$100k): {mid_income_adj:.3f}")  
        print(f"  High income ($100k-$300k):  {high_income_adj:.3f}")
        
    else:
        print("✅ All brackets within ±10% of targets!")
    
    return adjustment_recommendations

if __name__ == '__main__':
    analyze_current_vs_targets()
