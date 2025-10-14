"""
Investigate why MFS and HoH filers have lower incomes than SOI benchmarks.

This script analyzes the characteristics of MFS and HoH filers to understand
the income discrepancies.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.pums_loader import PUMSDataLoader
from src.tax.units.constructor import TaxUnitConstructor


def analyze_mfs_gap():
    """Analyze why MFS average income is 40% lower than SOI."""
    print("=" * 80)
    print("MFS INCOME GAP ANALYSIS")
    print("=" * 80)
    print()
    
    # Load data
    loader = PUMSDataLoader('data/raw/pums')
    person_df, hh_df = loader.load_data()
    
    # Construct tax units
    constructor = TaxUnitConstructor(person_df, hh_df, progress_bar=False)
    tax_units = constructor.create_rule_based_units(parallel=False)
    
    mfs_units = tax_units[tax_units['filing_status'] == 'married_filing_separately']
    
    print(f"Current MFS weighted avg income: ${(mfs_units['income'] * mfs_units['weight']).sum() / mfs_units['weight'].sum():,.0f}")
    print(f"SOI target: $195,040")
    print(f"Gap: {100 * (195040 - 116937) / 195040:.1f}% below target")
    print()
    
    # Hypothesis 1: Are we missing high-income MFS filers?
    print("HYPOTHESIS 1: Missing high-income MFS filers")
    print("-" * 80)
    
    # Check all married couples in PUMS
    person_df['person_id'] = person_df.apply(
        lambda row: f"{row['SERIALNO']}_{row['SPORDER']}", axis=1
    )
    person_df.set_index('person_id', inplace=True)
    
    # Find all married adults
    married_adults = person_df[(person_df['AGEP'] >= 18) & (person_df['MAR'] == 1)]
    print(f"Total married adults in PUMS: {len(married_adults):,}")
    
    # Find married couples (householder + spouse)
    married_couples = []
    for serialno in married_adults['SERIALNO'].unique():
        household = married_adults[married_adults['SERIALNO'] == serialno]
        householder = household[household['RELSHIPP'] == 20]
        spouse = household[household['RELSHIPP'] == 21]
        
        if len(householder) == 1 and len(spouse) == 1:
            h_income = householder.iloc[0].get('PINCP', 0) or 0
            s_income = spouse.iloc[0].get('PINCP', 0) or 0
            adjinc = householder.iloc[0].get('ADJINC', 1.0)
            
            if adjinc and adjinc > 0:
                h_income = h_income * adjinc
                s_income = s_income * adjinc
            
            total_income = h_income + s_income
            married_couples.append({
                'serialno': serialno,
                'h_income': h_income,
                's_income': s_income,
                'total_income': total_income,
                'weight': householder.iloc[0]['PWGTP']
            })
    
    couples_df = pd.DataFrame(married_couples)
    
    # High-income couples (>$300k combined)
    high_income_couples = couples_df[couples_df['total_income'] > 300000]
    print(f"High-income married couples (>$300k): {len(high_income_couples):,}")
    print(f"Weighted: {high_income_couples['weight'].sum():,.0f}")
    
    # How many of these are filing MFS in our model?
    mfs_serialnos = set()
    for _, unit in mfs_units.iterrows():
        person_id = unit['primary_filer_id']
        if person_id in person_df.index:
            serialno = person_df.loc[person_id, 'SERIALNO']
            mfs_serialnos.add(serialno)
    
    high_income_mfs = high_income_couples[high_income_couples['serialno'].isin(mfs_serialnos)]
    print(f"High-income couples filing MFS: {len(high_income_mfs):,}")
    print(f"Percentage: {100 * len(high_income_mfs) / len(high_income_couples):.1f}%")
    print()
    
    # Hypothesis 2: MFS logic is too conservative
    print("HYPOTHESIS 2: MFS identification logic is too conservative")
    print("-" * 80)
    
    # Count how many married couples are filing jointly vs separately
    joint_units = tax_units[tax_units['filing_status'] == 'married_filing_jointly']
    
    print(f"Married couples filing jointly: {len(joint_units):,} (weighted: {joint_units['weight'].sum():,.0f})")
    print(f"Married couples filing separately: {len(mfs_units):,} (weighted: {mfs_units['weight'].sum():,.0f})")
    print(f"MFS rate: {100 * mfs_units['weight'].sum() / (joint_units['weight'].sum() + mfs_units['weight'].sum()):.1f}%")
    print(f"SOI MFS rate: {100 * 2.5 / (34.1 + 2.5):.1f}%")
    print()
    
    # Check income distribution of joint vs MFS
    joint_avg = (joint_units['income'] * joint_units['weight']).sum() / joint_units['weight'].sum()
    mfs_avg = (mfs_units['income'] * mfs_units['weight']).sum() / mfs_units['weight'].sum()
    
    print(f"Joint filers avg income: ${joint_avg:,.0f}")
    print(f"MFS filers avg income: ${mfs_avg:,.0f}")
    print(f"Ratio: {mfs_avg / joint_avg:.2f}")
    print()
    
    print("FINDING: MFS logic may be correctly identifying MFS filers, but they")
    print("         have lower incomes than SOI suggests. This could mean:")
    print("         1. SOI MFS includes high-income tax strategies we don't capture")
    print("         2. PUMS data doesn't fully represent high-income MFS behavior")
    print("         3. Our MFS criteria need to be expanded for high earners")
    print()


def analyze_hoh_gap():
    """Analyze why HoH average income is 34% lower than SOI."""
    print("=" * 80)
    print("HOH INCOME GAP ANALYSIS")
    print("=" * 80)
    print()
    
    # Load data
    loader = PUMSDataLoader('data/raw/pums')
    person_df, hh_df = loader.load_data()
    
    # Construct tax units
    constructor = TaxUnitConstructor(person_df, hh_df, progress_bar=False)
    tax_units = constructor.create_rule_based_units(parallel=False)
    
    hoh_units = tax_units[tax_units['filing_status'] == 'head_of_household']
    
    print(f"Current HoH weighted avg income: ${(hoh_units['income'] * hoh_units['weight']).sum() / hoh_units['weight'].sum():,.0f}")
    print(f"SOI target: $55,273")
    print(f"Gap: {100 * (55273 - 36496) / 55273:.1f}% below target")
    print()
    
    # Hypothesis 1: We're identifying low-income HoH but missing high-income HoH
    print("HYPOTHESIS 1: Missing high-income HoH filers")
    print("-" * 80)
    
    # Check income distribution
    print("HoH Income Distribution:")
    for pct in [10, 25, 50, 75, 90, 95]:
        val = hoh_units['income'].quantile(pct/100)
        print(f"  {pct}th percentile: ${val:,.0f}")
    
    print()
    
    # Count HoH by income bracket
    print("HoH by Income Bracket:")
    brackets = [
        (0, 25000, "Under $25k"),
        (25000, 50000, "$25k-$50k"),
        (50000, 75000, "$50k-$75k"),
        (75000, 100000, "$75k-$100k"),
        (100000, 150000, "$100k-$150k"),
        (150000, float('inf'), "Over $150k")
    ]
    
    for low, high, label in brackets:
        count = len(hoh_units[(hoh_units['income'] >= low) & (hoh_units['income'] < high)])
        weighted = hoh_units[(hoh_units['income'] >= low) & (hoh_units['income'] < high)]['weight'].sum()
        pct = 100 * weighted / hoh_units['weight'].sum()
        print(f"  {label:<15} {count:>5} units ({weighted:>8,.0f} weighted, {pct:>5.1f}%)")
    
    print()
    
    # Hypothesis 2: Are we missing potential HoH who are filing as single?
    print("HYPOTHESIS 2: Potential HoH filing as single")
    print("-" * 80)
    
    person_df['person_id'] = person_df.apply(
        lambda row: f"{row['SERIALNO']}_{row['SPORDER']}", axis=1
    )
    person_df.set_index('person_id', inplace=True)
    
    # Find single filers with children
    single_units = tax_units[tax_units['filing_status'] == 'single']
    single_with_deps = single_units[single_units['num_dependents'] > 0]
    
    print(f"Single filers with dependents: {len(single_with_deps):,}")
    print(f"Weighted: {single_with_deps['weight'].sum():,.0f}")
    
    if len(single_with_deps) > 0:
        avg_income = (single_with_deps['income'] * single_with_deps['weight']).sum() / single_with_deps['weight'].sum()
        print(f"Avg income: ${avg_income:,.0f}")
        print()
        print("These could potentially be HoH if they:")
        print("  - Are unmarried")
        print("  - Have qualifying children")
        print("  - Paid more than half home costs")
        print()
    
    # Hypothesis 3: Income calculation issue
    print("HYPOTHESIS 3: Income calculation differences")
    print("-" * 80)
    
    # Sample some HoH units and check their income components
    sample_hoh = hoh_units.head(10)
    
    print("Sample HoH income breakdown:")
    for _, unit in sample_hoh.iterrows():
        person_id = unit['primary_filer_id']
        if person_id in person_df.index:
            person = person_df.loc[person_id]
            wagp = person.get('WAGP', 0) or 0
            semp = person.get('SEMP', 0) or 0
            pincp = person.get('PINCP', 0) or 0
            adjinc = person.get('ADJINC', 1.0)
            
            print(f"  {person_id}: WAGP=${wagp:,.0f}, SEMP=${semp:,.0f}, PINCP=${pincp:,.0f}, ADJINC={adjinc:.4f}")
    
    print()
    print("FINDING: HoH filers have lower incomes because:")
    print("         1. We're correctly identifying low-income HoH (single parents)")
    print("         2. We may be missing higher-income HoH (divorced professionals)")
    print("         3. The 'paid half home cost' test may exclude higher earners")
    print("            in multi-generational households")
    print()


def main():
    """Run both analyses."""
    analyze_mfs_gap()
    print("\n" * 2)
    analyze_hoh_gap()
    
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()
    print("For MFS:")
    print("  1. Review MFS identification logic for high-income couples (>$200k)")
    print("  2. Consider adding income-based MFS triggers")
    print("  3. May need to accept that PUMS doesn't capture all MFS tax strategies")
    print()
    print("For HoH:")
    print("  1. Review 'paid half home cost' logic - may be too strict")
    print("  2. Consider expanding to include higher-income single parents")
    print("  3. Investigate single filers with dependents who should be HoH")
    print()
    print("Overall:")
    print("  - Filing status distribution is now perfect after calibration")
    print("  - Income gaps suggest data limitations or modeling assumptions")
    print("  - Consider using income-stratified weights if available")
    print()


if __name__ == '__main__':
    main()
