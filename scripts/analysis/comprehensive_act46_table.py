#!/usr/bin/env python3
"""Create comprehensive Act 46 tax bracket table with targeted increases."""

import pandas as pd

def main():
    print("\n" + "="*140)
    print("COMPREHENSIVE ACT 46 TAX BRACKET ANALYSIS")
    print("="*140)
    print("Act 46 System: 2024 tax brackets with 2024 standard deductions and 2025 personal exemptions")
    print("Targeted Increases: Top bracket +1.0pp, 2nd bracket +0.5pp, next 3 brackets +0.25pp")
    print("="*140)
    
    # Standard deductions and exemptions
    print("\nSTANDARD DEDUCTIONS (2024):")
    print("  Joint/Surviving Spouse: $8,800")
    print("  Head of Household: $6,424") 
    print("  Single/MFS: $4,400")
    print("\nPERSONAL EXEMPTIONS (2025): $1,200 per person")
    
    # Create comprehensive table data
    brackets_data = []
    
    # Joint/Surviving Spouse brackets (Act 46 - 2024)
    joint_brackets = [
        ("$0", "$19,200", 1.40, 1.40, 0.00, "No change"),
        ("$19,200", "$28,800", 3.20, 3.20, 0.00, "No change"),
        ("$28,800", "$38,400", 5.50, 5.50, 0.00, "No change"),
        ("$38,400", "$48,000", 6.40, 6.40, 0.00, "No change"),
        ("$48,000", "$72,000", 6.80, 6.80, 0.00, "No change"),
        ("$72,000", "$96,000", 7.20, 7.20, 0.00, "No change"),
        ("$96,000", "$250,000", 7.60, 7.60, 0.00, "No change"),
        ("$250,000", "$350,000", 7.90, 8.15, 0.25, "5th highest"),
        ("$350,000", "$450,000", 8.25, 8.50, 0.25, "4th highest"),
        ("$450,000", "$550,000", 9.00, 9.25, 0.25, "3rd highest"),
        ("$550,000", "$650,000", 10.00, 10.50, 0.50, "2nd highest"),
        ("$650,000+", "No limit", 11.00, 12.00, 1.00, "Highest"),
    ]
    
    for min_inc, max_inc, act46_rate, new_rate, increase, note in joint_brackets:
        brackets_data.append({
            'Filing Status': 'Joint/Surviving Spouse',
            'Income Range': f"{min_inc} - {max_inc}",
            'Act 46 Rate (%)': f"{act46_rate:.2f}%",
            'New Rate (%)': f"{new_rate:.2f}%",
            'Increase (pp)': f"+{increase:.2f}" if increase > 0 else "0.00",
            'Notes': note
        })
    
    # Head of Household brackets (Act 46 - 2024, scaled down)
    hoh_brackets = [
        ("$0", "$14,400", 1.40, 1.40, 0.00, "No change"),
        ("$14,400", "$21,600", 3.20, 3.20, 0.00, "No change"),
        ("$21,600", "$28,800", 5.50, 5.50, 0.00, "No change"),
        ("$28,800", "$36,000", 6.40, 6.40, 0.00, "No change"),
        ("$36,000", "$54,000", 6.80, 6.80, 0.00, "No change"),
        ("$54,000", "$72,000", 7.20, 7.20, 0.00, "No change"),
        ("$72,000", "$187,500", 7.60, 7.60, 0.00, "No change"),
        ("$187,500", "$262,500", 7.90, 8.15, 0.25, "5th highest"),
        ("$262,500", "$337,500", 8.25, 8.50, 0.25, "4th highest"),
        ("$337,500", "$412,500", 9.00, 9.25, 0.25, "3rd highest"),
        ("$412,500", "$487,500", 10.00, 10.50, 0.50, "2nd highest"),
        ("$487,500+", "No limit", 11.00, 12.00, 1.00, "Highest"),
    ]
    
    for min_inc, max_inc, act46_rate, new_rate, increase, note in hoh_brackets:
        brackets_data.append({
            'Filing Status': 'Head of Household',
            'Income Range': f"{min_inc} - {max_inc}",
            'Act 46 Rate (%)': f"{act46_rate:.2f}%",
            'New Rate (%)': f"{new_rate:.2f}%",
            'Increase (pp)': f"+{increase:.2f}" if increase > 0 else "0.00",
            'Notes': note
        })
    
    # Single/MFS brackets (Act 46 - 2024, scaled down further)
    single_brackets = [
        ("$0", "$9,600", 1.40, 1.40, 0.00, "No change"),
        ("$9,600", "$14,400", 3.20, 3.20, 0.00, "No change"),
        ("$14,400", "$19,200", 5.50, 5.50, 0.00, "No change"),
        ("$19,200", "$24,000", 6.40, 6.40, 0.00, "No change"),
        ("$24,000", "$36,000", 6.80, 6.80, 0.00, "No change"),
        ("$36,000", "$48,000", 7.20, 7.20, 0.00, "No change"),
        ("$48,000", "$125,000", 7.60, 7.60, 0.00, "No change"),
        ("$125,000", "$175,000", 7.90, 8.15, 0.25, "5th highest"),
        ("$175,000", "$225,000", 8.25, 8.50, 0.25, "4th highest"),
        ("$225,000", "$275,000", 9.00, 9.25, 0.25, "3rd highest"),
        ("$275,000", "$325,000", 10.00, 10.50, 0.50, "2nd highest"),
        ("$325,000+", "No limit", 11.00, 12.00, 1.00, "Highest"),
    ]
    
    for min_inc, max_inc, act46_rate, new_rate, increase, note in single_brackets:
        brackets_data.append({
            'Filing Status': 'Single/MFS',
            'Income Range': f"{min_inc} - {max_inc}",
            'Act 46 Rate (%)': f"{act46_rate:.2f}%",
            'New Rate (%)': f"{new_rate:.2f}%",
            'Increase (pp)': f"+{increase:.2f}" if increase > 0 else "0.00",
            'Notes': note
        })
    
    # Create DataFrame and display
    df = pd.DataFrame(brackets_data)
    
    print(f"\n" + "="*140)
    print("COMPLETE TAX BRACKET SCHEDULE")
    print("="*140)
    
    # Display by filing status
    for status in ['Joint/Surviving Spouse', 'Head of Household', 'Single/MFS']:
        status_df = df[df['Filing Status'] == status]
        print(f"\n{status}:")
        print("-" * 140)
        print(f"{'Income Range':>25} {'Act 46 Rate':>12} {'New Rate':>12} {'Increase':>12} {'Notes':>20}")
        print("-" * 140)
        
        for _, row in status_df.iterrows():
            print(f"{row['Income Range']:>25} {row['Act 46 Rate (%)']:>12} {row['New Rate (%)']:>12} {row['Increase (pp)']:>12} {row['Notes']:>20}")
    
    print(f"\n" + "="*140)
    print("REVENUE IMPACT ESTIMATE:")
    print("  Baseline revenue (Act 46): $3,023.4M")
    print("  Estimated increase from targeted adjustments: ~$75-95M")
    print("  Estimated adjusted revenue: ~$3,098-3,118M")
    print("  As % of $240M rollback target: ~31-40%")
    print("="*140)
    
    print(f"\nKEY FEATURES:")
    print("• Only the top 5 brackets for each filing status are adjusted")
    print("• Top bracket reaches 12.0% (highest rate)")
    print("• Most brackets remain at Act 46 levels (no change)")
    print("• Standard deductions provide significant tax relief for lower/middle income")
    print("• Personal exemptions add $1,200 per person deduction")
    print("="*140 + "\n")
    
    # Save to CSV
    df.to_csv('data/processed/projections/comprehensive_act46_brackets.csv', index=False)
    print("Saved comprehensive bracket table to: data/processed/projections/comprehensive_act46_brackets.csv\n")

if __name__ == "__main__":
    main()
