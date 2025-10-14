"""
Summarize Hawaii CTC Analysis Results

Clean display of the full population analysis results.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))


def load_and_summarize_results():
    """Load and summarize the full population CTC analysis results."""
    
    # Load the full analysis data
    try:
        analysis_df = pd.read_parquet('data/processed/hawaii_ctc_full_population_20250722_161935.parquet')
        print("Hawaii Child Tax Credit Analysis - Full Population Results")
        print("="*65)
        
        # Overall statewide summary
        total_units = len(analysis_df)
        units_with_ctc = len(analysis_df[analysis_df['ctc_ctc_total'] > 0])
        total_ctc = analysis_df['ctc_ctc_total'].sum()
        weighted_total_ctc = (analysis_df['ctc_ctc_total'] * analysis_df['PWGTP']).sum()
        weighted_total_units = analysis_df['PWGTP'].sum()
        total_children = analysis_df['ctc_qualifying_children'].sum()
        
        print(f"\nSTATEWIDE SUMMARY:")
        print(f"Total Tax Units: {total_units:,}")
        print(f"Units with CTC: {units_with_ctc:,} ({units_with_ctc/total_units*100:.1f}%)")
        print(f"Total CTC Amount: ${total_ctc:,.2f}")
        print(f"Weighted CTC Amount: ${weighted_total_ctc:,.2f}")
        print(f"Average CTC per Unit: ${total_ctc/total_units:.2f}")
        print(f"Average CTC per Recipient: ${total_ctc/units_with_ctc:.2f}")
        print(f"Total Qualifying Children: {total_children:,}")
        print(f"Estimated Population Tax Units: {weighted_total_units:,.0f}")
        
        # County-level analysis
        print(f"\nCOUNTY-LEVEL BREAKDOWN:")
        print("-" * 25)
        print(f"{'County':<12} {'Units':<8} {'CTC Recipients':<15} {'Total CTC':<15} {'Weighted CTC':<15} {'Avg CTC':<10}")
        print("-" * 90)
        
        for county in sorted(analysis_df['county'].dropna().unique()):
            subset = analysis_df[analysis_df['county'] == county]
            count = len(subset)
            recipients = len(subset[subset['ctc_ctc_total'] > 0])
            total_ctc_county = subset['ctc_ctc_total'].sum()
            weighted_ctc = (subset['ctc_ctc_total'] * subset['PWGTP']).sum()
            avg_ctc = subset['ctc_ctc_total'].mean()
            
            print(f"{county:<12} {count:>6} {recipients:>13} ${total_ctc_county:>12,.0f} ${weighted_ctc:>12,.0f} ${avg_ctc:>8.0f}")
        
        # Senate district analysis
        print(f"\nSENATE DISTRICT BREAKDOWN:")
        print("-" * 28)
        print(f"{'District':<10} {'Name':<25} {'Units':<8} {'Recipients':<12} {'Total CTC':<15} {'Avg CTC':<10}")
        print("-" * 95)
        
        for district in sorted(analysis_df['senate_district'].dropna().unique()):
            subset = analysis_df[analysis_df['senate_district'] == district]
            count = len(subset)
            recipients = len(subset[subset['ctc_ctc_total'] > 0])
            total_ctc_district = subset['ctc_ctc_total'].sum()
            avg_ctc = subset['ctc_ctc_total'].mean()
            district_name = subset['senate_name'].iloc[0] if len(subset) > 0 else 'Unknown'
            
            print(f"{district:<10} {district_name:<25} {count:>6} {recipients:>10} ${total_ctc_district:>12,.0f} ${avg_ctc:>8.0f}")
        
        # House district analysis
        print(f"\nHOUSE DISTRICT BREAKDOWN:")
        print("-" * 26)
        print(f"{'District':<10} {'Name':<25} {'Units':<8} {'Recipients':<12} {'Total CTC':<15} {'Avg CTC':<10}")
        print("-" * 95)
        
        for district in sorted(analysis_df['house_district'].dropna().unique()):
            subset = analysis_df[analysis_df['house_district'] == district]
            count = len(subset)
            recipients = len(subset[subset['ctc_ctc_total'] > 0])
            total_ctc_district = subset['ctc_ctc_total'].sum()
            avg_ctc = subset['ctc_ctc_total'].mean()
            district_name = subset['house_name'].iloc[0] if len(subset) > 0 else 'Unknown'
            
            print(f"{district:<10} {district_name:<25} {count:>6} {recipients:>10} ${total_ctc_district:>12,.0f} ${avg_ctc:>8.0f}")
        
        # Filing status breakdown
        print(f"\nFILING STATUS BREAKDOWN:")
        print("-" * 25)
        print(f"{'Status':<25} {'Units':<8} {'CTC Recipients':<15} {'Total CTC':<15} {'Avg CTC':<10}")
        print("-" * 75)
        
        for status in analysis_df['filing_status'].unique():
            subset = analysis_df[analysis_df['filing_status'] == status]
            count = len(subset)
            recipients = len(subset[subset['ctc_ctc_total'] > 0])
            total_ctc_status = subset['ctc_ctc_total'].sum()
            avg_ctc = subset['ctc_ctc_total'].mean()
            
            print(f"{status:<25} {count:>6} {recipients:>13} ${total_ctc_status:>12,.0f} ${avg_ctc:>8.0f}")
        
        # Geographic insights
        print(f"\nGEOGRAPHIC INSIGHTS:")
        print("-" * 19)
        
        # Highest and lowest CTC areas
        puma_stats = analysis_df.groupby(['PUMA', 'region']).agg({
            'ctc_ctc_total': ['count', 'sum', 'mean'],
            'ctc_qualifying_children': 'sum'
        }).round(2)
        
        puma_stats.columns = ['_'.join(col).strip() for col in puma_stats.columns]
        puma_stats = puma_stats.reset_index()
        
        # Top areas by total CTC
        top_total = puma_stats.nlargest(3, 'ctc_ctc_total_sum')
        print("Top 3 Areas by Total CTC:")
        for _, row in top_total.iterrows():
            print(f"  {row['region']} (PUMA {row['PUMA']}): ${row['ctc_ctc_total_sum']:,.0f}")
        
        # Top areas by average CTC
        top_avg = puma_stats.nlargest(3, 'ctc_ctc_total_mean')
        print("\nTop 3 Areas by Average CTC per Unit:")
        for _, row in top_avg.iterrows():
            print(f"  {row['region']} (PUMA {row['PUMA']}): ${row['ctc_ctc_total_mean']:,.0f}")
        
        # Policy implications
        print(f"\nPOLICY IMPLICATIONS:")
        print("-" * 19)
        print(f"• Estimated annual CTC impact: ${weighted_total_ctc:,.2f}")
        print(f"• Average benefit per recipient family: ${total_ctc/units_with_ctc:,.2f}")
        print(f"• Geographic variation: {analysis_df.groupby('county')['ctc_ctc_total'].mean().std()/analysis_df.groupby('county')['ctc_ctc_total'].mean().mean()*100:.1f}% coefficient of variation across counties")
        print(f"• Legislative districts with highest CTC impact: {analysis_df.groupby('senate_district')['ctc_ctc_total'].sum().idxmax()}")
        
        return analysis_df
        
    except FileNotFoundError:
        print("Analysis results file not found. Please run the full population analysis first.")
        return None


def create_executive_summary(analysis_df):
    """Create an executive summary for policymakers."""
    if analysis_df is None:
        return
    
    print(f"\n" + "="*65)
    print("EXECUTIVE SUMMARY FOR POLICYMAKERS")
    print("="*65)
    
    total_ctc = analysis_df['ctc_ctc_total'].sum()
    weighted_total_ctc = (analysis_df['ctc_ctc_total'] * analysis_df['PWGTP']).sum()
    units_with_ctc = len(analysis_df[analysis_df['ctc_ctc_total'] > 0])
    total_children = analysis_df['ctc_qualifying_children'].sum()
    
    print(f"\nKEY FINDINGS:")
    print(f"1. ECONOMIC IMPACT:")
    print(f"   • Total annual CTC benefits to Hawaii families: ${weighted_total_ctc:,.2f}")
    print(f"   • Number of families receiving benefits: {units_with_ctc:,}")
    print(f"   • Children benefiting from CTC: {total_children:,}")
    
    print(f"\n2. GEOGRAPHIC DISTRIBUTION:")
    county_totals = analysis_df.groupby('county')['ctc_ctc_total'].sum().sort_values(ascending=False)
    print(f"   • Highest CTC county: {county_totals.index[0]} (${county_totals.iloc[0]:,.2f})")
    print(f"   • Lowest CTC county: {county_totals.index[-1]} (${county_totals.iloc[-1]:,.2f})")
    
    print(f"\n3. LEGISLATIVE DISTRICTS:")
    senate_totals = analysis_df.groupby('senate_district')['ctc_ctc_total'].sum().sort_values(ascending=False)
    house_totals = analysis_df.groupby('house_district')['ctc_ctc_total'].sum().sort_values(ascending=False)
    print(f"   • Senate district with highest CTC: {senate_totals.index[0]} (${senate_totals.iloc[0]:,.2f})")
    print(f"   • House district with highest CTC: {house_totals.index[0]} (${house_totals.iloc[0]:,.2f})")
    
    print(f"\n4. FAMILY CHARACTERISTICS:")
    filing_status_stats = analysis_df[analysis_df['ctc_ctc_total'] > 0]['filing_status'].value_counts()
    print(f"   • Most common filing status among CTC recipients: {filing_status_stats.index[0]} ({filing_status_stats.iloc[0]:,} families)")
    print(f"   • Average CTC benefit per recipient family: ${total_ctc/units_with_ctc:,.2f}")


def main():
    """Main function to summarize results."""
    analysis_df = load_and_summarize_results()
    create_executive_summary(analysis_df)


if __name__ == "__main__":
    main()
