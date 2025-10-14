import pandas as pd
import numpy as np
from pathlib import Path

def analyze_current_weighting():
    """Analyze current tax unit weighting and design hybrid approach."""
    
    print("Loading data...")
    person_df = pd.read_parquet('data/processed/pums_person_processed.parquet')
    tax_units = pd.read_parquet('src/data/processed/tax_units_rule_based.parquet')
    
    # Current analysis
    print("\n=== Current Tax Unit Weighting Analysis ===")
    print(f"Tax units count: {len(tax_units):,}")
    print(f"Current weight sum: {tax_units['weight'].sum():,.0f}")
    print(f"Average weight per tax unit: {tax_units['weight'].mean():.2f}")
    
    # Person weight analysis
    total_person_weight = person_df['PWGTP'].sum()
    print(f"\nTotal person weight (PWGTP): {total_person_weight:,.0f}")
    
    # Filing status distribution
    filing_status_dist = tax_units['filing_status'].value_counts(normalize=True) * 100
    print(f"\nCurrent filing status distribution:")
    for status, pct in filing_status_dist.items():
        print(f"  {status}: {pct:.1f}%")
    
    # Expected SOI benchmarks for Hawaii
    soi_benchmarks = {
        'single': 51.0,
        'joint': 36.0,
        'head_of_household': 9.6,
        'married_filing_separately': 3.4
    }
    
    print(f"\nSOI Benchmarks:")
    for status, pct in soi_benchmarks.items():
        print(f"  {status}: {pct:.1f}%")
    
    return person_df, tax_units, soi_benchmarks

def design_hybrid_weighting(person_df, tax_units, soi_benchmarks):
    """Design hybrid person/household weighting system."""
    
    print("\n=== Hybrid Weighting Design ===")
    
    # Approach 1: Aggregate person weights by tax unit
    print("\n1. Person Weight Aggregation Approach:")
    
    # Create a dictionary to map person IDs to tax unit indices (more efficient than row-wise operations)
    person_to_tax_unit = {}
    
    # Process primary and secondary filers in bulk
    for col in ['primary_filer_id', 'secondary_filer_id']:
        # Get non-null filer IDs and their corresponding tax unit indices
        valid_filers = tax_units[col].dropna()
        person_to_tax_unit.update(zip(valid_filers, valid_filers.index))
    
    # Create a Series mapping person IDs to tax unit indices
    person_tax_unit_map = pd.Series(person_to_tax_unit).reset_index()
    person_tax_unit_map.columns = ['person_id', 'tax_unit_idx']
    
    # Extract the numeric part of the person ID for joining with person_df
    person_tax_unit_map['person_num'] = person_tax_unit_map['person_id'].str.extract('_(\d+)$').astype(int)
    
    # Join with person_df to get PWGTP values
    person_weights = person_df[['PWGTP']].reset_index()
    merged = person_tax_unit_map.merge(person_weights, 
                                     left_on='person_num', 
                                     right_on='index', 
                                     how='left')
    
    # Group by tax unit and sum PWGTP
    person_weight_sums = merged.groupby('tax_unit_idx')['PWGTP'].sum()
    
    # Assign to tax_units
    tax_units['person_weight_sum'] = 0.0
    tax_units.loc[person_weight_sums.index, 'person_weight_sum'] = person_weight_sums
    
    print(f"Person weight sum for tax units: {tax_units['person_weight_sum'].sum():,.0f}")
    
    # Approach 2: Hybrid weighting formula
    print("\n2. Hybrid Weighting Formula:")
    
    # Formula: hybrid_weight = alpha * household_weight + (1-alpha) * person_weight_sum
    # Where alpha balances between household and person weighting
    
    alphas = [0.0, 0.3, 0.5, 0.7, 1.0]
    
    for alpha in alphas:
        hybrid_weight = alpha * tax_units['weight'] + (1 - alpha) * tax_units['person_weight_sum']
        total_hybrid = hybrid_weight.sum()
        print(f"  Alpha {alpha:.1f}: Total weight = {total_hybrid:,.0f}")
    
    # Approach 3: Filing status calibration
    print("\n3. Filing Status Calibration Approach:")
    
    # Calculate current weighted distribution
    current_weighted = {}
    for status in soi_benchmarks.keys():
        status_mask = tax_units['filing_status'] == status
        current_weighted[status] = tax_units.loc[status_mask, 'weight'].sum() / tax_units['weight'].sum() * 100
    
    print("Current weighted distribution vs SOI:")
    for status in soi_benchmarks.keys():
        current = current_weighted.get(status, 0)
        target = soi_benchmarks[status]
        gap = current - target
        print(f"  {status}: {current:.1f}% vs {target:.1f}% (gap: {gap:+.1f}%)")
    
    # Calculate adjustment factors
    adjustment_factors = {}
    for status in soi_benchmarks.keys():
        current = current_weighted.get(status, 0.01)  # Avoid division by zero
        target = soi_benchmarks[status]
        adjustment_factors[status] = target / current if current > 0 else 1.0
    
    print("\nSuggested adjustment factors:")
    for status, factor in adjustment_factors.items():
        print(f"  {status}: {factor:.3f}")
    
    return adjustment_factors

def recommend_hybrid_system(adjustment_factors):
    """Recommend specific hybrid weighting implementation."""
    
    print("\n=== RECOMMENDED HYBRID WEIGHTING SYSTEM ===")
    
    print("""
**Hybrid Person/Household Weighting Approach:**

1. **Base Weight Calculation:**
   - Start with household weight (WGTP) as base
   - Add person weight aggregation for tax unit members
   - Formula: base_weight = WGTP + sum(PWGTP for tax unit members)

2. **Filing Status Calibration:**
   - Apply adjustment factors to align with SOI benchmarks
   - Multiply base weight by filing status adjustment factor
   - This corrects systematic biases in filing status distribution

3. **Implementation Steps:**
   a) Modify tax unit constructor to aggregate PWGTP for each tax unit
   b) Apply filing status adjustment factors
   c) Validate against SOI benchmarks
   d) Test impact on CTC estimates

4. **Adjustment Factors (based on current gaps):**""")
    
    for status, factor in adjustment_factors.items():
        print(f"   - {status}: {factor:.3f}")
    
    print("""
5. **Benefits:**
   - Incorporates both household and individual representation
   - Corrects filing status distribution biases
   - Maintains population total alignment
   - Improves tax unit sample representativeness

6. **Validation Metrics:**
   - Filing status distribution vs SOI benchmarks
   - Total weighted population alignment
   - CTC estimate stability
   - Geographic distribution consistency
""")

if __name__ == "__main__":
    person_df, tax_units, soi_benchmarks = analyze_current_weighting()
    adjustment_factors = design_hybrid_weighting(person_df, tax_units, soi_benchmarks)
    recommend_hybrid_system(adjustment_factors)
