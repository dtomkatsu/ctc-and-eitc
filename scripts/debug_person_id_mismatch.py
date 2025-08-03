#!/usr/bin/env python3
"""
Debug person ID mismatch between PUMS data and tax unit construction.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.tax.units.utils import create_person_id

def main():
    print("Loading data...")
    
    # Load original PUMS data
    person_df = pd.read_parquet('data/processed/pums_person_processed.parquet')
    tax_units = pd.read_parquet('src/data/processed/tax_units_rule_based.parquet')
    
    print(f"PUMS person data: {len(person_df):,} people")
    print(f"Tax units: {len(tax_units):,} units")
    
    # Check current person IDs in PUMS data
    print(f"\nPUMS data index info:")
    print(f"  Index name: {person_df.index.name}")
    print(f"  Index type: {type(person_df.index)}")
    print(f"  Sample index values: {list(person_df.index[:5])}")
    
    # Check if person_id column exists
    if 'person_id' in person_df.columns:
        print(f"  person_id column exists: {person_df['person_id'].dtype}")
        print(f"  Sample person_id values: {list(person_df['person_id'][:5])}")
    else:
        print(f"  No person_id column found")
    
    # Check tax unit person IDs
    print(f"\nTax unit person ID info:")
    print(f"  Sample primary_filer_id values: {list(tax_units['primary_filer_id'][:5])}")
    if 'secondary_filer_id' in tax_units.columns:
        secondary_ids = tax_units['secondary_filer_id'].dropna()
        if len(secondary_ids) > 0:
            print(f"  Sample secondary_filer_id values: {list(secondary_ids[:5])}")
    
    # Test person ID generation
    print(f"\n" + "="*80)
    print("TESTING PERSON ID GENERATION")
    print("="*80)
    
    # Create person IDs using the utility function
    test_person_ids = create_person_id(person_df)
    print(f"Generated person IDs: {len(test_person_ids):,}")
    print(f"Sample generated IDs: {list(test_person_ids[:5])}")
    print(f"Generated ID type: {type(test_person_ids.iloc[0])}")
    
    # Compare with tax unit IDs
    tax_unit_primary_ids = set(tax_units['primary_filer_id'].astype(str))
    generated_ids = set(test_person_ids.astype(str))
    
    print(f"\nID Set Comparison:")
    print(f"  Tax unit primary IDs: {len(tax_unit_primary_ids):,}")
    print(f"  Generated person IDs: {len(generated_ids):,}")
    
    # Find overlap
    overlap = tax_unit_primary_ids.intersection(generated_ids)
    print(f"  Overlapping IDs: {len(overlap):,}")
    print(f"  Overlap percentage: {len(overlap)/len(tax_unit_primary_ids)*100:.1f}%")
    
    if len(overlap) < len(tax_unit_primary_ids) * 0.5:
        print(f"  ⚠️  LOW OVERLAP - Person ID mismatch confirmed!")
        
        # Show examples of mismatched IDs
        print(f"\nSample tax unit IDs not in generated IDs:")
        missing_in_generated = tax_unit_primary_ids - generated_ids
        for i, missing_id in enumerate(list(missing_in_generated)[:10]):
            print(f"    {missing_id}")
            
        print(f"\nSample generated IDs not in tax units:")
        missing_in_tax_units = generated_ids - tax_unit_primary_ids
        for i, missing_id in enumerate(list(missing_in_tax_units)[:10]):
            print(f"    {missing_id}")
    
    # Check specific householder/spouse pairs
    print(f"\n" + "="*80)
    print("CHECKING HOUSEHOLDER/SPOUSE PAIR IDS")
    print("="*80)
    
    # Find a household with married adults
    married_adults = person_df[(person_df['MAR'] == 1) & (person_df['AGEP'] >= 18)]
    
    for hh_id, hh_group in married_adults.groupby('SERIALNO'):
        householders = hh_group[hh_group['RELSHIPP'] == 20]
        spouses = hh_group[hh_group['RELSHIPP'] == 21]
        
        if len(householders) >= 1 and len(spouses) >= 1:
            householder = householders.iloc[0]
            spouse = spouses.iloc[0]
            
            print(f"Example household {hh_id}:")
            print(f"  Householder:")
            print(f"    PUMS index: {householder.name}")
            print(f"    SERIALNO: {householder['SERIALNO']}")
            print(f"    SPORDER: {householder['SPORDER']}")
            
            # Generate person ID for this householder
            hh_person_id = f"{householder['SERIALNO']}_{householder['SPORDER']}"
            print(f"    Generated person_id: {hh_person_id}")
            
            print(f"  Spouse:")
            print(f"    PUMS index: {spouse.name}")
            print(f"    SERIALNO: {spouse['SERIALNO']}")
            print(f"    SPORDER: {spouse['SPORDER']}")
            
            # Generate person ID for this spouse
            sp_person_id = f"{spouse['SERIALNO']}_{spouse['SPORDER']}"
            print(f"    Generated person_id: {sp_person_id}")
            
            # Check if these IDs exist in tax units
            hh_in_tax_units = tax_units[tax_units['primary_filer_id'] == hh_person_id]
            sp_in_tax_units = tax_units[tax_units['primary_filer_id'] == sp_person_id]
            
            print(f"  Tax unit status:")
            print(f"    Householder {hh_person_id} in tax units: {len(hh_in_tax_units) > 0}")
            print(f"    Spouse {sp_person_id} in tax units: {len(sp_in_tax_units) > 0}")
            
            if len(hh_in_tax_units) > 0:
                print(f"    Householder filing status: {hh_in_tax_units.iloc[0]['filing_status']}")
            if len(sp_in_tax_units) > 0:
                print(f"    Spouse filing status: {sp_in_tax_units.iloc[0]['filing_status']}")
            
            # Check for joint filing
            joint_units = tax_units[
                ((tax_units['primary_filer_id'] == hh_person_id) & 
                 (tax_units['secondary_filer_id'] == sp_person_id)) |
                ((tax_units['primary_filer_id'] == sp_person_id) & 
                 (tax_units['secondary_filer_id'] == hh_person_id))
            ]
            print(f"    Joint filing: {len(joint_units) > 0}")
            
            break  # Just check one example
    
    # Check the tax unit constructor's person ID handling
    print(f"\n" + "="*80)
    print("ANALYZING TAX UNIT CONSTRUCTOR PERSON ID HANDLING")
    print("="*80)
    
    # Load a small sample and trace through the constructor
    sample_hh = person_df[person_df['SERIALNO'] == hh_id].copy()
    print(f"Sample household {hh_id} has {len(sample_hh)} people")
    
    # Check what happens when we set person_id as index
    sample_hh['person_id'] = create_person_id(sample_hh)
    print(f"Generated person_ids for sample: {list(sample_hh['person_id'])}")
    
    # Set as index
    sample_hh.set_index('person_id', inplace=True)
    print(f"After setting as index: {list(sample_hh.index)}")
    print(f"Index name: {sample_hh.index.name}")
    
    return {
        'pums_person_count': len(person_df),
        'tax_unit_count': len(tax_units),
        'id_overlap': len(overlap),
        'overlap_pct': len(overlap)/len(tax_unit_primary_ids)*100 if len(tax_unit_primary_ids) > 0 else 0
    }

if __name__ == "__main__":
    main()
