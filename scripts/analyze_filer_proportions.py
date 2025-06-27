#!/usr/bin/env python3
"""
Analyze the distribution of filer types in the constructed tax units.

This script loads PUMS data, constructs tax units, and analyzes the
proportions of different filing statuses.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import tax unit constructor
from src.tax.units.constructor import TaxUnitConstructor
from src.tax.units.base import FILING_STATUS

# Import test data generation functions
sys.path.append(str(Path(__file__).parent.parent / 'tests'))
from test_smoke import create_test_person, create_test_household

def load_sample_data():
    """Load sample PUMS data for analysis."""
    # Create sample data with numeric SERIALNO
    hh_df = pd.DataFrame([
        create_test_household(NP=2, HINCP=50000, SERIALNO=1),
        create_test_household(NP=4, HINCP=80000, SERIALNO=2),
        create_test_household(NP=1, HINCP=30000, SERIALNO=3),
        create_test_household(NP=3, HINCP=60000, SERIALNO=4),
    ])
    
    # Ensure SERIALNO is numeric and set as string for consistency
    hh_df['SERIALNO'] = hh_df['SERIALNO'].astype(str)
    
    person_data = [
        # Single filer
        create_test_person(
            SPORDER=1, SERIALNO='1',
            RELSHIPP=1, MAR=5,  # Single
            PINCP=40000, WAGP=40000,
            AGEP=30, SEX=1, ESR=1,
            PWGTP=1, RAC1P=1, SCHL=21, INTP=0, RETP=0, OIP=0, PAP=0, SSP=0, SSIP=0
        ),
        
        # Married couple
        create_test_person(
            SPORDER=1, SERIALNO='2',
            RELSHIPP=1, MAR=1,  # Married
            PINCP=60000, WAGP=60000,
            AGEP=35, SEX=1, ESR=1,
            PWGTP=1, RAC1P=1, SCHL=21, INTP=0, RETP=0, OIP=0, PAP=0, SSP=0, SSIP=0
        ),
        create_test_person(
            SPORDER=2, SERIALNO='2',
            RELSHIPP=2, MAR=1,  # Spouse
            PINCP=20000, WAGP=20000,
            AGEP=32, SEX=2, ESR=1,
            PWGTP=1, RAC1P=1, SCHL=21, INTP=0, RETP=0, OIP=0, PAP=0, SSP=0, SSIP=0
        ),
        
        # Single person
        create_test_person(
            SPORDER=1, SERIALNO='3',
            RELSHIPP=1, MAR=5,  # Single
            PINCP=30000, WAGP=30000,
            AGEP=25, SEX=1, ESR=1,
            PWGTP=1, RAC1P=1, SCHL=21, INTP=0, RETP=0, OIP=0, PAP=0, SSP=0, SSIP=0
        ),
        
        # Head of household with child
        create_test_person(
            SPORDER=1, SERIALNO='4',
            RELSHIPP=1, MAR=5,  # Single
            PINCP=100000, WAGP=100000,
            AGEP=40, SEX=1, ESR=1,
            PWGTP=1, RAC1P=1, SCHL=21, INTP=0, RETP=0, OIP=0, PAP=0, SSP=0, SSIP=0
        ),
        create_test_person(
            SPORDER=2, SERIALNO='4',
            RELSHIPP=3, MAR=5,  # Child
            PINCP=0, WAGP=0,
            AGEP=10, SEX=1, ESR=0,
            PWGTP=1, RAC1P=1, SCHL=21, INTP=0, RETP=0, OIP=0, PAP=0, SSP=0, SSIP=0
        ),
        create_test_person(
            SPORDER=3, SERIALNO='4',
            RELSHIPP=3, MAR=5,  # Child
            PINCP=0, WAGP=0,
            AGEP=15, SEX=2, ESR=0,
            PWGTP=1, RAC1P=1, SCHL=21, INTP=0, RETP=0, OIP=0, PAP=0, SSP=0, SSIP=0
        )
    ]
    
    person_df = pd.DataFrame(person_data)
    person_df['SERIALNO'] = person_df['SERIALNO'].astype(str)
    person_df['SPORDER'] = person_df['SPORDER'].astype(int)
    
    # Set index to person_id for easier reference
    person_df['person_id'] = person_df['SERIALNO'] + '_' + person_df['SPORDER'].astype(str)
    person_df.set_index('person_id', inplace=True)
    
    return person_df, hh_df

def analyze_filer_proportions(person_df, hh_df):
    """Analyze the proportions of different filer types."""
    # Create tax units
    constructor = TaxUnitConstructor(person_df, hh_df)
    tax_units = constructor.create_rule_based_units()
    
    if tax_units is None or tax_units.empty:
        print("No tax units were created.")
        return
    
    # Calculate proportions
    total_units = len(tax_units)
    status_counts = tax_units['filing_status'].value_counts()
    status_proportions = (status_counts / total_units * 100).round(1)
    
    # Print results
    print("\nFiling Status Distribution:")
    print("-" * 30)
    for status, count in status_counts.items():
        print(f"{status}: {count} units ({status_proportions[status]}%)")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    status_proportions.plot(kind='bar', color='skyblue')
    plt.title('Filing Status Distribution')
    plt.xlabel('Filing Status')
    plt.ylabel('Percentage of Tax Units')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save plot
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    plot_path = output_dir / 'filing_status_distribution.png'
    plt.savefig(plot_path)
    print(f"\nPlot saved to: {plot_path}")
    
    return tax_units

if __name__ == "__main__":
    print("Loading sample data...")
    person_df, hh_df = load_sample_data()
    
    print("\nSample Person Data:")
    print(person_df[['SERIALNO', 'SPORDER', 'RELSHIPP', 'MAR', 'PINCP']])
    
    print("\nSample Household Data:")
    print(hh_df[['SERIALNO', 'NP', 'HINCP']])
    
    print("\nAnalyzing filer proportions...")
    tax_units = analyze_filer_proportions(person_df, hh_df)
    
    if tax_units is not None:
        print("\nTax Units Created:")
        print(tax_units[['filer_id', 'filing_status', 'income', 'num_dependents']])
