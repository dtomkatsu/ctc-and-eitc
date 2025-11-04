#!/usr/bin/env python3
"""
Calculate total hybrid weights from tax units.
"""
import pandas as pd
from pathlib import Path

# File paths
OUTPUT_DIR = Path('output/hybrid_analysis')
TAX_UNITS_FILE = OUTPUT_DIR / 'all_tax_units.csv'

# Check if tax units file exists
if not TAX_UNITS_FILE.exists():
    raise FileNotFoundError(f"Tax units file not found at {TAX_UNITS_FILE}")

# Load tax units
print(f"Loading tax units from {TAX_UNITS_FILE}")
tax_units = pd.read_csv(TAX_UNITS_FILE)

# Calculate totals
total_hybrid_weight = tax_units['weight'].sum()
total_hh_weight = tax_units['hh_weight'].sum()
total_person_weight = tax_units['person_weight_sum'].sum()

# Calculate by filing status
status_totals = tax_units.groupby('filing_status')['weight'].sum()

# Print results
print("\n=== Hybrid Weight Summary ===")
print(f"Total Hybrid Weight: {total_hybrid_weight:,.2f}")
print(f"Total Household Weight: {total_hh_weight:,.2f}")
print(f"Total Person Weight: {total_person_weight:,.2f}")
print(f"Total Tax Units: {len(tax_units):,}")

print("\n=== By Filing Status ===")
for status, weight in status_totals.items():
    print(f"{status}: {weight:,.2f}")

# Save results
with open(OUTPUT_DIR / 'hybrid_weight_summary.txt', 'w') as f:
    f.write("=== Hybrid Weight Summary ===\n")
    f.write(f"Total Hybrid Weight: {total_hybrid_weight:,.2f}\n")
    f.write(f"Total Household Weight: {total_hh_weight:,.2f}\n")
    f.write(f"Total Person Weight: {total_person_weight:,.2f}\n")
    f.write(f"Total Tax Units: {len(tax_units):,}\n\n")
    f.write("=== By Filing Status ===\n")
    for status, weight in status_totals.items():
        f.write(f"{status}: {weight:,.2f}\n")

print(f"\nResults saved to {OUTPUT_DIR}/hybrid_weight_summary.txt")
