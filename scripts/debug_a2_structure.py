#!/usr/bin/env python3
import pandas as pd

file_path = 'data/raw/Selected Resident Return Data/Dotax Soi 2022 - A2-1.csv'
df = pd.read_csv(file_path, header=None)

print("Examining key rows to understand structure:\n")

# TOTAL row (should be row 20)
print("Row 20 (TOTAL RESIDENT TAXABLE):")
for j in range(20):
    print(f"  col[{j}] = {repr(df.iloc[20, j])}")

print("\n\nRow 29 (TOTAL - ALL RESIDENT RETURNS):")
for j in range(20):
    print(f"  col[{j}] = {repr(df.iloc[29, j])}")

print("\n\nSample data row 8 ($0 to under $10k):")
for j in range(20):
    print(f"  col[{j}] = {repr(df.iloc[8, j])}")
