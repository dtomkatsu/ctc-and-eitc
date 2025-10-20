#!/usr/bin/env python3
import pandas as pd

file_path = 'data/raw/Selected Resident Return Data/Dotax Soi 2022 - A9-2.csv'
df = pd.read_csv(file_path, header=None)

print("Examining rows 10-15 to understand bracket structure:\n")
for i in range(10, 16):
    print(f"Row {i}:")
    for j in range(5):
        print(f"  col[{j}] = {repr(df.iloc[i, j])}")
    print()
