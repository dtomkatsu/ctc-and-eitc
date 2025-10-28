# Capital Gains Methodology

## Overview

This document explains how capital gains are incorporated into the Hawaii tax model and what we're actually measuring when we compare to DOTax Table 21.

## Key Concept: We Compare AMOUNTS, Not Tax Liability

**Important**: DOTax Table 21 shows the **dollar amounts** of capital gains by AGI bracket, NOT the tax liability on those gains. We calibrate our model to match these amounts.

## Data Flow

### 1. Starting Point: PUMS Data
- PUMS provides wage/salary income, self-employment income, etc.
- PUMS does **NOT** include capital gains in AGI
- We calculate initial AGI from PUMS income sources

### 2. Capital Gains Estimation
```
AGI (from PUMS, no cap gains)
    ↓
+ Estimated Capital Gains (from Table 21 calibration)
    ↓
= AGI with Capital Gains
```

### 3. Tax Calculation Flow
```
AGI with Capital Gains
    ↓
- Standard/Itemized Deductions
    ↓
= Taxable Income
    ↓
Apply Hawaii Tax Brackets
    ↓
= Tax Liability (before credits)
```

## What We're Comparing

### Capital Gains Amounts (Table 21)
- **DOTax Table 21**: Shows total capital gains amounts by AGI bracket
  - Example: $400k+ bracket has $2,210.3M in capital gains
- **Our Model**: Estimates capital gains amounts by AGI bracket
  - We calibrate scaling factors to match these amounts
- **Comparison**: Dollar amounts of capital gains (NOT tax on them)

### Tax Liability (Table 12A)
- **DOTax Table 12A**: Shows total tax liability by AGI bracket
  - This INCLUDES the tax effect of capital gains
  - Higher AGI from capital gains → higher tax liability
- **Our Model**: Calculates tax liability using Hawaii tax brackets
  - Capital gains increase AGI → increase taxable income → increase tax
- **Comparison**: Total tax liability (which indirectly reflects capital gains)

## Why This Approach is Correct

1. **Apples-to-apples comparison**: DOTax Table 21 shows capital gains amounts, so we compare amounts
2. **Tax effect is captured**: The tax impact of capital gains shows up in Table 12A tax liability comparison
3. **No separate capital gains tax**: Hawaii doesn't have a separate capital gains tax rate - capital gains are taxed as ordinary income
4. **Realistic modeling**: We add capital gains to AGI, which flows through to taxable income naturally

## Calibration Process

### Step 1: Match Filer Counts
- Use participation rates to match number of filers with capital gains per bracket
- Target: 43,443 total filers with capital gains (from Table 21)

### Step 2: Match Dollar Amounts
- Use bracket-specific scaling factors to match capital gains amounts
- Target: $2,995M total capital gains (from Table 21)

### Step 3: Validate Tax Impact
- Check that total tax liability is reasonable (Table 12A)
- Capital gains should increase tax liability for high-income filers

## Current Results

### Capital Gains Amounts (vs Table 21)
- **Total amount**: $3,838M vs $2,995M target (+27.3%)
- **Total filers**: 42,010 vs 43,443 target (-3.3%)
- **Distribution**: Properly concentrated in high-income brackets

### Tax Liability (vs Table 12A)
- **Total tax**: $2,981M vs $3,029M benchmark (-1.6%) ✅
- **By bracket**: Most brackets within ±10%
- **Impact**: Capital gains increase tax liability for high earners as expected

## Deterministic Assignment

We use `deterministic_top` assignment method:
- Assigns capital gains to highest income filers within each bracket
- Most realistic (wealthy filers more likely to have capital gains)
- Completely reproducible results
- Eliminates random variation

## Summary

**What we measure**:
- ✅ Capital gains **amounts** (dollars) by bracket → compare to Table 21
- ✅ Tax **liability** (includes effect of capital gains) → compare to Table 12A

**What we DON'T measure**:
- ❌ Tax liability specifically from capital gains (not available in DOTax data)
- ❌ Separate capital gains tax rate (Hawaii doesn't have one)

**Why this works**:
- Capital gains are added to AGI
- Higher AGI → higher taxable income → higher tax liability
- Tax effect is captured in Table 12A comparison
- Amounts are calibrated to Table 21 benchmarks
