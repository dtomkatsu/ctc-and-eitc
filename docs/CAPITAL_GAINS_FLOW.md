# Capital Gains Data Flow - Visual Guide

## The Complete Picture

```
┌─────────────────────────────────────────────────────────────────┐
│                     PUMS DATA (Input)                           │
│  - Wage income, self-employment, etc.                          │
│  - NO capital gains included                                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              STEP 1: Calculate Base AGI                         │
│  AGI = wages + self-employment + other income                  │
│  (This is "agi_without_cap_gains")                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│         STEP 2: Estimate Capital Gains (Table 21)               │
│  - Assign capital gains to selected filers                     │
│  - Amount based on bracket-specific scaling factors            │
│  - Calibrated to match Table 21 amounts                        │
│                                                                 │
│  ✅ WHAT WE COMPARE HERE: Capital Gains AMOUNTS                │
│     Model: $3,840M vs Table 21: $2,995M                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│           STEP 3: Add Capital Gains to AGI                      │
│  AGI (with cap gains) = AGI (base) + Capital Gains             │
│  (This is "agi_with_cap_gains" or just "agi")                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│         STEP 4: Calculate Taxable Income                        │
│  Taxable Income = AGI - Deductions                             │
│  (Standard or itemized, whichever is higher)                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│         STEP 5: Apply Hawaii Tax Brackets                       │
│  Tax Liability = f(Taxable Income, Filing Status)              │
│  - Progressive tax brackets                                     │
│  - Capital gains taxed as ordinary income                      │
│  - Higher AGI → Higher Tax                                     │
│                                                                 │
│  ✅ WHAT WE COMPARE HERE: Total Tax Liability                  │
│     Model: $2,981M vs Table 12A: $3,029M (-1.6%)              │
│     (This INCLUDES the tax effect of capital gains)            │
└─────────────────────────────────────────────────────────────────┘
```

## Two Separate Comparisons

### Comparison #1: Capital Gains Amounts (Table 21)
```
DOTax Table 21          Our Model               What We Compare
─────────────────       ─────────────           ───────────────
$2,995M total      vs   $3,840M total      →    Dollar amounts
43,443 filers      vs   41,686 filers      →    Filer counts
By AGI bracket     vs   By AGI bracket     →    Distribution
```

**This is NOT a tax comparison** - it's comparing the raw dollar amounts of capital gains.

### Comparison #2: Tax Liability (Table 12A)
```
DOTax Table 12A         Our Model               What We Compare
─────────────────       ─────────────           ───────────────
$3,029M total      vs   $2,981M total      →    Tax liability
By AGI bracket     vs   By AGI bracket     →    Tax by bracket
By filing status   vs   By filing status   →    Tax by status
```

**This IS a tax comparison** - and it includes the tax effect of capital gains because:
- Filers with capital gains have higher AGI
- Higher AGI → higher taxable income → higher tax

## Key Insights

### Why We Don't Measure "Tax on Capital Gains" Separately

1. **Hawaii doesn't have separate capital gains tax rates**
   - Capital gains are taxed as ordinary income
   - Same progressive brackets apply

2. **DOTax doesn't report it separately**
   - Table 21 shows capital gains amounts (not tax)
   - Table 12A shows total tax (including effect of cap gains)

3. **The effect is captured in total tax**
   - When we compare total tax liability (Table 12A)
   - We're implicitly comparing the tax effect of capital gains
   - Because filers with cap gains pay more tax

### Example: How Capital Gains Affect Tax

```
Filer A (No Capital Gains):
  Wages: $100,000
  AGI: $100,000
  Taxable Income: $91,200 (after $8,800 standard deduction)
  Tax: ~$5,500

Filer B (With Capital Gains):
  Wages: $100,000
  Capital Gains: $50,000
  AGI: $150,000
  Taxable Income: $141,200 (after $8,800 standard deduction)
  Tax: ~$9,500

Difference: $4,000 more tax due to capital gains
```

This $4,000 difference shows up in our Table 12A comparison, even though we don't calculate it separately.

## Summary

**What we're doing**:
1. ✅ Estimating capital gains **amounts** → compare to Table 21
2. ✅ Adding those amounts to AGI
3. ✅ Calculating tax on the higher AGI → compare to Table 12A

**What we're NOT doing**:
- ❌ Calculating a separate "capital gains tax"
- ❌ Comparing tax liability specifically from capital gains
- ❌ Using different tax rates for capital gains

**Why this is correct**:
- Matches how Hawaii actually taxes capital gains (as ordinary income)
- Matches what DOTax reports (amounts in Table 21, total tax in Table 12A)
- Captures the full effect of capital gains on tax liability
