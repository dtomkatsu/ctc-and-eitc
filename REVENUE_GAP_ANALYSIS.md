# Analysis: Why 2017 vs 2026 Policy Shows 29.9% Revenue Gap

## The Numbers
- **2017 Policy on 2026 Incomes:** $4.87B
- **2026 Policy on 2026 Incomes:** $3.41B  
- **Gap:** -$1.46B (-29.9%)

---

## Is This Gap Too Wide?

### Standard Deduction Changes:

| Filing Status | 2017 | 2026 | Increase |
|---------------|------|------|----------|
| **Joint** | $4,400 | $16,000 | **+263%** |
| **Single** | $2,200 | $8,000 | **+264%** |
| **Head of Household** | $3,212 | $12,000 | **+274%** |

**These are MASSIVE increases** - standard deductions increased by ~3.6×

### Tax Bracket Width Changes (Joint Filers):

**2017 Brackets:**
```
$0-$4,800: 1.4%
$4,800-$9,600: 3.2%
$9,600-$19,200: 5.5%
$19,200-$28,800: 6.4%
```
First bracket: $4,800 wide

**2026 Brackets:**
```
$0-$28,800: 1.4%
$28,800-$38,400: 3.2%
$38,400-$48,000: 5.5%
```
First bracket: $28,800 wide (**6× wider**)

---

## Why The Gap is Actually Reasonable

### Example: $75K Joint Filer

**2017 Policy:**
- Gross income: $75,000
- Standard deduction: $4,400
- **Taxable income: $70,600**
- Income passes through 7+ brackets
- Much of income taxed at 5.5%, 6.4%, 6.8%+

**2026 Policy:**
- Gross income: $75,000
- Standard deduction: $16,000
- **Taxable income: $59,000**
- Stays mostly in first 3 brackets
- Most income taxed at 1.4%, 3.2%

**Tax Savings Components:**
1. **Deduction increase:** $11,600 more deduction
2. **Bracket expansion:** More income at low rates

---

## Mathematical Validation

### Component 1: Standard Deduction Impact
- Average deduction increase: ~$8,000-$10,000 per filer
- At ~6% average marginal rate: $480-$600 savings per filer
- For 700K filers: **~$336M-$420M** in savings

### Component 2: Bracket Expansion Impact  
- Wider low-rate brackets keep more income at 1.4%-3.2%
- Instead of hitting 5.5%-6.8% rates
- Saves 2-4% on significant portions of income
- Estimated impact: **~$500M-$700M** in savings

### Component 3: Compounding Effect
- Lower taxable base + more favorable rate structure
- **Total expected: ~$800M-$1.1B in savings**

### Actual Result: **$1.46B savings**

---

## Potential Issues to Check

### 1. Are 2017 brackets being applied correctly?
✓ **VERIFY:** Load a few sample tax units and manually calculate their 2017 vs 2026 tax

### 2. Is the adjustment protocol applied consistently?
- Both scenarios use same 4.3% adjustment rate ✓
- Itemized deductions, AGI adjustments, credits all applied ✓

### 3. Are we double-counting any adjustments?
- Need to verify adjustments are % of pre-credit tax, not stacking incorrectly

### 4. Is there an issue with bracket logic?
- 2017 has higher top bracket threshold ($300K vs $250K in 2026)
- This could create reverse effect for very high earners
- **CHECK:** Are high earners getting MORE taxed under 2026?

---

## What To Investigate Next

### Quick Diagnostic:
1. **Export sample of 100 tax units** with both calculations
2. **Manually verify** a few calculations
3. **Check income distribution impact:**
   - Break down by income groups
   - See if high earners distort the average

### Expected Finding:
The 29.9% gap is likely **legitimate** given the policy changes, BUT there might be:
- A calculation error in bracket application
- An issue with how adjustments compound
- Edge cases with very high earners

---

## Quick Test Command

Run this to see income-group breakdown:
```python
results = pd.read_parquet('data/processed/policy_analysis/enhanced_2017_deductions_2026_brackets.parquet')

def income_group(income):
    if income < 50000: return 'Under $50K'
    elif income < 100000: return '$50K-$100K'
    elif income < 200000: return '$100K-$200K'
    else: return 'Over $200K'

results['income_group'] = results['income'].apply(income_group)

comparison = results.groupby('income_group').agg({
    'scenario_2017_final_tax': 'mean',
    'scenario_2026_final_tax': 'mean',
    'income': 'mean'
})

comparison['pct_reduction'] = (1 - comparison['scenario_2026_final_tax'] / comparison['scenario_2017_final_tax']) * 100
print(comparison)
```

This will show if certain income groups are driving the large gap.
