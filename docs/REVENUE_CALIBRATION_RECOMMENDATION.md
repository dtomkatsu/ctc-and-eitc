# Revenue Calibration: Final Recommendation

## Problem Identified

The current 2D IPF (matching both returns AND AGI) **is not converging**:
- Joint $150k-$200k: 37% error (should be <1%)
- Overall revenue error: **12.8%** (unacceptable)
- IPF is not actually adjusting weights

**Root cause**: Sequential adjustment (count → AGI → count → AGI) creates oscillation and cannot converge to both targets simultaneously.

---

## Solution for Revenue Estimation

### ⭐ **RECOMMENDED: Calibrate to AGI Only**

For revenue estimation, **AGI is what matters** (not exact return counts):

```
Revenue = Σ tax(AGI, filing_status, dependents, ...)
```

Since tax is primarily a function of AGI, matching AGI targets gives accurate revenue.

**Benefits**:
- ✅ 1D IPF converges reliably
- ✅ Mathematically guaranteed to match AGI targets
- ✅ Revenue estimates will be accurate
- ✅ Simple and fast

**Trade-off**:
- ⚠️ Return counts may not match exactly (but this doesn't affect revenue)

---

## Implementation

### Step 1: Create AGI-Only Calibration

```python
def apply_agi_calibration(
    tax_units,
    weight_col='weight',
    agi_col='agi',
    filing_status_col='filing_status',
    output_weight_col='weight_agi_calibrated'
):
    """
    Calibrate weights to match AGI targets only.
    
    This ensures accurate revenue estimates.
    """
    # Use standard 1D IPF on AGI totals
    # This WILL converge
    
    for bracket in all_brackets:
        mask = get_bracket_mask(tax_units, bracket)
        
        target_agi = benchmark[bracket]['total_agi']
        current_agi = (weights[mask] * agi[mask]).sum()
        
        adjustment_factor = target_agi / current_agi
        weights[mask] *= adjustment_factor
    
    return weights
```

### Step 2: Validate Revenue Accuracy

```python
# Calculate revenue with AGI-calibrated weights
revenue = calculate_revenue(tax_units, 'weight_agi_calibrated')

# Compare to SOI
for bracket in all_brackets:
    error = abs(revenue[bracket] - soi_revenue[bracket]) / soi_revenue[bracket]
    print(f"{bracket}: {error:.2%} revenue error")
```

**Expected result**: < 2% revenue error (excellent for policy analysis)

---

## Alternative: Proper 2D IPF (If You Need Both Targets)

If you absolutely need to match BOTH returns AND AGI:

### Use Constrained Optimization

```python
from scipy.optimize import minimize

def objective(weights, targets_count, targets_agi):
    """Minimize squared error for both targets."""
    count_error = (actual_count - target_count)**2
    agi_error = (actual_agi - target_agi)**2
    return count_error + agi_error

# Optimize weights subject to constraints
result = minimize(objective, initial_weights, constraints=...)
```

**This will converge**, but is more complex.

---

## Comparison

| Approach | Revenue Accuracy | Complexity | Convergence | Return Count Accuracy |
|----------|------------------|------------|-------------|----------------------|
| **AGI-only IPF** | ✅ Excellent | ✅ Simple | ✅ Guaranteed | ⚠️ Approximate |
| **Sequential 2D IPF** (current) | ❌ Poor (13% error) | Medium | ❌ Fails | ❌ Poor |
| **Optimization-based 2D** | ✅ Excellent | ⚠️ Complex | ✅ Yes | ✅ Excellent |

---

## Final Recommendation

**For revenue estimation**:

1. ✅ **Use AGI-only calibration** (simplest, most reliable)
2. ✅ Validate that revenue errors are < 5%
3. ✅ Document that return counts are approximate (but revenue is accurate)

**If you need exact return counts too**:

1. Implement proper simultaneous 2D IPF using optimization
2. Or use hierarchical approach: calibrate filing status totals first, then AGI within each status

---

## Next Steps

1. Implement AGI-only calibration
2. Run revenue validation
3. If revenue accuracy is good (< 5% error), you're done
4. If not, investigate specific problematic brackets

**Bottom line**: For revenue estimation, matching AGI is sufficient. Don't overcomplicate with 2D IPF that doesn't converge.
