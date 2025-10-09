# Single Filer Overcount Analysis

## Executive Summary

**Problem**: Single filers are 4.8 percentage points above the IRS SOI benchmark (56.5% vs 51.7%), while joint filers are 4.9 percentage points below (30.3% vs 35.1%).

**Root Cause**: 585 married people with spouses in the same household are filing as single instead of jointly. This is the PRIMARY ISSUE causing the filing status distribution gap.

## Current Filing Status Distribution

| Status | Model | SOI Target | Gap |
|--------|-------|------------|-----|
| **Single** | 56.5% | 51.7% | **+4.8pp** |
| **Joint** | 30.3% | 35.1% | **-4.9pp** |
| **Head of Household** | 10.0% | 10.4% | -0.4pp ✅ |
| **Married Filing Sep** | 3.2% | 2.7% | +0.5pp ✅ |

## Key Findings

### 1. Married People Filing as Single
- **Total**: 2,400 married people filing as single (9.4% of all single filers)
- **With spouse in household**: 585 people
- **Without spouse in household**: 1,815 people

### 2. Relationship Codes of Married Singles
The 2,400 married people filing as single have the following relationship codes:
- **Householder**: 665 (27.7%) - These should be paired with spouses
- **Other relative**: 230 (9.6%)
- **Child**: 175 (7.3%)
- **Unknown codes (31-38)**: 1,103 (46.0%) - Group quarters and other categories
- **Institutionalized GQ**: 115 (4.8%)
- **Other nonrelative**: 68 (2.8%)

### 3. The 585 Missing Joint Filers
**Critical Issue**: 585 married people have their spouse in the same household but are filing as single.

This indicates a bug in the `_identify_joint_filers()` method in `constructor.py`.

## Logic Analysis of `_identify_joint_filers()`

### Current Implementation (Lines 727-819)

The method has two passes:

#### **Pass 1**: Householder + Spouse pairs (Lines 746-774)
```python
# Look for householders who are married (RELSHIPP=20, MAR=1)
# Then find their spouse (RELSHIPP=21, MAR=1)
```
✅ This logic is correct and should catch most married couples.

#### **Pass 2**: Other married couples (Lines 776-817)
```python
# For remaining married adults (MAR=1):
#   - Check age difference < 15 years
#   - Check opposite sex
#   - If both conditions met, pair them
```
⚠️ This logic is **too permissive** and may create false positives.

### Why 585 Married Couples Are Missed

Possible reasons:

1. **Data Quality Issues**:
   - Some spouses may have `RELSHIPP != 21` (not coded as spouse)
   - Some householders may have `RELSHIPP != 20`
   - Marital status codes may be inconsistent

2. **Logic Gaps**:
   - Pass 1 only looks for householder (20) + spouse (21) pairs
   - If either person has a different relationship code, they won't be paired
   - Pass 2 requires age < 15 years AND opposite sex, which may be too restrictive

3. **Processing Order**:
   - If one spouse is processed as a single filer before the pairing logic runs, they won't be available for pairing

## Recommendations

### Immediate Fix: Enhance Pass 1 Logic

Modify `_identify_joint_filers()` to be more robust:

```python
# Pass 1: Find all married couples (householder + spouse)
for id1 in adult_ids:
    person1 = adults.loc[id1]
    
    # Look for married householders OR married people with RELSHIPP=20
    if person1.get('MAR') == 1 and person1.get('RELSHIPP') == 20:
        # Look for their spouse (RELSHIPP == 21 OR any married person in same household)
        for id2 in adult_ids:
            person2 = adults.loc[id2]
            
            # Primary check: RELSHIPP=21 (spouse)
            if person2.get('RELSHIPP') == 21 and person2.get('MAR') == 1:
                # Pair them
                
            # Secondary check: Both married, similar age, opposite sex
            elif person2.get('MAR') == 1:
                age_diff = abs(person1.get('AGEP', 0) - person2.get('AGEP', 0))
                opposite_sex = person1.get('SEX', 1) != person2.get('SEX', 1)
                
                if age_diff < 20 and opposite_sex:  # Relaxed age threshold
                    # Pair them
```

### Additional Improvements

1. **Relax Age Threshold**: Increase from 15 to 20 years to catch more couples
2. **Add Same-Sex Support**: Remove opposite sex requirement for modern households
3. **Check Spouse Presence**: Before filing someone as single, verify they don't have a spouse in the household
4. **Logging**: Add detailed logging to track why couples aren't being paired

## Expected Impact

Converting the 585 missing joint filers would:
- **Reduce single filers**: 56.5% → 54.2% (closer to 51.7% target)
- **Increase joint filers**: 30.3% → 31.6% (closer to 35.1% target)
- **Close the gap by ~50%**

## Next Steps

1. ✅ **Completed**: Diagnosed the root cause (585 missing joint filers)
2. **TODO**: Implement enhanced pairing logic in `_identify_joint_filers()`
3. **TODO**: Add validation to prevent married people with spouses from filing as single
4. **TODO**: Re-run analysis to verify improvements
5. **TODO**: Fine-tune MFS thresholds if needed to reach 2.7% target
