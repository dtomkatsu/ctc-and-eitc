# Filing Status Analysis: Residents vs Nonresidents

## Source Data
**DOTAX Table 4:** Types and Number of Returns Filed by Filing Status of Taxpayer for Tax Year 2022

## Key Findings

### Total Returns (2022)
- **All Returns:** 743,109
- **Residents:** 635,117 (85.5%)
- **Nonresidents:** 107,992 (14.5%)

---

## Filing Status Distribution

### Nonresidents (107,992 total)

| Filing Status | Returns | % of Nonresidents |
|--------------|---------|-------------------|
| **Married Filing Jointly** | 50,872 | **47.1%** |
| **Single** | 43,852 | **40.6%** |
| **Married Filing Separately** | 6,909 | **6.4%** |
| **Head of Household** | 4,305 | **4.0%** |
| **Qualifying Widow(er)** | 26 | **0.0%** |
| **Composite** | 2,028 | **1.9%** |

### Residents (635,117 total)

| Filing Status | Returns | % of Residents |
|--------------|---------|----------------|
| **Single** | 335,198 | **52.8%** |
| **Married Filing Jointly** | 216,358 | **34.1%** |
| **Head of Household** | 67,393 | **10.6%** |
| **Married Filing Separately** | 16,007 | **2.5%** |
| **Qualifying Widow(er)** | 161 | **0.0%** |

---

## Key Differences: Nonresidents vs Residents

### 1. Joint Filers
- **Nonresidents:** 47.1%
- **Residents:** 34.1%
- **Difference:** +13.0 percentage points

**Interpretation:** Nonresidents are significantly MORE likely to file jointly. This suggests:
- Many nonresidents are married couples with investment properties in Hawaii
- Higher-income married couples (vacation homes, rental properties)
- Military families stationed in Hawaii but claiming residency elsewhere

### 2. Single Filers
- **Nonresidents:** 40.6%
- **Residents:** 52.8%
- **Difference:** -12.2 percentage points

**Interpretation:** Nonresidents are LESS likely to be single filers. This is counterintuitive but makes sense when considering:
- Single investors may form LLCs or partnerships (composite returns)
- Many single nonresidents file as composite (1.9% vs 0% for residents)
- Married couples more likely to own vacation/investment properties

### 3. Married Filing Separately
- **Nonresidents:** 6.4%
- **Residents:** 2.5%
- **Difference:** +3.9 percentage points

**Interpretation:** Nonresidents have a MUCH HIGHER MFS rate. Possible reasons:
- Tax optimization strategies for high-income couples
- One spouse is resident, one is nonresident
- Complex income situations (e.g., military, multi-state income)

### 4. Head of Household
- **Nonresidents:** 4.0%
- **Residents:** 10.6%
- **Difference:** -6.6 percentage points

**Interpretation:** Nonresidents have a MUCH LOWER HoH rate. This makes sense because:
- HoH requires a qualifying person living with the taxpayer
- Nonresidents' dependents typically don't live in Hawaii
- Most nonresidents are investors/property owners without dependents in HI

### 5. Composite Returns
- **Nonresidents:** 1.9% (2,028 returns)
- **Residents:** Not applicable (n/a)

**Interpretation:** Composite returns are UNIQUE to nonresidents. These are:
- Partnerships, S-corps, trusts filing on behalf of nonresident members
- Complex business structures
- High-income investors with Hawaii business interests

---

## Implications for Tax Modeling

### 1. Synthetic Nonresident Population
The `NonresidentSynthesizer` class now uses the **actual observed distribution** from Table 4:

```python
NONRES_FILING_STATUS = {
    'joint': 0.471,      # 47.1%
    'single': 0.406,     # 40.6%
    'mfs': 0.064,        # 6.4%
    'hoh': 0.040,        # 4.0%
    'widow': 0.000,      # 0.0% (26 returns)
    'composite': 0.019   # 1.9%
}
```

### 2. Calibration Targets
When calibrating the full population model (residents + nonresidents), use:

**Combined Targets (743,109 total):**
- Joint: 267,230 (36.0%)
- Single: 379,050 (51.0%)
- MFS: 22,916 (3.1%)
- HoH: 71,698 (9.6%)
- Widow: 187 (0.0%)
- Composite: 2,028 (0.3%)

**Resident-Only Targets (635,117 total):**
- Joint: 216,358 (34.1%)
- Single: 335,198 (52.8%)
- MFS: 16,007 (2.5%)
- HoH: 67,393 (10.6%)
- Widow: 161 (0.0%)

### 3. Policy Analysis Considerations

**When analyzing policies affecting nonresidents:**
- Focus on joint filers (47.1% of nonresidents)
- Consider MFS implications (6.4% vs 2.5% residents)
- HoH policies have minimal impact on nonresidents (4.0%)

**When analyzing policies affecting all filers:**
- Nonresidents skew the distribution toward joint filers
- Combined distribution is different from resident-only distribution
- Use combined targets (Table 4) for validation

---

## Validation Checklist

When building the full population model, validate:

- [ ] Total returns = 743,109
- [ ] Resident returns = 635,117 (85.5%)
- [ ] Nonresident returns = 107,992 (14.5%)
- [ ] Nonresident joint filers ≈ 50,872 (47.1% of nonresidents)
- [ ] Nonresident single filers ≈ 43,852 (40.6% of nonresidents)
- [ ] Nonresident MFS ≈ 6,909 (6.4% of nonresidents)
- [ ] Nonresident HoH ≈ 4,305 (4.0% of nonresidents)
- [ ] Nonresident composite ≈ 2,028 (1.9% of nonresidents)
- [ ] Combined filing status distribution matches Table 4

---

## Year-over-Year Comparison (2021 vs 2022)

### Nonresidents

| Filing Status | 2021 | 2022 | Change |
|--------------|------|------|--------|
| Joint | 50,297 (46.7%) | 50,872 (47.1%) | +1.1% |
| Single | 43,922 (40.8%) | 43,852 (40.6%) | -0.2% |
| MFS | 7,126 (6.6%) | 6,909 (6.4%) | -3.0% |
| HoH | 4,277 (4.0%) | 4,305 (4.0%) | +0.7% |
| Widow | 30 (0.0%) | 26 (0.0%) | -13.3% |
| Composite | 2,039 (1.9%) | 2,028 (1.9%) | -0.5% |
| **TOTAL** | **107,691** | **107,992** | **+0.3%** |

**Key Observations:**
- Nonresident returns increased slightly (+0.3%)
- Joint filers increased (+1.1%)
- MFS decreased (-3.0%)
- Overall distribution remained stable

---

## Summary

✅ **Nonresidents have a VERY different filing status distribution than residents**

**Key Takeaways:**
1. **Joint filers dominate** among nonresidents (47.1% vs 34.1% residents)
2. **Single filers are less common** among nonresidents (40.6% vs 52.8% residents)
3. **MFS is much higher** among nonresidents (6.4% vs 2.5% residents)
4. **HoH is much lower** among nonresidents (4.0% vs 10.6% residents)
5. **Composite returns are unique** to nonresidents (1.9%)

**Modeling Impact:**
- The synthetic nonresident population now uses **actual observed data** from Table 4
- No more assumptions or guesswork about filing status distribution
- Validation targets are clear and precise
- Model will accurately reflect the full 743,109 Hawaii tax returns

**Files Updated:**
- `src/tax/units/nonresident_synthesizer.py` - Uses actual Table 4 distribution
- `src/tax/calibration/dotax_soi_parser.py` - Added `parse_filing_status_by_residency()` method
- `docs/NONRESIDENT_MODELING_STRATEGIES.md` - Updated assumptions section
- `README_FULL_POPULATION.md` - Updated key assumptions
