# Project Reorganization Plan

## Current Issues

After analyzing the project structure, I've identified several organizational issues that impact maintainability and clarity:

### 1. **Outdated Project Name**
- Directory: `ctc-and-eitc` (Child Tax Credit and Earned Income Tax Credit)
- Actual scope: Hawaii state-wide tax estimation (CTC/EITC removed in October 2025)
- **Impact**: Misleading name causes confusion about project purpose

### 2. **Script Sprawl** (51+ scripts in `/scripts/`)
- Many one-off analysis scripts from iterative development
- Difficult to identify core pipeline vs. exploratory scripts
- Examples of clutter:
  - `analyze_excess_single_filers.py`
  - `analyze_hoh_bottlenecks.py`
  - `analyze_married_misclassification.py`
  - `calculate_weighted_sum_fixed.py`
  - `calculate_weighted_totals_corrected.py`
  - Multiple versions of similar scripts (v2, step1, step3, etc.)

### 3. **Duplicate/Nested Project Structure**
- `/tax-credits-final/` subdirectory contains duplicate structure
- Unclear which is the "real" project root
- Causes confusion about where to add new code

### 4. **Mixed Output Locations**
- `/output/` - Some analysis results
- `/analysis_results/` - Other analysis results
- No clear organization of what goes where

### 5. **Unclear Module Boundaries**
- `/src/tax/units/` has 22 items (some could be better organized)
- `/src/data/` mixes loaders with processing logic
- `/config/` exists both at root and in `/src/config/`

### 6. **Test Organization**
- `/tests/credits/` - Tests for archived CTC module
- `/tests/unit/` and `/tests/units/` - Unclear distinction
- Some test files may be outdated

---

## Recommended Reorganization

### Phase 1: Rename Project (High Priority)

**Current:** `ctc-and-eitc`  
**Proposed:** `hawaii-tax-estimation`

**Rationale:**
- Accurately reflects current scope
- Removes confusion about CTC/EITC focus
- Aligns with README.md title

**Action:**
```bash
cd /Users/dtomkatsu/CascadeProjects/
mv ctc-and-eitc hawaii-tax-estimation
```

---

### Phase 2: Reorganize Scripts Directory

**Current Structure:** 51+ scripts in flat `/scripts/` directory

**Proposed Structure:**
```
scripts/
├── pipeline/              # Core production pipeline
│   ├── 01_construct_tax_units.py
│   ├── 02_calibrate_weights.py
│   ├── 03_validate_results.py
│   └── 04_generate_reports.py
│
├── analysis/              # Analysis and exploration
│   ├── filing_status/
│   │   ├── analyze_filing_status_gaps.py
│   │   ├── analyze_joint_filing_disparity.py
│   │   └── analyze_hoh_qualification.py
│   ├── income/
│   │   ├── analyze_income_distribution.py
│   │   ├── analyze_high_income.py
│   │   └── analyze_millionaires.py
│   └── validation/
│       ├── compare_to_soi.py
│       └── validate_tax_units.py
│
├── calibration/           # Calibration testing
│   ├── demo_ipf_calibration.py
│   ├── test_ipf_calibration.py
│   └── compare_calibration_methods.py
│
├── data_prep/             # Data download and preparation
│   ├── download_pums.py
│   └── create_crosswalks.py
│
└── archived/              # Old/deprecated scripts
    └── (move old analysis scripts here)
```

**Benefits:**
- Clear separation of production vs. exploratory code
- Easy to find relevant scripts
- Numbered pipeline scripts show execution order
- Grouped by functional area

---

### Phase 3: Clean Up Duplicate Structures

**Remove:**
- `/tax-credits-final/` - Appears to be old duplicate structure
- Verify it's not being used, then delete

**Consolidate:**
- Move any unique code from `/tax-credits-final/` to main structure
- Update any references

---

### Phase 4: Standardize Output Directories

**Current:** `/output/` and `/analysis_results/`

**Proposed:**
```
output/
├── production/            # Production pipeline outputs
│   ├── tax_units/
│   ├── calibrated_weights/
│   └── reports/
│
├── analysis/              # Ad-hoc analysis results
│   ├── filing_status/
│   ├── income_distribution/
│   └── validation/
│
└── archived/              # Old analysis results
    └── (move old results here)
```

**Action:**
- Merge `/analysis_results/` into `/output/analysis/`
- Add `.gitignore` entries for output files

---

### Phase 5: Reorganize Source Modules

**Current `/src/` structure:**
```
src/
├── analysis/
├── config/
├── data/
├── tax/
└── visualization/
```

**Proposed improvements:**

#### A. Consolidate Config
```
config/                    # Move to root level only
├── income_growth.py
├── tax_brackets.py
└── benchmarks.py
```
Remove `/src/config/` duplication.

#### B. Reorganize `/src/tax/units/`
**Current:** 22 files in flat structure

**Proposed:**
```
src/tax/units/
├── __init__.py
├── constructor.py         # Main constructor
├── base.py               # Base classes
│
├── core/                 # Core construction logic
│   ├── dependencies.py
│   ├── income.py
│   ├── relationships.py
│   └── utils.py
│
├── filing_status/        # Filing status determination
│   ├── __init__.py
│   ├── single.py
│   ├── joint.py         # Merged from mfj.py
│   ├── separate.py      # Merged from mfs.py
│   └── hoh.py
│
├── calibration/          # Weight calibration
│   ├── __init__.py
│   ├── soi_calibration.py
│   ├── ipf_calibration.py
│   └── gentle_calibration.py
│
├── enhancement/          # Income enhancement
│   ├── __init__.py
│   └── income_enhancement.py
│
├── validation/           # Validation logic
│   ├── __init__.py
│   ├── base_validator.py
│   └── ml_validator.py
│
└── nonresident/          # Non-resident (not used)
    └── synthesizer.py    # Clearly separated
```

**Benefits:**
- Logical grouping by function
- Easier to navigate
- Clear separation of concerns
- Non-resident code isolated

#### C. Reorganize `/src/data/`
```
src/data/
├── loaders/              # Data loading
│   ├── __init__.py
│   ├── pums_loader.py
│   ├── dotax_loader.py
│   └── irs_soi_loader.py
│
├── processing/           # Data processing
│   ├── __init__.py
│   └── statistical_matching.py
│
└── parsers/              # Data parsing
    ├── __init__.py
    └── dotax_soi_parser.py
```

---

### Phase 6: Clean Up Tests

**Current issues:**
- `/tests/credits/` - Tests for archived CTC module
- `/tests/unit/` vs `/tests/units/` confusion

**Proposed:**
```
tests/
├── unit/                 # Unit tests
│   ├── test_constructor.py
│   ├── test_filing_status.py
│   ├── test_income.py
│   └── test_calibration.py
│
├── integration/          # Integration tests
│   ├── test_pipeline.py
│   └── test_data_loading.py
│
├── validation/           # Validation tests
│   ├── test_soi_comparison.py
│   └── test_benchmarks.py
│
└── archived/             # Old tests
    └── credits/          # Move archived CTC tests here
```

---

## Implementation Priority

### High Priority (Do First)
1. ✅ **Rename project directory** - Fixes fundamental naming issue
2. ✅ **Reorganize scripts/** - Biggest immediate impact on usability
3. ✅ **Remove `/tax-credits-final/`** - Eliminates confusion

### Medium Priority (Do Soon)
4. **Consolidate output directories** - Cleaner organization
5. **Reorganize `/src/tax/units/`** - Better module structure
6. **Clean up config duplication** - Remove confusion

### Low Priority (Nice to Have)
7. **Reorganize `/src/data/`** - Minor improvement
8. **Clean up tests** - Can be done incrementally

---

## Migration Strategy

### Step 1: Create New Structure (Non-Breaking)
- Create new directories alongside old ones
- Copy/move files to new locations
- Update imports in moved files

### Step 2: Update References
- Update import statements throughout codebase
- Update script paths in documentation
- Update `.gitignore` if needed

### Step 3: Test
- Run test suite to ensure nothing broke
- Test main pipeline scripts
- Verify imports work correctly

### Step 4: Remove Old Structure
- Delete old directories
- Clean up any remaining references
- Update README.md with new structure

### Step 5: Document
- Update README.md with new structure
- Create CONTRIBUTING.md with guidelines
- Document where new code should go

---

## Specific File Actions

### Scripts to Archive (Examples)
Move to `/scripts/archived/`:
- `adjust_hoh_criteria_step3.py`
- `analyze_excess_single_filers.py`
- `analyze_hoh_bottlenecks.py`
- `analyze_married_misclassification.py`
- `calculate_weighted_sum_fixed.py`
- `calculate_weighted_totals_corrected.py`
- All `*_step1.py`, `*_step3.py`, `*_v2.py` variants

### Scripts to Keep in Pipeline
- `construct_tax_units.py` → `pipeline/01_construct_tax_units.py`
- `validate_tax_units.py` → `pipeline/03_validate_results.py`
- `compare_to_soi.py` → `analysis/validation/compare_to_soi.py`

### Scripts to Keep in Analysis
- `analyze_filing_status_gaps.py` → `analysis/filing_status/`
- `analyze_income_distribution.py` → `analysis/income/`
- `analyze_high_income.py` → `analysis/income/`

---

## Benefits of Reorganization

### For Development
- **Faster navigation**: Find relevant code quickly
- **Clear boundaries**: Know where new code belongs
- **Reduced confusion**: No duplicate structures
- **Better onboarding**: New developers understand structure

### For Maintenance
- **Easier refactoring**: Logical groupings make changes easier
- **Better testing**: Clear test organization
- **Simpler debugging**: Know where to look for issues
- **Version control**: Cleaner git history

### For Collaboration
- **Clear conventions**: Everyone follows same structure
- **Better code review**: Easier to review organized code
- **Documentation**: Structure is self-documenting
- **Reduced conflicts**: Less chance of merge conflicts

---

## Next Steps

1. **Review this plan** - Get feedback on proposed structure
2. **Prioritize changes** - Decide which phases to implement first
3. **Create migration scripts** - Automate file moves and import updates
4. **Test thoroughly** - Ensure nothing breaks
5. **Update documentation** - Reflect new structure in README.md

---

## Questions to Consider

1. **Project name**: Is `hawaii-tax-estimation` the best name?
2. **Script organization**: Should we use numbered prefixes for pipeline scripts?
3. **Module depth**: Is the proposed nesting too deep or too shallow?
4. **Backward compatibility**: Do we need to maintain old import paths?
5. **Git history**: Should we preserve git history when moving files?

---

## Estimated Effort

- **Phase 1 (Rename)**: 30 minutes
- **Phase 2 (Scripts)**: 2-3 hours
- **Phase 3 (Duplicates)**: 1 hour
- **Phase 4 (Outputs)**: 1 hour
- **Phase 5 (Modules)**: 3-4 hours
- **Phase 6 (Tests)**: 2 hours

**Total**: ~10-12 hours of focused work

**Recommendation**: Do Phases 1-3 first (high priority), then evaluate before continuing.
