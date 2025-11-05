# Comprehensive Tax Calibration Optimization Plan

## Objective
Systematically optimize all AGI brackets to align three key metrics with DOTAX Table A8 targets:
1. **Filer counts** by bracket
2. **AGI totals** by bracket
3. **Tax liability** by bracket

## Problem Analysis

### Current Status
- **$400k+ bracket**: Manually optimized to excellent alignment (Joint +5.6%, Single -0.9%, HoH +1.9%)
- **Other brackets**: Need systematic optimization across all 12 AGI brackets
- **Challenge**: Must evaluate against FULL pipeline output, not intermediate data

### Key Insight from Previous Optimization
The ultra-high-income optimizer initially failed because it evaluated against pre-calibration data. The successful approach was to:
1. Run the ENTIRE `regenerate_tax_units.py` pipeline
2. Extract final results after all calibration stages
3. Compare against targets
4. Iterate parameters

## Optimization Strategy

### Phase 1: Parameter Identification
Identify all tunable parameters across the calibration pipeline:

#### Stage 1: Itemized Deduction Reduction
- **Parameters**: Deduction reduction rates by income bracket
- **Current**: 40-60% reduction across all brackets
- **Optimization opportunity**: Bracket-specific reduction rates
- **File**: `src/tax/deductions/itemized_estimator.py`

#### Stage 2: Comprehensive Bracket Calibration (Pareto)
- **Parameters**: Target filer counts per bracket (from DOTAX Table A2)
- **Current**: Fixed targets from DOTAX
- **Optimization opportunity**: None (these are targets, not parameters)
- **File**: `src/tax/adjustments/pareto_calibration.py`

#### Stage 3: Income Distribution Calibration
- **Parameters**: 
  - Income adjustment factors by bracket
  - Percentile-based redistribution weights
- **Current**: Systematic shifts to match effective rates
- **Optimization opportunity**: Fine-tune adjustment magnitudes
- **File**: `src/tax/adjustments/income_distribution_calibrator.py`

#### Stage 4: Weight Calibration (Low/Middle Income)
- **Parameters**:
  - Weight adjustment factors for $0-$200k brackets
  - Filing status-specific adjustments
- **Current**: Direct bracket-level weight adjustment
- **Optimization opportunity**: Optimize adjustment factors
- **File**: `src/tax/adjustments/comprehensive_weight_calibrator.py`

#### Stage 5: Ultra-High-Income Synthesis
- **Parameters**: Weight factors for synthetic filers (MFJ, Single, HoH)
- **Current**: Manually optimized Iteration 9 (MFJ 35/28/22/15%, etc.)
- **Optimization opportunity**: Keep current (already optimized)
- **File**: `src/tax/adjustments/ultra_high_income_synthesizer_v2.py`

#### Stage 6: Final Gap Closer
- **Parameters**:
  - Middle-income weight reduction (8-10%)
  - High-income deduction reduction (15%)
  - Tax multipliers by bracket
- **Current**: Hybrid solution C parameters
- **Optimization opportunity**: Fine-tune all three components
- **File**: `src/tax/adjustments/final_gap_closer.py`

### Phase 2: Objective Function Design

#### Multi-Objective Optimization
Minimize weighted sum of squared percentage errors across three dimensions:

```python
def objective_function(params):
    # Run full pipeline with params
    tax_units = run_full_pipeline(params)
    
    # Calculate errors for each bracket
    errors = []
    for bracket in ALL_BRACKETS:
        # Filer count error (weight: 0.3)
        filer_error = ((model_count - target_count) / target_count) ** 2
        
        # AGI error (weight: 0.3)
        agi_error = ((model_agi - target_agi) / target_agi) ** 2
        
        # Tax liability error (weight: 0.4 - most important)
        tax_error = ((model_tax - target_tax) / target_tax) ** 2
        
        # Weighted sum for this bracket
        bracket_error = 0.3 * filer_error + 0.3 * agi_error + 0.4 * tax_error
        errors.append(bracket_error)
    
    # Total error with bracket-specific weights
    # (higher income brackets more important due to revenue impact)
    bracket_weights = get_bracket_weights()  # Based on revenue contribution
    total_error = np.sum(np.array(errors) * bracket_weights)
    
    return total_error
```

#### Bracket Weighting Strategy
- **$0-$50k**: Weight = 0.5 (lower importance, small revenue impact)
- **$50k-$200k**: Weight = 1.0 (standard weight)
- **$200k-$400k**: Weight = 1.5 (higher importance)
- **$400k+**: Weight = 2.0 (highest importance, largest revenue impact)

### Phase 3: Optimization Implementation

#### Full-Pipeline Wrapper
Create `scripts/optimize_full_calibration.py`:

```python
class FullCalibrationOptimizer:
    """Optimize all calibration parameters across the full pipeline."""
    
    def __init__(self):
        self.targets = load_dotax_targets()  # Table A8 targets
        
    def run_full_pipeline(self, params):
        """
        Run regenerate_tax_units.py with custom parameters.
        Returns final calibrated tax units.
        """
        # 1. Construct tax units from PUMS
        tax_units = constructor.create_tax_units()
        
        # 2. Apply itemized deduction reduction (parameterized)
        tax_units = apply_itemized_deduction_reduction(
            tax_units,
            reduction_rates=params['deduction_rates']
        )
        
        # 3. Calculate initial taxes
        tax_units = calculator.calculate_tax_units_batch(tax_units)
        
        # 4. Apply Pareto calibration (fixed targets)
        tax_units = apply_pareto_calibration(tax_units)
        
        # 5. Apply income distribution calibration (parameterized)
        tax_units = apply_income_distribution_calibration(
            tax_units,
            adjustment_factors=params['income_adjustments']
        )
        
        # 6. Apply weight calibration (parameterized)
        tax_units = apply_comprehensive_weight_calibration(
            tax_units,
            weight_adjustments=params['weight_adjustments']
        )
        
        # 7. Apply ultra-high-income synthesis (keep current)
        tax_units = apply_ultra_high_income_synthesis(tax_units)
        
        # 8. Apply final gap closer (parameterized)
        tax_units = apply_final_gap_closer(
            tax_units,
            gap_closer_params=params['gap_closer']
        )
        
        return tax_units
    
    def evaluate(self, params):
        """Evaluate parameter set against targets."""
        tax_units = self.run_full_pipeline(params)
        
        # Calculate errors for all metrics
        total_error = 0
        for bracket in self.targets['brackets']:
            bracket_mask = (tax_units['agi'] >= bracket['min']) & \
                          (tax_units['agi'] < bracket['max'])
            
            # Model values
            model_count = tax_units.loc[bracket_mask, 'weight'].sum()
            model_agi = (tax_units.loc[bracket_mask, 'agi'] * \
                        tax_units.loc[bracket_mask, 'weight']).sum()
            model_tax = (tax_units.loc[bracket_mask, 'hi_state_tax'] * \
                        tax_units.loc[bracket_mask, 'weight']).sum()
            
            # Target values
            target_count = bracket['target_count']
            target_agi = bracket['target_agi']
            target_tax = bracket['target_tax']
            
            # Calculate weighted error
            error = self._calculate_bracket_error(
                model_count, model_agi, model_tax,
                target_count, target_agi, target_tax,
                bracket['weight']
            )
            
            total_error += error
        
        return total_error
    
    def optimize(self, method='differential_evolution'):
        """Run optimization."""
        if method == 'differential_evolution':
            result = differential_evolution(
                self.evaluate,
                bounds=self._get_parameter_bounds(),
                maxiter=50,
                popsize=20,
                tol=0.001,
                workers=1  # Sequential to avoid conflicts
            )
        
        return result
```

#### Parameter Structure
```python
params = {
    'deduction_rates': {
        # Reduction rate by bracket (0.0 = no reduction, 1.0 = 100% reduction)
        '0-50k': 0.40,
        '50-100k': 0.45,
        '100-200k': 0.50,
        '200-400k': 0.55,
        '400k+': 0.60,
    },
    'income_adjustments': {
        # Income adjustment factor by bracket (-0.3 to +0.3)
        '100-150k': 0.0,
        '150-200k': -0.05,
        '200-300k': -0.10,
        '300-400k': -0.08,
        '400k+': 0.0,
    },
    'weight_adjustments': {
        # Weight multiplier by bracket (0.7 to 1.3)
        '0-10k': 1.0,
        '10-20k': 0.95,
        '20-30k': 0.95,
        '30-40k': 0.98,
        '40-50k': 0.98,
        '50-75k': 1.0,
        '75-100k': 1.02,
        '100-200k': 1.0,
    },
    'gap_closer': {
        'middle_income_weight_reduction': 0.09,  # 8-10%
        'high_income_deduction_reduction': 0.15,  # 15%
        'tax_multipliers': {
            # Tax multiplier by bracket (0.8 to 1.2)
            '0-50k': 1.0,
            '50-100k': 1.0,
            '100-200k': 0.95,
            '200-400k': 0.90,
            '400k+': 1.0,
        }
    }
}
```

### Phase 4: Iterative Refinement

#### Process
1. **Initial run**: Start with current manual parameters
2. **Optimization**: Run differential_evolution for 50 iterations
3. **Validation**: Check results against targets
4. **Iteration**: If not acceptable, adjust bounds and re-optimize
5. **Repeat**: Until all brackets within acceptable thresholds

#### Acceptable Thresholds
- **Filer count**: ±5% of target
- **AGI**: ±7% of target
- **Tax liability**: ±10% of target for low brackets, ±5% for high brackets

#### Iteration Strategy
- **Iteration 1**: Broad search (large bounds)
- **Iteration 2**: Narrow around best result from iteration 1
- **Iteration 3**: Fine-tuning (very narrow bounds)
- **Continue**: Until convergence or acceptable performance

### Phase 5: Validation & Testing

#### Validation Checks
1. **Bracket-by-bracket comparison** against DOTAX Table A8
2. **Filing status distribution** preserved
3. **Income source distributions** reasonable
4. **No extreme outliers** in adjusted values
5. **Pipeline stability** (reproducible results)

#### Testing Scenarios
- **Baseline**: Current manual configuration
- **Optimized**: After optimization
- **Comparison**: Document improvements in each metric

## Implementation Timeline

### Step 1: Setup (1 hour)
- [ ] Create optimization script structure
- [ ] Define parameter bounds
- [ ] Implement full-pipeline wrapper
- [ ] Set up evaluation metrics

### Step 2: Initial Optimization (2-4 hours)
- [ ] Run differential_evolution optimization
- [ ] Monitor convergence
- [ ] Extract optimal parameters

### Step 3: Validation (1 hour)
- [ ] Run regenerate_tax_units.py with optimal params
- [ ] Compare all brackets against targets
- [ ] Document improvements

### Step 4: Iteration (2-6 hours)
- [ ] If not acceptable, adjust bounds
- [ ] Re-run optimization
- [ ] Repeat until convergence

### Step 5: Finalization (1 hour)
- [ ] Apply optimal parameters to source files
- [ ] Update documentation
- [ ] Create memory of final configuration

## Expected Outcomes

### Success Criteria
- **All brackets** within acceptable thresholds (±10% tax liability)
- **Total tax accuracy** improved to within ±5%
- **Filing status distributions** preserved
- **Reproducible** results

### Potential Improvements
- Current: 50% of brackets within ±10% on tax liability
- Target: 80%+ of brackets within ±10% on tax liability
- Current: -4.1% total tax accuracy
- Target: ±2% total tax accuracy

## Risk Mitigation

### Known Risks
1. **Long optimization time**: Each pipeline run takes ~30 seconds
   - Mitigation: Use coarse grid search first, then fine-tune
2. **Local optima**: May get stuck in suboptimal solution
   - Mitigation: Multiple optimization runs with different seeds
3. **Parameter interactions**: Changes in one stage affect others
   - Mitigation: Sequential optimization (optimize one stage at a time)
4. **Overfitting**: Too many parameters may overfit to current data
   - Mitigation: Limit parameters, use regularization

### Fallback Strategy
If full optimization doesn't converge:
1. **Sequential optimization**: Optimize one stage at a time
2. **Manual refinement**: Use optimizer to find direction, manually fine-tune
3. **Hybrid approach**: Optimize most sensitive parameters only

## Next Steps
1. Create `scripts/optimize_full_calibration.py` 
2. Run initial optimization
3. Validate results
4. Iterate until acceptable
5. Document final configuration
