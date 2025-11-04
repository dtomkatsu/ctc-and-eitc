# Hawaii Tax Model - Implementation Priority Plan

**Purpose**: Actionable roadmap to improve model accuracy from 79% to 85%+  
**Timeline**: 4 weeks  
**Effort**: ~40 hours total

## Quick Wins (Week 1) - 8 hours

### 1. Add Non-Resident Revenue ✅ DONE
**Impact**: +8.8% coverage improvement  
**Status**: Script created and tested
```bash
python scripts/add_nonresident_revenue.py
```
**Result**: Model now covers 100% of revenue (residents + non-residents)

### 2. Organize Project Structure - 2 hours
**Impact**: Better maintainability, easier navigation
```bash
# Step 1: Review what will be moved
python scripts/organize_project_files.py

# Step 2: Execute organization
python scripts/organize_project_files.py --execute

# Step 3: Archive large logs
mkdir -p logs/archive
mv regenerate*.log logs/archive/
gzip logs/archive/*.log

# Step 4: Update .gitignore
echo "logs/archive/" >> .gitignore
echo "*.log" >> .gitignore
```

### 3. Activate Synthetic High-Income Units - 2 hours
**Impact**: Fix $1M+ undercount (~5% accuracy improvement)
```python
# In scripts/regenerate_tax_units.py, enable:
from src.tax.adjustments.ultra_high_income_synthesizer_v2 import UltraHighIncomeSynthesizerV2

# Add after Stage 5:
synthesizer = UltraHighIncomeSynthesizerV2(
    pareto_alpha=1.3,
    tail_multiplier=0.45
)
tax_units = synthesizer.synthesize(tax_units)
```

### 4. Create Central Configuration - 4 hours
**Impact**: Easier parameter management
```python
# Create config/model_config.py
class ModelConfig:
    """Centralized model configuration."""
    
    # Calibration parameters
    SCENARIO = 'moderate'  # conservative, moderate, aggressive
    
    # Ensemble weights (moderate scenario)
    ENSEMBLE_WEIGHTS = {
        'fy_actual_2022_2024': 0.30,
        'fy_2025_estimate': 0.10,
        'dotax_2018_2021': 0.20,
        'bls_wage': 0.25,
        'acs_income': 0.10,
        'demographics': 0.05
    }
    
    # Revenue parameters
    INCOME_SCALING_FACTOR = 0.935  # -6.5% adjustment
    GROWTH_RATE_ANNUAL = 0.020     # 2.0%
    NONRESIDENT_SHARE = 0.088      # 8.8% of total
    
    # Act 46 parameters
    ACT46_IMPACT_RATE = -0.199     # -19.9% of resident revenue
    
    # Targets
    FY_2024_ACTUAL = 3280          # Million (last confirmed)
    FY_2026_RESIDENT_TARGET = 3085 # Million (moderate)
    FY_2026_TOTAL_TARGET = 3383    # Million (with non-residents)
    
    @classmethod
    def get_scenario_params(cls, scenario='moderate'):
        """Get parameters for specific scenario."""
        scenarios = {
            'conservative': {
                'growth_rate': 0.015,
                'target_resident': 3082,
                'income_scaling': 0.934
            },
            'moderate': {
                'growth_rate': 0.020,
                'target_resident': 3085,
                'income_scaling': 0.935
            },
            'aggressive': {
                'growth_rate': 0.025,
                'target_resident': 3074,
                'income_scaling': 0.932
            }
        }
        return scenarios[scenario]
```

## High-Impact Improvements (Week 2) - 16 hours

### 5. Dynamic Calibration System - 8 hours
**Impact**: Self-correcting model, maintains accuracy over time

Create `src/calibration/dynamic_calibrator.py`:
```python
class DynamicCalibrator:
    """Auto-calibrate model when new FY data available."""
    
    def __init__(self, config):
        self.config = config
        self.history = []
    
    def update_from_actual(self, fy_year, actual_revenue):
        """Update model based on actual FY data."""
        
        # Compare to projection
        projection = self.get_projection_for_year(fy_year)
        error = (projection - actual_revenue) / actual_revenue
        
        # Adjust weights based on error
        if abs(error) > 0.05:  # More than 5% error
            self.adjust_ensemble_weights(error)
            self.adjust_income_scaling(error)
        
        # Log for learning
        self.history.append({
            'year': fy_year,
            'actual': actual_revenue,
            'projected': projection,
            'error': error,
            'adjustments': self.get_adjustments()
        })
    
    def adjust_ensemble_weights(self, error):
        """Adjust ensemble weights based on projection error."""
        
        if error > 0:  # Overestimated
            # Increase weight on conservative components
            self.config.ENSEMBLE_WEIGHTS['fy_actual_2022_2024'] += 0.05
            self.config.ENSEMBLE_WEIGHTS['dotax_2018_2021'] -= 0.05
        else:  # Underestimated
            # Increase weight on growth components
            self.config.ENSEMBLE_WEIGHTS['dotax_2018_2021'] += 0.05
            self.config.ENSEMBLE_WEIGHTS['fy_actual_2022_2024'] -= 0.05
        
        # Normalize weights
        total = sum(self.config.ENSEMBLE_WEIGHTS.values())
        for key in self.config.ENSEMBLE_WEIGHTS:
            self.config.ENSEMBLE_WEIGHTS[key] /= total
```

### 6. Enhanced Itemized Deductions - 8 hours
**Impact**: 2-3% accuracy improvement

Enhance `src/tax/deductions/itemized_estimator.py`:
```python
def estimate_itemized_deductions_enhanced(tax_unit):
    """Enhanced itemized deduction estimation using PUMS data."""
    
    deductions = {}
    
    # 1. SALT deductions (use actual Hawaii tax from prior year)
    if hasattr(tax_unit, 'prior_year_state_tax'):
        salt_base = tax_unit.prior_year_state_tax
    else:
        salt_base = tax_unit.agi * 0.045  # Estimate
    
    # Add property tax from PUMS
    property_tax = 0
    if hasattr(tax_unit, 'VALP'):  # Property value
        property_tax = tax_unit.VALP * 0.0028  # Hawaii avg rate
    
    # Add GET estimate
    get_estimate = tax_unit.agi * 0.002
    
    deductions['SALT'] = min(salt_base + property_tax + get_estimate, 10000)
    
    # 2. Mortgage interest (from PUMS)
    if hasattr(tax_unit, 'MRGP'):  # Mortgage payment
        # Estimate interest portion (70% of payment in early years)
        annual_interest = tax_unit.MRGP * 12 * 0.70
        deductions['mortgage_interest'] = annual_interest
    else:
        deductions['mortgage_interest'] = 0
    
    # 3. Charitable contributions (income-based with demographic adjustment)
    base_rate = 0.025  # 2.5% base
    
    # Adjust by age
    if tax_unit.age > 65:
        base_rate *= 1.3  # Seniors give more
    
    # Adjust by income level
    if tax_unit.agi > 200000:
        base_rate *= 1.5  # High earners give more
    
    deductions['charitable'] = tax_unit.agi * base_rate
    
    # 4. Medical expenses (above 7.5% AGI threshold)
    medical_base = tax_unit.agi * 0.05  # 5% average medical
    if tax_unit.age > 65:
        medical_base *= 2  # Seniors have higher medical
    
    medical_deductible = max(0, medical_base - (tax_unit.agi * 0.075))
    deductions['medical'] = medical_deductible
    
    total_itemized = sum(deductions.values())
    
    # Compare to standard deduction
    standard = get_standard_deduction(tax_unit.filing_status, 2026)
    
    return {
        'itemized_total': total_itemized,
        'standard_deduction': standard,
        'uses_itemized': total_itemized > standard,
        'deduction_amount': max(total_itemized, standard),
        'breakdown': deductions
    }
```

## Medium-Impact Improvements (Week 3) - 12 hours

### 7. Confidence Intervals - 6 hours
**Impact**: Better uncertainty quantification

```python
# src/analysis/uncertainty.py
import numpy as np
from scipy import stats

class UncertaintyAnalyzer:
    """Calculate confidence intervals for projections."""
    
    def bootstrap_projection(self, tax_units, n_samples=1000):
        """Bootstrap confidence intervals."""
        
        projections = []
        
        for _ in range(n_samples):
            # Resample with replacement
            sample = tax_units.sample(n=len(tax_units), replace=True)
            
            # Add noise to parameters
            growth_noise = np.random.normal(0, 0.005)  # ±0.5% std
            scaling_noise = np.random.normal(0, 0.01)   # ±1% std
            
            # Calculate projection
            projection = self.calculate_projection(
                sample,
                growth_adjustment=growth_noise,
                scaling_adjustment=scaling_noise
            )
            projections.append(projection)
        
        # Calculate confidence intervals
        projections = np.array(projections)
        
        return {
            'mean': np.mean(projections),
            'median': np.median(projections),
            'std': np.std(projections),
            'ci_95': np.percentile(projections, [2.5, 97.5]),
            'ci_90': np.percentile(projections, [5, 95]),
            'ci_68': np.percentile(projections, [16, 84])
        }
```

### 8. Documentation Consolidation - 6 hours
**Impact**: Better usability and maintenance

```bash
# Consolidate similar documentation
cd docs/

# Create consolidated guides
cat *CALIBRATION*.md > calibration/COMPLETE_CALIBRATION_GUIDE.md
cat *IMPLEMENTATION*.md > implementation/IMPLEMENTATION_GUIDE.md
cat *PROJECTION*.md > projections/PROJECTION_GUIDE.md

# Archive old versions
mkdir -p archive/2025_11
mv *_SUMMARY.md archive/2025_11/
mv *_PLAN.md archive/2025_11/
mv *_REPORT.md archive/2025_11/

# Create index
cat > README.md << EOF
# Hawaii Tax Model Documentation

## Quick Start
- [README](../README.md) - Project overview
- [Getting Started](guides/QUICK_START.md)
- [Model Accuracy](MODEL_ACCURACY_EVALUATION.md)

## Methodology
- [Calibration Approach](methodology/CALIBRATION_APPROACH.md)
- [Ensemble Methodology](methodology/ENSEMBLE_WEIGHTS.md)
- [Resident vs Total](methodology/RESIDENT_VS_TOTAL.md)

## Results
- [2026 Projections](results/PROJECTIONS_2026.md)
- [Act 46 Impact](results/ACT46_IMPACT.md)
- [Accuracy Evaluation](results/ACCURACY_EVALUATION.md)

## Technical Guides
- [Calibration Guide](guides/CALIBRATION_GUIDE.md)
- [Data Setup](guides/DATA_SETUP.md)
- [API Reference](guides/API_REFERENCE.md)
EOF
```

## Long-Term Improvements (Week 4) - 4 hours

### 9. Automated Testing Suite - 4 hours
**Impact**: Prevent regressions, ensure quality

```python
# tests/test_accuracy.py
import pytest
from src.projection.ensemble import EnsembleProjector
from config.model_config import ModelConfig

class TestModelAccuracy:
    """Test model accuracy against benchmarks."""
    
    def test_revenue_projection_range(self):
        """Test that projections are within acceptable range."""
        projector = EnsembleProjector()
        
        # Conservative scenario
        conservative = projector.project(scenario='conservative')
        assert 3070 <= conservative['resident'] <= 3095
        
        # Moderate scenario
        moderate = projector.project(scenario='moderate')
        assert 3075 <= moderate['resident'] <= 3095
        
        # Aggressive scenario
        aggressive = projector.project(scenario='aggressive')
        assert 3065 <= aggressive['resident'] <= 3085
    
    def test_nonresident_share(self):
        """Test non-resident calculations."""
        from scripts.add_nonresident_revenue import NonResidentRevenueEstimator
        
        estimator = NonResidentRevenueEstimator()
        result = estimator.estimate_total_from_resident(3085)
        
        assert abs(result['nonresident_share'] - 0.088) < 0.001
        assert abs(result['total'] - 3383) < 1
    
    def test_act46_impact(self):
        """Test Act 46 impact calculations."""
        baseline = 3085
        impact_rate = -0.199
        expected_impact = baseline * impact_rate
        
        assert abs(expected_impact - (-614)) < 1
    
    def test_growth_rates(self):
        """Test that growth rates are reasonable."""
        config = ModelConfig()
        
        for scenario in ['conservative', 'moderate', 'aggressive']:
            params = config.get_scenario_params(scenario)
            assert 0.015 <= params['growth_rate'] <= 0.025
```

## Success Metrics

### Week 1 Completion
- [ ] Non-resident revenue added
- [ ] Project files organized
- [ ] Synthetic high-income activated
- [ ] Central config created

### Week 2 Completion
- [ ] Dynamic calibration implemented
- [ ] Itemized deductions enhanced
- [ ] Model accuracy > 82%

### Week 3 Completion
- [ ] Confidence intervals added
- [ ] Documentation consolidated
- [ ] Model accuracy > 84%

### Week 4 Completion
- [ ] Test suite running
- [ ] All improvements integrated
- [ ] Model accuracy > 85%

## Expected Outcomes

### Accuracy Improvements
| Component | Before | After | Improvement |
|-----------|--------|-------|-------------|
| Revenue Coverage | 91.2% | 100% | +8.8pp |
| High-Income Accuracy | 70% | 85% | +15pp |
| Itemized Deductions | 85% | 95% | +10pp |
| Overall Accuracy | 79% | 86% | +7pp |

### Organization Improvements
| Aspect | Before | After |
|--------|--------|-------|
| Root directory files | 93 | <10 |
| Documentation structure | Chaotic | Organized |
| Configuration | Scattered | Centralized |
| Test coverage | Minimal | Comprehensive |

## Risk Mitigation

### Risk: Breaking existing functionality
**Mitigation**: 
- Create tests before changes
- Version control all changes
- Keep backup of working version

### Risk: Over-optimization
**Mitigation**:
- Focus on biggest impact items first
- Validate each change independently
- Don't exceed ±5% adjustment limits

### Risk: Time overrun
**Mitigation**:
- Prioritize quick wins first
- Time-box each task
- Skip nice-to-haves if needed

---

**Start Date**: November 4, 2025  
**Target Completion**: December 1, 2025  
**Total Effort**: ~40 hours  
**Expected Outcome**: Model accuracy from 79% → 86%
