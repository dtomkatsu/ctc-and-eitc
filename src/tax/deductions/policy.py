"""
Hawaii Tax Policy Parameters for Deductions and Exemptions.

Configurable policy parameters to enable revenue modeling of policy changes.
"""

from typing import Dict, Optional
import json
from pathlib import Path


class DeductionPolicy:
    """
    Configurable policy parameters for deductions and exemptions.
    
    Enables revenue modeling of policy changes like:
    - Increasing/decreasing standard deduction
    - Eliminating/modifying personal exemptions
    - Capping itemized deductions
    
    Example:
        # Baseline 2022
        policy = DeductionPolicy(year=2022)
        
        # Scenario: Increase standard deduction by 20%
        policy.set_policy_change('standard_deduction.single', 2640)
        policy.set_policy_change('standard_deduction.joint', 5280)
    """
    
    def __init__(self, year: int = 2022):
        """
        Initialize policy parameters.
        
        Args:
            year: Tax year for baseline parameters
        """
        self.year = year
        self._set_baseline_parameters()
        self.modifications = []  # Track policy changes
    
    def _set_baseline_parameters(self):
        """Set baseline 2022 Hawaii tax parameters."""
        
        # Standard deductions (2022 Hawaii values)
        # Source: Hawaii Department of Taxation
        self.standard_deduction = {
            'single': 2200,
            'joint': 4400,
            'hoh': 3212,
            'mfs': 2200
        }
        
        # Additional standard deduction for age 65+
        # Hawaii allows additional deduction for elderly taxpayers
        self.standard_deduction_additional_age = {
            'single': 1300,
            'joint': 1050,  # per person (both spouses can claim)
            'hoh': 1300,
            'mfs': 1300
        }
        
        # Personal exemption amount per person
        # Hawaii allows exemptions for self, spouse, and dependents
        self.personal_exemption = 1144  # 2022 value
        
        # Itemized deduction limits (if any)
        self.itemized_deduction_limits = {
            'medical_agi_threshold': 0.075,  # 7.5% AGI floor for medical expenses
            'charitable_agi_limit': 0.5,     # 50% AGI limit for charitable contributions
            'misc_agi_threshold': 0.02,      # 2% AGI floor for misc deductions
        }
        
        # Itemized deduction cap (for high earners)
        self.itemized_deduction_cap = None  # None = no cap
        self.itemized_deduction_cap_threshold = None  # AGI threshold for cap
        
        # Exemption phaseout (if any)
        self.exemption_phaseout_start = None  # None = no phaseout
        self.exemption_phaseout_rate = None
    
    def set_policy_change(self, parameter: str, value):
        """
        Modify policy parameter for scenario analysis.
        
        Args:
            parameter: Parameter path (e.g., 'personal_exemption', 'standard_deduction.single')
            value: New value
        
        Examples:
            policy.set_policy_change('personal_exemption', 1500)  # Increase exemption
            policy.set_policy_change('standard_deduction.single', 2500)  # Increase std ded
            policy.set_policy_change('itemized_deduction_cap', 20000)  # Cap itemized
        """
        # Track modification
        self.modifications.append({
            'parameter': parameter,
            'old_value': self._get_parameter_value(parameter),
            'new_value': value
        })
        
        # Apply modification
        if '.' in parameter:
            # Handle nested parameters (e.g., 'standard_deduction.single')
            parts = parameter.split('.')
            obj = self
            for part in parts[:-1]:
                obj = getattr(obj, part)
            # Handle dict assignment
            if isinstance(obj, dict):
                obj[parts[-1]] = value
            else:
                setattr(obj, parts[-1], value)
        else:
            setattr(self, parameter, value)
    
    def _get_parameter_value(self, parameter: str):
        """Get current value of a parameter."""
        if '.' in parameter:
            parts = parameter.split('.')
            obj = self
            for part in parts[:-1]:
                obj = getattr(obj, part)
            # Handle dict access
            if isinstance(obj, dict):
                return obj.get(parts[-1])
            else:
                return getattr(obj, parts[-1])
        else:
            return getattr(self, parameter)
    
    def get_standard_deduction(self, filing_status: str, age_65_plus_count: int = 0) -> float:
        """
        Calculate standard deduction including age adjustments.
        
        Args:
            filing_status: 'single', 'joint', 'hoh', or 'mfs'
            age_65_plus_count: Number of taxpayers aged 65+ (0, 1, or 2 for joint)
        
        Returns:
            Total standard deduction amount
        """
        base = self.standard_deduction[filing_status]
        
        if age_65_plus_count > 0:
            additional = self.standard_deduction_additional_age[filing_status]
            if filing_status == 'joint':
                # Joint filers get additional per person (both can be 65+)
                base += additional * min(age_65_plus_count, 2)
            else:
                # Single/HOH/MFS get one additional
                base += additional
        
        return base
    
    def get_total_exemptions(self, num_exemptions: int, agi: float = 0) -> float:
        """
        Calculate total personal exemption amount.
        
        Args:
            num_exemptions: Number of exemptions claimed
            agi: Adjusted Gross Income (for phaseout calculation)
        
        Returns:
            Total exemption amount (may be reduced by phaseout)
        """
        base_exemption = self.personal_exemption * num_exemptions
        
        # Apply phaseout if applicable
        if self.exemption_phaseout_start and agi > self.exemption_phaseout_start:
            excess = agi - self.exemption_phaseout_start
            reduction = excess * self.exemption_phaseout_rate
            base_exemption = max(0, base_exemption - reduction)
        
        return base_exemption
    
    def apply_itemized_deduction_cap(self, itemized_amount: float, agi: float) -> float:
        """
        Apply itemized deduction cap if applicable.
        
        Args:
            itemized_amount: Total itemized deductions before cap
            agi: Adjusted Gross Income
        
        Returns:
            Itemized deductions after applying cap
        """
        if self.itemized_deduction_cap is None:
            return itemized_amount
        
        # Only apply cap if AGI exceeds threshold
        if self.itemized_deduction_cap_threshold and agi < self.itemized_deduction_cap_threshold:
            return itemized_amount
        
        return min(itemized_amount, self.itemized_deduction_cap)
    
    def is_baseline(self) -> bool:
        """Check if this is the baseline policy (no modifications)."""
        return len(self.modifications) == 0
    
    def get_modifications_summary(self) -> str:
        """Get summary of policy modifications."""
        if self.is_baseline():
            return f"Baseline {self.year} Hawaii tax policy (no modifications)"
        
        summary = [f"Modified {self.year} Hawaii tax policy:"]
        for mod in self.modifications:
            summary.append(f"  - {mod['parameter']}: {mod['old_value']} → {mod['new_value']}")
        
        return "\n".join(summary)
    
    def save_to_file(self, filepath: str):
        """Save policy parameters to JSON file."""
        policy_dict = {
            'year': self.year,
            'standard_deduction': self.standard_deduction,
            'standard_deduction_additional_age': self.standard_deduction_additional_age,
            'personal_exemption': self.personal_exemption,
            'itemized_deduction_limits': self.itemized_deduction_limits,
            'itemized_deduction_cap': self.itemized_deduction_cap,
            'itemized_deduction_cap_threshold': self.itemized_deduction_cap_threshold,
            'exemption_phaseout_start': self.exemption_phaseout_start,
            'exemption_phaseout_rate': self.exemption_phaseout_rate,
            'modifications': self.modifications
        }
        
        with open(filepath, 'w') as f:
            json.dump(policy_dict, f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'DeductionPolicy':
        """Load policy parameters from JSON file."""
        with open(filepath, 'r') as f:
            policy_dict = json.load(f)
        
        policy = cls(year=policy_dict['year'])
        
        # Apply saved parameters
        for key, value in policy_dict.items():
            if key != 'year' and key != 'modifications':
                setattr(policy, key, value)
        
        policy.modifications = policy_dict.get('modifications', [])
        
        return policy
    
    def __repr__(self) -> str:
        return f"DeductionPolicy(year={self.year}, modifications={len(self.modifications)})"


def create_baseline_policy_file():
    """Create and save baseline 2022 policy file."""
    policy = DeductionPolicy(year=2022)
    
    output_dir = Path('data/policy')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'baseline_2022_deductions.json'
    policy.save_to_file(str(output_path))
    
    print(f"✅ Saved baseline 2022 policy to: {output_path}")
    return policy


if __name__ == '__main__':
    create_baseline_policy_file()
