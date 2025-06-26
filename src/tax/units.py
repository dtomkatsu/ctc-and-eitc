"""
Tax Unit Construction Module

This module provides functionality to construct tax units from PUMS data
using rule-based and machine learning approaches. Includes Hawaii-specific
adjustments for extended family structures and dependency rules.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
import logging

# Relationship codes with Hawaii-specific extensions
RELSHIPP_CODES = {
    '20': 'Reference person',
    '21': 'Opposite-sex spouse',
    '22': 'Opposite-sex unmarried partner',
    '23': 'Same-sex spouse',
    '24': 'Same-sex unmarried partner',
    '25': 'Biological son or daughter',
    '26': 'Adopted son or daughter',
    '27': 'Stepson or stepdaughter',
    '28': 'Brother or sister',
    '29': 'Father or mother',
    '30': 'Grandchild',
    '31': 'Parent-in-law',
    '32': 'Son-in-law or daughter-in-law',
    '33': 'Other relative',
    '34': 'Foster child',
    '39': 'Grandparent (Hawaii extended family)',
    '40': 'Aunt/Uncle (Hawaii extended family)',
    '41': 'Niece/Nephew (Hawaii extended family)',
    '42': 'Cousin (Hawaii extended family)',
    '43': 'In-law (Hawaii extended family)',
    '44': 'Other relative (Hawaii extended family)'
}

# Hawaii-specific income thresholds (2023)
HAWAII_INCOME_THRESHOLDS = {
    'single': 15000,      # Lower threshold for single filers in Hawaii
    'joint': 30000,       # Lower threshold for joint filers in Hawaii
    'hoh': 22500,         # Lower threshold for head of household in Hawaii
    'dependent': 5000,    # Lower threshold for dependents in Hawaii
    'support': 20000,     # Support test threshold for Hawaii
    'relative_income': 5000  # Gross income test for qualifying relatives in Hawaii
}

# Filing status codes
FILING_STATUS = {
    'SINGLE': 1,
    'JOINT': 2,
    'SEPARATE': 3,
    'HEAD_HOUSEHOLD': 4,
    'WIDOW': 5
}

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('tax_units.log')
    ]
)
logger = logging.getLogger(__name__)

class TaxUnitConstructor:
    """Class for constructing tax units from PUMS data."""
    
    def __init__(self, person_df: pd.DataFrame, hh_df: pd.DataFrame):
        """Initialize with person and household data."""
        self.person_df = person_df.copy()
        self.hh_df = hh_df.copy()
        self.tax_units = None
        
        # Ensure required columns exist
        self._validate_inputs()
    
    def _validate_inputs(self) -> None:
        """Validate that required columns exist in the input DataFrames."""
        required_person_cols = ['SERIALNO', 'SPORDER', 'AGEP', 'SEX', 'MAR', 'RELSHIPP', 'PINCP']
        required_hh_cols = ['SERIALNO', 'HINCP', 'NP']
        
        missing_person = [col for col in required_person_cols if col not in self.person_df.columns]
        missing_hh = [col for col in required_hh_cols if col not in self.hh_df.columns]
        
        if missing_person:
            raise ValueError(f"Missing required columns in person data: {missing_person}")
        if missing_hh:
            raise ValueError(f"Missing required columns in household data: {missing_hh}")
    
    def create_rule_based_units(self) -> pd.DataFrame:
        """
        Create tax units using rule-based approach based on IRS filing rules.
        
        Returns:
            DataFrame with one row per tax unit and relevant attributes
        """
        logger.info("Creating tax units using rule-based approach...")
        
        # Make a copy of the person data to avoid modifying the original
        persons = self.person_df.copy()
        
        # Add a unique person ID if it doesn't exist
        if 'person_id' not in persons.columns:
            persons['person_id'] = persons['SERIALNO'] + '_' + persons['SPORDER'].astype(str)
        
        # Initialize list to store tax units
        tax_units = []
        
        # Process each household
        for hh_id, hh in self.hh_df.iterrows():
            hh_id = hh['SERIALNO']
            hh_members = persons[persons['SERIALNO'] == hh_id].copy()
            
            # Identify potential filers (adults 18+)
            adults = hh_members[hh_members['AGEP'] >= 18].copy()
            
            # Skip households with no adults (should be rare in PUMS)
            if len(adults) == 0:
                logger.warning(f"Household {hh_id} has no adults, skipping")
                continue
                
            # Sort adults by relationship to householder (RELSHIPP)
            # 0 = Reference person, 1 = Spouse, etc.
            adults = adults.sort_values('RELSHIPP')
            
            # Create tax units based on household composition
            if len(adults) == 1:
                # Single filer
                tax_unit = self._create_single_filer(adults.iloc[0], hh_members, hh)
                tax_units.append(tax_unit)
            else:
                # For multiple adults, try to identify married couples
                tax_units.extend(self._identify_joint_filers(adults, hh_members, hh))
        
        # Convert to DataFrame
        self.tax_units = pd.DataFrame(tax_units)
        logger.info(f"Created {len(self.tax_units)} tax units")
        
        return self.tax_units
    
    def _get_filing_status(self, adult: pd.Series, has_qualifying_child: bool, 
                          is_married: bool = False) -> int:
        """Determine the correct filing status for the tax unit."""
        age = adult.get('AGEP', 0)
        
        if is_married:
            return FILING_STATUS['JOINT']
        elif has_qualifying_child:
            # Check if qualifies for Head of Household
            # Must be unmarried or considered unmarried on the last day of the year
            # Must have paid more than half the cost of keeping up a home
            # Must have a qualifying person living with them for more than half the year
            return FILING_STATUS['HEAD_HOUSEHOLD']
        else:
            return FILING_STATUS['SINGLE']
    
    def _create_single_filer(self, adult: pd.Series, hh_members: pd.DataFrame, 
                           hh_data: pd.Series) -> Dict[str, Any]:
        """Create a tax unit for a single filer with detailed dependency tests."""
        # Initialize lists to track dependents
        qualifying_children = []
        other_dependents = []
        
        # Check each household member for dependency status
        for _, member in hh_members.iterrows():
            if member['person_id'] == adult['person_id']:
                continue  # Skip the adult themselves
                
            # Check if this is a qualifying child
            if self._is_qualifying_child(member, adult, hh_members):
                qualifying_children.append(member)
            # Check if this is a qualifying relative (only if not a qualifying child)
            elif self._is_qualifying_relative(member, adult, hh_members):
                other_dependents.append(member)
        
        # Determine filing status
        has_qualifying_child = len(qualifying_children) > 0
        filing_status = self._get_filing_status(adult, has_qualifying_child)
        
        # Calculate total number of dependents (children + other dependents)
        total_dependents = len(qualifying_children) + len(other_dependents)
        
        return {
            'taxunit_id': f"{hh_data['SERIALNO']}_1",
            'SERIALNO': hh_data['SERIALNO'],
            'filing_status': filing_status,  # Use the determined filing status
            'adult1_id': adult['person_id'],
            'adult2_id': None,
            'num_dependents': total_dependents,
            'num_children': len(qualifying_children),
            'num_other_dependents': len(other_dependents),
            'hh_income': hh_data['HINCP'],
            'wages': adult.get('WAGP', 0) or 0,
            'self_employment_income': adult.get('SEMP', 0) or 0,
            'interest_income': adult.get('INTP', 0) or 0,
            'ss_income': (adult.get('SSP', 0) or 0) + (adult.get('SSIP', 0) or 0),
            'other_income': (adult.get('OIP', 0) or 0) + (adult.get('PAP', 0) or 0) + (adult.get('RETP', 0) or 0),
            'total_income': adult.get('PINCP', 0) or 0
        }
    
    def _identify_joint_filers(self, adults: pd.DataFrame, hh_members: pd.DataFrame, 
                             hh_data: pd.Series) -> List[Dict[str, Any]]:
        """Identify potential joint filers within a household."""
        tax_units = []
        processed = set()
        
        # First, identify potential married couples
        for i, adult1 in adults.iterrows():
            if i in processed:
                continue
                
            # Look for a spouse
            spouse = None
            for j, adult2 in adults.loc[i+1:].iterrows():
                if j in processed:
                    continue
                    
                # Check if these adults appear to be spouses
                if self._is_potential_spouse(adult1, adult2):
                    spouse = adult2
                    processed.add(j)
                    break
            
            if spouse is not None:
                # Create joint return
                tax_unit = self._create_joint_filer(adult1, spouse, hh_members, hh_data)
                tax_units.append(tax_unit)
                processed.add(i)
            else:
                # No spouse found, create single return
                tax_unit = self._create_single_filer(adult1, hh_members, hh_data)
                tax_units.append(tax_unit)
                processed.add(i)
        
        return tax_units
    
    def _is_qualifying_child(self, child: pd.Series, potential_parent: pd.Series, 
                           hh_members: pd.DataFrame) -> bool:
        """
        Determine if a person qualifies as a child for tax purposes (EITC/CTC).
        
        IRS tests for qualifying child:
        1. Relationship test
        2. Age test
        3. Residency test (assumed met if in same household)
        4. Support test (child cannot provide > 50% of own support)
        5. Joint return test (child cannot file joint return)
        """
        # Relationship test - must be child, stepchild, adopted child, foster child,
        # sibling, half-sibling, stepsibling, or descendant of any of these
        valid_relationships = ['25', '26', '27', '30', '34']  # Direct children and grandchildren
        
        # Add Hawaii-specific extended family relationships
        valid_relationships.extend(['39', '40', '41', '42'])  # Grandparents, aunts/uncles, nieces/nephews, cousins
        
        # For siblings who may be dependents (e.g., younger siblings of householder)
        if child.get('RELSHIPP') == '28':  # Brother or sister
            # Check if significantly younger and could be claimed
            age_diff = potential_parent.get('AGEP', 0) - child.get('AGEP', 0)
            if age_diff < 10:  # Arbitrary threshold for sibling dependents
                return False
            valid_relationships.append('28')
        
        if str(child.get('RELSHIPP')) not in valid_relationships:
            return False
        
        # Age test
        age = child.get('AGEP', 0)
        
        # Student status
        schl = child.get('SCHL', 0)
        is_student = 16 <= schl <= 20  # Undergrad or grad student
        
        # Disability status
        is_disabled = child.get('DIS', 0) == 1
        
        # Age requirements
        if age < 19:
            age_qualified = True
        elif age < 24 and is_student:
            age_qualified = True
        elif is_disabled:  # No age limit for disabled
            age_qualified = True
        else:
            age_qualified = False
        
        if not age_qualified:
            return False
        
        # Joint return test - check if child is married
        if child.get('MAR') == 1:
            return False
        
        # Support test - use income as proxy
        child_income = 0
        for inc_type in ['WAGP', 'SEMP', 'INTP', 'RETP', 'SSP', 'SSIP', 'OIP', 'PAP']:
            if inc_type in child:
                child_income += abs(float(child.get(inc_type, 0) or 0))
        
        if child_income > HAWAII_INCOME_THRESHOLDS['support']:
            return False
        
        return True
    
    def _is_qualifying_relative(self, person: pd.Series, potential_claimer: pd.Series, 
                              hh_members: pd.DataFrame) -> bool:
        """Determine if a person qualifies as a dependent relative."""
        # Relationship test - many relationships qualify
        qualifying_relationships = [
            '25', '26', '27',  # Children (who don't meet qualifying child)
            '28',  # Siblings
            '29',  # Parents
            '30',  # Grandchildren
            '31',  # In-laws
            '32',  # Son/daughter-in-law
            '33',  # Other relatives
            '39', '40', '41', '42'  # Hawaii extended family
        ]
        
        # Member of household test
        if str(person.get('RELSHIPP')) not in qualifying_relationships:
            # Could still qualify if lived with taxpayer all year
            # Exclude roommates/housemates
            if str(person.get('RELSHIPP')) in ['35', '36', '37']:
                return False
        
        # Gross income test - use Hawaii threshold
        person_income = 0
        for inc_type in ['WAGP', 'SEMP', 'INTP', 'RETP', 'OIP', 'PAP']:
            if inc_type in person:
                person_income += abs(float(person.get(inc_type, 0) or 0))
        
        if person_income > HAWAII_INCOME_THRESHOLDS['relative_income']:
            return False
        
        # Support test - use income as proxy
        if person_income > HAWAII_INCOME_THRESHOLDS['dependent']:
            return False
        
        return True
    
    def _is_potential_spouse(self, person1: pd.Series, person2: pd.Series) -> bool:
        """Determine if two adults are likely spouses or partners."""
        # Check if they are in a spousal or partner relationship
        rel1 = str(person1.get('RELSHIPP', ''))
        rel2 = str(person2.get('RELSHIPP', ''))
        
        # Spouse/partner relationships (including same-sex)
        is_spouse_pair = (
            (rel1 == '21' and rel2 == '20') or  # Opposite-sex spouse
            (rel1 == '20' and rel2 == '21') or
            (rel1 == '23' and rel2 == '20') or  # Same-sex spouse
            (rel1 == '20' and rel2 == '23') or
            (rel1 == '22' and rel2 == '20') or  # Unmarried partner
            (rel1 == '20' and rel2 == '22') or
            (rel1 == '24' and rel2 == '20') or  # Same-sex unmarried partner
            (rel1 == '20' and rel2 == '24')
        )
        
        # Additional check for married status
        is_married = (person1.get('MAR') == 1 and person2.get('MAR') == 1)
        
        # Age difference check (spouses/partners are typically close in age)
        age_diff = abs(person1.get('AGEP', 0) - person2.get('AGEP', 0))
        reasonable_age_diff = age_diff <= 25  # More lenient for Hawaii's diverse families
        
        return (is_spouse_pair or is_married) and reasonable_age_diff
    
    def _create_joint_filer(self, adult1: pd.Series, adult2: pd.Series, 
                          hh_members: pd.DataFrame, hh_data: pd.Series) -> Dict[str, Any]:
        """Create a tax unit for a joint filer."""
        # Count qualifying children (under 17)
        children = hh_members[(hh_members['AGEP'] < 17) & 
                            (hh_members['AGEP'] > 0) &  # Exclude infants if needed
                            (hh_members['RELSHIPP'].isin([2, 3, 4, 5, 6]))]  # Own child/stepchild/adopted child
        
        # Count other dependents (17+ or other relatives)
        other_deps = hh_members[
            ((hh_members['AGEP'] >= 17) & (hh_members['AGEP'] < 19) |  # 17-18 year olds
             (hh_members['AGEP'] >= 19) & (hh_members['AGEP'] < 24) &  # 19-23 year old students
             hh_members['SCHL'].between(15, 18))  # Enrolled in school
        ]
        
        # Determine primary and secondary filer (typically higher earner first)
        if adult1.get('PINCP', 0) >= adult2.get('PINCP', 0):
            primary, secondary = adult1, adult2
        else:
            primary, secondary = adult2, adult1
        
        return {
            'taxunit_id': f"{hh_data['SERIALNO']}_2",
            'SERIALNO': hh_data['SERIALNO'],
            'filing_status': 2,  # Married filing jointly
            'adult1_id': primary['person_id'],
            'adult2_id': secondary['person_id'],
            'num_dependents': len(children) + len(other_deps),
            'num_children': len(children),
            'num_other_dependents': len(other_deps),
            'hh_income': hh_data['HINCP'],
            'wages': primary.get('WAGP', 0) + secondary.get('WAGP', 0),
            'self_employment_income': primary.get('SEMP', 0) + secondary.get('SEMP', 0),
            'interest_income': primary.get('INTP', 0) + secondary.get('INTP', 0),
            'ss_income': (primary.get('SSP', 0) + primary.get('SSIP', 0) +
                         secondary.get('SSP', 0) + secondary.get('SSIP', 0)),
            'other_income': (primary.get('OIP', 0) + primary.get('PAP', 0) + primary.get('RETP', 0) +
                           secondary.get('OIP', 0) + secondary.get('PAP', 0) + secondary.get('RETP', 0)),
            'total_income': primary.get('PINCP', 0) + secondary.get('PINCP', 0)
        }

def load_processed_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load processed PUMS data."""
    # Look for data in the project root's data/processed directory
    data_dir = Path('/Users/dtomkatsu/CascadeProjects/ctc-and-eitc/data/processed')
    person_file = data_dir / 'pums_person_processed.parquet'
    hh_file = data_dir / 'pums_household_processed.parquet'
    
    logger.info(f"Looking for person data at: {person_file}")
    logger.info(f"Looking for household data at: {hh_file}")
    
    person_df = pd.read_parquet(person_file)
    hh_df = pd.read_parquet(hh_file)
    
    return person_df, hh_df

def main():
    """Main function to demonstrate tax unit creation."""
    try:
        # Load processed data
        logger.info("Loading processed PUMS data...")
        person_df, hh_df = load_processed_data()
        
        # Create tax units
        constructor = TaxUnitConstructor(person_df, hh_df)
        tax_units = constructor.create_rule_based_units()
        
        # Save results
        output_file = Path(__file__).parent.parent / 'data/processed/tax_units_rule_based.parquet'
        tax_units.to_parquet(output_file, index=False)
        logger.info(f"Saved {len(tax_units)} tax units to {output_file}")
        
        # Print summary
        print("\nTax Unit Summary:")
        print(f"Total tax units: {len(tax_units)}")
        print(f"Single filers: {len(tax_units[tax_units['filing_status'] == 1])}")
        print(f"Joint filers: {len(tax_units[tax_units['filing_status'] == 2])}")
        print(f"Total dependents: {tax_units['num_dependents'].sum()}")
        
    except Exception as e:
        logger.error(f"Error creating tax units: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
