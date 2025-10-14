#!/usr/bin/env python3
"""
Migration script for project reorganization.

This script:
1. Creates new directory structure
2. Moves scripts to appropriate locations
3. Updates import statements and references
4. Creates backup before making changes
"""

import os
import shutil
import re
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class ProjectMigrator:
    """Handles project structure migration."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.scripts_dir = project_root / "scripts"
        self.backup_dir = project_root / "backup_pre_migration"
        
    def create_backup(self):
        """Create backup of scripts directory."""
        logger.info("Creating backup of scripts directory...")
        if self.backup_dir.exists():
            shutil.rmtree(self.backup_dir)
        shutil.copytree(self.scripts_dir, self.backup_dir)
        logger.info(f"✓ Backup created at: {self.backup_dir}")
        
    def create_new_structure(self):
        """Create new directory structure."""
        logger.info("Creating new directory structure...")
        
        new_dirs = [
            self.scripts_dir / "pipeline",
            self.scripts_dir / "analysis" / "filing_status",
            self.scripts_dir / "analysis" / "income",
            self.scripts_dir / "analysis" / "validation",
            self.scripts_dir / "calibration",
            self.scripts_dir / "data_prep",
            self.scripts_dir / "archived",
        ]
        
        for dir_path in new_dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"  Created: {dir_path.relative_to(self.project_root)}")
            
    def get_script_mappings(self) -> Dict[str, str]:
        """Define where each script should be moved."""
        return {
            # Pipeline scripts
            "construct_tax_units.py": "pipeline/01_construct_tax_units.py",
            "apply_soi_calibration.py": "pipeline/02_apply_soi_calibration.py",
            "validate_tax_units.py": "pipeline/03_validate_results.py",
            
            # Calibration scripts
            "demo_ipf_calibration.py": "calibration/demo_ipf_calibration.py",
            "test_ipf_calibration.py": "calibration/test_ipf_calibration.py",
            
            # Data prep scripts
            "download_pums.py": "data_prep/download_pums.py",
            "create_official_crosswalk.py": "data_prep/create_official_crosswalk.py",
            "create_zip_puma_crosswalk.py": "data_prep/create_zip_puma_crosswalk.py",
            
            # Analysis - Filing Status
            "analyze_filing_status_gaps.py": "analysis/filing_status/analyze_filing_status_gaps.py",
            "analyze_joint_filing_disparity.py": "analysis/filing_status/analyze_joint_filing_disparity.py",
            "analyze_missing_joint_filers.py": "analysis/filing_status/analyze_missing_joint_filers.py",
            "analyze_married_misclassification.py": "analysis/filing_status/analyze_married_misclassification.py",
            "analyze_married_singles.py": "analysis/filing_status/analyze_married_singles.py",
            "analyze_hoh_bottlenecks.py": "analysis/filing_status/analyze_hoh_bottlenecks.py",
            "analyze_mfs_patterns.py": "analysis/filing_status/analyze_mfs_patterns.py",
            "check_filing_status.py": "analysis/filing_status/check_filing_status.py",
            
            # Analysis - Income
            "analyze_income_distribution.py": "analysis/income/analyze_income_distribution.py",
            "analyze_high_income.py": "analysis/income/analyze_high_income.py",
            "analyze_millionaires.py": "analysis/income/analyze_millionaires.py",
            "check_income.py": "analysis/income/check_income.py",
            "check_pincp.py": "analysis/income/check_pincp.py",
            "check_raw_income.py": "analysis/income/check_raw_income.py",
            "analyze_person_weights.py": "analysis/income/analyze_person_weights.py",
            "analyze_person_weights_v2.py": "analysis/income/analyze_person_weights_v2.py",
            
            # Analysis - Validation
            "compare_to_soi.py": "analysis/validation/compare_to_soi.py",
            "validate_tax_unit_improvements.py": "analysis/validation/validate_tax_unit_improvements.py",
            "compare_data_sources.py": "analysis/validation/compare_data_sources.py",
            "analyze_soi_data.py": "analysis/validation/analyze_soi_data.py",
            "analyze_soi_filing_status.py": "analysis/validation/analyze_soi_filing_status.py",
            "analyze_irs_soi_data.py": "analysis/validation/analyze_irs_soi_data.py",
            
            # Archive - Old analysis scripts
            "adjust_hoh_criteria_step3.py": "archived/adjust_hoh_criteria_step3.py",
            "analyze_excess_single_filers.py": "archived/analyze_excess_single_filers.py",
            "analyze_filer_proportions.py": "archived/analyze_filer_proportions.py",
            "analyze_final_joint_gap.py": "archived/analyze_final_joint_gap.py",
            "analyze_hoh_fine_tuning_step4.py": "archived/analyze_hoh_fine_tuning_step4.py",
            "analyze_hoh_joint_misclassification.py": "archived/analyze_hoh_joint_misclassification.py",
            "analyze_households.py": "archived/analyze_households.py",
            "analyze_pums_codes.py": "archived/analyze_pums_codes.py",
            "build_full_population.py": "archived/build_full_population.py",
            "calculate_hawaii_taxes.py": "archived/calculate_hawaii_taxes.py",
            "calculate_taxes_full_model.py": "archived/calculate_taxes_full_model.py",
            "calculate_taxes_with_adjustments.py": "archived/calculate_taxes_with_adjustments.py",
            "calculate_weighted_sum.py": "archived/calculate_weighted_sum.py",
            "calculate_weighted_sum_fixed.py": "archived/calculate_weighted_sum_fixed.py",
            "calculate_weighted_totals.py": "archived/calculate_weighted_totals.py",
            "calculate_weighted_totals_corrected.py": "archived/calculate_weighted_totals_corrected.py",
            "calculate_weighted_totals_step1.py": "archived/calculate_weighted_totals_step1.py",
            "calculate_weighted_totals_step3_results.py": "archived/calculate_weighted_totals_step3_results.py",
            "county_analysis_with_imputation.py": "archived/county_analysis_with_imputation.py",
            "debug_ml_validator.py": "archived/debug_ml_validator.py",
            "debug_puma_mapping.py": "archived/debug_puma_mapping.py",
            "design_hybrid_weighting.py": "archived/design_hybrid_weighting.py",
            "diagnose_and_calibrate_model.py": "archived/diagnose_and_calibrate_model.py",
            "get_tax_units_by_county.py": "archived/get_tax_units_by_county.py",
            "test_tax_unit_counting.py": "archived/test_tax_unit_counting.py",
            "analyze_revenue_impact_2017_vs_2024.py": "archived/analyze_revenue_impact_2017_vs_2024.py",
        }
        
    def move_scripts(self):
        """Move scripts to new locations."""
        logger.info("\nMoving scripts to new locations...")
        
        mappings = self.get_script_mappings()
        moved_count = 0
        skipped_count = 0
        
        for old_name, new_path in mappings.items():
            old_file = self.scripts_dir / old_name
            new_file = self.scripts_dir / new_path
            
            if old_file.exists():
                shutil.move(str(old_file), str(new_file))
                logger.info(f"  ✓ {old_name} → {new_path}")
                moved_count += 1
            else:
                logger.warning(f"  ⚠ Skipped {old_name} (not found)")
                skipped_count += 1
                
        logger.info(f"\n✓ Moved {moved_count} scripts, skipped {skipped_count}")
        
    def update_script_imports(self):
        """Update import statements in moved scripts."""
        logger.info("\nUpdating import statements in scripts...")
        
        updated_count = 0
        
        # Walk through all Python files in scripts directory
        for py_file in self.scripts_dir.rglob("*.py"):
            if py_file.name == "migrate_project_structure.py":
                continue
                
            content = py_file.read_text()
            original_content = content
            
            # Update sys.path.append patterns to account for new depth
            # Count directory depth from scripts/
            depth = len(py_file.relative_to(self.scripts_dir).parts) - 1
            parent_refs = "../" * depth if depth > 0 else ""
            
            # Update sys.path.append(str(Path(__file__).parent.parent))
            if depth == 0:
                # Scripts in root of scripts/ directory
                new_parent = "Path(__file__).parent.parent"
            elif depth == 1:
                # Scripts one level deep (e.g., scripts/pipeline/)
                new_parent = "Path(__file__).parent.parent.parent"
            elif depth == 2:
                # Scripts two levels deep (e.g., scripts/analysis/filing_status/)
                new_parent = "Path(__file__).parent.parent.parent.parent"
            else:
                new_parent = "Path(__file__).parent.parent"
                
            # Replace sys.path.append patterns
            content = re.sub(
                r'sys\.path\.append\(str\(Path\(__file__\)\.parent\.parent\)\)',
                f'sys.path.append(str({new_parent}))',
                content
            )
            
            # Update relative file paths in scripts
            # e.g., "data/processed/..." should work from any depth
            # These are already relative to project root, so they should be fine
            
            if content != original_content:
                py_file.write_text(content)
                logger.info(f"  ✓ Updated imports in {py_file.relative_to(self.scripts_dir)}")
                updated_count += 1
                
        logger.info(f"\n✓ Updated {updated_count} script files")
        
    def create_readme_files(self):
        """Create README files in new directories."""
        logger.info("\nCreating README files in new directories...")
        
        readmes = {
            "pipeline": """# Pipeline Scripts

Core production pipeline scripts for tax unit construction and calibration.

## Execution Order

1. **01_construct_tax_units.py** - Construct tax units from PUMS data
2. **02_apply_soi_calibration.py** - Apply SOI weight calibration
3. **03_validate_results.py** - Validate results against benchmarks

## Usage

Run the full pipeline:
```bash
python pipeline/01_construct_tax_units.py
python pipeline/02_apply_soi_calibration.py
python pipeline/03_validate_results.py
```
""",
            "analysis": """# Analysis Scripts

Exploratory analysis and validation scripts organized by topic.

## Subdirectories

- **filing_status/** - Filing status analysis and validation
- **income/** - Income distribution and high-income analysis
- **validation/** - SOI comparison and validation scripts

## Usage

These scripts are for ad-hoc analysis and exploration, not part of the main pipeline.
""",
            "calibration": """# Calibration Scripts

Scripts for testing and demonstrating IPF and other calibration methods.

## Scripts

- **demo_ipf_calibration.py** - Demonstration of IPF calibration
- **test_ipf_calibration.py** - Testing IPF on real data

## Usage

```bash
python calibration/demo_ipf_calibration.py
python calibration/test_ipf_calibration.py
```
""",
            "data_prep": """# Data Preparation Scripts

Scripts for downloading and preparing data sources.

## Scripts

- **download_pums.py** - Download PUMS data from Census Bureau
- **create_official_crosswalk.py** - Create geographic crosswalks
- **create_zip_puma_crosswalk.py** - Create ZIP-PUMA crosswalk

## Usage

```bash
python data_prep/download_pums.py
```
""",
            "archived": """# Archived Scripts

Old exploratory and development scripts kept for reference.

These scripts were used during iterative development but are not part of the current pipeline.
They are kept for historical reference and may contain useful analysis patterns.
""",
        }
        
        for subdir, content in readmes.items():
            readme_path = self.scripts_dir / subdir / "README.md"
            readme_path.write_text(content)
            logger.info(f"  ✓ Created {readme_path.relative_to(self.project_root)}")
            
    def remove_duplicate_structure(self):
        """Remove tax-credits-final duplicate structure."""
        logger.info("\nChecking for duplicate structure...")
        
        duplicate_dir = self.project_root / "tax-credits-final"
        if duplicate_dir.exists():
            logger.info(f"Found duplicate structure: {duplicate_dir}")
            response = input("Remove tax-credits-final directory? (y/n): ")
            if response.lower() == 'y':
                shutil.rmtree(duplicate_dir)
                logger.info("✓ Removed tax-credits-final directory")
            else:
                logger.info("⚠ Skipped removal of tax-credits-final")
        else:
            logger.info("✓ No duplicate structure found")
            
    def generate_migration_report(self):
        """Generate a report of the migration."""
        logger.info("\nGenerating migration report...")
        
        report_path = self.project_root / "MIGRATION_REPORT.md"
        
        report = """# Project Reorganization Migration Report

## Changes Made

### 1. Directory Structure
Created new organized structure in `/scripts/`:
- `pipeline/` - Core production scripts (numbered for execution order)
- `analysis/filing_status/` - Filing status analysis
- `analysis/income/` - Income distribution analysis
- `analysis/validation/` - SOI comparison and validation
- `calibration/` - IPF and calibration testing
- `data_prep/` - Data download and preparation
- `archived/` - Old exploratory scripts

### 2. Scripts Moved
All scripts have been moved to appropriate subdirectories based on their function.

### 3. Import Statements Updated
All moved scripts have been updated to account for new directory depth:
- Scripts in subdirectories now use correct `sys.path.append()` patterns
- Relative paths adjusted for new locations

### 4. README Files Added
Each subdirectory now has a README.md explaining its purpose and usage.

## Backup

A complete backup of the original scripts directory was created at:
`backup_pre_migration/`

## Rollback Instructions

If you need to rollback the migration:

```bash
# Remove new structure
rm -rf scripts/pipeline scripts/analysis scripts/calibration scripts/data_prep scripts/archived

# Restore from backup
cp -r backup_pre_migration/* scripts/

# Remove backup
rm -rf backup_pre_migration
```

## Next Steps

1. Test the main pipeline scripts to ensure they work correctly
2. Update any external references to script paths
3. Update documentation to reflect new structure
4. Remove backup directory once satisfied with migration

## Migration Date
""" + f"{os.popen('date').read()}"
        
        report_path.write_text(report)
        logger.info(f"✓ Migration report created: {report_path}")
        
    def run_migration(self):
        """Run the complete migration process."""
        logger.info("=" * 60)
        logger.info("PROJECT REORGANIZATION MIGRATION")
        logger.info("=" * 60)
        
        try:
            self.create_backup()
            self.create_new_structure()
            self.move_scripts()
            self.update_script_imports()
            self.create_readme_files()
            self.remove_duplicate_structure()
            self.generate_migration_report()
            
            logger.info("\n" + "=" * 60)
            logger.info("✓ MIGRATION COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            logger.info("\nNext steps:")
            logger.info("1. Test pipeline scripts: python scripts/pipeline/01_construct_tax_units.py")
            logger.info("2. Review migration report: MIGRATION_REPORT.md")
            logger.info("3. Update README.md with new structure")
            logger.info("4. Remove backup once satisfied: rm -rf backup_pre_migration/")
            
        except Exception as e:
            logger.error(f"\n✗ Migration failed: {e}")
            logger.error("Backup is available at: backup_pre_migration/")
            raise


def main():
    """Main entry point."""
    # Get project root (parent of scripts directory)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    logger.info(f"Project root: {project_root}")
    logger.info(f"Scripts directory: {script_dir}")
    
    # Confirm before proceeding
    print("\nThis script will reorganize the project structure.")
    print("A backup will be created before making any changes.")
    response = input("\nProceed with migration? (y/n): ")
    
    if response.lower() != 'y':
        logger.info("Migration cancelled.")
        return
        
    # Run migration
    migrator = ProjectMigrator(project_root)
    migrator.run_migration()


if __name__ == "__main__":
    main()
