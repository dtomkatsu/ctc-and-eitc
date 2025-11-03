#!/usr/bin/env python3
"""
Organize project files and clean up root directory.

This script:
1. Moves documentation files to docs/
2. Archives old log files
3. Removes duplicate/obsolete files
4. Creates a cleaner project structure
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def organize_files(dry_run=True):
    """Organize project files into appropriate directories."""
    
    project_root = Path(__file__).parent.parent
    
    # Create necessary directories
    dirs_to_create = [
        'docs/archive',
        'docs/analysis',
        'docs/calibration',
        'docs/projections',
        'logs/archive',
        'archive/old_docs',
        'archive/old_scripts'
    ]
    
    for dir_path in dirs_to_create:
        full_path = project_root / dir_path
        if not full_path.exists():
            if not dry_run:
                full_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"{'Would create' if dry_run else 'Created'}: {dir_path}")
    
    # Move documentation files to docs/
    doc_files_to_move = {
        # Analysis docs
        'REVENUE_GAP_ANALYSIS.md': 'docs/analysis/',
        'REVENUE_OVERESTIMATION_ANALYSIS.md': 'docs/analysis/',
        'REVENUE_ANALYSIS_SUMMARY.md': 'docs/analysis/',
        'COVERAGE_GAP_ANALYSIS.md': 'docs/analysis/',
        'COVERAGE_GAP_ASSESSMENT.md': 'docs/analysis/',
        'COVERAGE_GAP_REASSESSMENT_DOTAX.md': 'docs/analysis/',
        'SOI_ALIGNMENT_ASSESSMENT.md': 'docs/analysis/',
        'SOI_INCOME_BRACKET_VALIDATION.md': 'docs/analysis/',
        'TAX_LIABILITY_VALIDATION_FINDINGS.md': 'docs/analysis/',
        
        # Calibration docs
        'CALIBRATION_COMPLETE_SUMMARY.md': 'docs/calibration/',
        'CALIBRATION_OPTIONS.md': 'docs/calibration/',
        'CALIBRATION_RESULTS.md': 'docs/calibration/',
        'FILING_STATUS_CALIBRATION_PLAN.md': 'docs/calibration/',
        'FILING_STATUS_CALIBRATION_PROPOSAL.md': 'docs/calibration/',
        'FINAL_CALIBRATION_SUMMARY.md': 'docs/calibration/',
        'SOI_CALIBRATION_IMPLEMENTATION.md': 'docs/calibration/',
        'MODEL_CALIBRATION_REPORT.md': 'docs/calibration/',
        
        # Implementation docs
        'IMPLEMENTATION_SUMMARY.md': 'docs/',
        'AGI_IMPLEMENTATION_SUMMARY.md': 'docs/',
        'CAPITAL_GAINS_IMPLEMENTATION_SUMMARY.md': 'docs/',
        'CAPITAL_GAINS_FLOW.md': 'docs/',
        'CAPITAL_GAINS_METHODOLOGY.md': 'docs/',
        'CAPITAL_GAINS_MODELING.md': 'docs/',
        'CAPITAL_GAINS_VALIDATION_SUMMARY.md': 'docs/',
        
        # Projection docs
        'PROJECTION_2026_RESULTS.md': 'docs/projections/',
        'REVISED_2026_PROJECTION_WITH_ITEMIZED_DEDUCTIONS.md': 'docs/projections/',
        'ENSEMBLE_GROWTH_PROJECTION_PLAN.md': 'docs/projections/',
        'POPULATION_GROWTH_IMPLEMENTATION.md': 'docs/projections/',
        'AGE_SPECIFIC_POPULATION_GROWTH.md': 'docs/projections/',
        
        # Stage implementation docs
        'STAGE3_IMPLEMENTATION_SUMMARY.md': 'docs/calibration/',
        'STAGE4_IMPLEMENTATION_SUMMARY.md': 'docs/calibration/',
        'STAGE5_IMPLEMENTATION_SUMMARY.md': 'docs/calibration/',
        
        # Other important docs
        'DEDUCTIONS_MODULE_DESIGN.md': 'docs/',
        'DEDUCTIONS_MODULE_SUMMARY.md': 'docs/',
        'TAX_BRACKET_IMPLEMENTATION_SUMMARY.md': 'docs/',
        'TAX_SCENARIO_CALCULATION_STANDARD.md': 'docs/',
        'HAWAII_TAX_POLICY_IMPACT_2017_VS_2024.md': 'docs/',
        
        # Summaries and fixes
        'HOH_FIX_SUMMARY.md': 'docs/archive/',
        'MFS_DIAGNOSIS_SUMMARY.md': 'docs/archive/',
        'HOUSEHOLD_CAP_FIX_RESULTS.md': 'docs/archive/',
        'OVERCOUNTING_FIX_SUMMARY.md': 'docs/archive/',
        'INVESTIGATION_COMPLETE_SUMMARY.md': 'docs/archive/',
        'TAX_UNITS_REGENERATION_SUMMARY.md': 'docs/archive/',
        'FINAL_REGENERATION_SUMMARY.md': 'docs/archive/',
        
        # Wage growth docs
        'WAGE_GROWTH_FINAL_SUMMARY.md': 'docs/projections/',
        'WAGE_GROWTH_PHASE1_SUMMARY.md': 'docs/projections/',
        'WAGE_GROWTH_BRACKET_SPECIFIC.md': 'docs/projections/',
        
        # Other analysis
        '50K_75K_INVESTIGATION_SUMMARY.md': 'docs/archive/',
        'INCOME_BRACKET_ISSUE_ANALYSIS.md': 'docs/archive/',
        'MIDDLE_INCOME_OVERESTIMATION_FINDINGS.md': 'docs/archive/',
        'ACS_TIMESERIES_VALIDATION_REPORT.md': 'docs/archive/',
        'PUMS_BLS_MAPPING_ASSESSMENT.md': 'docs/archive/',
        'ITEMIZED_DEDUCTION_MODELING_PLAN.md': 'docs/archive/',
        'IRS_SOI_INTEGRATION.md': 'docs/archive/',
        'FINAL_100_PERCENT_COVERAGE.md': 'docs/archive/',
        
        # Quick references
        'AGI_QUICK_REFERENCE.md': 'docs/',
        'QUICK_FIX_GUIDE.md': 'docs/',
        'DATA_SETUP_SUMMARY.md': 'docs/',
        'MIGRATION_REPORT.md': 'docs/',
    }
    
    for file_name, dest_dir in doc_files_to_move.items():
        src = project_root / file_name
        dest = project_root / dest_dir / file_name
        
        if src.exists():
            if not dry_run:
                # Create destination directory if needed
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dest))
            logger.info(f"{'Would move' if dry_run else 'Moved'}: {file_name} → {dest_dir}")
    
    # Move log files
    log_files = [
        'county_analysis.log',
        'coverage_gap_analysis.log',
        'diagnosis.log',
        'district_ctc_estimates.log',
        'download_pums.log',
        'fix_puma_mapping.log',
        'hybrid_weight_analysis.log',
        'married_singles_analysis.log',
        'process_pums.log',
        'soi_validation.log',
        'tax_unit_construction.log',
        'tax_unit_test.log',
        'tax_units.log',
        'validate_tax_units.log'
    ]
    
    # Move large regenerate logs to archive
    large_logs = [
        'regenerate_no_cap.log',
        'regenerate_output.log',
        'regenerate_priority1_fix.log',
        'regenerate_with_cap6.log',
    ]
    
    for log_file in log_files + large_logs:
        src = project_root / log_file
        if log_file in large_logs:
            dest = project_root / 'logs' / 'archive' / log_file
        else:
            dest = project_root / 'logs' / log_file
        
        if src.exists():
            if not dry_run:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dest))
            logger.info(f"{'Would move' if dry_run else 'Moved'}: {log_file} → logs/")
    
    # Move test scripts to archive
    test_scripts = [
        'analyze_hybrid_weights.py',
        'calculate_hybrid_weight_totals.py',
        'calculate_hybrid_weights_simple.py',
        'debug_joint_filers.py',
        'debug_spouse.py',
        'test_full_pipeline_hybrid.py',
        'test_hybrid_weighting.py',
        'test_joint_filer_fix.py'
    ]
    
    for script in test_scripts:
        src = project_root / script
        dest = project_root / 'archive' / 'old_scripts' / script
        
        if src.exists():
            if not dry_run:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dest))
            logger.info(f"{'Would move' if dry_run else 'Moved'}: {script} → archive/old_scripts/")
    
    # Move CSV files
    csv_files = [
        'households_without_adults_detailed.csv',
        'households_without_adults_sample.csv',
        'married_singles_analysis.csv',
        'multiprocessing_test_results.csv'
    ]
    
    for csv_file in csv_files:
        src = project_root / csv_file
        dest = project_root / 'archive' / csv_file
        
        if src.exists():
            if not dry_run:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dest))
            logger.info(f"{'Would move' if dry_run else 'Moved'}: {csv_file} → archive/")
    
    # Move other files
    other_files = [
        'pincp_distribution.png',
        'tract_puma_relationship.txt',
        'tract_puma_relationship_2024.txt'
    ]
    
    for other_file in other_files:
        src = project_root / other_file
        dest = project_root / 'archive' / other_file
        
        if src.exists():
            if not dry_run:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dest))
            logger.info(f"{'Would move' if dry_run else 'Moved'}: {other_file} → archive/")
    
    # Report summary
    logger.info("\n" + "="*50)
    if dry_run:
        logger.info("DRY RUN COMPLETE - No files were actually moved")
        logger.info("Run with --execute to actually move files")
    else:
        logger.info("FILE ORGANIZATION COMPLETE")
    
    # List remaining files in root
    remaining_files = []
    for item in project_root.iterdir():
        if item.is_file() and item.name not in ['.gitignore', 'README.md', 'requirements.txt', 
                                                'requirements-dev.txt', 'requirements-ml.txt', 
                                                'setup.py', '.coverage', '.env.example']:
            if not item.name.startswith('.'):
                remaining_files.append(item.name)
    
    if remaining_files:
        logger.info(f"\nRemaining files in root ({len(remaining_files)}):")
        for f in sorted(remaining_files):
            logger.info(f"  - {f}")
    else:
        logger.info("\nRoot directory is clean!")
    
    return not dry_run


def main():
    """Main execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Organize project files')
    parser.add_argument('--execute', action='store_true', 
                       help='Actually move files (default is dry run)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("PROJECT FILE ORGANIZATION")
    print("="*60)
    
    if args.execute:
        print("⚠️  EXECUTING FILE MOVES - This will reorganize your project")
        response = input("Are you sure? (yes/no): ")
        if response.lower() != 'yes':
            print("Aborted.")
            return
    else:
        print("🔍 DRY RUN MODE - No files will be moved")
    
    print()
    organize_files(dry_run=not args.execute)
    
    if not args.execute:
        print("\n✅ Dry run complete. Run with --execute to move files.")


if __name__ == "__main__":
    main()
