#!/usr/bin/env python3
"""
Extract key insights from IRS SOI ZIP code data for Hawaii and propose integration with tax unit construction.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_examine_soi_data():
    """Load the IRS SOI data and examine its structure."""
    # Load the raw Excel file with proper parsing
    data_path = Path('data/irs_soi/22zp12hi.xlsx')
    
    # Try different approaches to find the actual data
    logger.info("Examining Excel file structure...")
    
    # Read multiple rows to find headers
    for skip_rows in range(10):
        try:
            df = pd.read_excel(data_path, skiprows=skip_rows, nrows=20)
            logger.info(f"Skip {skip_rows} - First column: {df.columns[0]}")
            logger.info(f"First few values: {df.iloc[:3, 0].tolist()}")
            
            # Check if we found ZIP codes
            first_col_vals = df.iloc[:, 0].astype(str)
            if any(val.isdigit() and len(val) == 5 for val in first_col_vals):
                logger.info(f"Found ZIP codes at skip_rows={skip_rows}")
                # Load full dataset
                full_df = pd.read_excel(data_path, skiprows=skip_rows)
                return full_df, skip_rows
                
        except Exception as e:
            continue
    
    # If no ZIP codes found, try manual parsing
    logger.info("Manual parsing approach...")
    df_raw = pd.read_excel(data_path, header=None)
    
    # Look for the row with ZIP codes
    for i in range(min(15, len(df_raw))):
        row_vals = df_raw.iloc[i].astype(str).tolist()
        if any('zip' in str(val).lower() for val in row_vals):
            logger.info(f"Found ZIP header at row {i}: {row_vals[:5]}")
            # Use next row as data start
            df = pd.read_excel(data_path, skiprows=i+1)
            return df, i+1
    
    return None, None

def identify_key_columns(df):
    """Identify key columns in the IRS SOI data."""
    logger.info("Identifying key data columns...")
    
    # The first column should be ZIP codes
    zip_col = df.columns[0]
    logger.info(f"ZIP column: {zip_col}")
    
    # Look at the data structure - typically IRS SOI has:
    # - ZIP code
    # - Income brackets with return counts and amounts
    # - Filing status breakdowns
    
    # Sample some rows to understand structure
    sample_data = df.head(10)
    logger.info("Sample data structure:")
    for i, row in sample_data.iterrows():
        if pd.notna(row.iloc[0]) and str(row.iloc[0]).isdigit():
            logger.info(f"ZIP {row.iloc[0]}: {row.iloc[1:6].tolist()}")
    
    # Common IRS SOI column patterns
    column_mapping = {
        'zip_code': 0,  # First column
        'total_returns': None,
        'agi_amounts': [],
        'filing_status_cols': []
    }
    
    # Try to identify patterns in column names or positions
    for i, col in enumerate(df.columns[1:20], 1):  # Check first 20 columns
        col_str = str(col).lower()
        if 'return' in col_str or 'count' in col_str:
            if column_mapping['total_returns'] is None:
                column_mapping['total_returns'] = i
        elif 'agi' in col_str or 'income' in col_str:
            column_mapping['agi_amounts'].append(i)
    
    return column_mapping

def extract_filing_patterns(df, column_mapping):
    """Extract filing status patterns from the IRS data."""
    logger.info("Extracting filing status patterns...")
    
    # Clean ZIP codes
    df_clean = df.copy()
    df_clean['ZIP'] = df_clean.iloc[:, 0].astype(str).str.strip()
    
    # Filter to valid Hawaii ZIP codes (96xxx)
    hawaii_zips = df_clean[df_clean['ZIP'].str.match(r'^96\d{3}$')].copy()
    logger.info(f"Found {len(hawaii_zips)} Hawaii ZIP codes")
    
    if len(hawaii_zips) == 0:
        logger.warning("No Hawaii ZIP codes found")
        return None
    
    # Extract key statistics
    zip_stats = []
    
    for _, row in hawaii_zips.iterrows():
        zip_code = row['ZIP']
        
        # Try to extract return counts and income data
        # This is dataset-specific and may need adjustment
        row_data = {
            'zip_code': zip_code,
            'total_returns': None,
            'total_agi': None,
            'avg_agi': None
        }
        
        # Look for numeric columns that might be return counts
        numeric_cols = []
        for i in range(1, min(20, len(row))):
            try:
                val = pd.to_numeric(row.iloc[i], errors='coerce')
                if pd.notna(val) and val > 0:
                    numeric_cols.append((i, val))
            except:
                continue
        
        if numeric_cols:
            # Assume first large number is total returns
            row_data['total_returns'] = numeric_cols[0][1] if numeric_cols else None
            
        zip_stats.append(row_data)
    
    return pd.DataFrame(zip_stats)

def create_zip_to_puma_crosswalk():
    """Create a crosswalk from ZIP codes to PUMAs for Hawaii."""
    logger.info("Creating ZIP-to-PUMA crosswalk strategy...")
    
    # This would require external data sources
    crosswalk_strategy = {
        'data_sources_needed': [
            'HUD USPS ZIP-to-County crosswalk',
            'Census Bureau ZIP-to-tract relationships', 
            'Our existing tract-to-PUMA mappings'
        ],
        'approach': [
            '1. Download HUD crosswalk for ZIP-to-County mapping',
            '2. Use Census tract-ZIP relationships',
            '3. Join with our tract-to-PUMA crosswalk',
            '4. Handle many-to-many relationships with population weighting'
        ],
        'challenges': [
            'ZIP codes cross PUMA boundaries',
            'Need population weights for allocation',
            'Some ZIPs may not map cleanly to PUMAs'
        ]
    }
    
    return crosswalk_strategy

def propose_tax_unit_integration(zip_stats):
    """Propose how to integrate IRS SOI data with tax unit construction."""
    logger.info("Developing integration proposals...")
    
    proposals = [
        {
            'name': 'Geographic Filing Status Calibration',
            'description': 'Use ZIP-level filing status distributions to improve geographic accuracy',
            'implementation': [
                'Map ZIP codes to PUMAs using crosswalk',
                'Calculate filing status percentages by PUMA from IRS data', 
                'Replace current state-level SOI targets with PUMA-level targets',
                'Implement PUMA-specific weight raking for filing status'
            ],
            'benefits': [
                'More accurate geographic filing patterns',
                'Accounts for urban/rural differences in Hawaii',
                'Better representation of Honolulu vs neighbor islands'
            ],
            'data_requirements': [
                'ZIP-to-PUMA crosswalk',
                'Filing status counts by ZIP from IRS data',
                'Population weights for ZIP-PUMA allocation'
            ]
        },
        {
            'name': 'Income Distribution Validation',
            'description': 'Validate PUMS income against actual IRS income by geography',
            'implementation': [
                'Extract AGI distributions by ZIP from IRS data',
                'Aggregate to PUMA level using crosswalk',
                'Compare PUMS AGI totals to IRS AGI totals by PUMA',
                'Apply scaling factors if systematic differences found'
            ],
            'benefits': [
                'Corrects PUMS income underreporting',
                'More accurate CTC eligibility calculations',
                'Better income-based tax estimates'
            ],
            'data_requirements': [
                'AGI amounts by ZIP from IRS data',
                'ZIP-to-PUMA population weights',
                'Current PUMS income totals by PUMA'
            ]
        },
        {
            'name': 'CTC Benchmarking and Validation',
            'description': 'Use actual IRS CTC claims to validate our estimates',
            'implementation': [
                'Extract CTC amounts and recipient counts by ZIP',
                'Aggregate to PUMA/county level',
                'Compare with our CTC estimates',
                'Identify systematic biases in dependent counting'
            ],
            'benefits': [
                'Direct validation of CTC calculations',
                'Identifies errors in dependent assignment',
                'Provides confidence intervals for estimates'
            ],
            'data_requirements': [
                'CTC data by ZIP from IRS SOI',
                'Our current CTC estimates by geography',
                'Dependent counts by ZIP if available'
            ]
        },
        {
            'name': 'Enhanced Geographic Targeting',
            'description': 'Enable more precise geographic analysis using ZIP-level data',
            'implementation': [
                'Create comprehensive ZIP-PUMA-County mapping',
                'Develop ZIP-level CTC impact estimates',
                'Enable legislative district analysis via ZIP codes',
                'Create interactive mapping capabilities'
            ],
            'benefits': [
                'Higher geographic precision',
                'Better policy targeting',
                'More detailed impact analysis'
            ],
            'data_requirements': [
                'Complete ZIP-to-geography crosswalks',
                'ZIP-level population and household data',
                'Legislative district-ZIP relationships'
            ]
        }
    ]
    
    return proposals

def prioritize_implementation(proposals, zip_stats):
    """Prioritize implementation based on data availability and impact."""
    logger.info("Prioritizing implementation approaches...")
    
    # Score each proposal
    scored_proposals = []
    
    for proposal in proposals:
        score = {
            'name': proposal['name'],
            'feasibility_score': 0,  # 0-10
            'impact_score': 0,       # 0-10
            'data_availability': 0,  # 0-10
            'implementation_effort': 0  # 0-10 (lower is better)
        }
        
        # Score based on proposal characteristics
        if 'Filing Status' in proposal['name']:
            score['feasibility_score'] = 8
            score['impact_score'] = 9
            score['data_availability'] = 7
            score['implementation_effort'] = 6
        elif 'Income Distribution' in proposal['name']:
            score['feasibility_score'] = 7
            score['impact_score'] = 8
            score['data_availability'] = 8
            score['implementation_effort'] = 5
        elif 'CTC Benchmarking' in proposal['name']:
            score['feasibility_score'] = 9
            score['impact_score'] = 10
            score['data_availability'] = 6
            score['implementation_effort'] = 4
        elif 'Geographic Targeting' in proposal['name']:
            score['feasibility_score'] = 6
            score['impact_score'] = 7
            score['data_availability'] = 5
            score['implementation_effort'] = 8
        
        # Calculate overall priority score
        score['priority_score'] = (
            score['feasibility_score'] * 0.3 +
            score['impact_score'] * 0.4 +
            score['data_availability'] * 0.2 +
            (10 - score['implementation_effort']) * 0.1
        )
        
        scored_proposals.append(score)
    
    # Sort by priority score
    scored_proposals.sort(key=lambda x: x['priority_score'], reverse=True)
    
    return scored_proposals

def main():
    """Main analysis function."""
    logger.info("Starting IRS SOI data integration analysis...")
    
    # Load and examine the data
    df, skip_rows = load_and_examine_soi_data()
    if df is None:
        logger.error("Could not parse IRS SOI data")
        return
    
    logger.info(f"Loaded IRS data with {len(df)} rows and {len(df.columns)} columns")
    
    # Identify key columns
    column_mapping = identify_key_columns(df)
    
    # Extract filing patterns
    zip_stats = extract_filing_patterns(df, column_mapping)
    
    # Create crosswalk strategy
    crosswalk_strategy = create_zip_to_puma_crosswalk()
    
    # Develop integration proposals
    proposals = propose_tax_unit_integration(zip_stats)
    
    # Prioritize implementation
    prioritized = prioritize_implementation(proposals, zip_stats)
    
    # Output results
    logger.info("\n" + "="*60)
    logger.info("IRS SOI INTEGRATION ANALYSIS RESULTS")
    logger.info("="*60)
    
    if zip_stats is not None:
        logger.info(f"\nHawaii ZIP codes found: {len(zip_stats)}")
        logger.info("Sample ZIP statistics:")
        print(zip_stats.head().to_string())
    
    logger.info(f"\nCrosswalk Strategy:")
    for key, value in crosswalk_strategy.items():
        logger.info(f"  {key}:")
        if isinstance(value, list):
            for item in value:
                logger.info(f"    - {item}")
        else:
            logger.info(f"    {value}")
    
    logger.info(f"\nPrioritized Implementation Recommendations:")
    for i, proposal in enumerate(prioritized, 1):
        logger.info(f"\n{i}. {proposal['name']} (Priority Score: {proposal['priority_score']:.1f})")
        logger.info(f"   Feasibility: {proposal['feasibility_score']}/10")
        logger.info(f"   Impact: {proposal['impact_score']}/10")
        logger.info(f"   Data Available: {proposal['data_availability']}/10")
        logger.info(f"   Implementation Effort: {proposal['implementation_effort']}/10")
    
    # Save results
    results = {
        'zip_stats': zip_stats,
        'crosswalk_strategy': crosswalk_strategy,
        'proposals': proposals,
        'prioritized': prioritized
    }
    
    # Save ZIP stats if available
    if zip_stats is not None:
        output_path = Path('data/irs_soi/hawaii_zip_statistics.csv')
        zip_stats.to_csv(output_path, index=False)
        logger.info(f"\nZIP statistics saved to: {output_path}")
    
    return results

if __name__ == "__main__":
    results = main()
