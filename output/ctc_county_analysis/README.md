# Hawaii Child Tax Credit District Analysis Summary

## Statewide Totals
- **Total CTC Amount**: $395,881,606
- **Total Tax Units**: 27,646
- **CTC Recipient Rate**: 16.4%
- **Average CTC per Tax Unit**: $623
- **Average CTC per Recipient**: $3,545
- **Total Qualifying Children**: 201,901

## House District Analysis
- **Number of Districts**: 9
- **Highest CTC District**: 1 ($120,882,977)
- **Lowest CTC District**: 4 ($20,221,694)

## County Analysis
- **Number of Counties**: 3
- **Highest CTC County**: Honolulu

### CTC by County:
- **Hawaii**: $54,923,703
- **Honolulu**: $274,998,629
- **Maui_Kalawao_Kauai**: $65,959,274

## Files Generated
- `hawaii_house_districts_ctc_summary.csv`: House district CTC estimates
- `hawaii_senate_districts_ctc_summary.csv`: Senate district CTC estimates  
- `hawaii_county_districts_ctc_summary.csv`: County CTC estimates
- `house/`, `senate/`, `county/`: Individual district profile files
- `hawaii_ctc_summary_report.json`: Machine-readable summary statistics

## Methodology
- Based on 2023 PUMS data for Hawaii
- Tax units constructed using rule-based approach with SOI calibration
- CTC calculated using 2023 tax law parameters
- Population estimates using PUMS household weights
- Geographic assignment via PUMA-to-district crosswalk

Generated on: 2025-09-03 09:58:16
