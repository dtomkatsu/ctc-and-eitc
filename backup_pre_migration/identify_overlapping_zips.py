import pandas as pd
from collections import defaultdict

def find_overlapping_zips(csv_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Group by ZIP code and collect unique districts
    zip_districts = df.groupby('zip_code').agg({
        'house_district': lambda x: sorted(x.unique().tolist()),
        'senate_district': lambda x: sorted(x.unique().tolist())
    }).reset_index()
    
    # Find ZIPs in multiple house districts
    multi_house = zip_districts[zip_districts['house_district'].apply(len) > 1]
    multi_senate = zip_districts[zip_districts['senate_district'].apply(len) > 1]
    
    # Count overlaps
    house_overlaps = multi_house[['zip_code', 'house_district']].to_dict('records')
    senate_overlaps = multi_senate[['zip_code', 'senate_district']].to_dict('records')
    
    # Count total overlapping ZIPs
    total_house_overlaps = len(multi_house)
    total_senate_overlaps = len(multi_senate)
    
    # Find ZIPs with the most overlaps
    if not multi_house.empty:
        multi_house['num_districts'] = multi_house['house_district'].apply(len)
        max_house_overlaps = multi_house.sort_values('num_districts', ascending=False).iloc[0]
    else:
        max_house_overlaps = None
        
    if not multi_senate.empty:
        multi_senate['num_districts'] = multi_senate['senate_district'].apply(len)
        max_senate_overlaps = multi_senate.sort_values('num_districts', ascending=False).iloc[0]
    else:
        max_senate_overlaps = None
    
    return {
        'house_overlaps': {
            'total': total_house_overlaps,
            'examples': house_overlaps[:5],  # Show first 5 examples
            'max_overlaps': max_house_overlaps.to_dict() if max_house_overlaps is not None else None
        },
        'senate_overlaps': {
            'total': total_senate_overlaps,
            'examples': senate_overlaps[:5],  # Show first 5 examples
            'max_overlaps': max_senate_overlaps.to_dict() if max_senate_overlaps is not None else None
        },
        'total_unique_zips': len(zip_districts)
    }

if __name__ == "__main__":
    csv_path = "data/crosswalks/hawaii_zip_puma_district_crosswalk.csv"
    results = find_overlapping_zips(csv_path)
    
    print("\n=== ZIP Code Overlap Analysis ===")
    print(f"Total unique ZIP codes: {results['total_unique_zips']}")
    
    print("\n--- House District Overlaps ---")
    print(f"ZIPs in multiple house districts: {results['house_overlaps']['total']}")
    if results['house_overlaps']['max_overlaps']:
        print(f"ZIP with most house districts: {results['house_overlaps']['max_overlaps']['zip_code']} "
              f"(in {results['house_overlaps']['max_overlaps']['num_districts']} districts)")
    print("Example overlaps:")
    for item in results['house_overlaps']['examples']:
        print(f"  ZIP {item['zip_code']}: Districts {item['house_district']}")
    
    print("\n--- Senate District Overlaps ---")
    print(f"ZIPs in multiple senate districts: {results['senate_overlaps']['total']}")
    if results['senate_overlaps']['max_overlaps']:
        print(f"ZIP with most senate districts: {results['senate_overlaps']['max_overlaps']['zip_code']} "
              f"(in {results['senate_overlaps']['max_overlaps']['num_districts']} districts)")
    print("Example overlaps:")
    for item in results['senate_overlaps']['examples']:
        print(f"  ZIP {item['zip_code']}: Districts {item['senate_district']}")
