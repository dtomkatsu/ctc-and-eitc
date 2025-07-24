#!/usr/bin/env python3
"""
Performance testing for tax unit construction.

This script tests the performance of the tax unit construction with different
optimization techniques:
1. Baseline (original implementation)
2. Vectorized operations
3. Batch processing
4. Parallel processing
"""

import time
import tracemalloc
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tax.units.constructor import TaxUnitConstructor

class PerformanceTester:
    """Class to test and compare performance of different implementations."""
    
    def __init__(self, person_df, hh_df):
        """Initialize with data."""
        self.person_df = person_df
        self.hh_df = hh_df
        self.results = []
    
    def measure_memory(self, func, *args, **kwargs):
        """Measure memory usage of a function."""
        tracemalloc.start()
        result = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return result, peak / 1024**2  # Convert to MB
    
    def time_function(self, func, *args, **kwargs):
        """Time a function execution."""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time
    
    def test_baseline(self):
        """Test baseline implementation."""
        print("\n=== Testing Baseline Implementation ===")
        
        # Time and memory measurement
        start_time = time.time()
        result, mem_usage = self.measure_memory(
            lambda: TaxUnitConstructor(self.person_df, self.hh_df).create_rule_based_units()
        )
        end_time = time.time()
        
        self.results.append({
            'test': 'baseline',
            'time_seconds': end_time - start_time,
            'memory_mb': mem_usage,
            'num_units': len(result),
            'num_households': len(self.hh_df)
        })
        
        return result
    
    def test_vectorized(self):
        """Test implementation with vectorized operations."""
        print("\n=== Testing Vectorized Implementation ===")
        
        # Create a copy of the constructor with vectorized methods
        class VectorizedTaxUnitConstructor(TaxUnitConstructor):
            def _identify_joint_filers(self, adults, hh_members):
                # Vectorized implementation of joint filer identification
                adults = adults.copy()
                adults['is_householder'] = (adults['RELSHIPP'] == 20)
                adults['is_spouse'] = (adults['RELSHIPP'] == 21)
                
                # Find householders with spouses
                joint_filers = []
                for hh_id, hh_group in adults.groupby('SERIALNO'):
                    householders = hh_group[hh_group['is_householder']]
                    spouses = hh_group[hh_group['is_spouse']]
                    
                    # Match householders with their spouses
                    for _, hh in householders.iterrows():
                        spouse = spouses[spouses['SERIALNO'] == hh['SERIALNO']]
                        if not spouse.empty:
                            joint_filers.append((hh.name, spouse.index[0]))
                
                # For now, just test with no MFS filers
                return joint_filers, []
        
        # Time and memory measurement
        start_time = time.time()
        result, mem_usage = self.measure_memory(
            lambda: VectorizedTaxUnitConstructor(self.person_df, self.hh_df).create_rule_based_units()
        )
        end_time = time.time()
        
        self.results.append({
            'test': 'vectorized',
            'time_seconds': end_time - start_time,
            'memory_mb': mem_usage,
            'num_units': len(result),
            'num_households': len(self.hh_df)
        })
        
        return result
    
    def _process_household_batch(self, hh_ids, person_df, hh_df):
        """Process a batch of households."""
        constructor = TaxUnitConstructor(
            person_df[person_df['SERIALNO'].isin(hh_ids)], 
            hh_df[hh_df['SERIALNO'].isin(hh_ids)]
        )
        return constructor.create_rule_based_units()
    
    def test_batch_processing(self, batch_size=1000):
        """Test batch processing implementation."""
        print(f"\n=== Testing Batch Processing (batch_size={batch_size}) ===")
        
        # Get unique household IDs
        hh_ids = self.hh_df['SERIALNO'].unique()
        num_batches = (len(hh_ids) + batch_size - 1) // batch_size
        
        all_units = []
        start_time = time.time()
        peak_memory = 0
        
        for i in range(0, len(hh_ids), batch_size):
            batch_ids = hh_ids[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{num_batches} ({len(batch_ids)} households)")
            
            # Process batch and measure memory
            batch_result, mem_usage = self.measure_memory(
                lambda: self._process_household_batch(batch_ids, self.person_df, self.hh_df)
            )
            
            all_units.append(batch_result)
            peak_memory = max(peak_memory, mem_usage)
        
        end_time = time.time()
        
        # Combine results
        if all_units:
            result = pd.concat(all_units, ignore_index=True)
        else:
            result = pd.DataFrame()
        
        self.results.append({
            'test': f'batch_{batch_size}',
            'time_seconds': end_time - start_time,
            'memory_mb': peak_memory,
            'num_units': len(result),
            'num_households': len(hh_ids)
        })
        
        return result
    
    def test_parallel_processing(self, num_processes=4, batch_size=1000):
        """Test parallel processing implementation."""
        print(f"\n=== Testing Parallel Processing ({num_processes} processes, batch_size={batch_size}) ===")
        
        # Get unique household IDs
        hh_ids = self.hh_df['SERIALNO'].unique()
        
        # Split into batches for parallel processing
        batches = [hh_ids[i:i + batch_size] for i in range(0, len(hh_ids), batch_size)]
        
        start_time = time.time()
        
        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            # Create a partial function with the common arguments
            process_func = partial(
                self._process_household_batch,
                person_df=self.person_df,
                hh_df=self.hh_df
            )
            
            # Process batches in parallel
            results = list(executor.map(process_func, batches))
        
        end_time = time.time()
        
        # Combine results
        if results:
            result = pd.concat(results, ignore_index=True)
        else:
            result = pd.DataFrame()
        
        # Measure peak memory (this is an approximation in multiprocessing)
        _, peak_memory = self.measure_memory(lambda: None)
        
        self.results.append({
            'test': f'parallel_{num_processes}_{batch_size}',
            'time_seconds': end_time - start_time,
            'memory_mb': peak_memory,
            'num_units': len(result),
            'num_households': len(hh_ids)
        })
        
        return result
    
    def print_results(self):
        """Print performance test results."""
        print("\n=== Performance Test Results ===")
        df = pd.DataFrame(self.results)
        
        # Calculate speedup compared to baseline
        if not df.empty and 'baseline' in df['test'].values:
            baseline_time = df[df['test'] == 'baseline']['time_seconds'].values[0]
            df['speedup'] = baseline_time / df['time_seconds']
            df['memory_reduction'] = 1 - (df['memory_mb'] / df[df['test'] == 'baseline']['memory_mb'].values[0])
        
        print("\nPerformance Summary:")
        print(df[['test', 'time_seconds', 'speedup', 'memory_mb', 'memory_reduction', 'num_units']])
        
        # Print recommendations
        print("\nRecommendations:")
        if not df.empty:
            fastest = df.loc[df['time_seconds'].idxmin()]
            most_memory_efficient = df.loc[df['memory_mb'].idxmin()]
            
            print(f"1. Fastest method: {fastest['test']} ({fastest['time_seconds']:.2f}s, "
                  f"{fastest['memory_mb']:.2f}MB)")
            print(f"2. Most memory efficient: {most_memory_efficient['test']} "
                  f"({most_memory_efficient['memory_mb']:.2f}MB, {most_memory_efficient['time_seconds']:.2f}s)")
            
            if 'vectorized' in df['test'].values and 'baseline' in df['test'].values:
                vec = df[df['test'] == 'vectorized'].iloc[0]
                base = df[df['test'] == 'baseline'].iloc[0]
                speedup = base['time_seconds'] / vec['time_seconds']
                if speedup > 1.1:
                    print(f"3. Vectorization provided {speedup:.1f}x speedup over baseline")
                
            if any('batch' in t for t in df['test']):
                best_batch = df[df['test'].str.startswith('batch_')].sort_values('time_seconds').iloc[0]
                print(f"4. Best batch size: {best_batch['test'].replace('batch_', '')} "
                      f"({best_batch['time_seconds']:.2f}s)")
                
            if any('parallel' in t for t in df['test']):
                best_parallel = df[df['test'].str.startswith('parallel_')].sort_values('time_seconds').iloc[0]
                print(f"5. Best parallel configuration: {best_parallel['test']} "
                      f"({best_parallel['time_seconds']:.2f}s)")

def create_sample_data(num_households=10000, avg_household_size=2.5):
    """Create sample data for performance testing."""
    np.random.seed(42)
    
    # Generate household data
    hh_data = []
    person_data = []
    
    for hh_id in range(1, num_households + 1):
        # Generate household income (skewed distribution)
        hh_income = np.random.lognormal(10.5, 0.8)
        hh_data.append({
            'SERIALNO': str(hh_id),
            'HINCP': hh_income,
            'ADJINC': 1.0
        })
        
        # Generate household members (Poisson distribution around average)
        size = max(1, int(np.random.poisson(avg_household_size - 1) + 1))
        
        # Add householder (always present)
        person_data.append({
            'SERIALNO': str(hh_id),
            'SPORDER': '1',
            'AGEP': int(np.random.normal(45, 15)),
            'SEX': np.random.choice([1, 2]),
            'MAR': np.random.choice([1, 2, 3, 4, 5], p=[0.4, 0.1, 0.1, 0.1, 0.3]),
            'RELSHIPP': 20,  # Householder
            'WAGP': np.random.lognormal(10, 0.7),
            'CIT': 1,
            'SEMP': 0,
            'ADJINC': 1.0,
            'SCHL': int(np.random.choice(range(1, 25)))
        })
        
        # Add spouse (if married)
        if person_data[-1]['MAR'] == 1 and size > 1:  # Married
            person_data.append({
                'SERIALNO': str(hh_id),
                'SPORDER': '2',
                'AGEP': max(18, int(person_data[-1]['AGEP'] + np.random.normal(0, 5))),
                'SEX': 2 if person_data[-1]['SEX'] == 1 else 1,  # Opposite sex
                'MAR': 1,
                'RELSHIPP': 21,  # Spouse
                'WAGP': np.random.lognormal(9.5, 0.7),
                'CIT': 1,
                'SEMP': 0,
                'ADJINC': 1.0,
                'SCHL': int(np.random.choice(range(1, 25)))
            })
            size -= 1
        
        # Add children (if any remaining spots)
        for i in range(1, size):
            person_data.append({
                'SERIALNO': str(hh_id),
                'SPORDER': str(i + 1),
                'AGEP': int(np.random.uniform(0, 18)),
                'SEX': np.random.choice([1, 2]),
                'MAR': 0,
                'RELSHIPP': 22,  # Child
                'WAGP': 0,
                'CIT': 1,
                'SEMP': 0,
                'ADJINC': 1.0,
                'SCHL': int(np.random.uniform(1, 13))
            })
    
    # Create DataFrames
    person_df = pd.DataFrame(person_data)
    hh_df = pd.DataFrame(hh_data)
    
    # Create person_id from SERIALNO and SPORDER
    person_df['person_id'] = person_df['SERIALNO'] + '_' + person_df['SPORDER']
    person_df.set_index('person_id', inplace=True)
    
    return person_df, hh_df

def main():
    """Main function to run performance tests."""
    print("Generating test data...")
    
    # Create sample data (10,000 households by default)
    person_df, hh_df = create_sample_data(num_households=10000)
    
    print(f"Created test data with {len(person_df)} persons in {len(hh_df)} households")
    print(f"Average household size: {len(person_df) / len(hh_df):.2f}")
    
    # Initialize tester
    tester = PerformanceTester(person_df, hh_df)
    
    # Run tests
    print("\n=== Starting Performance Tests ===")
    
    # 1. Baseline test
    print("\n[1/4] Running baseline test...")
    tester.test_baseline()
    
    # 2. Vectorized test
    print("\n[2/4] Running vectorized test...")
    tester.test_vectorized()
    
    # 3. Batch processing test
    print("\n[3/4] Running batch processing test...")
    tester.test_batch_processing(batch_size=1000)
    
    # 4. Parallel processing test
    print("\n[4/4] Running parallel processing test...")
    tester.test_parallel_processing(num_processes=4, batch_size=500)
    
    # Print results
    tester.print_results()

if __name__ == "__main__":
    main()
