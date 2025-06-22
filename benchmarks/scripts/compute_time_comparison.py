#!/usr/bin/env python3
"""
Compute Time Comparison: JSON-Tables vs CSV vs JSON

Focused analysis of processing times across different formats.
"""

import sys
import os
import time
import json
import pandas as pd
import csv
import statistics
import numpy as np

# Add parent directory to path to import jsontables
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from jsontables import JSONTablesEncoder, JSONTablesDecoder
from jsontables.core import JSONTablesV2Encoder, JSONTablesV2Decoder

def time_operation_multiple(func, iterations=5, *args, **kwargs):
    """Time a function multiple times and return statistics"""
    times = []
    result = None
    
    for i in range(iterations):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to milliseconds
    
    return {
        'mean': statistics.mean(times),
        'stdev': statistics.stdev(times) if len(times) > 1 else 0,
        'min': min(times),
        'max': max(times),
        'times': times
    }

def load_boston_dataset():
    """Load the Boston real estate dataset"""
    boston_path = os.path.join(os.path.dirname(__file__), '..', 'data', '8000-boston.csv')
    
    if not os.path.exists(boston_path):
        print(f"❌ Boston dataset not found at {boston_path}")
        return None
    
    try:
        # Read with proper handling of mixed types
        df = pd.read_csv(boston_path, dtype=str, na_values=[''])
        
        # Clean up the data for better compatibility
        numeric_cols = ['ST_NUM', 'ZIP_CODE', 'BLDG_SEQ', 'NUM_BLDGS', 'RES_FLOOR', 'CD_FLOOR', 
                       'RES_UNITS', 'LAND_SF', 'GROSS_AREA', 'LIVING_AREA', 'LAND_VALUE', 
                       'BLDG_VALUE', 'TOTAL_VALUE', 'GROSS_TAX', 'YR_BUILT', 'YR_REMODEL',
                       'BED_RMS', 'FULL_BTH', 'HLF_BTH', 'KITCHENS', 'TT_RMS', 'FIREPLACES', 'NUM_PARKING']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert to records format, replacing NaN with None for JSON compatibility
        data = df.where(pd.notnull(df), None).to_dict('records')
        print(f"✅ Loaded Boston dataset: {len(data):,} rows, {len(df.columns)} columns")
        return data
    except Exception as e:
        print(f"❌ Error loading Boston dataset: {e}")
        return None

def generate_test_data(rows, cols=7):
    """Generate test data similar to our datasets"""
    np.random.seed(42)  # For reproducible results
    
    data = []
    for i in range(rows):
        row = {
            'id': f'ID_{i:06d}',
            'name': f'Person_{i}',
            'age': int(np.random.randint(18, 80)),
            'salary': int(np.random.randint(30000, 120000)),
            'department': np.random.choice(['Engineering', 'Sales', 'Marketing', 'HR']),
            'active': bool(np.random.choice([True, False])),
            'score': float(round(np.random.uniform(0, 100), 2))
        }
        # Add extra columns for larger tests
        for j in range(7, cols):
            row[f'field_{j}'] = int(np.random.randint(0, 1000))
        data.append(row)
    
    return data

def benchmark_format(data, format_name, encode_func, decode_func, iterations=3):
    """Benchmark a specific format"""
    print(f"  Testing {format_name}...")
    
    # Encode timing
    encode_times = time_operation_multiple(encode_func, iterations)
    
    # Decode timing
    decode_times = time_operation_multiple(decode_func, iterations)
    
    return {
        'encode': encode_times,
        'decode': decode_times
    }

def run_compute_comparison():
    """Run comprehensive compute time comparison"""
    
    print("⚡ COMPUTE TIME COMPARISON")
    print("=" * 60)
    print("Detailed performance analysis vs CSV and JSON baselines")
    print()
    
    # Test datasets
    datasets = [
        ("Medium Synthetic (1K×7)", generate_test_data(1000, 7)),
        ("Large Synthetic (5K×7)", generate_test_data(5000, 7)),
        ("Real Boston (8K×90)", load_boston_dataset())
    ]
    
    all_results = {}
    
    for dataset_name, data in datasets:
        if data is None:
            continue
            
        print(f"📊 {dataset_name}")
        print("-" * 50)
        
        results = {}
        
        # CSV benchmark
        def encode_csv():
            with open('temp.csv', 'w', newline='') as f:
                if data:
                    writer = csv.DictWriter(f, fieldnames=data[0].keys())
                    writer.writeheader()
                    writer.writerows(data)
        
        def decode_csv():
            with open('temp.csv', 'r') as f:
                return list(csv.DictReader(f))
        
        results['CSV'] = benchmark_format(data, 'CSV', encode_csv, decode_csv)
        
        # JSON benchmark
        def encode_json():
            with open('temp.json', 'w') as f:
                json.dump(data, f)
        
        def decode_json():
            with open('temp.json', 'r') as f:
                return json.load(f)
        
        results['JSON'] = benchmark_format(data, 'JSON', encode_json, decode_json)
        
        # JSON-Tables v1 benchmark
        def encode_jt_v1():
            encoded = JSONTablesEncoder.from_records(data)
            with open('temp_jt_v1.json', 'w') as f:
                json.dump(encoded, f)
        
        def decode_jt_v1():
            with open('temp_jt_v1.json', 'r') as f:
                encoded = json.load(f)
            return JSONTablesDecoder.to_records(encoded)
        
        results['JSON-Tables v1'] = benchmark_format(data, 'JSON-Tables v1', encode_jt_v1, decode_jt_v1)
        
        # JSON-Tables v2 benchmark
        def encode_jt_v2():
            encoded = JSONTablesV2Encoder.from_records_v2(data)
            with open('temp_jt_v2.json', 'w') as f:
                json.dump(encoded, f)
        
        def decode_jt_v2():
            with open('temp_jt_v2.json', 'r') as f:
                encoded = json.load(f)
            return JSONTablesV2Decoder.to_records(encoded)
        
        results['JSON-Tables v2'] = benchmark_format(data, 'JSON-Tables v2', encode_jt_v2, decode_jt_v2)
        
        all_results[dataset_name] = results
        
        # Display results for this dataset
        display_dataset_results(dataset_name, results)
        
        # Cleanup temp files
        for f in ['temp.csv', 'temp.json', 'temp_jt_v1.json', 'temp_jt_v2.json']:
            if os.path.exists(f):
                os.remove(f)
        
        print()
    
    # Overall analysis
    print("\n🎯 OVERALL COMPUTE TIME ANALYSIS")
    print("=" * 40)
    analyze_compute_results(all_results)
    
    return all_results

def display_dataset_results(dataset_name, results):
    """Display results for a single dataset"""
    print(f"\nPerformance Results (milliseconds, mean±stdev):")
    print(f"{'Format':<16} {'Encode':<15} {'Decode':<15}")
    print("-" * 50)
    
    csv_encode = results['CSV']['encode']['mean']
    csv_decode = results['CSV']['decode']['mean']
    json_encode = results['JSON']['encode']['mean']
    json_decode = results['JSON']['decode']['mean']
    
    for format_name, data in results.items():
        encode_mean = data['encode']['mean']
        encode_stdev = data['encode']['stdev']
        decode_mean = data['decode']['mean']
        decode_stdev = data['decode']['stdev']
        
        encode_str = f"{encode_mean:.2f}±{encode_stdev:.2f}"
        decode_str = f"{decode_mean:.2f}±{decode_stdev:.2f}"
        
        print(f"{format_name:<16} {encode_str:<15} {decode_str:<15}")
    
    print(f"\nRelative to CSV:")
    for format_name, data in results.items():
        if format_name != 'CSV':
            encode_ratio = data['encode']['mean'] / csv_encode
            decode_ratio = data['decode']['mean'] / csv_decode
            print(f"  {format_name}: {encode_ratio:.2f}x encode, {decode_ratio:.2f}x decode")
    
    print(f"\nRelative to JSON:")
    for format_name, data in results.items():
        if format_name != 'JSON':
            encode_ratio = data['encode']['mean'] / json_encode
            decode_ratio = data['decode']['mean'] / json_decode
            print(f"  {format_name}: {encode_ratio:.2f}x encode, {decode_ratio:.2f}x decode")

def analyze_compute_results(all_results):
    """Analyze results across all datasets"""
    
    print("\n📈 SCALING ANALYSIS:")
    print("-" * 25)
    
    # Analyze how each format scales
    dataset_sizes = {
        "Medium Synthetic (1K×7)": 1000,
        "Large Synthetic (5K×7)": 5000,
        "Real Boston (8K×90)": 8000
    }
    
    formats = ['CSV', 'JSON', 'JSON-Tables v1', 'JSON-Tables v2']
    
    for fmt in formats:
        print(f"\n{fmt} Scaling:")
        encode_times = []
        decode_times = []
        sizes = []
        
        for dataset_name, results in all_results.items():
            if fmt in results and dataset_name in dataset_sizes:
                encode_times.append(results[fmt]['encode']['mean'])
                decode_times.append(results[fmt]['decode']['mean'])
                sizes.append(dataset_sizes[dataset_name])
        
        if len(encode_times) >= 2:
            # Calculate scaling factor between datasets
            for i in range(1, len(encode_times)):
                size_ratio = sizes[i] / sizes[i-1]
                encode_ratio = encode_times[i] / encode_times[i-1]
                decode_ratio = decode_times[i] / decode_times[i-1]
                
                encode_efficiency = size_ratio / encode_ratio
                decode_efficiency = size_ratio / decode_ratio
                
                print(f"  {sizes[i-1]} → {sizes[i]} rows:")
                print(f"    Encode: {encode_ratio:.2f}x time ({encode_efficiency:.2f} efficiency)")
                print(f"    Decode: {decode_ratio:.2f}x time ({decode_efficiency:.2f} efficiency)")
    
    print("\n💡 KEY INSIGHTS:")
    print("-" * 15)
    
    print("\n🏆 Consistent Performance Leaders:")
    # Find formats that consistently perform well
    encode_leaders = {}
    decode_leaders = {}
    
    for dataset_name, results in all_results.items():
        # Find fastest encode/decode for this dataset
        encode_times = [(fmt, data['encode']['mean']) for fmt, data in results.items()]
        decode_times = [(fmt, data['decode']['mean']) for fmt, data in results.items()]
        
        encode_fastest = min(encode_times, key=lambda x: x[1])[0]
        decode_fastest = min(decode_times, key=lambda x: x[1])[0]
        
        encode_leaders[encode_fastest] = encode_leaders.get(encode_fastest, 0) + 1
        decode_leaders[decode_fastest] = decode_leaders.get(decode_fastest, 0) + 1
    
    print(f"Encode winners: {dict(sorted(encode_leaders.items(), key=lambda x: -x[1]))}")
    print(f"Decode winners: {dict(sorted(decode_leaders.items(), key=lambda x: -x[1]))}")
    
    print("\n⚖️  Cost vs Benefit Analysis:")
    # Calculate average performance ratios vs CSV and JSON
    csv_ratios = {'encode': {}, 'decode': {}}
    json_ratios = {'encode': {}, 'decode': {}}
    
    for dataset_name, results in all_results.items():
        csv_encode = results['CSV']['encode']['mean']
        csv_decode = results['CSV']['decode']['mean']
        json_encode = results['JSON']['encode']['mean']
        json_decode = results['JSON']['decode']['mean']
        
        for fmt in ['JSON-Tables v1', 'JSON-Tables v2']:
            if fmt in results:
                # vs CSV
                encode_ratio = results[fmt]['encode']['mean'] / csv_encode
                decode_ratio = results[fmt]['decode']['mean'] / csv_decode
                
                if fmt not in csv_ratios['encode']:
                    csv_ratios['encode'][fmt] = []
                    csv_ratios['decode'][fmt] = []
                
                csv_ratios['encode'][fmt].append(encode_ratio)
                csv_ratios['decode'][fmt].append(decode_ratio)
                
                # vs JSON
                encode_ratio = results[fmt]['encode']['mean'] / json_encode
                decode_ratio = results[fmt]['decode']['mean'] / json_decode
                
                if fmt not in json_ratios['encode']:
                    json_ratios['encode'][fmt] = []
                    json_ratios['decode'][fmt] = []
                
                json_ratios['encode'][fmt].append(encode_ratio)
                json_ratios['decode'][fmt].append(decode_ratio)
    
    print("\nAverage performance vs CSV:")
    for fmt in ['JSON-Tables v1', 'JSON-Tables v2']:
        if fmt in csv_ratios['encode']:
            avg_encode = statistics.mean(csv_ratios['encode'][fmt])
            avg_decode = statistics.mean(csv_ratios['decode'][fmt])
            print(f"  {fmt}: {avg_encode:.2f}x encode, {avg_decode:.2f}x decode")
    
    print("\nAverage performance vs JSON:")
    for fmt in ['JSON-Tables v1', 'JSON-Tables v2']:
        if fmt in json_ratios['encode']:
            avg_encode = statistics.mean(json_ratios['encode'][fmt])
            avg_decode = statistics.mean(json_ratios['decode'][fmt])
            print(f"  {fmt}: {avg_encode:.2f}x encode, {avg_decode:.2f}x decode")

if __name__ == "__main__":
    run_compute_comparison() 