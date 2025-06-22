import time
import json
import pandas as pd
import csv
import os
import gzip
import statistics
from jsontables import JSONTablesEncoder, JSONTablesDecoder
from jsontables.core import JSONTablesV2Encoder, JSONTablesV2Decoder
import numpy as np

def time_operation(func, *args, **kwargs):
    """Time a function and return (result, time_in_seconds)"""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return result, end - start

def time_operation_multiple(func, iterations=5, *args, **kwargs):
    """Time a function multiple times and return statistics"""
    times = []
    result = None
    
    for i in range(iterations):
        result, elapsed = time_operation(func, *args, **kwargs)
        times.append(elapsed * 1000)  # Convert to milliseconds
    
    return {
        'mean': statistics.mean(times),
        'stdev': statistics.stdev(times) if len(times) > 1 else 0,
        'min': min(times),
        'max': max(times),
        'times': times
    }

def load_boston_dataset():
    """Load the real Boston real estate dataset"""
    boston_path = "/Users/admin/Desktop/tetra-ws/featrix/merge/mosaic/etc/sample_data/small-data-sets/8000-boston.csv"
    
    if not os.path.exists(boston_path):
        print(f"❌ Boston dataset not found at {boston_path}")
        return None
    
    try:
        # Read with proper handling of mixed types
        df = pd.read_csv(boston_path, dtype=str, na_values=[''])
        
        # Clean up the data for better compatibility
        # Convert numeric columns back to numbers where possible
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
            'age': int(np.random.randint(18, 80)),  # Convert to native int
            'salary': int(np.random.randint(30000, 120000)),  # Convert to native int
            'department': np.random.choice(['Engineering', 'Sales', 'Marketing', 'HR']),
            'active': bool(np.random.choice([True, False])),  # Convert to native bool
            'score': float(round(np.random.uniform(0, 100), 2))  # Convert to native float
        }
        # Add extra columns for larger tests
        for j in range(7, cols):
            row[f'field_{j}'] = int(np.random.randint(0, 1000))  # Convert to native int
        data.append(row)
    
    return data

def create_new_row_for_dataset(data):
    """Create a new row compatible with the dataset structure"""
    if not data or len(data) == 0:
        return {}
    
    # Get the first row to understand the structure
    sample_row = data[0]
    
    # Check if this is synthetic data (has our test fields)
    if 'id' in sample_row and 'name' in sample_row:
        # Synthetic data
        new_row = {'id': 'NEW_001', 'name': 'New Person', 'age': 25, 'salary': 50000, 
                   'department': 'Engineering', 'active': True, 'score': 85.5}
        # Add extra fields if they exist
        for key in sample_row.keys():
            if key not in new_row:
                new_row[key] = 999
        return new_row
    
    # Real dataset - create a new row based on the actual structure
    new_row = {}
    for key, value in sample_row.items():
        if key in ['ST_NUM', 'ZIP_CODE']:
            new_row[key] = 9999
        elif key in ['ST_NAME', 'CITY']:
            new_row[key] = 'NEW_ADDRESS'
        elif key in ['LAND_VALUE', 'BLDG_VALUE', 'TOTAL_VALUE']:
            new_row[key] = 100000
        elif isinstance(value, (int, float)) and value is not None:
            new_row[key] = 1
        elif isinstance(value, str):
            new_row[key] = 'NEW'
        else:
            new_row[key] = None
    
    return new_row

def benchmark_csv(data, iterations=5):
    """Benchmark CSV operations with multiple iterations"""
    results = {}
    
    # 1. ENCODE (write)
    def encode_csv():
        with open('temp.csv', 'w', newline='') as f:
            if data:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
    
    results['encode'] = time_operation_multiple(encode_csv, iterations)
    
    # 2. DECODE (read) 
    def decode_csv():
        with open('temp.csv', 'r') as f:
            return list(csv.DictReader(f))
    
    results['decode'] = time_operation_multiple(decode_csv, iterations)
    
    # 3. ADD ROW (optimized append)
    new_row = create_new_row_for_dataset(data)
    
    def add_row_csv_optimized():
        # Optimized: Just append a line to the CSV file
        with open('temp.csv', 'a', newline='') as f:
            if data:  # Make sure we know the fieldnames
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writerow(new_row)
    
    results['add_row'] = time_operation_multiple(add_row_csv_optimized, iterations)
    
    # 4. GZIP COMPRESSION
    def compress_csv():
        with open('temp.csv', 'rb') as f_in:
            with gzip.open('temp.csv.gz', 'wb') as f_out:
                f_out.write(f_in.read())
    
    results['gzip_compress'] = time_operation_multiple(compress_csv, iterations)
    
    # 5. GZIP DECOMPRESSION
    def decompress_csv():
        with gzip.open('temp.csv.gz', 'rb') as f_in:
            with open('temp_decompressed.csv', 'wb') as f_out:
                f_out.write(f_in.read())
    
    results['gzip_decompress'] = time_operation_multiple(decompress_csv, iterations)
    
    return results

def benchmark_json(data, iterations=5):
    """Benchmark standard JSON operations with multiple iterations"""
    results = {}
    
    # 1. ENCODE
    def encode_json():
        with open('temp.json', 'w') as f:
            json.dump(data, f)
    
    results['encode'] = time_operation_multiple(encode_json, iterations)
    
    # 2. DECODE
    def decode_json():
        with open('temp.json', 'r') as f:
            return json.load(f)
    
    results['decode'] = time_operation_multiple(decode_json, iterations)
    
    # 3. ADD ROW (still requires full rewrite for JSON arrays)
    new_row = create_new_row_for_dataset(data)
    
    def add_row_json():
        with open('temp.json', 'r') as f:
            data = json.load(f)
        data.append(new_row)
        with open('temp.json', 'w') as f:
            json.dump(data, f)
    
    results['add_row'] = time_operation_multiple(add_row_json, iterations)
    
    # 4. GZIP COMPRESSION
    def compress_json():
        with open('temp.json', 'rb') as f_in:
            with gzip.open('temp.json.gz', 'wb') as f_out:
                f_out.write(f_in.read())
    
    results['gzip_compress'] = time_operation_multiple(compress_json, iterations)
    
    # 5. GZIP DECOMPRESSION  
    def decompress_json():
        with gzip.open('temp.json.gz', 'rb') as f_in:
            with open('temp_decompressed.json', 'wb') as f_out:
                f_out.write(f_in.read())
    
    results['gzip_decompress'] = time_operation_multiple(decompress_json, iterations)
    
    return results

def benchmark_jsontables(data, iterations=5):
    """Benchmark JSON-Tables operations with multiple iterations"""
    results = {}
    
    # 1. ENCODE
    def encode_jsontables():
        encoded = JSONTablesEncoder.from_records(data)
        with open('temp_jsontables.json', 'w') as f:
            json.dump(encoded, f)
    
    results['encode'] = time_operation_multiple(encode_jsontables, iterations)
    
    # 2. DECODE
    def decode_jsontables():
        with open('temp_jsontables.json', 'r') as f:
            encoded = json.load(f)
        return JSONTablesDecoder.to_records(encoded)
    
    results['decode'] = time_operation_multiple(decode_jsontables, iterations)
    
    # 3. ADD ROW (still requires full rewrite for JSON-Tables)
    new_row = create_new_row_for_dataset(data)
    
    def add_row_jsontables():
        # Read existing
        with open('temp_jsontables.json', 'r') as f:
            encoded = json.load(f)
        data = JSONTablesDecoder.to_records(encoded)
        
        # Add row
        data.append(new_row)
        
        # Re-encode and save
        encoded = JSONTablesEncoder.from_records(data)
        with open('temp_jsontables.json', 'w') as f:
            json.dump(encoded, f)
    
    results['add_row'] = time_operation_multiple(add_row_jsontables, iterations)
    
    # 4. GZIP COMPRESSION
    def compress_jsontables():
        with open('temp_jsontables.json', 'rb') as f_in:
            with gzip.open('temp_jsontables.json.gz', 'wb') as f_out:
                f_out.write(f_in.read())
    
    results['gzip_compress'] = time_operation_multiple(compress_jsontables, iterations)
    
    # 5. GZIP DECOMPRESSION
    def decompress_jsontables():
        with gzip.open('temp_jsontables.json.gz', 'rb') as f_in:
            with open('temp_decompressed_jsontables.json', 'wb') as f_out:
                f_out.write(f_in.read())
    
    results['gzip_decompress'] = time_operation_multiple(decompress_jsontables, iterations)
    
    return results

def benchmark_jsontables_v2(data, iterations=5):
    """Benchmark JSON-Tables v2 operations with multiple iterations"""
    results = {}
    
    # 1. ENCODE
    def encode_jsontables_v2():
        encoded = JSONTablesV2Encoder.from_records_v2(data)
        with open('temp_jsontables_v2.json', 'w') as f:
            json.dump(encoded, f)
    
    results['encode'] = time_operation_multiple(encode_jsontables_v2, iterations)
    
    # 2. DECODE
    def decode_jsontables_v2():
        with open('temp_jsontables_v2.json', 'r') as f:
            encoded = json.load(f)
        return JSONTablesV2Decoder.to_records(encoded)
    
    results['decode'] = time_operation_multiple(decode_jsontables_v2, iterations)
    
    # 3. ADD ROW (still requires full rewrite for JSON-Tables v2)
    new_row = create_new_row_for_dataset(data)
    
    def add_row_jsontables_v2():
        # Read existing
        with open('temp_jsontables_v2.json', 'r') as f:
            encoded = json.load(f)
        data = JSONTablesV2Decoder.to_records(encoded)
        
        # Add row
        data.append(new_row)
        
        # Re-encode and save
        encoded = JSONTablesV2Encoder.from_records_v2(data)
        with open('temp_jsontables_v2.json', 'w') as f:
            json.dump(encoded, f)
    
    results['add_row'] = time_operation_multiple(add_row_jsontables_v2, iterations)
    
    # 4. GZIP COMPRESSION
    def compress_jsontables_v2():
        with open('temp_jsontables_v2.json', 'rb') as f_in:
            with gzip.open('temp_jsontables_v2.json.gz', 'wb') as f_out:
                f_out.write(f_in.read())
    
    results['gzip_compress'] = time_operation_multiple(compress_jsontables_v2, iterations)
    
    # 5. GZIP DECOMPRESSION
    def decompress_jsontables_v2():
        with gzip.open('temp_jsontables_v2.json.gz', 'rb') as f_in:
            with open('temp_decompressed_jsontables_v2.json', 'wb') as f_out:
                f_out.write(f_in.read())
    
    results['gzip_decompress'] = time_operation_multiple(decompress_jsontables_v2, iterations)
    
    return results

def benchmark_parquet(data, iterations=5):
    """Benchmark Parquet operations with multiple iterations"""
    results = {}
    
    try:
        # Convert to DataFrame with better type handling
        df = pd.DataFrame(data)
        
        # Clean up data types for Parquet compatibility
        for col in df.columns:
            # Handle mixed types by converting to string first, then to appropriate type
            if df[col].dtype == 'object':
                # Check if it looks like it should be numeric
                try:
                    # Try to convert to numeric, keep as string if it fails
                    numeric_series = pd.to_numeric(df[col], errors='coerce')
                    # If more than 50% can be converted to numeric, use numeric
                    if numeric_series.notna().sum() > len(df) * 0.5:
                        df[col] = numeric_series
                    else:
                        # Keep as string but ensure no None values
                        df[col] = df[col].astype(str).replace('None', '')
                except:
                    # Fallback to string
                    df[col] = df[col].astype(str).replace('None', '')
        
        # 1. ENCODE
        def encode_parquet():
            df.to_parquet('temp.parquet', index=False)
        
        results['encode'] = time_operation_multiple(encode_parquet, iterations)
        
        # 2. DECODE
        def decode_parquet():
            return pd.read_parquet('temp.parquet').to_dict('records')
        
        results['decode'] = time_operation_multiple(decode_parquet, iterations)
        
        # 3. ADD ROW
        new_row = create_new_row_for_dataset(data)
        
        def add_row_parquet():
            # Read existing
            df = pd.read_parquet('temp.parquet')
            # Add row - ensure types match
            new_row_df = pd.DataFrame([new_row])
            # Align data types
            for col in df.columns:
                if col in new_row_df.columns:
                    try:
                        new_row_df[col] = new_row_df[col].astype(df[col].dtype)
                    except:
                        # If type conversion fails, convert both to string
                        df[col] = df[col].astype(str)
                        new_row_df[col] = new_row_df[col].astype(str)
            
            combined_df = pd.concat([df, new_row_df], ignore_index=True)
            combined_df.to_parquet('temp.parquet', index=False)
        
        results['add_row'] = time_operation_multiple(add_row_parquet, iterations)
        
    except Exception as e:
        print(f"  ⚠️  Parquet setup failed: {e}")
        results = {
            'encode': {'mean': None, 'stdev': None},
            'decode': {'mean': None, 'stdev': None}, 
            'add_row': {'mean': None, 'stdev': None}
        }
    
    return results

def format_timing_result(timing_stats):
    """Format timing statistics for display"""
    mean = timing_stats['mean']
    stdev = timing_stats['stdev']
    
    if mean < 1:
        return f"{mean:.3f}±{stdev:.3f}"
    elif mean < 10:
        return f"{mean:.2f}±{stdev:.2f}"
    else:
        return f"{mean:.1f}±{stdev:.1f}"

def get_file_sizes():
    """Get file sizes for all temporary files to analyze storage efficiency"""
    sizes = {}
    files = {
        'CSV': 'temp.csv',
        'JSON': 'temp.json', 
        'JSON-Tables v1': 'temp_jsontables.json',
        'JSON-Tables v2': 'temp_jsontables_v2.json',
        'CSV.gz': 'temp.csv.gz',
        'JSON.gz': 'temp.json.gz',
        'JSON-Tables v1.gz': 'temp_jsontables.json.gz',
        'JSON-Tables v2.gz': 'temp_jsontables_v2.json.gz'
    }
    
    for format_name, filename in files.items():
        if os.path.exists(filename):
            sizes[format_name] = os.path.getsize(filename)
        else:
            sizes[format_name] = None
    
    return sizes

def run_benchmarks():
    """Run comprehensive benchmarks with multiple iterations"""
    
    print("📊 COMPREHENSIVE FORMAT ANALYSIS")
    print("=" * 60)
    print("Performance, storage, and compression measurements")
    print()
    
    # Test configurations: (data_generator, label, iterations)
    test_configs = [
        (lambda: generate_test_data(1000, 7), "Medium Synthetic (1K rows, 7 cols)", 3),
        (lambda: generate_test_data(5000, 7), "Large Synthetic (5K rows, 7 cols)", 3),
        (load_boston_dataset, "Real Boston Dataset (8K rows, 90 cols)", 2),
    ]
    
    all_results = {}
    
    for data_generator, label, iterations in test_configs:
        print(f"📋 {label}")
        print("-" * 55)
        
        # Generate or load test data
        if callable(data_generator):
            data = data_generator()
        else:
            data = data_generator
            
        if data is None:
            print("❌ Skipping - data unavailable")
            print()
            continue
            
        print(f"Running {iterations} iterations per operation...")
        
        # Benchmark each format (excluding Parquet for now to focus on text formats)
        formats = [
            ("CSV", benchmark_csv),
            ("JSON", benchmark_json),
            ("JSON-Tables v1", benchmark_jsontables),
            ("JSON-Tables v2", benchmark_jsontables_v2),
        ]
        
        results = {}
        for format_name, benchmark_func in formats:
            try:
                print(f"  Testing {format_name}...")
                results[format_name] = benchmark_func(data, iterations)
                print(f"  ✅ {format_name} completed")
            except Exception as e:
                print(f"  ❌ {format_name} failed: {e}")
                results[format_name] = {
                    'encode': {'mean': None, 'stdev': None},
                    'decode': {'mean': None, 'stdev': None}, 
                    'add_row': {'mean': None, 'stdev': None},
                    'gzip_compress': {'mean': None, 'stdev': None},
                    'gzip_decompress': {'mean': None, 'stdev': None}
                }
        
        # Get file sizes after encoding
        print("  Measuring storage sizes...")
        file_sizes = get_file_sizes()
        
        all_results[label] = {
            'performance': results,
            'storage': file_sizes
        }
        
        # Display results
        print()
        print("PERFORMANCE RESULTS (milliseconds, mean±stdev):")
        print(f"{'Format':<16} {'Encode':<12} {'Decode':<12} {'Add Row':<12} {'Gzip':<12} {'Gunzip':<12}")
        print("-" * 80)
        
        for fmt in ["CSV", "JSON", "JSON-Tables v1", "JSON-Tables v2"]:
            if fmt in results:
                r = results[fmt]
                
                encode_str = format_timing_result(r['encode']) if r['encode']['mean'] is not None else "FAIL"
                decode_str = format_timing_result(r['decode']) if r['decode']['mean'] is not None else "FAIL"
                add_row_str = format_timing_result(r['add_row']) if r['add_row']['mean'] is not None else "FAIL"
                gzip_str = format_timing_result(r['gzip_compress']) if r['gzip_compress']['mean'] is not None else "FAIL"
                gunzip_str = format_timing_result(r['gzip_decompress']) if r['gzip_decompress']['mean'] is not None else "FAIL"
                
                print(f"{fmt:<16} {encode_str:<12} {decode_str:<12} {add_row_str:<12} {gzip_str:<12} {gunzip_str:<12}")
        
        print()
        print("STORAGE EFFICIENCY (bytes):")
        print(f"{'Format':<20} {'Uncompressed':<12} {'Compressed':<12} {'Compression':<12}")
        print("-" * 60)
        
        for fmt in ["CSV", "JSON", "JSON-Tables v1", "JSON-Tables v2"]:
            uncompressed = file_sizes.get(fmt)
            compressed = file_sizes.get(f"{fmt}.gz")
            
            if uncompressed is not None and compressed is not None:
                ratio = compressed / uncompressed
                compression_pct = (1 - ratio) * 100
                print(f"{fmt:<20} {uncompressed:<12,} {compressed:<12,} {compression_pct:<11.1f}%")
            else:
                print(f"{fmt:<20} {'FAIL':<12} {'FAIL':<12} {'FAIL':<12}")
        
        print()
        print()
    
    # Cleanup
    cleanup_files = [
        'temp.csv', 'temp.json', 'temp_jsontables.json', 'temp_jsontables_v2.json',
        'temp.csv.gz', 'temp.json.gz', 'temp_jsontables.json.gz', 'temp_jsontables_v2.json.gz',
        'temp_decompressed.csv', 'temp_decompressed.json', 'temp_decompressed_jsontables.json', 
        'temp_decompressed_jsontables_v2.json'
    ]
    for f in cleanup_files:
        if os.path.exists(f):
            os.remove(f)
    
    return all_results

if __name__ == "__main__":
    results = run_benchmarks()
