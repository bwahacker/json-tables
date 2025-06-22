#!/usr/bin/env python3
"""
Comprehensive profiling analysis of JSON-Tables operations.

Uses the profiling framework to identify performance bottlenecks
and track where time is spent across different operations.
"""

import sys
import os
import tempfile
import pandas as pd

# Add parent directory to path to import jsontables
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from jsontables import JSONTablesEncoder, JSONTablesDecoder
from jsontables.core import append_to_json_table_file
from jsontables.profiling import profiling_session, save_profile_results

def generate_test_data(num_rows: int, num_cols: int = 7) -> list:
    """Generate test data for profiling."""
    data = []
    for i in range(num_rows):
        record = {
            'id': f'ID_{i:06d}',
            'name': f'Person_{i}',
            'email': f'person{i}@example.com',
            'age': 20 + (i % 60),
            'salary': 30000 + (i % 100000),
            'department': ['Engineering', 'Sales', 'Marketing', 'HR'][i % 4],
            'active': i % 3 == 0
        }
        data.append(record)
    return data

def profile_encoding_operations():
    """Profile JSON-Tables encoding operations."""
    print("\n🔍 Profiling Encoding Operations")
    print("=" * 40)
    
    test_sizes = [100, 1000, 5000]
    
    for size in test_sizes:
        print(f"\n📊 Dataset size: {size} rows")
        
        with profiling_session(f"encoding_{size}_rows"):
            # Generate test data
            test_data = generate_test_data(size)
            
            # Test records encoding
            encoded = JSONTablesEncoder.from_records(test_data)
            
            # Test DataFrame encoding
            df = pd.DataFrame(test_data)
            encoded_df = JSONTablesEncoder.from_dataframe(df)
            
            # Test columnar encoding
            encoded_columnar = JSONTablesEncoder.from_records(test_data, columnar=True)
        
        # Save detailed results
        save_profile_results(f'benchmarks/results/profile_encoding_{size}.json')

def profile_decoding_operations():
    """Profile JSON-Tables decoding operations."""
    print("\n🔍 Profiling Decoding Operations")
    print("=" * 40)
    
    test_sizes = [100, 1000, 5000]
    
    for size in test_sizes:
        print(f"\n📊 Dataset size: {size} rows")
        
        # Prepare test data
        test_data = generate_test_data(size)
        encoded = JSONTablesEncoder.from_records(test_data)
        encoded_columnar = JSONTablesEncoder.from_records(test_data, columnar=True)
        
        with profiling_session(f"decoding_{size}_rows"):
            # Test row-oriented decoding
            df_from_rows = JSONTablesDecoder.to_dataframe(encoded)
            records_from_rows = JSONTablesDecoder.to_records(encoded)
            
            # Test columnar decoding
            df_from_cols = JSONTablesDecoder.to_dataframe(encoded_columnar)
            records_from_cols = JSONTablesDecoder.to_records(encoded_columnar)
        
        # Save detailed results
        save_profile_results(f'benchmarks/results/profile_decoding_{size}.json')

def profile_append_operations():
    """Profile JSON-Tables append operations."""
    print("\n🔍 Profiling Append Operations")
    print("=" * 40)
    
    test_sizes = [100, 1000, 5000]
    
    for size in test_sizes:
        print(f"\n📊 Initial dataset size: {size} rows")
        
        with profiling_session(f"append_{size}_rows"):
            # Create initial dataset
            initial_data = generate_test_data(size)
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                temp_file = f.name
                encoded = JSONTablesEncoder.from_records(initial_data)
                import json
                json.dump(encoded, f)
            
            # Test multiple appends
            for i in range(10):  # 10 append operations
                new_rows = [generate_test_data(1)[0]]  # Single row append
                new_rows[0]['id'] = f'APPEND_{i}'
                append_to_json_table_file(temp_file, new_rows)
            
            # Cleanup
            os.unlink(temp_file)
        
        # Save detailed results
        save_profile_results(f'benchmarks/results/profile_append_{size}.json')

def profile_library_overhead():
    """Profile library overhead vs pure Python operations."""
    print("\n🔍 Profiling Library vs Python Overhead")
    print("=" * 45)
    
    size = 1000
    test_data = generate_test_data(size)
    
    with profiling_session("library_overhead", enable_library_instrumentation=True):
        # Operations that heavily use pandas
        df = pd.DataFrame(test_data)
        encoded_df = JSONTablesEncoder.from_dataframe(df)
        
        # Operations that use JSON extensively
        encoded_records = JSONTablesEncoder.from_records(test_data)
        decoded_records = JSONTablesDecoder.to_records(encoded_records)
        
        # Mix of operations
        for i in range(5):
            subset = test_data[i*100:(i+1)*100]
            encoded = JSONTablesEncoder.from_records(subset)
            df_subset = JSONTablesDecoder.to_dataframe(encoded)
    
    # Save detailed results
    save_profile_results('benchmarks/results/profile_library_overhead.json')

def analyze_performance_patterns():
    """Analyze performance patterns across different operation types."""
    print("\n📈 Performance Pattern Analysis")
    print("=" * 35)
    
    # Test scaling behavior
    sizes = [50, 100, 500, 1000, 2000]
    
    for size in sizes:
        test_data = generate_test_data(size)
        
        with profiling_session(f"scaling_{size}", enable_library_instrumentation=False):
            # Quick encode/decode cycle
            encoded = JSONTablesEncoder.from_records(test_data)
            decoded = JSONTablesDecoder.to_records(encoded)
        
        # Save scaling results
        save_profile_results(f'benchmarks/results/profile_scaling_{size}.json')

def run_comprehensive_profile():
    """Run comprehensive profiling analysis."""
    print("🚀 JSON-Tables Comprehensive Performance Profile")
    print("=" * 55)
    
    # Ensure results directory exists
    os.makedirs('benchmarks/results', exist_ok=True)
    
    # Run all profiling tests
    profile_encoding_operations()
    profile_decoding_operations()
    profile_append_operations() 
    profile_library_overhead()
    analyze_performance_patterns()
    
    print("\n✅ Comprehensive profiling complete!")
    print("📁 Results saved to benchmarks/results/profile_*.json")
    
    # Generate summary report
    generate_summary_report()

def generate_summary_report():
    """Generate a summary report from all profiling results."""
    print("\n📋 Generating Summary Report")
    print("=" * 30)
    
    import json
    import glob
    
    # Collect all profile results
    profile_files = glob.glob('benchmarks/results/profile_*.json')
    
    summary_data = {
        'encoding_performance': {},
        'decoding_performance': {},
        'append_performance': {},
        'scaling_analysis': {},
        'library_overhead': {}
    }
    
    for file_path in profile_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            filename = os.path.basename(file_path)
            
            if 'encoding' in filename:
                size = filename.split('_')[2].split('.')[0]
                summary_data['encoding_performance'][size] = {
                    'total_time': data['summary']['total_time_ms'],
                    'total_calls': data['summary']['total_calls'],
                    'top_operations': data['operations'][:3]
                }
            elif 'decoding' in filename:
                size = filename.split('_')[2].split('.')[0]
                summary_data['decoding_performance'][size] = {
                    'total_time': data['summary']['total_time_ms'],
                    'total_calls': data['summary']['total_calls'],
                    'top_operations': data['operations'][:3]
                }
            elif 'append' in filename:
                size = filename.split('_')[2].split('.')[0]
                summary_data['append_performance'][size] = {
                    'total_time': data['summary']['total_time_ms'],
                    'total_calls': data['summary']['total_calls'],
                    'top_operations': data['operations'][:3]
                }
            elif 'scaling' in filename:
                size = filename.split('_')[2].split('.')[0]
                summary_data['scaling_analysis'][size] = {
                    'total_time': data['summary']['total_time_ms'],
                    'avg_per_call': data['summary']['avg_time_per_call_ms']
                }
            elif 'library_overhead' in filename:
                summary_data['library_overhead'] = {
                    'total_time': data['summary']['total_time_ms'],
                    'library_calls': [op for op in data['operations'] if 'pandas' in op['name'] or 'json' in op['name']]
                }
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Save summary report
    with open('benchmarks/results/performance_summary_report.json', 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    # Print key insights
    print_key_insights(summary_data)

def print_key_insights(summary_data):
    """Print key performance insights."""
    print("\n🎯 Key Performance Insights")
    print("=" * 30)
    
    # Encoding performance trends
    if summary_data['encoding_performance']:
        print("\n📊 Encoding Performance:")
        for size, data in summary_data['encoding_performance'].items():
            print(f"  {size} rows: {data['total_time']:.2f}ms total")
    
    # Decoding performance trends
    if summary_data['decoding_performance']:
        print("\n📋 Decoding Performance:")
        for size, data in summary_data['decoding_performance'].items():
            print(f"  {size} rows: {data['total_time']:.2f}ms total")
    
    # Scaling analysis
    if summary_data['scaling_analysis']:
        print("\n📈 Scaling Behavior:")
        sizes = sorted(summary_data['scaling_analysis'].keys(), key=int)
        for size in sizes:
            data = summary_data['scaling_analysis'][size]
            print(f"  {size} rows: {data['avg_per_call']:.2f}ms per operation")
    
    # Library overhead
    if summary_data['library_overhead']:
        print("\n⚙️ Library Overhead:")
        lib_calls = summary_data['library_overhead']['library_calls'][:5]
        for call in lib_calls:
            print(f"  {call['name']}: {call['total_time_ms']:.2f}ms")

if __name__ == "__main__":
    run_comprehensive_profile() 