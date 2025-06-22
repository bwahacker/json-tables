#!/usr/bin/env python3
"""
Test and benchmark efficient JSON-Tables append functionality.
"""

import sys
import os
import time
import json
import tempfile
import statistics

# Add parent directory to path to import jsontables
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from jsontables import JSONTablesEncoder, JSONTablesDecoder
from jsontables.core import JSONTablesAppender, append_to_json_table_file

def time_operation(func, *args, **kwargs):
    """Time a single operation"""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return result, (end - start) * 1000  # Convert to milliseconds

def test_efficient_append():
    """Test that the efficient append functionality works correctly."""
    print("🧪 Testing Efficient JSON-Tables Append")
    print("=" * 50)
    
    # Create initial test data
    initial_data = [
        {'name': 'Alice', 'age': 30, 'city': 'New York'},
        {'name': 'Bob', 'age': 25, 'city': 'Boston'},
        {'name': 'Carol', 'age': 35, 'city': 'Chicago'}
    ]
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = f.name
        
        # Write initial JSON-Tables data
        encoded = JSONTablesEncoder.from_records(initial_data)
        json.dump(encoded, f)
    
    print(f"✅ Created initial file with {len(initial_data)} rows")
    
    # Test appending new rows
    new_rows = [
        {'name': 'David', 'age': 28, 'city': 'Denver'},
        {'name': 'Eve', 'age': 32, 'city': 'Seattle'}
    ]
    
    # Test efficient append
    success = append_to_json_table_file(temp_file, new_rows)
    
    if success:
        print(f"✅ Successfully appended {len(new_rows)} rows")
        
        # Verify the result
        with open(temp_file, 'r') as f:
            result_data = json.load(f)
        
        decoded = JSONTablesDecoder.to_records(result_data)
        
        print(f"✅ Verification: File now contains {len(decoded)} rows")
        
        # Check that all original + new data is present
        expected_names = ['Alice', 'Bob', 'Carol', 'David', 'Eve']
        actual_names = [row['name'] for row in decoded]
        
        if actual_names == expected_names:
            print("✅ All data preserved correctly")
        else:
            print(f"❌ Data mismatch: expected {expected_names}, got {actual_names}")
    else:
        print("❌ Append operation failed")
    
    # Cleanup
    os.unlink(temp_file)
    
    return success

def benchmark_append_methods():
    """Benchmark efficient append vs traditional full rewrite."""
    print("\n⚡ Benchmarking Append Methods")
    print("=" * 40)
    
    # Test with different dataset sizes
    test_sizes = [100, 500, 1000, 2000]
    
    for size in test_sizes:
        print(f"\n📊 Testing with {size} initial rows:")
        
        # Generate test data
        initial_data = []
        for i in range(size):
            initial_data.append({
                'id': f'ID_{i:06d}',
                'name': f'Person_{i}',
                'value': i * 2.5,
                'category': ['A', 'B', 'C'][i % 3]
            })
        
        new_rows = [
            {'id': 'NEW_001', 'name': 'New Person', 'value': 999.9, 'category': 'D'}
        ]
        
        # Test 1: Traditional full rewrite approach
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            traditional_file = f.name
            encoded = JSONTablesEncoder.from_records(initial_data)
            json.dump(encoded, f)
        
        def traditional_append():
            # Read entire file
            with open(traditional_file, 'r') as f:
                data = json.load(f)
            
            # Decode to records
            records = JSONTablesDecoder.to_records(data)
            
            # Add new row
            records.extend(new_rows)
            
            # Re-encode and write
            encoded = JSONTablesEncoder.from_records(records)
            with open(traditional_file, 'w') as f:
                json.dump(encoded, f)
        
        _, traditional_time = time_operation(traditional_append)
        
        # Test 2: Efficient append approach
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            efficient_file = f.name
            encoded = JSONTablesEncoder.from_records(initial_data)
            json.dump(encoded, f)
        
        def efficient_append():
            return append_to_json_table_file(efficient_file, new_rows)
        
        success, efficient_time = time_operation(efficient_append)
        
        # Results
        if success:
            speedup = traditional_time / efficient_time
            print(f"  Traditional approach: {traditional_time:.2f}ms")
            print(f"  Efficient approach:   {efficient_time:.2f}ms")
            print(f"  Speedup: {speedup:.1f}x faster")
        else:
            print(f"  Traditional approach: {traditional_time:.2f}ms")
            print(f"  Efficient approach:   FAILED")
        
        # Cleanup
        os.unlink(traditional_file)
        os.unlink(efficient_file)

def explore_append_friendly_format():
    """Explore an even more append-friendly JSON-Tables variant."""
    print("\n🚀 Exploring Append-Friendly Format Design")
    print("=" * 45)
    
    print("Current JSON-Tables structure:")
    print('''{
  "__dict_type": "table",
  "cols": ["name", "age"],
  "row_data": [
    ["Alice", 30],
    ["Bob", 25]
  ]
}''')
    
    print("\nPossible append-friendly alternatives:")
    
    print("\n1. JSONL-style with header:")
    print('{"__dict_type": "table", "cols": ["name", "age"]}')
    print('["Alice", 30]')
    print('["Bob", 25]')
    print('["Carol", 35]  # <- Can append this line easily')
    
    print("\n2. Streaming JSON-Tables:")
    print('''{
  "__dict_type": "table",
  "cols": ["name", "age"],
  "rows": [
    ["Alice", 30],
    ["Bob", 25],
    // EOF marker allows easy append
  ]
}''')
    
    print("\n3. Append log format:")
    print('''{
  "__dict_type": "table",
  "cols": ["name", "age"],
  "row_data": [["Alice", 30], ["Bob", 25]],
  "appends": [
    {"timestamp": "2024-01-01", "rows": [["Carol", 35]]},
    // New appends can be added here
  ]
}''')
    
    print("\nThe current efficient append modifies the JSON structure directly,")
    print("achieving near-optimal performance without changing the format.")

if __name__ == "__main__":
    # Run tests
    success = test_efficient_append()
    
    if success:
        benchmark_append_methods()
    
    explore_append_friendly_format() 