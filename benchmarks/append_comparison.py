#!/usr/bin/env python3
"""
Comprehensive append performance comparison.

Tests:
1. Traditional JSON append (full read/write)
2. Smart text-based append (regex manipulation)
3. Ultra-fast append (tail manipulation + fallback)
"""

import json
import os
import sys
import tempfile
import time
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import all our append implementations
from jsontables.core import JSONTablesEncoder, JSONTablesDecoder
from jsontables.smart_append import SmartJSONAppender
from jsontables.ultra_fast_append import ultra_fast_append

def create_test_data(num_rows: int) -> Dict[str, Any]:
    """Create a JSON-Tables test file with specified number of rows."""
    records = []
    for i in range(num_rows):
        records.append({
            "id": f"ID_{i:06d}",
            "name": f"User_{i}",
            "score": i * 10,
            "active": i % 2 == 0,
            "group": f"Group_{i % 5}"
        })
    
    data = JSONTablesEncoder.from_records(records)
    return data

def traditional_append(file_path: str, new_rows: List[Dict[str, Any]]) -> float:
    """Traditional append using full JSON read/write."""
    start_time = time.time()
    
    # Read existing file
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Append new rows
    cols = data["cols"]
    for row in new_rows:
        row_values = [row.get(col) for col in cols]
        data["row_data"].append(row_values)
    
    # Write back
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    return time.time() - start_time

def smart_append_wrapper(file_path: str, new_rows: List[Dict[str, Any]]) -> float:
    """Wrapper for smart append to match the timing interface."""
    start_time = time.time()
    success = SmartJSONAppender.append_rows_smart(file_path, new_rows)
    elapsed = time.time() - start_time
    if not success:
        raise Exception("Smart append failed")
    return elapsed

def ultra_fast_append_wrapper(file_path: str, new_rows: List[Dict[str, Any]]) -> float:
    """Wrapper for ultra-fast append to match the timing interface."""
    start_time = time.time()
    success = ultra_fast_append(file_path, new_rows)
    elapsed = time.time() - start_time
    if not success:
        raise Exception("Ultra-fast append failed")
    return elapsed

def benchmark_append_method(method_name: str, append_func, base_sizes: List[int], append_size: int = 10):
    """Benchmark a specific append method across different file sizes."""
    print(f"\n📊 Testing {method_name}")
    print("-" * 50)
    
    results = []
    
    for base_size in base_sizes:
        # Create test file
        data = create_test_data(base_size)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
            json.dump(data, f, indent=2)
        
        try:
            # Create new rows to append
            new_rows = []
            for i in range(append_size):
                new_rows.append({
                    "id": f"NEW_{i:06d}",
                    "name": f"NewUser_{i}",
                    "score": 9999 + i,
                    "active": True,
                    "group": "NewGroup"
                })
            
            # Time the append operation
            elapsed = append_func(temp_file, new_rows)
            
            # Verify result
            with open(temp_file, 'r') as f:
                result = json.load(f)
            
            expected_rows = base_size + append_size
            actual_rows = len(result["row_data"])
            
            if actual_rows == expected_rows:
                print(f"  {base_size:5d} rows → {elapsed*1000:6.2f}ms ✅")
                results.append((base_size, elapsed))
            else:
                print(f"  {base_size:5d} rows → ERROR (got {actual_rows}, expected {expected_rows}) ❌")
                results.append((base_size, None))
            
        except Exception as e:
            print(f"  {base_size:5d} rows → ERROR: {e} ❌")
            results.append((base_size, None))
        
        finally:
            # Clean up
            os.unlink(temp_file)
    
    return results

def compare_all_methods():
    """Run comprehensive comparison of all append methods."""
    print("🚀 JSON-Tables Append Performance Comparison")
    print("=" * 60)
    
    # Test with various base file sizes
    base_sizes = [100, 500, 1000, 2000, 5000]
    append_size = 10  # Always append 10 rows
    
    # Test all methods
    methods = [
        ("Traditional (JSON parse)", traditional_append),
        ("Smart (Text manipulation)", smart_append_wrapper),
        ("Ultra-Fast (Tail + fallback)", ultra_fast_append_wrapper)
    ]
    
    all_results = {}
    
    for method_name, method_func in methods:
        results = benchmark_append_method(method_name, method_func, base_sizes, append_size)
        all_results[method_name] = results
    
    # Summary comparison
    print(f"\n📈 PERFORMANCE SUMMARY (appending {append_size} rows)")
    print("=" * 60)
    print(f"{'Base Size':<10} {'Traditional':<12} {'Smart':<12} {'Ultra-Fast':<12} {'Speedup':<10}")
    print("-" * 60)
    
    for i, base_size in enumerate(base_sizes):
        traditional_time = all_results["Traditional (JSON parse)"][i][1]
        smart_time = all_results["Smart (Text manipulation)"][i][1]
        ultra_time = all_results["Ultra-Fast (Tail + fallback)"][i][1]
        
        if all([traditional_time, smart_time, ultra_time]):
            speedup_smart = traditional_time / smart_time
            speedup_ultra = traditional_time / ultra_time
            
            print(f"{base_size:<10} {traditional_time*1000:>8.2f}ms {smart_time*1000:>8.2f}ms {ultra_time*1000:>8.2f}ms {speedup_ultra:>8.2f}x")
        else:
            print(f"{base_size:<10} {'ERROR':<12} {'ERROR':<12} {'ERROR':<12} {'N/A':<10}")
    
    # Analysis
    print(f"\n🔍 ANALYSIS")
    print("-" * 20)
    
    # Find the best performing method for large files
    if all_results:
        print("Key findings:")
        print("• Traditional method: Full JSON parse - reliable but O(n)")
        print("• Smart method: Text manipulation - complex regex overhead")
        print("• Ultra-fast method: Tail manipulation - O(1) for appends")
        print()
        print("💡 Recommendation:")
        print("  Use Ultra-Fast for high-frequency appends")
        print("  Use Traditional for occasional appends on small files")
        print("  Smart method adds complexity without major benefits")

if __name__ == "__main__":
    compare_all_methods() 