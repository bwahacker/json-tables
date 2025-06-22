#!/usr/bin/env python3
"""
Debug and demonstrate JSON append approaches.

Key insight: For JSON-Tables, traditional JSON parsing is actually quite efficient
for small to medium files, and text manipulation adds overhead without much benefit
until you get to very large files.
"""

import json
import tempfile
import time
import os

def create_sample_file(num_rows: int):
    """Create a sample JSON-Tables file."""
    data = {
        "__dict_type": "table",
        "cols": ["id", "name", "value"],
        "row_data": [
            [f"ID_{i}", f"Name_{i}", i * 10]
            for i in range(num_rows)
        ]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(data, f, indent=2)
        return f.name

def traditional_append_demo(file_path: str):
    """Demonstrate traditional append."""
    print("🔍 Traditional Approach Analysis:")
    
    start = time.perf_counter()
    
    # Read and parse - this is what we thought was expensive
    with open(file_path, 'r') as f:
        data = json.load(f)
    parse_time = time.perf_counter()
    
    # Modify in memory - this is very fast
    new_row = ["NEW_001", "New Name", 999]
    data["row_data"].append(new_row)
    modify_time = time.perf_counter()
    
    # Write back - this is what actually takes time
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    write_time = time.perf_counter()
    
    total_time = (write_time - start) * 1000
    parse_duration = (parse_time - start) * 1000
    modify_duration = (modify_time - parse_time) * 1000
    write_duration = (write_time - modify_time) * 1000
    
    print(f"  Parse JSON:  {parse_duration:.2f}ms")
    print(f"  Modify data: {modify_duration:.3f}ms") 
    print(f"  Write JSON:  {write_duration:.2f}ms")
    print(f"  Total:       {total_time:.2f}ms")
    
    return total_time

def analyze_bottlenecks():
    """Analyze where time is actually spent in JSON operations."""
    print("🚀 JSON-Tables Append Performance Analysis")
    print("=" * 50)
    
    file_sizes = [100, 1000, 5000, 10000]
    
    for size in file_sizes:
        print(f"\n📊 {size} rows:")
        
        # Create test file
        test_file = create_sample_file(size)
        file_size = os.path.getsize(test_file)
        print(f"  File size: {file_size:,} bytes")
        
        # Analyze traditional approach
        total_time = traditional_append_demo(test_file)
        
        # Calculate throughput
        throughput = file_size / (total_time / 1000) / (1024 * 1024)  # MB/s
        print(f"  Throughput: {throughput:.1f} MB/s")
        
        # Cleanup
        os.unlink(test_file)

def key_insights():
    """Share key insights about JSON append performance."""
    print("\n🎯 Key Performance Insights")
    print("=" * 35)
    
    print("\n💡 Why Traditional JSON Parsing is Fast:")
    print("  • Modern JSON parsers are highly optimized")
    print("  • Python's json module is implemented in C")
    print("  • Memory operations are faster than file I/O")
    print("  • Data structures are already optimized for manipulation")
    
    print("\n⚠️ Why Text Manipulation Can Be Slower:")
    print("  • File I/O overhead (read + write)")
    print("  • String manipulation overhead")
    print("  • Regex parsing complexity")
    print("  • Need to maintain JSON validity")
    
    print("\n📈 When Each Approach Wins:")
    print("  Traditional JSON:")
    print("    ✅ Small to medium files (<1MB)")
    print("    ✅ Occasional appends")
    print("    ✅ Complex data structures")
    print("    ✅ Need validation/error handling")
    
    print("\n  Text Manipulation:")
    print("    ✅ Very large files (>10MB)")
    print("    ✅ Frequent appends (streaming)")
    print("    ✅ Simple, regular data patterns")
    print("    ✅ When JSON parsing becomes bottleneck")
    
    print("\n🚀 Optimal Strategy:")
    print("  • Use traditional JSON for most cases")
    print("  • Switch to text manipulation for large files or high-frequency appends")
    print("  • Consider JSONL for true streaming scenarios")
    print("  • Benchmark with your actual data sizes and patterns")

def demonstrate_scaling():
    """Demonstrate how performance scales with file size."""
    print("\n📈 Performance Scaling Demonstration")
    print("=" * 40)
    
    sizes = [100, 500, 1000, 2000, 5000]
    times = []
    
    for size in sizes:
        test_file = create_sample_file(size)
        
        start = time.perf_counter()
        with open(test_file, 'r') as f:
            data = json.load(f)
        data["row_data"].append(["NEW", "New", 999])
        with open(test_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        duration = (time.perf_counter() - start) * 1000
        times.append(duration)
        
        os.unlink(test_file)
    
    print("Size   | Time    | Time/Row")
    print("-------|---------|----------")
    for i, (size, time_ms) in enumerate(zip(sizes, times)):
        time_per_row = time_ms / size
        print(f"{size:5d}  | {time_ms:5.1f}ms | {time_per_row:6.3f}ms")
    
    # Calculate if it's linear
    if len(times) >= 2:
        ratio = times[-1] / times[0]
        size_ratio = sizes[-1] / sizes[0]
        linearity = ratio / size_ratio
        
        print(f"\nScaling analysis:")
        print(f"  Time ratio: {ratio:.1f}x")
        print(f"  Size ratio: {size_ratio:.1f}x") 
        print(f"  Linearity factor: {linearity:.2f}")
        if linearity < 1.2:
            print("  ✅ Nearly linear scaling - traditional approach is fine")
        else:
            print("  ⚠️ Super-linear scaling - consider text manipulation for large files")

if __name__ == "__main__":
    analyze_bottlenecks()
    key_insights()
    demonstrate_scaling() 