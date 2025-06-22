#!/usr/bin/env python3
"""
Streaming JSON-Tables: A format designed for efficient append operations.

This explores an alternative JSON-Tables format that enables true O(1) append
operations while maintaining JSON compatibility and human readability.
"""

import json
import time
import tempfile
import os
from typing import List, Dict, Any

class StreamingJSONTables:
    """
    Streaming JSON-Tables format for efficient append operations.
    
    Format 1: JSONL-style with header
    {"__dict_type": "table", "cols": ["name", "age"]}
    ["Alice", 30]
    ["Bob", 25]
    
    Format 2: Append-optimized JSON
    {
      "__dict_type": "streaming_table",
      "cols": ["name", "age"],
      "rows": [
        ["Alice", 30],
        ["Bob", 25]
      ]
    }
    # Append marker: new rows can be added after the closing ]
    """
    
    @staticmethod
    def create_jsonl_format(file_path: str, records: List[Dict[str, Any]]) -> None:
        """Create a JSONL-style JSON-Tables file."""
        if not records:
            return
        
        cols = list(records[0].keys())
        
        with open(file_path, 'w') as f:
            # Write header
            header = {"__dict_type": "table", "cols": cols}
            f.write(json.dumps(header) + '\n')
            
            # Write rows
            for record in records:
                row = [record.get(col) for col in cols]
                f.write(json.dumps(row) + '\n')
    
    @staticmethod
    def append_to_jsonl_format(file_path: str, new_records: List[Dict[str, Any]]) -> float:
        """Append to JSONL format - true O(1) operation."""
        if not new_records:
            return 0.0
        
        start_time = time.perf_counter()
        
        # Read header to get column order
        with open(file_path, 'r') as f:
            header_line = f.readline()
            header = json.loads(header_line)
            cols = header.get("cols", [])
        
        # Append new rows - this is O(1)!
        with open(file_path, 'a') as f:
            for record in new_records:
                row = [record.get(col) for col in cols]
                f.write(json.dumps(row) + '\n')
        
        end_time = time.perf_counter()
        return (end_time - start_time) * 1000  # milliseconds
    
    @staticmethod
    def read_jsonl_format(file_path: str) -> List[Dict[str, Any]]:
        """Read JSONL format back to records."""
        records = []
        
        with open(file_path, 'r') as f:
            # Read header
            header_line = f.readline()
            header = json.loads(header_line)
            cols = header.get("cols", [])
            
            # Read rows
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    record = dict(zip(cols, row))
                    records.append(record)
        
        return records
    
    @staticmethod
    def create_append_optimized_format(file_path: str, records: List[Dict[str, Any]]) -> None:
        """Create an append-optimized JSON format with special markers."""
        if not records:
            return
        
        cols = list(records[0].keys())
        row_data = []
        
        for record in records:
            row = [record.get(col) for col in cols]
            row_data.append(row)
        
        # Create the base structure
        table_data = {
            "__dict_type": "streaming_table",
            "cols": cols,
            "rows": row_data,
            "__append_marker": "ROWS_END"
        }
        
        with open(file_path, 'w') as f:
            json.dump(table_data, f, indent=2)
    
    @staticmethod
    def append_to_optimized_format(file_path: str, new_records: List[Dict[str, Any]]) -> float:
        """Append to optimized format using file seeking."""
        if not new_records:
            return 0.0
        
        start_time = time.perf_counter()
        
        # Read the file to understand structure
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Parse to get column info
        data = json.loads(content)
        cols = data.get("cols", [])
        
        # Convert new records to rows
        new_rows = []
        for record in new_records:
            row = [record.get(col) for col in cols]
            new_rows.append(row)
        
        # Find the position to insert - look for the __append_marker
        marker_pos = content.find('"__append_marker"')
        if marker_pos == -1:
            # Fallback - find the end of the rows array
            rows_end = content.rfind(']', 0, marker_pos if marker_pos > 0 else len(content))
            if rows_end == -1:
                return 0.0  # Failed
            
            # Insert new rows before the closing ]
            new_rows_json = [json.dumps(row) for row in new_rows]
            
            # Check if array is empty
            if '"rows": []' in content:
                # Empty array
                new_content = content[:rows_end] + ',\n    '.join([''] + new_rows_json) + content[rows_end:]
            else:
                # Array has content
                new_content = content[:rows_end] + ',\n    ' + ',\n    '.join(new_rows_json) + content[rows_end:]
            
            with open(file_path, 'w') as f:
                f.write(new_content)
        
        end_time = time.perf_counter()
        return (end_time - start_time) * 1000

def benchmark_streaming_formats():
    """Benchmark the streaming formats against traditional approaches."""
    print("🚀 Benchmarking Streaming JSON-Tables Formats")
    print("=" * 50)
    
    # Test data
    test_sizes = [100, 500, 1000, 2000, 5000]
    
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
        
        new_row = {'id': 'NEW_001', 'name': 'New Person', 'value': 999.9, 'category': 'D'}
        
        # Test 1: JSONL format
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            jsonl_file = f.name
        
        StreamingJSONTables.create_jsonl_format(jsonl_file, initial_data)
        jsonl_time = StreamingJSONTables.append_to_jsonl_format(jsonl_file, [new_row])
        
        # Verify JSONL worked
        jsonl_result = StreamingJSONTables.read_jsonl_format(jsonl_file)
        jsonl_success = len(jsonl_result) == size + 1
        
        # Test 2: Traditional JSON-Tables (for comparison)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            traditional_file = f.name
        
        # Simulate traditional append time
        start = time.perf_counter()
        # Read + parse + modify + write simulation
        table_data = {
            "__dict_type": "table",
            "cols": list(initial_data[0].keys()),
            "row_data": [[row[col] for col in initial_data[0].keys()] for row in initial_data]
        }
        with open(traditional_file, 'w') as f:
            json.dump(table_data, f)
        
        # Simulate append
        with open(traditional_file, 'r') as f:
            data = json.load(f)
        
        new_row_data = [new_row[col] for col in data["cols"]]
        data["row_data"].append(new_row_data)
        
        with open(traditional_file, 'w') as f:
            json.dump(data, f)
        
        traditional_time = (time.perf_counter() - start) * 1000
        
        # Results
        print(f"  JSONL format:       {jsonl_time:.3f}ms {'✅' if jsonl_success else '❌'}")
        print(f"  Traditional format: {traditional_time:.3f}ms")
        
        if jsonl_success and jsonl_time > 0:
            speedup = traditional_time / jsonl_time
            print(f"  JSONL speedup: {speedup:.1f}x faster")
        
        # Cleanup
        os.unlink(jsonl_file)
        os.unlink(traditional_file)

def demonstrate_formats():
    """Demonstrate the different streaming formats."""
    print("\n📝 Format Demonstrations")
    print("=" * 30)
    
    sample_data = [
        {'name': 'Alice', 'age': 30, 'city': 'NYC'},
        {'name': 'Bob', 'age': 25, 'city': 'LA'}
    ]
    
    print("\n1. Standard JSON-Tables:")
    standard = {
        "__dict_type": "table",
        "cols": ["name", "age", "city"],
        "row_data": [["Alice", 30, "NYC"], ["Bob", 25, "LA"]]
    }
    print(json.dumps(standard, indent=2))
    
    print("\n2. JSONL-style (truly append-friendly):")
    print('{"__dict_type": "table", "cols": ["name", "age", "city"]}')
    print('["Alice", 30, "NYC"]')
    print('["Bob", 25, "LA"]')
    print('["Carol", 35, "CHI"]  # <- O(1) append!')
    
    print("\n3. Hybrid approach (JSON + append log):")
    hybrid = {
        "__dict_type": "table", 
        "cols": ["name", "age", "city"],
        "row_data": [["Alice", 30, "NYC"], ["Bob", 25, "LA"]],
        "append_log": [
            {"timestamp": "2024-01-01T10:00:00Z", "rows": [["Carol", 35, "CHI"]]},
            # New appends here - O(1) operation
        ]
    }
    print(json.dumps(hybrid, indent=2))

if __name__ == "__main__":
    demonstrate_formats()
    benchmark_streaming_formats() 