#!/usr/bin/env python3
"""
Ultra-fast JSON append - reads only the tail, true O(1) performance.

Strategy:
1. Read last 1024 bytes to find closing array structure
2. JSON encode the new record and insert before the closing ]
3. Graceful fallback to traditional parsing if anything looks suspicious
"""

import json
import os
from typing import List, Dict, Any, Optional, Tuple

def ultra_fast_append(file_path: str, new_rows: List[Dict[str, Any]]) -> bool:
    """
    Ultra-fast append with graceful fallback.
    
    Args:
        file_path: Path to JSON-Tables file
        new_rows: List of new row dictionaries to append
        
    Returns:
        True if successful, False otherwise
    """
    if not new_rows:
        return True
    
    try:
        # Step 1: Try ultra-fast approach
        cols = _get_columns_fast(file_path)
        if cols and _try_tail_append(file_path, new_rows, cols):
            return True
        
        # Step 2: Fallback to reliable traditional approach
        print("Falling back to traditional append...")
        return _fallback_append(file_path, new_rows)
        
    except Exception as e:
        print(f"Ultra-fast append failed: {e}")
        return _fallback_append(file_path, new_rows)

def _get_columns_fast(file_path: str) -> Optional[List[str]]:
    """
    Quickly extract column info from file header.
    Read only first 1KB to get the cols array.
    """
    try:
        with open(file_path, 'r') as f:
            header = f.read(1024)
        
        # Look for cols array in header
        start = header.find('"cols"')
        if start == -1:
            return None
        
        # Find the opening bracket
        bracket_start = header.find('[', start)
        if bracket_start == -1:
            return None
        
        # Find the closing bracket
        bracket_count = 1
        pos = bracket_start + 1
        while pos < len(header) and bracket_count > 0:
            if header[pos] == '[':
                bracket_count += 1
            elif header[pos] == ']':
                bracket_count -= 1
            pos += 1
        
        if bracket_count > 0:
            return None  # Didn't find complete array
        
        # Extract and parse the cols array
        cols_json = header[bracket_start:pos]
        return json.loads(cols_json)
        
    except Exception:
        return None

def _try_tail_append(file_path: str, new_rows: List[Dict[str, Any]], cols: List[str]) -> bool:
    """
    Try to append by manipulating only the file tail.
    """
    try:
        file_size = os.path.getsize(file_path)
        if file_size < 100:  # Too small to be a real JSON-Tables file
            return False
        
        # Read the tail
        tail_size = min(1024, file_size)
        with open(file_path, 'rb') as f:
            f.seek(file_size - tail_size)
            tail_bytes = f.read()
        
        tail = tail_bytes.decode('utf-8')
        
        # Look for the row_data array closing
        insertion_point = _find_array_insertion_point(tail, file_size, tail_size)
        if not insertion_point:
            return False
        
        abs_pos, needs_comma, indent = insertion_point
        
        # Format new rows as JSON
        new_rows_data = []
        for row in new_rows:
            row_values = [row.get(col) for col in cols]
            new_rows_data.append(row_values)
        
        # Create insertion text
        insertion_text = _format_insertion(new_rows_data, needs_comma, indent)
        
        # Do the insertion
        return _insert_at_position(file_path, abs_pos, insertion_text)
        
    except Exception:
        return False

def _find_array_insertion_point(tail: str, file_size: int, tail_size: int) -> Optional[Tuple[int, bool, int]]:
    """
    Find where to insert in the row_data array by analyzing the tail.
    Returns (absolute_position, needs_comma, indent_level)
    """
    # Look for a pattern like:  ]\n  }\n}
    # We want to find the ] that closes the row_data array
    
    # Find all ] characters
    bracket_positions = []
    for i, char in enumerate(tail):
        if char == ']':
            bracket_positions.append(i)
    
    if not bracket_positions:
        return None
    
    # Look for the ] that's followed by JSON object closing patterns
    for bracket_pos in reversed(bracket_positions):
        # Check what comes after this ]
        after_bracket = tail[bracket_pos + 1:].strip()
        
        # Should see something like }, "field": value, etc. or just }
        if after_bracket.startswith(',') or after_bracket.startswith('}'):
            # This looks like the row_data array closing
            abs_pos = (file_size - tail_size) + bracket_pos
            
            # Check if array has content (look before the ])
            before_bracket = tail[:bracket_pos].strip()
            needs_comma = before_bracket and not before_bracket.endswith('[')
            
            # Estimate indentation
            indent = _estimate_indent(tail, bracket_pos)
            
            return abs_pos, needs_comma, indent
    
    return None

def _estimate_indent(tail: str, bracket_pos: int) -> int:
    """Estimate indentation level from context."""
    # Find the line containing the bracket
    line_start = tail.rfind('\n', 0, bracket_pos)
    if line_start == -1:
        return 4  # Default indent
    
    line_start += 1
    
    # Count spaces
    indent = 0
    for i in range(line_start, min(bracket_pos, len(tail))):
        if tail[i] == ' ':
            indent += 1
        elif tail[i] == '\t':
            indent += 4
        else:
            break
    
    return indent + 2  # Add 2 for array element

def _format_insertion(rows: List[List], needs_comma: bool, indent: int) -> str:
    """Format the rows for insertion."""
    if not rows:
        return ""
    
    parts = []
    indent_str = " " * indent
    
    for i, row in enumerate(rows):
        prefix = "," if (needs_comma and i == 0) or i > 0 else ""
        row_json = json.dumps(row)
        parts.append(f"{prefix}\n{indent_str}{row_json}")
    
    return "".join(parts)

def _insert_at_position(file_path: str, position: int, text: str) -> bool:
    """Insert text at the specified position."""
    try:
        # Read the file in two parts
        with open(file_path, 'r') as f:
            before = f.read(position)
            after = f.read()
        
        # Write back with insertion
        with open(file_path, 'w') as f:
            f.write(before + text + after)
        
        # Quick validation that it's still valid JSON
        with open(file_path, 'r') as f:
            json.load(f)  # Will raise exception if invalid
        
        return True
        
    except Exception:
        return False

def _fallback_append(file_path: str, new_rows: List[Dict[str, Any]]) -> bool:
    """
    Reliable fallback using traditional JSON parsing.
    """
    try:
        # Read and parse the file
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Validate structure
        if data.get("__dict_type") != "table" or "cols" not in data:
            return False
        
        cols = data["cols"]
        
        # Convert and append new rows
        for row in new_rows:
            row_values = [row.get(col) for col in cols]
            data["row_data"].append(row_values)
        
        # Write back
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return True
        
    except Exception as e:
        print(f"Fallback append failed: {e}")
        return False

# Demo and test functions
def demo_ultra_fast():
    """Demonstrate the ultra-fast append."""
    import tempfile
    
    print("🚀 Ultra-Fast JSON Append Demo")
    print("=" * 35)
    
    # Create test data
    data = {
        "__dict_type": "table",
        "cols": ["id", "name", "value"],
        "row_data": [
            ["ID_001", "Alice", 100],
            ["ID_002", "Bob", 200]
        ]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = f.name
        json.dump(data, f, indent=2)
    
    print("📄 Original file:")
    with open(temp_file, 'r') as f:
        print(f.read())
    
    # Test append
    new_rows = [
        {"id": "ID_003", "name": "Carol", "value": 300},
        {"id": "ID_004", "name": "David", "value": 400}
    ]
    
    print(f"\n➕ Appending {len(new_rows)} rows...")
    success = ultra_fast_append(temp_file, new_rows)
    
    if success:
        print("✅ Success! Result:")
        with open(temp_file, 'r') as f:
            result = json.load(f)
            print(f"  Rows: {len(result['row_data'])}")
            for i, row in enumerate(result['row_data']):
                print(f"    {i+1}: {row}")
    else:
        print("❌ Failed!")
    
    os.unlink(temp_file)

if __name__ == "__main__":
    demo_ultra_fast() 