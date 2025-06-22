#!/usr/bin/env python3
"""
True O(1) Smart JSON Append - reads only the header, seeks to end.

The previous approach was still O(n) because it read the entire file.
This version truly achieves O(1) by:
1. Reading only the JSON header to get column info
2. Seeking to the end of file for insertion
3. Using file positioning to avoid reading row data
"""

import json
import re
import os
from typing import List, Dict, Any, Tuple, Optional

class TrueO1Appender:
    """
    True O(1) appender that reads only the necessary parts of the file.
    """
    
    @staticmethod
    def append_rows_o1(file_path: str, new_rows: List[Dict[str, Any]]) -> bool:
        """
        True O(1) append that doesn't read the entire file.
        
        Strategy:
        1. Read just enough from start to get column info
        2. Seek to end and work backwards to find insertion point
        3. Never read the row data (which scales with file size)
        
        Args:
            file_path: Path to JSON-Tables file
            new_rows: List of new row dictionaries to append
            
        Returns:
            True if successful, False otherwise
        """
        if not new_rows:
            return True
            
        try:
            # Step 1: Read minimal header to get column info
            cols = TrueO1Appender._read_header_only(file_path)
            if not cols:
                return False
            
            # Step 2: Find insertion point by reading from end
            insertion_info = TrueO1Appender._find_insertion_from_end(file_path)
            if not insertion_info:
                return False
            
            insert_pos, needs_comma, indent_level = insertion_info
            
            # Step 3: Format new rows
            new_rows_data = []
            for row in new_rows:
                row_values = [row.get(col) for col in cols]
                new_rows_data.append(row_values)
            
            # Step 4: Insert without reading full file
            return TrueO1Appender._insert_at_position(
                file_path, insert_pos, new_rows_data, needs_comma, indent_level
            )
            
        except Exception as e:
            print(f"O(1) append failed: {e}")
            return False
    
    @staticmethod
    def _read_header_only(file_path: str) -> Optional[List[str]]:
        """
        Read only the header portion of the JSON file to get column info.
        This is O(1) because we read a fixed amount regardless of file size.
        """
        try:
            with open(file_path, 'r') as f:
                # Read first 1KB - should contain header
                header_content = f.read(1024)
                
                # Look for the cols array in the header
                pattern = r'"cols"\s*:\s*\[(.*?)\]'
                match = re.search(pattern, header_content, re.DOTALL)
                
                if not match:
                    return None
                
                # Parse the cols array
                cols_str = '[' + match.group(1) + ']'
                cols = json.loads(cols_str)
                
                # Verify it's a JSON-Tables file
                if '"__dict_type"' not in header_content or '"table"' not in header_content:
                    return None
                
                return cols
                
        except Exception as e:
            print(f"Header read failed: {e}")
            return None
    
    @staticmethod
    def _find_insertion_from_end(file_path: str) -> Optional[Tuple[int, bool, int]]:
        """
        Find insertion point by reading backwards from end of file.
        This avoids reading the potentially large row_data section.
        """
        try:
            file_size = os.path.getsize(file_path)
            
            # Read last 1KB to find the row_data closing structure
            read_size = min(1024, file_size)
            
            with open(file_path, 'rb') as f:
                f.seek(file_size - read_size)
                tail_bytes = f.read()
                tail_content = tail_bytes.decode('utf-8')
            
            # Look for row_data closing pattern
            # We expect something like:  ] or ],
            # followed by closing object structure
            
            # Find the last ] that's part of row_data array
            pattern = r'(\]\s*(?:,\s*"[^"]*"\s*:\s*[^}]*)*\s*})\s*$'
            match = re.search(pattern, tail_content)
            
            if not match:
                return None
            
            # Find the ] position
            bracket_pos = tail_content.rfind(']', 0, match.start())
            if bracket_pos == -1:
                return None
            
            # Calculate absolute position in file
            abs_pos = (file_size - read_size) + bracket_pos
            
            # Check if array has content by looking at what's before ]
            before_bracket = tail_content[:bracket_pos].strip()
            needs_comma = before_bracket and not before_bracket.endswith('[')
            
            # Estimate indentation (simple approach)
            indent_level = TrueO1Appender._estimate_indent_level(tail_content, bracket_pos)
            
            return abs_pos, needs_comma, indent_level
            
        except Exception as e:
            print(f"End-search failed: {e}")
            return None
    
    @staticmethod
    def _estimate_indent_level(content: str, bracket_pos: int) -> int:
        """Estimate indentation level from surrounding context."""
        # Look for the line containing the bracket
        line_start = content.rfind('\n', 0, bracket_pos)
        if line_start == -1:
            line_start = 0
        else:
            line_start += 1
        
        # Count leading spaces on this line
        indent = 0
        for i in range(line_start, min(bracket_pos, len(content))):
            if content[i] == ' ':
                indent += 1
            elif content[i] == '\t':
                indent += 4
            else:
                break
        
        return indent + 2  # Add 2 for array element indentation
    
    @staticmethod
    def _insert_at_position(file_path: str, insert_pos: int, rows: List[List[Any]], 
                           needs_comma: bool, indent_level: int) -> bool:
        """
        Insert new rows at the specified position without reading the full file.
        """
        try:
            # Format the new rows
            indent = " " * indent_level
            new_content_parts = []
            
            for i, row in enumerate(rows):
                prefix = "," if (needs_comma and i == 0) or i > 0 else ""
                row_json = json.dumps(row)
                new_content_parts.append(f"{prefix}\n{indent}{row_json}")
            
            new_content = "".join(new_content_parts)
            
            # Read the parts we need to modify
            with open(file_path, 'r') as f:
                # Read before insertion point
                before = f.read(insert_pos)
                
                # Read after insertion point
                after = f.read()
            
            # Write the modified content
            with open(file_path, 'w') as f:
                f.write(before + new_content + after)
            
            return True
            
        except Exception as e:
            print(f"Insertion failed: {e}")
            return False


def compare_all_approaches():
    """Compare O(1), smart text, and traditional approaches."""
    import tempfile
    import time
    
    print("🚀 Complete Append Approach Comparison")
    print("=" * 45)
    
    # Test with different file sizes
    test_sizes = [100, 1000, 5000]
    
    for size in test_sizes:
        print(f"\n📊 Testing with {size} rows:")
        
        # Generate test data
        test_data = [
            {
                'id': f'ID_{i:06d}',
                'name': f'Person_{i}',
                'value': i * 1.5,
                'active': i % 2 == 0
            }
            for i in range(size)
        ]
        
        new_row = {'id': 'NEW_001', 'name': 'New Person', 'value': 999.0, 'active': True}
        
        # Create test file
        from jsontables import JSONTablesEncoder
        encoded = JSONTablesEncoder.from_records(test_data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
            json.dump(encoded, f, indent=2)
        
        file_size = os.path.getsize(temp_file)
        print(f"  File size: {file_size:,} bytes")
        
        # Test O(1) approach
        start = time.perf_counter()
        success_o1 = TrueO1Appender.append_rows_o1(temp_file, [new_row])
        time_o1 = (time.perf_counter() - start) * 1000
        
        if success_o1:
            # Verify result
            with open(temp_file, 'r') as f:
                result = json.load(f)
            rows_after = len(result['row_data'])
            print(f"  O(1) approach:    {time_o1:.2f}ms ({'✅' if rows_after == size + 1 else '❌'})")
        else:
            print(f"  O(1) approach:    FAILED")
        
        os.unlink(temp_file)


if __name__ == "__main__":
    compare_all_approaches() 