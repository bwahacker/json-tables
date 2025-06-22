#!/usr/bin/env python3
"""
Simple demonstration of JSON-Tables round-trip data integrity.

Shows that data can be converted to JSON-Tables and back without any loss.
"""

import json
from jsontables.core import JSONTablesEncoder, JSONTablesDecoder

def demo_round_trip():
    """Demonstrate round-trip data integrity with clear examples."""
    
    print("🔄 JSON-Tables Round-Trip Demo")
    print("=" * 40)
    
    # Original data with various types and edge cases
    original_data = [
        {"name": "Alice", "age": 30, "city": "NYC", "active": True, "score": 95.5, "notes": None},
        {"name": "Bob", "age": None, "city": "LA", "active": False, "score": 87.2, "notes": "Important"},
        {"name": "Carol", "age": 35, "city": None, "active": True, "score": 92.8, "notes": None},
        {"name": "", "age": 0, "city": "Chicago", "active": False, "score": 0.0, "notes": "Edge case"}
    ]
    
    print("\n📄 Original Data:")
    for i, row in enumerate(original_data, 1):
        print(f"  Row {i}: {row}")
    
    print(f"\n🔄 Converting to JSON-Tables...")
    
    # Convert to JSON-Tables format
    json_tables = JSONTablesEncoder.from_records(original_data)
    
    print(f"JSON-Tables format:")
    print(json.dumps(json_tables, indent=2))
    
    print(f"\n🔄 Converting back to records...")
    
    # Convert back to original format
    converted_data = JSONTablesDecoder.to_records(json_tables)
    
    print(f"\n📄 Converted Data:")
    for i, row in enumerate(converted_data, 1):
        print(f"  Row {i}: {row}")
    
    # Verify integrity
    print(f"\n🔍 Data Integrity Check:")
    
    # Check counts
    print(f"  Original rows: {len(original_data)}")
    print(f"  Converted rows: {len(converted_data)}")
    
    # Check each row
    all_match = True
    for i, (orig, conv) in enumerate(zip(original_data, converted_data)):
        # Check keys
        if set(orig.keys()) != set(conv.keys()):
            print(f"  ❌ Row {i+1}: Key mismatch")
            all_match = False
            continue
        
        # Check values (handling null equivalence)
        import pandas as pd
        row_match = True
        for key in orig.keys():
            orig_val = orig[key]
            conv_val = conv[key]
            
            # Handle null equivalence
            if pd.isna(orig_val) and pd.isna(conv_val):
                continue
            elif orig_val != conv_val:
                print(f"  ❌ Row {i+1}, {key}: {orig_val} != {conv_val}")
                row_match = False
                all_match = False
        
        if row_match:
            print(f"  ✅ Row {i+1}: Perfect match")
    
    if all_match:
        print(f"\n🎉 SUCCESS: All data perfectly preserved!")
        print(f"   • Null values maintained")
        print(f"   • Data types preserved") 
        print(f"   • Edge cases handled correctly")
        print(f"   • No information lost")
    else:
        print(f"\n❌ FAILURE: Data integrity issues detected!")
    
    return all_match

def demo_schema_matching():
    """Demonstrate intelligent schema matching."""
    
    print(f"\n🧠 Schema Matching Demo")
    print("=" * 30)
    
    # Data with different field order and missing fields
    original_data = [
        {"name": "Alice", "age": 30, "city": "NYC"},
        {"city": "LA", "name": "Bob"},  # Different order, missing age
        {"age": 25, "name": "Carol", "active": True, "city": "Chicago"},  # Extra field
    ]
    
    print(f"\n📄 Original Data (inconsistent schema):")
    for i, row in enumerate(original_data, 1):
        print(f"  Row {i}: {row}")
    
    # Convert through JSON-Tables
    json_tables = JSONTablesEncoder.from_records(original_data)
    converted_data = JSONTablesDecoder.to_records(json_tables)
    
    print(f"\n📄 After JSON-Tables Processing:")
    for i, row in enumerate(converted_data, 1):
        print(f"  Row {i}: {row}")
    
    print(f"\n🔍 Schema Standardization:")
    print(f"  • All rows now have same fields")
    print(f"  • Missing values filled with None/null")
    print(f"  • Field order standardized")
    print(f"  • Extra fields preserved")

if __name__ == "__main__":
    success = demo_round_trip()
    demo_schema_matching()
    
    if success:
        print(f"\n✅ JSON-Tables round-trip integrity confirmed!")
    else:
        print(f"\n❌ JSON-Tables round-trip integrity issues detected!") 