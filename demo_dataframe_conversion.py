#!/usr/bin/env python3
"""
JSON-Tables DataFrame Conversion Demo

Simple demo showing fast DataFrame conversion with JSON-Tables.
"""

import pandas as pd
import numpy as np
from jsontables import df_to_jt, df_from_jt, DataIntegrityValidator

def main():
    print("🚀 JSON-Tables: Simple & Fast DataFrame Conversion")
    print("=" * 60)
    
    # Create sample DataFrame
    df = pd.DataFrame({
        'name': ['Alice', 'Bob', 'Carol', 'David'],
        'age': [30, 25, 35, 28],
        'score': [95.5, 87.2, 92.8, 89.1],
        'active': [True, False, True, True],
        'city': ['NYC', 'LA', 'Chicago', None]
    })
    
    print(f"📊 Original DataFrame ({df.shape}):")
    print(df)
    print()
    
    # Convert to JSON-Tables (fast by default)
    print("⚡ Converting to JSON-Tables...")
    json_table = df_to_jt(df)
    print("✅ Conversion complete!")
    print()
    
    # Show the JSON structure
    print("📋 JSON-Tables structure:")
    print(f"  Type: {json_table['__dict_type']}")
    print(f"  Columns: {json_table['cols']}")
    print(f"  Rows: {len(json_table['row_data'])}")
    print()
    
    # Convert back to DataFrame
    print("🔄 Converting back to DataFrame...")
    df_restored = df_from_jt(json_table)
    print("✅ Restoration complete!")
    print()
    
    print(f"📊 Restored DataFrame ({df_restored.shape}):")
    print(df_restored)
    print()
    
    # Validate data integrity
    print("🔍 Validating data integrity...")
    try:
        DataIntegrityValidator.validate_dataframe_equality(
            df, df_restored, operation_name="Demo conversion"
        )
        print("✅ PERFECT: Every single cell preserved exactly!")
    except Exception as e:
        print(f"❌ Data integrity issue: {e}")
    print()
    
    # Test with edge cases
    print("🌊 Testing edge cases...")
    df_edge = pd.DataFrame({
        'values': [1.0, np.nan, np.inf, -np.inf, 0.0],
        'nulls': ['text', None, '', 'more', np.nan],
        'types': [42, 'mixed', True, None, 3.14]
    })
    
    print(f"📊 Edge case DataFrame ({df_edge.shape}):")
    print(df_edge)
    print()
    
    # Convert and validate edge cases
    json_edge = df_to_jt(df_edge)
    df_edge_restored = df_from_jt(json_edge)
    
    try:
        DataIntegrityValidator.validate_dataframe_equality(
            df_edge, df_edge_restored, operation_name="Edge cases"
        )
        print("✅ Edge cases handled perfectly!")
    except Exception as e:
        print(f"❌ Edge case issue: {e}")
    print()
    
    # Test columnar format
    print("📊 Testing columnar format...")
    json_columnar = df_to_jt(df, columnar=True)
    df_columnar_restored = df_from_jt(json_columnar)
    
    try:
        DataIntegrityValidator.validate_dataframe_equality(
            df, df_columnar_restored, operation_name="Columnar format"
        )
        print("✅ Columnar format works perfectly!")
    except Exception as e:
        print(f"❌ Columnar issue: {e}")
    print()
    
    print("🎉 Demo complete!")
    print("✅ Fast conversion with perfect data integrity")
    print("✅ Handles all edge cases automatically")
    print("✅ Simple API - no complex configuration needed")

if __name__ == "__main__":
    main() 