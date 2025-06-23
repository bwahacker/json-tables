#!/usr/bin/env python3
"""
Demo: DataFrame ↔ JSON-Tables Conversion

Shows how easy it is to convert DataFrames to JSON-Tables and back.
"""

import pandas as pd
import numpy as np
from jsontables import df_to_jt, df_from_jt, to_json_table, from_json_table

def demo_basic_dataframe_conversion():
    """Demonstrate basic DataFrame conversion."""
    print("🔄 Basic DataFrame ↔ JSON-Tables Demo")
    print("=" * 45)
    
    # Create sample DataFrame
    df_original = pd.DataFrame({
        'name': ['Alice', 'Bob', 'Carol'],
        'age': [30, 25, 35],
        'score': [95.5, 87.2, 92.8],
        'active': [True, False, True]
    })
    
    print("📊 Original DataFrame:")
    print(df_original)
    print(f"Original dtypes:\n{df_original.dtypes}")
    
    # Convert to JSON-Tables
    json_table = df_to_jt(df_original)
    print(f"\n📋 JSON-Tables format:")
    print(f"  Columns: {json_table['cols']}")
    print(f"  Rows: {len(json_table['row_data'])}")
    print(f"  Has numpy metadata: {'_numpy_metadata' in json_table}")
    
    # Convert back to DataFrame
    df_restored = df_from_jt(json_table)
    print(f"\n📊 Restored DataFrame:")
    print(df_restored)
    print(f"Restored dtypes:\n{df_restored.dtypes}")
    
    # Check if data is preserved
    print(f"\n🔍 Data integrity check:")
    print(f"  Shape preserved: {df_original.shape == df_restored.shape}")
    print(f"  Columns preserved: {list(df_original.columns) == list(df_restored.columns)}")

def demo_numpy_dataframe_conversion():
    """Demonstrate DataFrame with numpy types conversion."""
    try:
        import numpy as np
    except ImportError:
        print("❌ Numpy not available - skipping numpy demo")
        return
    
    print("\n🔢 Numpy DataFrame ↔ JSON-Tables Demo")
    print("=" * 45)
    
    # Create DataFrame with numpy types and NaN values
    df_original = pd.DataFrame({
        'int_col': pd.array([1, 2, None], dtype='Int64'),
        'float_col': [1.1, np.nan, 3.3],
        'bool_col': pd.array([True, False, None], dtype='boolean'),
        'str_col': ['a', None, 'c']
    })
    
    print("📊 Original DataFrame with numpy types:")
    print(df_original)
    print(f"Original dtypes:\n{df_original.dtypes}")
    
    # Convert with automatic numpy handling
    json_table = df_to_jt(df_original, auto_numpy=True)
    df_restored = df_from_jt(json_table, auto_numpy=True)
    
    print(f"\n📊 After conversion:")
    print(df_restored)
    print(f"Restored dtypes:\n{df_restored.dtypes}")
    
    # Check data preservation
    print(f"\n🔍 Data Comparison:")
    print(f"  Original shape: {df_original.shape}")
    print(f"  Restored shape: {df_restored.shape}")
    print(f"  Columns match: {list(df_original.columns) == list(df_restored.columns)}")

def demo_manual_vs_convenience():
    """Demonstrate manual vs convenience function usage."""
    print("\n🔧 Manual vs Convenience Functions Demo")
    print("=" * 45)
    
    # Create sample DataFrame
    df = pd.DataFrame({
        'product': ['Widget A', 'Widget B', 'Widget C'],
        'category': ['Electronics', 'Electronics', 'Home'],
        'price': [19.99, 29.99, 9.99],
        'in_stock': [True, False, True]
    })
    
    print("📊 Original DataFrame:")
    print(df)
    
    print(f"\n🔧 Method 1: Convenience functions")
    json_table1 = df_to_jt(df)
    df_restored1 = df_from_jt(json_table1)
    print(f"  ✅ df_to_jt() → df_from_jt(): {df.shape} → {df_restored1.shape}")
    
    print(f"\n🔧 Method 2: Generic functions")
    json_table2 = to_json_table(df)
    df_restored2 = from_json_table(json_table2, as_dataframe=True)
    print(f"  ✅ to_json_table() → from_json_table(): {df.shape} → {df_restored2.shape}")
    
    print(f"\n🔧 Method 3: Get records instead")
    records = from_json_table(json_table2, as_dataframe=False)
    print(f"  ✅ from_json_table(as_dataframe=False): {len(records)} records")
    for i, record in enumerate(records):
        print(f"    Row {i}: {record}")

def demo_performance():
    """Demonstrate performance with larger DataFrame."""
    print("\n⚡ Performance Demo")
    print("=" * 45)
    
    # Create larger DataFrame
    import time
    
    n_rows = 10000
    df_large = pd.DataFrame({
        'id': [f"ID_{i:06d}" for i in range(n_rows)],
        'category': [f"Cat_{i % 10}" for i in range(n_rows)],
        'value': np.random.normal(100, 15, n_rows),
        'flag': np.random.choice([True, False], n_rows)
    })
    
    print(f"📊 Large DataFrame: {df_large.shape}")
    print(f"Memory usage: {df_large.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Time the conversion
    start_time = time.time()
    json_table = df_to_jt(df_large)
    mid_time = time.time()
    df_restored = df_from_jt(json_table)
    end_time = time.time()
    
    encode_time = (mid_time - start_time) * 1000
    decode_time = (end_time - mid_time) * 1000
    total_time = (end_time - start_time) * 1000
    
    print(f"\n⏱️  Performance results:")
    print(f"  df_to_jt():   {encode_time:.2f} ms ({n_rows / (mid_time - start_time):.0f} rows/sec)")
    print(f"  df_from_jt(): {decode_time:.2f} ms ({n_rows / (end_time - mid_time):.0f} rows/sec)")
    print(f"  Total:        {total_time:.2f} ms ({n_rows / (end_time - start_time):.0f} rows/sec)")
    
    # Verify data integrity
    print(f"\n🔍 Data integrity check:")
    print(f"  Shape preserved: {df_large.shape == df_restored.shape}")
    print(f"  Columns preserved: {list(df_large.columns) == list(df_restored.columns)}")

if __name__ == "__main__":
    demo_basic_dataframe_conversion()
    demo_numpy_dataframe_conversion()
    demo_manual_vs_convenience()
    demo_performance()
    
    print(f"\n🎉 DataFrame conversion demos complete!")
    print(f"💡 Use df_to_jt(df) and df_from_jt(json_table) for DataFrame conversions!") 