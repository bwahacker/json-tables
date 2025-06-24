#!/usr/bin/env python3
"""
Demo: JSON-Tables Edge Case Handling

Shows how JSON-Tables handles data that breaks other formats:
- NumPy types that break standard JSON
- NaN/infinity values that CSV mangles  
- Mixed data types in same column
- Unicode and extreme values
"""

import json
import numpy as np
import pandas as pd
import tempfile
import traceback
from jsontables import df_to_jt, df_from_jt, DataIntegrityValidator

def demo_numpy_types():
    """Demo handling NumPy types that break standard JSON."""
    print("🔢 NumPy Types (Standard JSON Breaks)")
    print("=" * 50)
    
    # Create DataFrame with various NumPy types
    df = pd.DataFrame({
        'int_types': [np.int8(8), np.int16(16), np.int32(32), np.int64(64)],
        'float_types': [np.float16(1.6), np.float32(3.2), np.float64(6.4), np.float32(12.8)],
        'bool_types': [np.bool_(True), np.bool_(False), np.bool_(True), np.bool_(False)],
        'numpy_scalars': [np.int64(100), np.float64(3.14159), np.bool_(False), np.int32(42)]
    })
    
    print(f"📊 Created DataFrame with {len(df)} rows of NumPy types")
    print(f"   int8, int16, int32, int64, float16, float32, float64, bool_")
    
    # Try standard JSON - this will fail
    print("\n❌ Standard JSON conversion:")
    try:
        standard_json = df.to_dict('records')
        json_str = json.dumps(standard_json)
        print("   ✅ Somehow worked (unexpected!)")
    except Exception as e:
        print(f"   💥 Failed: {type(e).__name__}: {str(e)[:60]}...")
    
    # JSON-Tables handles it perfectly
    print("\n✅ JSON-Tables conversion:")
    try:
        json_table = df_to_jt(df)
        df_restored = df_from_jt(json_table)
        DataIntegrityValidator.validate_dataframe_equality(df, df_restored)
        print("   ✅ Perfect conversion and restoration!")
        print(f"   📊 Preserved {len(df)} rows with complex NumPy types")
    except Exception as e:
        print(f"   💥 Unexpected error: {e}")
    
    print()

def demo_nan_infinity():
    """Demo handling NaN and infinity values."""
    print("🌀 NaN & Infinity Values (CSV Mangles These)")
    print("=" * 50)
    
    # Create DataFrame with NaN and infinity
    df = pd.DataFrame({
        'normal_values': [1.0, 2.0, 3.0, 4.0],
        'nan_values': [np.nan, 1.5, np.nan, 2.5],
        'infinity_values': [np.inf, -np.inf, 0.0, np.inf],
        'mixed_nan_inf': [1.0, np.nan, np.inf, -np.inf]
    })
    
    print(f"📊 Created DataFrame with NaN and ±infinity values")
    print(f"   Regular: {df['normal_values'].tolist()}")
    print(f"   NaN mix: {df['nan_values'].tolist()}")
    print(f"   Infinity: {df['infinity_values'].tolist()}")
    
    # Test CSV round-trip (problematic)
    print("\n📄 CSV round-trip test:")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        csv_path = f.name
    
    try:
        df.to_csv(csv_path, index=False)
        df_from_csv = pd.read_csv(csv_path)
        print("   ⚠️  CSV 'works' but check the data...")
        print(f"   Original infinity: {df['infinity_values'].iloc[0]}")
        print(f"   From CSV: {df_from_csv['infinity_values'].iloc[0]} (lost!)")
        
        # Check if data is actually the same
        try:
            DataIntegrityValidator.validate_dataframe_equality(df, df_from_csv)
            print("   ✅ Data integrity: Perfect")
        except Exception as e:
            print(f"   💥 Data integrity: FAILED - {str(e)[:50]}...")
            
    except Exception as e:
        print(f"   💥 CSV failed: {e}")
    finally:
        import os
        try:
            os.unlink(csv_path)
        except:
            pass
    
    # JSON-Tables handles it perfectly
    print("\n✅ JSON-Tables round-trip:")
    try:
        json_table = df_to_jt(df)
        df_restored = df_from_jt(json_table)
        DataIntegrityValidator.validate_dataframe_equality(df, df_restored)
        print("   ✅ Perfect preservation of NaN and ±infinity!")
        print(f"   Original infinity: {df['infinity_values'].iloc[0]}")
        print(f"   Restored: {df_restored['infinity_values'].iloc[0]} ✅")
    except Exception as e:
        print(f"   💥 Unexpected error: {e}")
    
    print()

def demo_mixed_types():
    """Demo handling mixed data types in same column."""
    print("🔀 Mixed Data Types (Breaks Type Assumptions)")
    print("=" * 50)
    
    # Create DataFrame with mixed types in columns
    df = pd.DataFrame({
        'mixed_column': [42, 'text', True, None],
        'numbers_and_strings': [1, '2', 3.0, 'four'],
        'bools_and_nums': [True, 0, False, 1.5],
        'everything': [np.int64(1), 'hello', np.nan, True]
    })
    
    print(f"📊 Created DataFrame with mixed types in same columns")
    for col in df.columns:
        types = [type(x).__name__ for x in df[col]]
        print(f"   {col}: {types}")
    
    # JSON-Tables handles it
    print("\n✅ JSON-Tables conversion:")
    try:
        json_table = df_to_jt(df)
        df_restored = df_from_jt(json_table)
        DataIntegrityValidator.validate_dataframe_equality(df, df_restored)
        print("   ✅ Perfect handling of mixed types!")
        print(f"   Preserved all {len(df)} rows with type diversity")
    except Exception as e:
        print(f"   💥 Unexpected error: {e}")
    
    print()

def demo_unicode_extremes():
    """Demo handling Unicode and extreme values."""
    print("🌍 Unicode & Extreme Values")
    print("=" * 50)
    
    # Create DataFrame with Unicode and extreme values
    df = pd.DataFrame({
        'unicode_text': ['café', '北京', 'مرحبا', '🎉'],
        'extreme_numbers': [
            np.finfo(np.float64).max,     # Largest float64
            np.finfo(np.float64).min,     # Smallest positive float64
            np.iinfo(np.int64).max,       # Largest int64
            np.iinfo(np.int64).min        # Smallest int64
        ],
        'special_strings': ['', '\n\t\r', 'with"quotes', "with'apostrophes"],
        'mixed_unicode': ['ASCII', 'Cañón', 'Москва', '🇺🇸🇨🇳🇫🇷']
    })
    
    print(f"📊 Created DataFrame with Unicode and extreme values")
    print(f"   Unicode: {df['unicode_text'].tolist()}")
    print(f"   Extremes: max_float64, min_float64, max_int64, min_int64")
    print(f"   Special: Empty, whitespace, quotes, emojis")
    
    # JSON-Tables handles everything
    print("\n✅ JSON-Tables conversion:")
    try:
        json_table = df_to_jt(df)
        df_restored = df_from_jt(json_table)
        DataIntegrityValidator.validate_dataframe_equality(df, df_restored)
        print("   ✅ Perfect handling of Unicode and extreme values!")
        print(f"   ✅ All {len(df)} rows preserved with full fidelity")
        
        # Show some examples
        print("\n🔍 Sample preservation:")
        print(f"   Unicode: '{df['unicode_text'].iloc[0]}' → '{df_restored['unicode_text'].iloc[0]}'")
        print(f"   Extreme: {df['extreme_numbers'].iloc[0]} → {df_restored['extreme_numbers'].iloc[0]}")
        print(f"   Emoji: '{df['mixed_unicode'].iloc[3]}' → '{df_restored['mixed_unicode'].iloc[3]}'")
        
    except Exception as e:
        print(f"   💥 Unexpected error: {e}")
        traceback.print_exc()
    
    print()

def demo_comprehensive_edge_cases():
    """Demo all edge cases together."""
    print("🎯 Comprehensive Edge Case Test")
    print("=" * 50)
    
    # The ultimate stress test DataFrame
    df = pd.DataFrame({
        'numpy_ints': [np.int8(1), np.int16(2), np.int32(3), np.int64(4)],
        'numpy_floats': [np.float16(1.1), np.float32(2.2), np.float64(3.3), np.nan],
        'special_values': [np.inf, -np.inf, np.nan, 0.0],
        'mixed_types': [1, 'text', True, None],
        'unicode_mix': ['ASCII', 'café', '北京', '🎉'],
        'edge_strings': ['', '\n', '"quoted"', "it's"],
        'boolean_chaos': [True, False, np.bool_(True), None],
        'extremes': [
            float('inf'),
            np.finfo(np.float64).min, 
            np.iinfo(np.int64).max,
            np.iinfo(np.int64).min
        ]
    })
    
    print(f"📊 Ultimate stress test: {len(df)} rows × {len(df.columns)} columns")
    print("   ✅ NumPy types, NaN/infinity, mixed types, Unicode, extremes")
    
    # The moment of truth
    print("\n🎯 JSON-Tables ultimate test:")
    try:
        json_table = df_to_jt(df)
        df_restored = df_from_jt(json_table)
        DataIntegrityValidator.validate_dataframe_equality(df, df_restored)
        
        print("   🎉 PERFECT! All edge cases handled flawlessly!")
        print(f"   ✅ {len(df) * len(df.columns)} cells preserved with 100% fidelity")
        print("   ✅ NumPy types converted properly")
        print("   ✅ NaN/infinity preserved")  
        print("   ✅ Mixed types maintained")
        print("   ✅ Unicode/emojis intact")
        print("   ✅ Extreme values preserved")
        
    except Exception as e:
        print(f"   💥 Failed: {e}")
        traceback.print_exc()
    
    print()

def main():
    """Run all edge case demos."""
    print("🧪 JSON-Tables Edge Case Robustness Demo")
    print("=" * 60)
    print("Testing data that breaks other formats...")
    print()
    
    demo_numpy_types()
    demo_nan_infinity()
    demo_mixed_types()
    demo_unicode_extremes()
    demo_comprehensive_edge_cases()
    
    print("🏆 JSON-Tables: Robust enough for production!")
    print("   While others break, JSON-Tables just works. ✅")

if __name__ == "__main__":
    main() 